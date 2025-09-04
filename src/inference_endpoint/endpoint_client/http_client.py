"""HTTP endpoint client implementation with multiprocessing and ZMQ."""

import asyncio
import logging
from collections.abc import Callable

import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


class AsyncHTTPEndpointClient:
    """
    Future-based wrapper around HTTPEndpointClient.

    Provides both callback and future-based interfaces for maximum flexibility.
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        complete_callback: Callable | None = None,
    ):
        """
        Initialize the future-based client.

        Args:
            config: HTTP client configuration
            aiohttp_config: aiohttp configuration
            zmq_config: ZMQ configuration
            complete_callback: Optional user callback for responses
        """
        self.user_callback = complete_callback
        self._pending_futures: dict[str, asyncio.Future] = {}

        # Create underlying client with our internal callback
        self._client = HTTPEndpointClient(
            config=config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            complete_callback=self._handle_response,
        )

    async def send_request(self, query: Query) -> asyncio.Future[QueryResult]:
        """
        Send a request and return a future for the response.

        The returned future can be:
        - Awaited directly: `result = await client.send_request(query)`
        - Checked for completion: `if future.done(): result = future.result()`
        - Used with asyncio utilities: `done, pending = await asyncio.wait([future])`

        Args:
            query: Query to send

        Returns:
            asyncio.Future that will contain the QueryResult
        """
        # Create future for this request
        future = asyncio.get_event_loop().create_future()
        self._pending_futures[query.id] = future

        # Send request through underlying client
        await self._client.send_request(query)

        return future

    async def _handle_response(self, result: QueryResult) -> None:
        """
        Internal callback that completes futures and calls user callback.

        Args:
            result: Response from the endpoint
        """
        # Complete the future if pending
        future = self._pending_futures.pop(result.query_id, None)
        if future and not future.done():
            if result.error:
                # Create exception for errors
                future.set_exception(Exception(result.error))
            else:
                future.set_result(result)

        # Also call user callback if provided
        if self.user_callback:
            try:
                await self.user_callback(result)
            except Exception as e:
                logger.error(f"Error in user callback: {e}")

    async def start(self) -> None:
        """Start the underlying client."""
        await self._client.start()

    async def shutdown(self) -> None:
        """
        Shutdown the client and cancel pending futures.
        """
        # Cancel all pending futures
        for future in self._pending_futures.values():
            if not future.done():
                future.cancel()

        self._pending_futures.clear()

        # Shutdown underlying client
        await self._client.shutdown()


class HTTPEndpointClient:
    """HTTP implementation of the EndpointClient interface."""

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        complete_callback: Callable | None = None,
    ):
        """
        Initialize HTTP endpoint client.

        Args:
            config: HTTP client configuration
            aiohttp_config: aiohttp configuration
            zmq_config: ZMQ configuration
            complete_callback: Optional callback for completed requests
        """
        self.config = config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.complete_callback = complete_callback
        self.zmq_context = zmq.asyncio.Context(io_threads=zmq_config.zmq_io_threads)
        self.worker_push_sockets: list[ZMQPushSocket] = []

        self.worker_manager: WorkerManager | None = None
        self.current_worker_idx = 0

        self._shutdown_event = asyncio.Event()
        self._response_handler_task: asyncio.Task | None = None

        # Create concurrency semaphore if configured
        self._concurrency_semaphore = None
        if config.max_concurrency > 0:
            self._concurrency_semaphore = asyncio.Semaphore(config.max_concurrency)

    async def send_request(self, query: Query) -> None:
        """
        Send a query to the endpoint. Non-blocking, results delivered via callback.

        Args:
            query: Query object containing request details
        """
        # Apply concurrency limit if configured
        if self._concurrency_semaphore:
            async with self._concurrency_semaphore:
                await self._send_request_impl(query)
        else:
            await self._send_request_impl(query)

    async def _send_request_impl(self, query: Query) -> None:
        """Internal implementation of send request."""
        # Round-robin to next worker
        worker_idx = self.current_worker_idx
        self.current_worker_idx = (self.current_worker_idx + 1) % len(
            self.worker_push_sockets
        )

        # Send query directly to worker's queue (non-blocking)
        await self.worker_push_sockets[worker_idx].send(query)

    async def start(self) -> None:
        """Initialize client and start worker manager."""
        # Initialize worker push sockets
        for i in range(self.config.num_workers):
            address = f"{self.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(self.zmq_context, address, self.zmq_config)
            self.worker_push_sockets.append(push_socket)

        # Start worker manager
        self.worker_manager = WorkerManager(
            self.config, self.aiohttp_config, self.zmq_config, self.zmq_context
        )
        await self.worker_manager.initialize()

        # Start response handler
        self._response_handler_task = asyncio.create_task(self._handle_responses())

    async def _handle_responses(self) -> None:
        """Handle responses from workers."""
        response_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_response_queue_addr,
            self.zmq_config,
            bind=True,
        )

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Blocking receive with short timeout for shutdown check
                    response = await asyncio.wait_for(
                        response_socket.receive(), timeout=1.0
                    )

                    # Execute client callback if configured
                    if self.complete_callback:
                        await self.complete_callback(response)

                except TimeoutError:
                    # Check shutdown and continue
                    continue
                except Exception as e:
                    logger.error(f"Error handling response: {e}")

        finally:
            response_socket.close()

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        self._shutdown_event.set()

        # Cancel response handler
        if self._response_handler_task:
            self._response_handler_task.cancel()
            try:
                await self._response_handler_task
            except asyncio.CancelledError:
                pass

        # Close push sockets
        for socket in self.worker_push_sockets:
            socket.close()

        # Shutdown worker manager
        if self.worker_manager:
            await self.worker_manager.shutdown()

        # Close ZMQ context
        self.zmq_context.term()
