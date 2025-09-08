"""Worker process implementation for HTTP endpoint client."""

import asyncio
import logging
import os
import signal
from multiprocessing import Process

import aiohttp
import orjson
import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


def worker_main(
    worker_id: int,
    http_config: HTTPClientConfig,
    aiohttp_config: AioHttpConfig,
    zmq_config: ZMQConfig,
    request_queue_addr: str,
    response_queue_addr: str,
):
    """Entry point for worker process."""
    # Install uvloop which also enables it
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        logger.info("uvloop not available, using default event loop")

    # Create and run worker
    worker = Worker(
        worker_id=worker_id,
        http_config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
        request_socket_addr=request_queue_addr,
        response_socket_addr=response_queue_addr,
    )

    # Run event loop
    asyncio.run(worker.run())


class Worker:
    """Worker process that performs actual HTTP requests."""

    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        request_socket_addr: str,
        response_socket_addr: str,
    ):
        """Initialize worker with configurations and ZMQ addresses."""
        self.worker_id = worker_id
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.request_socket_addr = request_socket_addr
        self.response_socket_addr = response_socket_addr
        self._shutdown = False
        self._session: aiohttp.ClientSession | None = None
        self._zmq_context: zmq.asyncio.Context | None = None
        self._request_socket: ZMQPullSocket | None = None
        self._response_socket: ZMQPushSocket | None = None
        self.tcp_connector: aiohttp.TCPConnector | None = None

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        # Initialize ZMQ context and sockets
        self._zmq_context = zmq.asyncio.Context()
        self._request_socket = ZMQPullSocket(
            self._zmq_context, self.request_socket_addr, self.zmq_config, bind=True
        )
        self._response_socket = ZMQPushSocket(
            self._zmq_context, self.response_socket_addr, self.zmq_config
        )

        # Create TCP connector
        self.tcp_connector = self.aiohttp_config.create_tcp_connector()

        # Create aiohttp session with TCP connector
        self._session = aiohttp.ClientSession(
            connector=self.tcp_connector,
            timeout=aiohttp.ClientTimeout(
                total=self.aiohttp_config.client_timeout_total,
                connect=self.aiohttp_config.client_timeout_connect,
                sock_read=self.aiohttp_config.client_timeout_sock_read,
            ),
            connector_owner=self.aiohttp_config.client_session_connector_owner,
            skip_auto_headers=self.aiohttp_config.skip_auto_headers,
        )

        try:
            # Signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            logger.info(f"Worker {self.worker_id} started")

            # Main processing loop
            while not self._shutdown:
                try:
                    # Pull query from queue with timeout
                    query = await asyncio.wait_for(
                        self._request_socket.receive(), timeout=1.0
                    )

                    # Process query asynchronously
                    asyncio.create_task(self._process_request(query))

                except TimeoutError:
                    # Check shutdown and continue
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")

        finally:
            # Cleanup
            await self._cleanup()

    async def _send_error_response(self, query_id: str, error_message: str) -> None:
        """Helper to create and send error responses."""
        error_response = QueryResult(
            query_id=query_id,
            response_output=None,
            error=error_message,
        )
        await self._response_socket.send(error_response)

    async def _check_http_status(self, response, query_id: str) -> bool:
        """Check HTTP status and send error if not 200. Returns True if OK."""
        if response.status != 200:
            error_text = await response.text()
            await self._send_error_response(
                query_id, f"HTTP {response.status}: {error_text}"
            )
            return False
        return True

    async def _parse_json_response(
        self, response_text: str, query_id: str
    ) -> dict | None:
        """Parse JSON response, send error if invalid. Returns None on error."""
        try:
            return orjson.loads(response_text)
        except (ValueError, TypeError) as e:
            await self._send_error_response(
                query_id, f"Failed to parse response: {str(e)}"
            )
            return None

    async def _process_request(self, query: Query) -> None:
        """Process a single query."""
        try:
            if query.stream:
                await self._handle_streaming_request(query)
            else:
                await self._handle_non_streaming_request(query)

        except Exception as e:
            await self._send_error_response(query.id, str(e))

    async def _handle_streaming_request(self, query: Query) -> None:
        """Handle streaming response."""
        url = self.http_config.endpoint_url

        async with self._session.post(
            url,
            json=query.to_json(),
            headers=query.headers if hasattr(query, "headers") else {},
        ) as response:
            # Check for HTTP errors
            if not await self._check_http_status(response, query.id):
                return

            # Stream chunks
            accumulated_content = []
            first_chunk_sent = False

            # Read lines from the stream
            async for line_bytes in response.content:
                # Decode line and handle multiple lines in one chunk
                line_str = line_bytes.decode("utf-8")

                # Split by newlines in case multiple lines come in one chunk
                for line in line_str.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    # Parse SSE format (data: ...)
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = orjson.loads(data_str)

                            # For streaming, check for content in choices
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        accumulated_content.append(content)

                                        # Send only the first chunk as a streaming indicator
                                        if not first_chunk_sent:
                                            first_chunk_response = QueryResult(
                                                query_id=query.id,
                                                response_output=content,
                                                metadata={
                                                    "first_chunk": True,
                                                    "final_chunk": False,
                                                },
                                            )
                                            await self._response_socket.send(
                                                first_chunk_response
                                            )
                                            first_chunk_sent = True

                        except (ValueError, TypeError):
                            continue

            # Send final complete response
            final_response = QueryResult(
                query_id=query.id,
                response_output="".join(accumulated_content),
                metadata={"first_chunk": False, "final_chunk": True},
            )
            await self._response_socket.send(final_response)

    async def _handle_non_streaming_request(self, query: Query) -> None:
        """Handle non-streaming response."""
        url = self.http_config.endpoint_url

        async with self._session.post(
            url,
            json=query.to_json(),
            headers=query.headers if hasattr(query, "headers") else {},
        ) as response:
            response_text = await response.text()

            # Check HTTP status
            if not await self._check_http_status(response, query.id):
                return

            # Parse JSON response
            response_data = await self._parse_json_response(response_text, query.id)
            if response_data is None:
                return

            # Ensure the response has the query ID
            if "id" not in response_data:
                response_data["id"] = query.id

            response_obj = QueryResult.from_json(response_data)

            # Override query_id to ensure it matches our query
            response_obj.query_id = query.id

            await self._response_socket.send(response_obj)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}")
        self._shutdown = True

    async def _cleanup(self):
        """Clean up resources."""
        logger.info(f"Worker {self.worker_id} shutting down...")

        # Close TCP connector
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

        # Close aiohttp session
        if self._session:
            await self._session.close()

        # Close ZMQ sockets
        if self._request_socket:
            self._request_socket.close()
        if self._response_socket:
            self._response_socket.close()

        # Terminate ZMQ context
        if self._zmq_context:
            self._zmq_context.term()


class WorkerManager:
    """Manages the lifecycle of worker processes."""

    def __init__(
        self,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        zmq_context: zmq.asyncio.Context,
    ):
        """Initialize worker manager."""
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.zmq_context = zmq_context
        self.workers: list[Process] = []
        self.worker_pids: dict[int, int] = {}  # worker_id -> pid
        self._shutdown_event = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize workers and ZMQ infrastructure."""
        # Spawn worker processes
        for i in range(self.http_config.num_workers):
            worker = self._spawn_worker(i)
            self.workers.append(worker)
            self.worker_pids[i] = worker.pid

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_workers())

        # Wait for workers to be ready
        await asyncio.sleep(0.5)

    def _spawn_worker(self, worker_id: int) -> Process:
        """Spawn a single worker process."""
        request_queue_addr = (
            f"{self.zmq_config.zmq_request_queue_prefix}_{worker_id}_requests"
        )
        response_queue_addr = self.zmq_config.zmq_response_queue_addr

        # Create worker process
        process = Process(
            target=worker_main,
            args=(
                worker_id,
                self.http_config,
                self.aiohttp_config,
                self.zmq_config,
                request_queue_addr,
                response_queue_addr,
            ),
            daemon=False,
        )
        process.start()
        return process

    async def _monitor_workers(self) -> None:
        """Monitor worker health and restart if needed."""
        while not self._shutdown_event.is_set():
            for i, worker in enumerate(self.workers):
                if not worker.is_alive():
                    logger.warning(f"Worker {i} died, restarting...")
                    # Terminate zombie process
                    if worker.pid:
                        try:
                            os.kill(worker.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

                    # Spawn new worker
                    new_worker = self._spawn_worker(i)
                    self.workers[i] = new_worker
                    self.worker_pids[i] = new_worker.pid

            await asyncio.sleep(2.0)  # Check every 2 seconds

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers."""
        self._shutdown_event.set()

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Send SIGTERM to all workers
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()

        # Wait for graceful shutdown
        await asyncio.sleep(0.5)

        # Force kill any remaining workers
        for worker in self.workers:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1.0)
