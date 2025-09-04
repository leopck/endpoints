# HTTP Endpoint Client Design

## Overview

This document describes the design of the HTTP endpoint client for the MLPerf Inference Endpoint Benchmarking System. The client leverages:

- **aiohttp** for async HTTP requests
- **ZMQ Push/Pull** sockets for inter-process communication
- **uvloop** for high-performance async event loops in workers
- **Multiprocessing** for true parallelism

The client is designed to be a pluggable component implementing the abstract endpoint client interface defined in `endpoint_client/interface.py`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTPEndpointClient                           │
│              (implements EndpointClient ABC)                    │
│  ┌─────────────────┐                                           │
│  │  send_request    │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ├─────ZMQ PUSH (Query)────▶ Worker 1 Queue           │
│           ├─────ZMQ PUSH (Query)────▶ Worker 2 Queue           │
│           └─────ZMQ PUSH (Query)────▶ Worker N Queue           │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                 ZMQ PULL (per worker)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WorkerManager                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Worker 1   │  │   Worker 2   │  │   Worker N   │  ...     │
│  │   (uvloop)   │  │   (uvloop)   │  │   (uvloop)   │         │
│  │   aiohttp    │  │   aiohttp    │  │   aiohttp    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│              ZMQ PUSH (QueryResult)                              │
│                            ▼                                     │
│                    ┌────────────────┐                           │
│                    │ Response Queue  │ (Shared)                 │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                   ZMQ PULL (blocking)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Handler                              │
│                 (calls complete_callback)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. HTTPEndpointClient

**Purpose**: HTTP implementation of the abstract EndpointClient interface.

**Key Attributes**:

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""
    endpoint_url: str
    num_workers: int = 4
    max_concurrency: int = -1  # -1 means unlimited, otherwise limits concurrent requests via semaphore

@dataclass
class AioHttpConfig:
    """Configuration for aiohttp client session and connectors."""
    # ClientSession configs
    client_session_connector_owner: bool = True

    # TCPConnector configs
    tcp_connector_limit: int = 0  # 0 means unlimited (Operating Systems have TCP client port limits in practice)
    tcp_connector_ttl_dns_cache: int = 300
    tcp_connector_enable_cleanup_closed: bool = True
    tcp_connector_force_close: bool = False  # Keep connections alive
    tcp_connector_keepalive_timeout: int = 30
    tcp_connector_use_dns_cache: bool = True
    tcp_connector_enable_tcp_nodelay: bool = True  # Disable Nagle's algorithm

    # ClientTimeout configs
    client_timeout_total: float = None  # None means no timeout
    client_timeout_connect: float = 10.0
    client_timeout_sock_read: float = None  # None means no timeout

    # Streaming configs
    streaming_buffer_size: int = 64 * 1024  # 64KB buffer for streaming

@dataclass
class ZMQConfig:
    """Configuration for ZMQ sockets and communication."""
    zmq_io_threads: int = 4  # Number of ZMQ IO threads
    zmq_request_queue_prefix: str = "ipc:///tmp/http_worker"
    zmq_response_queue_addr: str = "ipc:///tmp/http_responses"
    zmq_high_water_mark: int = 1000 # max msg queue size
    zmq_linger: int = 0  # Don't block on close
    zmq_send_timeout: int = -1  # Non-blocking send
    zmq_recv_timeout: int = -1  # Blocking receive
    zmq_recv_buffer_size: int = 10 * 1024 * 1024  # 10MB receive buffer
    zmq_send_buffer_size: int = 10 * 1024 * 1024  # 10MB send buffer
```

**Key Methods**:

```python
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import zmq.asyncio
from inference_endpoint.core.types import Query, QueryResult

class HTTPEndpointClient:
    """HTTP implementation of the EndpointClient interface."""

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        complete_callback: Optional[Callable] = None
    ):
        self.config = config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.complete_callback = complete_callback
        self.zmq_context = zmq.asyncio.Context(io_threads=zmq_config.zmq_io_threads)
        self.worker_push_sockets: List[ZMQPushSocket] = []

        self.worker_manager: Optional[WorkerManager] = None
        self.current_worker_idx = 0

        self._shutdown_event = asyncio.Event()

        # Create concurrency semaphore if configured
        self._concurrency_semaphore = None
        if config.max_concurrency > 0:
            self._concurrency_semaphore = asyncio.Semaphore(config.max_concurrency)

    async def send_request(
        self,
        query: Query
    ) -> None:
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
        self.current_worker_idx = (self.current_worker_idx + 1) % len(self.worker_push_sockets)

        # Send query directly to worker's queue (non-blocking)
        await self.worker_push_sockets[worker_idx].send(query)

    async def start(self) -> None:
        """Initialize client and start worker manager"""
        # Initialize worker push sockets
        for i in range(self.config.num_workers):
            address = f"{self.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(self.zmq_context, address, self.zmq_config)
            self.worker_push_sockets.append(push_socket)

        # Start worker manager
        self.worker_manager = WorkerManager(
            self.config,
            self.aiohttp_config,
            self.zmq_config,
            self.zmq_context
        )
        await self.worker_manager.initialize()

        # Start response handler
        asyncio.create_task(self._handle_responses())

    async def _handle_responses(self) -> None:
        """Handle responses from workers"""
        response_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_response_queue_addr,
            self.zmq_config,
            bind=True
        )

        try:
            while not self._shutdown_event.is_set():
                # Blocking receive - no timeout needed
                response = await response_socket.receive()

                # Execute client callback if configured
                if self.complete_callback:
                    await self.complete_callback(response)

        finally:
            response_socket.close()

    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        self._shutdown_event.set()

        # Close push sockets
        for socket in self.worker_push_sockets:
            socket.close()

        # Shutdown worker manager
        if self.worker_manager:
            await self.worker_manager.shutdown()

        # Close ZMQ context
        self.zmq_context.term()
```

### 2. WorkerManager

**Purpose**: Manages the lifecycle of worker processes and coordinates IPC.

**Key Responsibilities**:

- Initialize and manage worker processes
- Set up ZMQ Push/Pull sockets for request distribution
- Set up ZMQ Push/Pull sockets for response collection
- Handle worker health monitoring and restart
- Manage graceful shutdown

**Key Methods**:

```python
import multiprocessing as mp
from multiprocessing import Process
import signal
import os
from typing import List, Dict

class WorkerManager:
    def __init__(
        self,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        zmq_context: zmq.asyncio.Context
    ):
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.zmq_context = zmq_context
        self.workers: List[Process] = []
        self.worker_pids: Dict[int, int] = {}  # worker_id -> pid
        self._shutdown_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize workers and ZMQ infrastructure"""
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
        """Spawn a single worker process"""
        request_queue_addr = f"{self.zmq_config.zmq_request_queue_prefix}_{worker_id}_requests"
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
                response_queue_addr
            ),
            daemon=False
        )
        process.start()
        return process

    async def _monitor_workers(self) -> None:
        """Monitor worker health and restart if needed"""
        while not self._shutdown_event.is_set():
            for i, worker in enumerate(self.workers):
                if not worker.is_alive():
                    print(f"Worker {i} died, restarting...")
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

            await asyncio.sleep(5.0)  # Check every 5 seconds

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers"""
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


def worker_main(
    worker_id: int,
    http_config: HTTPClientConfig,
    aiohttp_config: AioHttpConfig,
    zmq_config: ZMQConfig,
    request_queue_addr: str,
    response_queue_addr: str
):
    """Entry point for worker process"""
    # Install uvloop
    import uvloop
    uvloop.install()

    # Create and run worker
    worker = Worker(
        worker_id=worker_id,
        http_config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
        request_socket_addr=request_queue_addr,
        response_socket_addr=response_queue_addr
    )

    # Run event loop
    asyncio.run(worker.run())
```

### 3. Worker

**Purpose**: Process that performs actual HTTP requests using aiohttp on uvloop.

**Key Features**:

- Runs on uvloop for maximum performance
- Pulls requests from shared ZMQ socket
- Supports both streaming and non-streaming requests
- Pushes responses back via ZMQ

**Streaming Behavior**:

- For streaming requests, the worker sends only two QueryResult messages:
  1. First chunk - `QueryResult` with `metadata["first_chunk"]: True` containing the first token
  2. Final response - `QueryResult` with `metadata["final_chunk"]: True` containing the complete accumulated output
- This minimizes inter-process communication while still providing streaming feedback

**Key Methods**:

```python
import aiohttp
import json
from typing import Optional

class Worker:
    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        request_socket_addr: str,
        response_socket_addr: str
    ):
        """Initialize worker with configurations and ZMQ addresses"""
        self.worker_id = worker_id
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.request_socket_addr = request_socket_addr
        self.response_socket_addr = response_socket_addr
        self._shutdown = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._zmq_context: Optional[zmq.asyncio.Context] = None
        self._request_socket: Optional[ZMQPullSocket] = None
        self._response_socket: Optional[ZMQPushSocket] = None

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses"""
        # Initialize ZMQ context and sockets
        self._zmq_context = zmq.asyncio.Context()
        self._request_socket = ZMQPullSocket(
            self._zmq_context,
            self.request_socket_addr,
            self.zmq_config,
            bind=True
        )
        self._response_socket = ZMQPushSocket(
            self._zmq_context,
            self.response_socket_addr,
            self.zmq_config
        )

        # Configure aiohttp session
        timeout = aiohttp.ClientTimeout(
            total=self.aiohttp_config.client_timeout_total,
            connect=self.aiohttp_config.client_timeout_connect,
            sock_read=self.aiohttp_config.client_timeout_sock_read
        )
        connector = aiohttp.TCPConnector(
            limit=self.aiohttp_config.tcp_connector_limit,
            ttl_dns_cache=self.aiohttp_config.tcp_connector_ttl_dns_cache,
            enable_cleanup_closed=self.aiohttp_config.tcp_connector_enable_cleanup_closed,
            force_close=self.aiohttp_config.tcp_connector_force_close,
            keepalive_timeout=self.aiohttp_config.tcp_connector_keepalive_timeout,
            use_dns_cache=self.aiohttp_config.tcp_connector_use_dns_cache,
            enable_tcp_nodelay=self.aiohttp_config.tcp_connector_enable_tcp_nodelay
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=self.aiohttp_config.client_session_connector_owner
        )

        try:
            # Signal handlers for graceful shutdown
            import signal
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            print(f"Worker {self.worker_id} started")

            # Main processing loop
            while not self._shutdown:
                try:
                    # Pull query from queue (blocking receive)
                    query = await self._request_socket.receive()

                    # Process query asynchronously
                    asyncio.create_task(self._process_request(query))

                except Exception as e:
                    print(f"Worker {self.worker_id} error: {e}")

        finally:
            # Cleanup
            await self._cleanup()

    async def _process_request(self, query: Query) -> None:
        """Process a single query"""
        try:
            if query.stream:
                await self._handle_streaming_request(query)
            else:
                await self._handle_non_streaming_request(query)

        except Exception as e:
            # Send error response
            error_response = QueryResult(
                query_id=query.id,
                response_output="",
                error=str(e)
            )
            await self._response_socket.send(error_response)

    async def _handle_streaming_request(
        self,
        query: Query
    ) -> None:
        """Handle streaming response"""
        url = self.http_config.endpoint_url

        try:
            async with self._session.post(
                url,
                json=query.to_json(),
                headers=query.headers
            ) as response:
                # Check for HTTP errors
                if response.status != 200:
                    error_text = await response.text()
                    error_response = QueryResult(
                        query_id=query.id,
                        response_output="",
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    await self._response_socket.send(error_response)
                    return

                # Stream chunks
                accumulated_content = []
                first_chunk_sent = False

                async for line in response.content:
                    # Decode line
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue

                    # Parse SSE format (data: ...)
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break

                        try:
                            chunk_data = json.loads(data_str)

                            # For streaming, we need to check if this is a partial message
                            # OpenAI streaming format has choices[0].delta.content
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        accumulated_content.append(content)

                                        # Send only the first chunk as a streaming indicator
                                        if not first_chunk_sent:
                                            first_chunk_response = QueryResult(
                                                query_id=query.id,
                                                response_output=content,
                                                metadata={"first_chunk": True, "final_chunk": False}
                                            )
                                            await self._response_socket.send(first_chunk_response)
                                            first_chunk_sent = True

                        except json.JSONDecodeError:
                            continue

                # Send final complete response
                final_response = QueryResult(
                    query_id=query.id,
                    response_output=''.join(accumulated_content),
                    metadata={"first_chunk": False, "final_chunk": True}
                )
                await self._response_socket.send(final_response)

        except Exception as e:
            raise

    async def _handle_non_streaming_request(
        self,
        query: Query
    ) -> None:
        """Handle non-streaming response"""
        url = self.http_config.endpoint_url

        async with self._session.post(
            url,
            json=query.to_json(),
            headers=query.headers
        ) as response:
            response_text = await response.text()

            if response.status != 200:
                # Send error response
                error_response = QueryResult(
                    query_id=query.id,
                    response_output="",
                    error=f"HTTP {response.status}: {response_text}"
                )
                await self._response_socket.send(error_response)
                return

            # Parse response using QueryResult.from_json
            try:
                response_data = json.loads(response_text)
                response_data['id'] = query.id
                response_obj = QueryResult.from_json(response_data)
                await self._response_socket.send(response_obj)

            except json.JSONDecodeError as e:
                # Send error response
                error_response = QueryResult(
                    query_id=query.id,
                    response_output="",
                    error=f"Failed to parse response: {str(e)}"
                )
                await self._response_socket.send(error_response)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        print(f"Worker {self.worker_id} received signal {signum}")
        self._shutdown = True

    async def _cleanup(self):
        """Clean up resources"""
        print(f"Worker {self.worker_id} shutting down...")

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
```

### 4. ZMQ Abstraction Layer

**Purpose**: Simplify ZMQ socket operations and provide clean APIs.

**Key Classes**:

```python
import zmq
import zmq.asyncio
import pickle
from typing import Any, Optional

class ZMQPushSocket:
    """Async wrapper for ZMQ PUSH socket"""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig):
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect(address)
        self.socket.setsockopt(zmq.SNDHWM, config.zmq_high_water_mark)
        self.socket.setsockopt(zmq.LINGER, config.zmq_linger)
        self.socket.setsockopt(zmq.SNDBUF, config.zmq_send_buffer_size)
        # Non-blocking send
        self.socket.setsockopt(zmq.SNDTIMEO, config.zmq_send_timeout)

    async def send(self, data: Any) -> None:
        """Serialize and send data through push socket (non-blocking)"""
        serialized = pickle.dumps(data)
        await self.socket.send(serialized, zmq.NOBLOCK)

    def close(self) -> None:
        """Close socket cleanly"""
        self.socket.close()

class ZMQPullSocket:
    """Async wrapper for ZMQ PULL socket"""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig, bind: bool = False):
        self.socket = context.socket(zmq.PULL)
        if bind:
            self.socket.bind(address)
        else:
            self.socket.connect(address)
        self.socket.setsockopt(zmq.RCVHWM, config.zmq_high_water_mark)
        self.socket.setsockopt(zmq.RCVBUF, config.zmq_recv_buffer_size)
        # Blocking receive (no timeout)
        self.socket.setsockopt(zmq.RCVTIMEO, config.zmq_recv_timeout)

    async def receive(self) -> Any:
        """Receive and deserialize data from pull socket (blocking)"""
        serialized = await self.socket.recv()
        return pickle.loads(serialized)

    def close(self) -> None:
        """Close socket cleanly"""
        self.socket.close()
```

### 5. AsyncHTTPEndpointClient (Wrapper)

**Purpose**: Provides a future-based interface on top of HTTPEndpointClient for more flexible usage patterns.

**Key Features**:

- Returns asyncio.Future objects from send_request
- Maintains mapping of query_id to futures
- Supports both callback and await patterns
- Thin wrapper with minimal overhead

**Implementation**:

```python
import asyncio
from typing import Dict, Optional, Callable
from inference_endpoint.core.types import Query, QueryResult

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
        complete_callback: Optional[Callable] = None
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
        self._pending_futures: Dict[str, asyncio.Future] = {}

        # Create underlying client with our internal callback
        self._client = HTTPEndpointClient(
            config=config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            complete_callback=self._handle_response
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
                print(f"Error in user callback: {e}")

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
```

**Usage Patterns**:

1. **Future-based (await pattern)**:

```python
client = AsyncHTTPEndpointClient(config, aiohttp_config, zmq_config)
await client.start()

# Send and await response
query = ChatCompletionQuery(prompt="Hello")
try:
    result = await client.send_request(query)
    print(f"Response: {result.response_output}")
except Exception as e:
    print(f"Error: {e}")
```

2. **Future-based (non-blocking check)**:

```python
# Send multiple requests
futures = []
for i in range(10):
    query = ChatCompletionQuery(prompt=f"Query {i}")
    future = await client.send_request(query)
    futures.append(future)

# Check completed futures
for future in futures:
    if future.done():
        try:
            result = future.result()
            print(f"Got result: {result.response_output}")
        except Exception as e:
            print(f"Got error: {e}")
```

3. **Callback-based (backwards compatible)**:

```python
async def handle_response(result: QueryResult):
    print(f"Callback received: {result.response_output}")

client = AsyncHTTPEndpointClient(
    config,
    aiohttp_config,
    zmq_config,
    complete_callback=handle_response
)
await client.start()

# Send request - both future and callback will work
future = await client.send_request(query)
# Can still await the future even with callback
result = await future
```

4. **Mixed usage with asyncio utilities**:

```python
# Send multiple requests and wait for first completion
queries = [ChatCompletionQuery(prompt=f"Q{i}") for i in range(5)]
futures = [await client.send_request(q) for q in queries]

# Wait for first to complete
done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
first_result = done.pop().result()

# Cancel remaining
for future in pending:
    future.cancel()
```

## Data Models

The client uses data models defined in `core/types.py`:

- `Query`: Base class for requests (e.g., `ChatCompletionQuery`)
- `QueryResult`: For all responses (both streaming and non-streaming)
  - Uses `metadata["first_chunk"]: True` to indicate first streaming chunk
  - Uses `metadata["final_chunk"]: True` to indicate final complete response
  - Non-streaming responses have neither flag set

Query objects are passed directly through ZMQ to workers, and QueryResult objects are sent back to the main process.

## Usage Examples

### Basic Usage with AsyncHTTPEndpointClient

```python
import asyncio
from typing import Union
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult

# Initialize configurations
http_config = HTTPClientConfig(
    endpoint_url="https://api.openai.com/v1",
    num_workers=4
)

aiohttp_config = AioHttpConfig()  # Use defaults
zmq_config = ZMQConfig()  # Use defaults

# Example 1: Future-based usage (recommended)
client = AsyncHTTPEndpointClient(http_config, aiohttp_config, zmq_config)
await client.start()

# Create a query
query = ChatCompletionQuery(
    id="req-001",
    model="gpt-4",
    prompt="What is the capital of France?",
    stream=False
)

# Send request and await response
try:
    result = await client.send_request(query)
    print(f"Response: {result.response_output}")
except Exception as e:
    print(f"Error: {e}")

await client.shutdown()

# Example 2: Callback-based usage (backward compatible)
async def handle_response(response: QueryResult):
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Response: {response.response_output}")

client = AsyncHTTPEndpointClient(
    http_config,
    aiohttp_config,
    zmq_config,
    complete_callback=handle_response
)
await client.start()

# Send request - both callback and future work
future = await client.send_request(query)
# Can still await even with callback
result = await future

await client.shutdown()

# Example 3: Mixed async patterns
client = AsyncHTTPEndpointClient(http_config, aiohttp_config, zmq_config)
await client.start()

# Send multiple queries
queries = [
    ChatCompletionQuery(prompt=f"Question {i}", model="gpt-4")
    for i in range(5)
]

# Collect futures
futures = []
for query in queries:
    future = await client.send_request(query)
    futures.append(future)

# Wait for all to complete with timeout
done, pending = await asyncio.wait(futures, timeout=10.0)

# Process completed
for future in done:
    result = future.result()
    print(f"Completed: {result.response_output}")

# Cancel pending
for future in pending:
    future.cancel()

await client.shutdown()
```
