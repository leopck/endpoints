# HTTP Endpoint Client

A high-performance HTTP client for the MLPerf Inference Endpoint Benchmarking System that leverages multiprocessing, async I/O, and ZMQ for efficient request handling.

## Features

- **High Performance**: Uses multiple worker processes with uvloop for maximum throughput
- **Async Support**: Built on aiohttp with full async/await support
- **Flexible API**: Provides both future-based and callback-based interfaces
- **Streaming Support**: Handles both streaming and non-streaming responses
- **Robust**: Includes worker health monitoring and automatic restart
- **Configurable**: Extensive configuration options for tuning performance

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTPEndpointClient                           │
│              (implements EndpointClient ABC)                    │
│  ┌─────────────────┐                                            │
│  │  issue_query    │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ├─────ZMQ PUSH (Query)────▶ Worker 1 Queue            │
│           ├─────ZMQ PUSH (Query)────▶ Worker 2 Queue            │
│           └─────ZMQ PUSH (Query)────▶ Worker N Queue            │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                 ZMQ PULL (per worker)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WorkerManager                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Worker 1   │  │   Worker 2   │  │   Worker N   │  ...      │
│  │   (uvloop)   │  │   (uvloop)   │  │   (uvloop)   │           │
│  │   aiohttp    │  │   aiohttp    │  │   aiohttp    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│                    ZMQ PUSH (QueryResult)                       │
│                            ▼                                    │
│                    ┌────────────────┐                           │
│                    │ Response Queue │ (Shared)                  │
│                    └────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                    ZMQ PULL (blocking)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Handler                             │
│                 (calls complete_callback)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Required dependencies
pip install aiohttp zmq orjson

# Optional for better performance
pip install uvloop
```

## Configuration

### HTTPClientConfig

Main configuration for the HTTP client:

```python
from inference_endpoint.endpoint_client import HTTPClientConfig

config = HTTPClientConfig(
    endpoint_url="https://api.openai.com/v1/chat/completions",
    num_workers=4,  # Number of worker processes
    max_concurrency=-1  # -1 for unlimited, or positive int to limit concurrent requests
)
```

### AioHttpConfig

Configuration for aiohttp session and TCP connections:

```python
from inference_endpoint.endpoint_client import AioHttpConfig

aiohttp_config = AioHttpConfig(
    # Timeout settings
    client_timeout_total=None,  # Total request timeout (None = no timeout)
    client_timeout_connect=None,  # Connection timeout
    client_timeout_sock_read=None,  # Socket read timeout

    # TCP Connector settings
    tcp_connector_limit=0,  # 0 = unlimited connections
    tcp_connector_limit_per_host=0,  # 0 = unlimited per host
    tcp_connector_keepalive_timeout=30,  # Keep connections alive for 30s
    tcp_connector_force_close=False,  # Reuse connections
    tcp_connector_enable_tcp_nodelay=True,  # Disable Nagle's algorithm

    # DNS caching
    tcp_connector_use_dns_cache=True,
    tcp_connector_ttl_dns_cache=300,  # Cache DNS for 5 minutes

    # Streaming
    streaming_buffer_size=64 * 1024  # 64KB buffer for streaming
)
```

### ZMQConfig

Configuration for ZMQ inter-process communication:

```python
from inference_endpoint.endpoint_client import ZMQConfig

zmq_config = ZMQConfig(
    zmq_io_threads=4,  # Number of ZMQ I/O threads
    zmq_high_water_mark=10_000,  # Max message queue size
    zmq_recv_buffer_size=10 * 1024 * 1024,  # 10MB receive buffer
    zmq_send_buffer_size=10 * 1024 * 1024,  # 10MB send buffer
    # Socket addresses are auto-generated if not specified
)
```

## Usage Examples

### Basic Usage

```python
import asyncio
from inference_endpoint.endpoint_client import (
    HTTPEndpointClient,
    HTTPClientConfig,
    AioHttpConfig,
    ZMQConfig
)
from inference_endpoint.core.types import ChatCompletionQuery

async def main():
    # Initialize configurations
    http_config = HTTPClientConfig(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        num_workers=4
    )
    aiohttp_config = AioHttpConfig()  # Use defaults
    zmq_config = ZMQConfig()  # Use defaults

    # Create client
    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    await client.start()

    # Create a query
    query = ChatCompletionQuery(
        id="req-001",
        model="gpt-4",
        prompt="What is the capital of France?",
        stream=False
    )

    # Send request and get future immediately
    future = client.issue_query(query)

    # Await the result
    try:
        result = await future
        print(f"Response: {result.response_output}")
    except Exception as e:
        print(f"Error: {e}")

    await client.shutdown()

asyncio.run(main())
```

### Callback-Based Usage

```python
from inference_endpoint.core.types import QueryResult, StreamChunk

def handle_response(response):
    """Callback function for handling responses (synchronous)."""
    match response:
        case QueryResult(error=error) if error:
            print(f"Error: {error}")
        case QueryResult(response_output=output):
            print(f"Response: {output}")
        case StreamChunk(response_chunk=chunk):
            print(f"Chunk: {chunk}")

async def main():
    # Create client with callback
    client = HTTPEndpointClient(
        http_config,
        aiohttp_config,
        zmq_config,
        complete_callback=handle_response
    )
    await client.start()

    # Send request - both callback and future work
    future = client.issue_query(query)

    # Can still await even with callback
    result = await future

    await client.shutdown()
```

Note: The callback is synchronous (not async) and receives both `StreamChunk` and `QueryResult` messages for streaming responses. Use pattern matching with `match` to handle different message types cleanly.

### Multiple Concurrent Requests

```python
async def main():
    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    await client.start()

    # Send multiple queries
    queries = [
        ChatCompletionQuery(
            id=f"req-{i}",
            prompt=f"Question {i}",
            model="gpt-4"
        )
        for i in range(10)
    ]

    # Collect futures
    futures = []
    for query in queries:
        future = client.issue_query(query)
        futures.append(future)

    # Wait for all to complete with timeout
    done, pending = await asyncio.wait(futures, timeout=30.0)

    # Process completed
    for future in done:
        result = future.result()
        print(f"Completed: {result.response_output}")

    # Cancel pending
    for future in pending:
        future.cancel()

    await client.shutdown()
```

### Streaming Requests

```python
async def main():
    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    await client.start()

    # Create streaming query
    query = ChatCompletionQuery(
        id="stream-001",
        model="gpt-4",
        prompt="Write a story about a robot",
        stream=True  # Enable streaming
    )

    # Send request - returns StreamingFuture for streaming queries
    future = client.issue_query(query)

    # Option 1: Just wait for complete response (same as non-streaming)
    result = await future
    print(f"Complete response: {result.response_output}")

    # Option 2: Access first chunk early
    first_chunk = await future.first
    print(f"AI started with: {first_chunk}")
    # ... do other work while waiting ...
    full_result = await future
    print(f"Complete: {full_result.response_output}")

    await client.shutdown()
```

### Advanced Streaming - Mixed Example

```python
async def process_with_early_feedback(query):
    """Show user immediate feedback when AI starts responding."""
    future = client.issue_query(query)

    # Get first chunk as soon as available
    first_chunk = await future.first
    update_ui(f"AI started responding: {first_chunk}...")

    # Continue processing while waiting for full response
    full_response = await future
    update_ui(f"Complete: {full_response.response_output}")

# Note: If the first chunk is marked as complete by the server (finish_reason is set),
# StreamChunk.is_complete will be True.
#
# Note: For empty streaming responses (no chunks), only a QueryResult is sent with
# metadata={"is_first": True} to indicate no chunks were produced.

# Racing multiple queries to same model
async def process_multiple_queries(model: str = "gpt-4"):
    """Send multiple queries to same model, wait for first chunk then all completions."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming",
        "What are the benefits of async programming?"
    ]

    queries = [
        ChatCompletionQuery(id=f"query-{i}", prompt=prompt, model=model, stream=True)
        for i, prompt in enumerate(prompts)
    ]

    # Issue all queries
    futures = [client.issue_query(q) for q in queries]

    # Race for the very first chunk from any query
    first_chunks = [f.first for f in futures]
    done, pending = await asyncio.wait(first_chunks, return_when=asyncio.FIRST_COMPLETED)

    # Print the first chunk that arrives
    first_task = done.pop()
    first_chunk = first_task.result()
    # Find which query it came from
    first_query_idx = first_chunks.index(first_task)
    print(f"First chunk received from query-{first_query_idx}: '{first_chunk}'")

    # Now wait for all queries to complete
    print("\nWaiting for all queries to complete...")
    all_results = await asyncio.gather(*futures)

    # Print all completed responses
    for result in all_results:
        print(f"\nQuery {result.query_id} completed:")
        print(f"  Prompt: {prompts[int(result.query_id.split('-')[1])]}")
        print(f"  Response: {result.response_output[:100]}...")
```

### Controlling runtime max-concurrency

```python
async def main():
    # Limit to 10 concurrent requests
    http_config = HTTPClientConfig(
        endpoint_url="https://api.example.com",
        num_workers=4,
        max_concurrency=10  # Limit concurrent requests
    )

    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    await client.start()

    # Send 100 requests - only 10 will run concurrently
    futures = []
    for i in range(100):
        query = ChatCompletionQuery(id=f"req-{i}", prompt=f"Query {i}")
        futures.append(client.issue_query(query))

    # Wait for all
    results = await asyncio.gather(*futures)

    await client.shutdown()
```

## Performance Tuning

### Worker Count

- Set `num_workers` based on your workload and available CPU cores
- More workers = higher throughput but also higher resource usage
- Recommended: Start with `num_workers = CPU cores` and adjust based on benchmarks

### Concurrency Limiting

- Use `max_concurrency` to prevent overwhelming the endpoint
- This is especially important when the endpoint has rate limits

## API Reference

### HTTPEndpointClient

```python
class HTTPEndpointClient:
    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        complete_callback: Callable | None = None
    )

    def issue_query(self, query: Query) -> asyncio.Future[QueryResult]
    async def start(self) -> None
    async def shutdown(self) -> None
```

### Query Types

The client accepts any `Query` subclass from `inference_endpoint.core.types`:

- `ChatCompletionQuery`: For chat completion requests
- `CompletionQuery`: For text completion requests
- Custom query types that inherit from `Query`

### QueryResult

All responses are returned as `QueryResult` objects with:

- `query_id`: ID of the original query
- `response_output`: The response content
- `error`: Error message if request failed
- `metadata`: Additional metadata (e.g., streaming flags)

NOTE: soon we will support arbitrary Query and QueryResult types
