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
│  └────────┬─────────┘                                           │
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
async def handle_response(response: QueryResult):
    """Callback function for handling responses."""
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Response: {response.response_output}")

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

    # Send request
    future = client.issue_query(query)

    # For streaming, the worker sends two QueryResult messages:
    # 1. First chunk with metadata["first_chunk"]: True
    # 2. Final response with metadata["final_chunk"]: True
    result = await future
    print(f"Complete response: {result.response_output}")

    await client.shutdown()
```

### Using Semaphore for Concurrency Control

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
