"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import asyncio
import json

# Add src to path for imports
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "environment": "test",
        "logging": {"level": "INFO", "output": "console"},
        "performance": {
            "max_concurrent_requests": 1000,
            "buffer_size": 10000,
            "memory_limit": "4GB",
        },
    }


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    from inference_endpoint.core.types import Query

    return Query(
        prompt="Hello, how are you?",
        model="gpt-3.5-turbo",
        max_tokens=50,
        temperature=0.7,
    )


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture
def mock_http_echo_server():
    """
    Mock HTTP server that echoes back the request payload in the appropriate format.

    This fixture creates a real HTTP server running on localhost that captures
    any HTTP request and returns the request payload as the response. Useful for
    testing HTTP clients with real network calls but controlled responses.

    Returns:
        A server instance with URL.

    Example:
        def test_my_http_client(mock_http_echo_server):
            server = mock_http_echo_server
            # Make real HTTP requests to server.url
            # The response will contain the exact payload you sent
    """

    class EchoServer:
        def __init__(self):
            self.host = "localhost"
            self.port = 12345  # Fixed port for consistency
            self.url = f"http://{self.host}:{self.port}"
            self.app = None
            self.runner = None
            self.site = None
            self._server_thread = None
            self._loop = None
            self._shutdown_event = threading.Event()

        async def _handle_echo_request(self, request: web.Request) -> web.Response:
            """Handle incoming HTTP requests and echo back the payload."""
            # Extract request data
            endpoint = request.path
            query_params = dict(request.query)
            headers = dict(request.headers)

            # Get request body
            try:
                if request.content_type == "application/json":
                    json_payload = await request.json()
                    raw_payload = json.dumps(json_payload)
                else:
                    raw_payload = await request.text()
                    try:
                        json_payload = json.loads(raw_payload)
                    except (json.JSONDecodeError, TypeError):
                        json_payload = None
            except Exception:
                json_payload = None
                raw_payload = ""

            request_data = {
                "method": request.method,
                "url": str(request.url),
                "endpoint": endpoint,
                "query_params": query_params,
                "headers": headers,
                "json_payload": json_payload,
                "raw_payload": raw_payload,
                "timestamp": time.time(),
            }
            print(f"Request data: {request_data}")

            # Default: echo back the request
            echo_response = {
                "echo": True,
                "request": request_data,
                "message": "Request payload echoed back successfully",
            }
            print(f"Echo response: {echo_response}")

            return web.json_response(
                echo_response,
                status=200,
            )

        async def _handle_echo_chat_completions_request(
            self, request: web.Request
        ) -> web.Response:
            """Handle incoming HTTP requests and echo back the payload."""
            # Extract request data
            endpoint = request.path
            query_params = dict(request.query)
            headers = dict(request.headers)

            # Get request body
            try:
                if request.content_type == "application/json":
                    json_payload = await request.json()
                    raw_payload = json.dumps(json_payload)
                else:
                    raw_payload = await request.text()
                    try:
                        json_payload = json.loads(raw_payload)
                    except (json.JSONDecodeError, TypeError):
                        json_payload = None
            except Exception:
                json_payload = None
                raw_payload = ""

            request_data = {
                "method": request.method,
                "url": str(request.url),
                "endpoint": endpoint,
                "query_params": query_params,
                "headers": headers,
                "json_payload": json_payload,
                "raw_payload": raw_payload,
                "timestamp": time.time(),
            }
            print(f"Request data: {request_data}")

            # Default: echo back the request
            echo_response = {
                "echo": True,
                "request": request_data,
                "message": "Request payload echoed back successfully",
            }
            print(f"Echo response: {echo_response}")

            return web.json_response(
                echo_response,
                status=200,
            )

        def _run_server(self):
            """Run the server in a separate thread."""
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            try:
                self._loop.run_until_complete(self._start_server())
            except Exception as e:
                print(f"Server error: {e}")

        async def _start_server(self):
            """Start the HTTP server."""
            # Create the web application
            self.app = web.Application()

            self.app.router.add_post(
                "/v1/chat/completions", self._handle_echo_chat_completions_request
            )
            self.app.router.add_post("/v1/completions", self._handle_echo_request)

            # Start the server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            print(
                f"==========================\nServer started at {self.url}\n==========================",
                flush=True,
            )

            # Wait for shutdown signal
            while not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)

            # Clean up
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()

        def start(self):
            """Start the server in a background thread."""
            self._server_thread = threading.Thread(target=self._run_server)
            self._server_thread.daemon = True
            self._server_thread.start()

            # Delay for the server to start before returning
            time.sleep(0.5)

        def stop(self):
            """Stop the HTTP server."""
            if self._shutdown_event:
                self._shutdown_event.set()
            if self._server_thread:
                self._server_thread.join(timeout=2)

    # Create and start the server
    server = EchoServer()
    server.start()

    try:
        yield server
    finally:
        server.stop()
