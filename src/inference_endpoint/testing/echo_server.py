import asyncio
import json
import threading
import time

from aiohttp import web

from inference_endpoint.core.types import ChatCompletionQuery, QueryResult


class EchoServer:
    def __init__(self, host: str = "localhost", port: int = 12345):
        self.host = host
        self.port = port
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
                # raw_payload = json.dumps(json_payload)
            else:
                raw_payload = await request.text()
                try:
                    json_payload = json.loads(raw_payload)
                except (json.JSONDecodeError, TypeError):
                    json_payload = None
        except Exception:
            json_payload = None
            # raw_payload = ""
        completion_request = ChatCompletionQuery.from_json(json_payload)
        response = QueryResult(
            query_id=completion_request.id, response_output=completion_request.prompt
        )

        request_data = {
            "method": request.method,
            "url": str(request.url),
            "endpoint": endpoint,
            "query_params": query_params,
            "headers": headers,
            "json_payload": response.to_json(),
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
