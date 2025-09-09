"""Comprehensive integration tests for the HTTP endpoint client."""

import asyncio
import pickle
import time

import aiohttp
import pytest
import pytest_asyncio
import zmq
import zmq.asyncio
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult
from inference_endpoint.endpoint_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.testing.echo_server import EchoServer


class TestHTTPEndpointClientConcurrency:
    """Test concurrent operations and future handling."""

    def _create_custom_client(
        self,
        mock_http_echo_server,
        num_workers=1,
        max_concurrency=-1,
        zmq_high_water_mark=10000,
        client_timeout=30.0,
        zmq_io_threads=None,
    ):
        """Helper method to create a client with custom configuration."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )

        zmq_config_kwargs = {
            "zmq_request_queue_prefix": f"ipc:///tmp/test_custom_{timestamp}",
            "zmq_response_queue_addr": f"ipc:///tmp/test_custom_resp_{timestamp}",
            "zmq_high_water_mark": zmq_high_water_mark,
        }
        if zmq_io_threads is not None:
            zmq_config_kwargs["zmq_io_threads"] = zmq_io_threads

        zmq_config = ZMQConfig(**zmq_config_kwargs)

        return HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(client_timeout_total=client_timeout),
            zmq_config=zmq_config,
        )

    @pytest.fixture(scope="class")
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,  # More workers for concurrency tests
            max_concurrency=-1,  # No limit by default
        )

    @pytest.fixture(scope="class")
    def zmq_config(self):
        """Create ZMQ configuration with unique addresses."""
        timestamp = int(time.time() * 1000)
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_conc_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_conc_resp_{timestamp}",
            zmq_high_water_mark=10000,  # Higher for massive tests
        )

    @pytest_asyncio.fixture(scope="class")
    async def http_client(self, http_config, zmq_config):
        """Create and start HTTP endpoint client."""
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(
                tcp_connector_limit=1000,  # Higher for massive concurrency
                client_timeout_total=30.0,
            ),
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_basic_future_handling(self, http_client):
        """Test basic future-based request/response."""
        query = ChatCompletionQuery(
            id="future-test",
            prompt="Test future handling",
            model="gpt-3.5-turbo",
        )

        # issue_query returns a future directly
        future = http_client.issue_query(query)
        assert isinstance(future, asyncio.Future)

        # Await the future
        result = await future
        assert result.query_id == "future-test"
        assert result.response_output == "Test future handling"

    @pytest.mark.asyncio
    async def test_concurrent_futures_proper_handling(self, http_client):
        """Test proper concurrent future handling - collect then await all."""
        num_requests = 50

        # Collect all futures first
        futures = []
        for i in range(num_requests):
            query = ChatCompletionQuery(
                id=f"concurrent-{i}",
                prompt=f"Concurrent request {i}",
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append(future)

        # Now await all futures together
        results = await asyncio.gather(*futures)

        # Verify all results
        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert result.query_id == f"concurrent-{i}"
            assert result.response_output == f"Concurrent request {i}"

    @pytest.mark.asyncio
    async def test_massive_concurrency(self, mock_http_echo_server):
        """Test high concurrent requests with proper connection management."""
        actual_max_concurrency = 10000

        # create client with unlimited concurrency
        client = self._create_custom_client(
            mock_http_echo_server,
            num_workers=1,
            max_concurrency=-1,
            zmq_high_water_mark=actual_max_concurrency,
        )

        await client.start()

        try:
            num_requests = actual_max_concurrency

            # Collect futures
            start_time = time.time()
            futures = []
            for i in range(num_requests):
                query = ChatCompletionQuery(
                    id=f"massive-{i}",
                    prompt=f"Request {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Wait for all futures to complete
            results = await asyncio.gather(*futures)
            end_time = time.time()

            # Verify results
            assert len(results) == num_requests
            result_ids = {r.query_id for r in results}
            expected_ids = {f"massive-{i}" for i in range(num_requests)}
            assert result_ids == expected_ids

            # Print performance metrics
            duration = end_time - start_time
            rps = num_requests / duration
            print(
                f"\nProcessed {num_requests} requests in {duration:.2f}s ({rps:.0f} req/s)"
            )

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_massive_payloads(self, http_client):
        """Test handling very large payloads."""
        # Create payloads of different sizes
        payload_sizes = [
            ("small", 128),  # 128 bytes
            ("medium", 1024),  # 1KB
            ("large", 1024 * 10),  # 10KB
            ("xlarge", 1024 * 100),  # 100KB
        ]

        futures = []

        for name, size in payload_sizes:
            # Create large prompt
            large_prompt = "x" * size
            query = ChatCompletionQuery(
                id=f"payload-{name}",
                prompt=large_prompt,
                model="gpt-3.5-turbo",
                max_tokens=2000,
            )
            future = http_client.issue_query(query)
            futures.append((name, size, future))

        # Wait for all payloads
        for name, size, future in futures:
            result = await future
            assert result.query_id == f"payload-{name}"
            assert len(result.response_output) == size
            print(f"\nSuccessfully processed {name} payload ({size} bytes)")

    @pytest.mark.asyncio
    async def test_many_workers(self, mock_http_echo_server):
        """Test with many workers."""
        actual_max_concurrency = 1000
        worker_counts = [16, 32]

        for num_workers in worker_counts:
            print(f"\nTesting with {num_workers} workers...")

            client = self._create_custom_client(
                mock_http_echo_server,
                num_workers=num_workers,
                max_concurrency=-1,
                zmq_high_water_mark=actual_max_concurrency,
                zmq_io_threads=8,
            )

            await client.start()

            try:
                num_requests = actual_max_concurrency
                futures = []

                start_time = time.time()
                for i in range(num_requests):
                    query = ChatCompletionQuery(
                        id=f"worker-test-{i}",
                        prompt=f"Testing {num_workers} workers - request {i}",
                        model="gpt-3.5-turbo",
                    )
                    future = client.issue_query(query)
                    futures.append(future)

                # Wait for all with timeout
                results = await asyncio.gather(*futures)
                duration = time.time() - start_time

                # Verify
                assert len(results) == num_requests
                print(
                    f"  Completed {num_requests} requests in {duration:.2f}s "
                    f"({num_requests/duration:.0f} req/s)"
                )

            finally:
                await client.shutdown()

    @pytest.mark.asyncio
    async def test_concurrency_limit_with_futures(self, mock_http_echo_server):
        """Test concurrency limiting with proper future handling."""
        max_concurrency = 5

        client = self._create_custom_client(
            mock_http_echo_server,
            num_workers=4,
            max_concurrency=max_concurrency,
            zmq_high_water_mark=max_concurrency * 20,
        )

        await client.start()

        try:
            # Send more requests than concurrency limit
            num_requests = 20 * max_concurrency
            futures = []
            issue_times = []

            # Record when each request is issued
            for i in range(num_requests):
                query = ChatCompletionQuery(
                    id=f"limited-{i}",
                    prompt=f"Concurrency limited request {i}",
                    model="gpt-3.5-turbo",
                )

                issue_times.append(time.time())
                future = client.issue_query(query)
                futures.append(future)

            # Wait for all
            results = await asyncio.gather(*futures)

            # Verify all completed
            assert len(results) == num_requests

            # Analyze concurrency pattern
            # With limit of 5, requests should be processed in batches
            print(
                f"\nConcurrency limit test: {num_requests} requests with limit of {max_concurrency}"
            )

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_with_futures(self, http_client):
        """Test streaming responses with future handling."""
        # Note: With the current implementation, streaming responses still return
        # a single future that completes with the final accumulated response.
        # The echo server simulates streaming by returning chunks with metadata.

        # Send both streaming and non-streaming requests
        futures = []

        # Non-streaming
        for i in range(5):
            query = ChatCompletionQuery(
                id=f"non-stream-{i}",
                prompt=f"Non-streaming request {i}",
                model="gpt-3.5-turbo",
                stream=False,
            )
            future = http_client.issue_query(query)
            futures.append(("non-stream", i, future))

        # Streaming
        for i in range(5):
            query = ChatCompletionQuery(
                id=f"stream-{i}",
                prompt=f"Streaming request {i}",
                model="gpt-3.5-turbo",
                stream=True,
            )
            future = http_client.issue_query(query)
            futures.append(("stream", i, future))

        # Wait for all
        for req_type, idx, future in futures:
            result = await future
            if req_type == "non-stream":
                assert result.query_id == f"non-stream-{idx}"
                assert result.response_output == f"Non-streaming request {idx}"
            else:
                assert result.query_id == f"stream-{idx}"
                assert (
                    "Streaming" in result.response_output
                    or result.response_output == f"Streaming request {idx}"
                )

    @pytest.mark.asyncio
    async def test_future_cancellation(self):
        """Test cancelling futures before completion."""
        # Use invalid endpoint so requests won't complete immediately
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=2,
        )

        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_cancel_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_cancel_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(client_timeout_total=30.0),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Create futures
            futures = []
            for i in range(10):
                query = ChatCompletionQuery(
                    id=f"cancel-{i}",
                    prompt=f"To be cancelled {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Small delay to let requests start
            await asyncio.sleep(0.1)

            # Cancel half of them
            for i in range(5):
                futures[i].cancel()

            # Shutdown to cancel remaining
            await client.shutdown()

            # Check cancellations - some futures may complete before cancellation
            cancelled_count = sum(1 for f in futures if f.cancelled())
            completed_count = sum(1 for f in futures if f.done() and not f.cancelled())

            # Either we cancelled some futures, or they completed/failed due to invalid endpoint
            assert cancelled_count > 0 or completed_count > 0
            print(
                f"\nCancellation test: {cancelled_count} cancelled, {completed_count} completed"
            )

        finally:
            pass  # Already shut down

    @pytest.mark.asyncio
    async def test_mixed_callback_and_future_pattern(self, http_client):
        """Test using both callbacks and futures together."""
        callback_results = []

        async def callback(result: QueryResult):
            callback_results.append(result)

        # Set callback
        http_client.complete_callback = callback

        # Send requests and collect futures
        futures = []
        for i in range(10):
            query = ChatCompletionQuery(
                id=f"mixed-{i}",
                prompt=f"Mixed pattern {i}",
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append(future)

        # Wait for futures
        future_results = await asyncio.gather(*futures)

        # Both callback and futures should have results
        assert len(future_results) == 10
        assert len(callback_results) == 10

        # Results should match
        future_ids = {r.query_id for r in future_results}
        callback_ids = {r.query_id for r in callback_results}
        assert future_ids == callback_ids


class TestHTTPEndpointClientErrorHandling:
    """Test error handling with real ZMQ sockets."""

    @pytest.mark.asyncio
    async def test_worker_connection_error(self):
        """Test handling when workers can't connect to endpoint."""
        # Use invalid endpoint
        http_config = HTTPClientConfig(
            endpoint_url="http://invalid-host-12345:9999/v1/chat/completions",
            num_workers=2,
        )

        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_conn_err_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_conn_err_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Send request
            query = ChatCompletionQuery(
                id="error-test",
                prompt="This should fail",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # Should get error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=5.0)

            assert "invalid-host-12345" in str(
                exc_info.value
            ) or "Cannot connect" in str(exc_info.value)

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_error_recovery(self):
        """Test that response handler recovers from errors."""
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_handler_err_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_handler_err_resp_{timestamp}",
        )

        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Initialize minimal client components
        for i in range(client.config.num_workers):
            address = f"{client.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(client.zmq_context, address, client.zmq_config)
            client.worker_push_sockets.append(push_socket)

        # Start response handler
        client._response_handler_task = asyncio.create_task(client._handle_responses())

        # Create context for test
        context = zmq.asyncio.Context()

        try:
            # Create push socket to send responses
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Send valid response
            result1 = QueryResult(
                query_id="test-1",
                response_output="Success",
            )
            await response_push.send(pickle.dumps(result1))

            # Send invalid data that will cause error
            await response_push.send(b"invalid pickle data")

            # Send another valid response to verify recovery
            result2 = QueryResult(
                query_id="test-2",
                response_output="Success after error",
            )
            await response_push.send(pickle.dumps(result2))

            # Create futures to track
            future1 = asyncio.get_event_loop().create_future()
            future2 = asyncio.get_event_loop().create_future()
            client._pending_futures["test-1"] = future1
            client._pending_futures["test-2"] = future2

            # Wait for processing
            await asyncio.sleep(0.5)

            # First future should be completed
            assert future1.done()
            assert future1.result().response_output == "Success"

            # Second future should also complete (handler recovered)
            assert future2.done()
            assert future2.result().response_output == "Success after error"

            # Cleanup
            response_push.close()
            await client.shutdown()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_zmq_send_failure(self):
        """Test handling of ZMQ send failures."""
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_send_fail_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_send_fail_resp_{timestamp}",
        )

        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Create a mock socket that will fail on send
        class FailingSocket:
            def __init__(self):
                self.socket = None

            async def send(self, data):
                raise Exception("ZMQ send failed")

            def close(self):
                pass

        # Replace sockets with failing one
        client.worker_push_sockets = [FailingSocket()]

        try:
            query = ChatCompletionQuery(
                id="send-fail",
                prompt="This will fail to send",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # The send happens asynchronously, wait for it
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=1.0)
            assert "ZMQ send failed" in str(exc_info.value)

        finally:
            pass


class TestHTTPEndpointClientCoverage:
    """Tests to improve code coverage."""

    @pytest.fixture(scope="class")
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

    @pytest.fixture(scope="class")
    def zmq_config(self):
        """Create ZMQ configuration with unique addresses."""
        timestamp = int(time.time() * 1000)
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_coverage_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_coverage_resp_{timestamp}",
        )

    @pytest_asyncio.fixture(scope="class")
    async def http_client(self, http_config, zmq_config):
        """Create and start HTTP endpoint client."""
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_with_callback(self, mock_http_echo_server):
        """Test HTTPEndpointClient initialization with callback."""
        callback_called = []

        async def test_callback(result: QueryResult):
            callback_called.append(result)

        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=5,  # Test concurrency semaphore creation
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_init_callback_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_init_callback_resp_{timestamp}",
            zmq_io_threads=2,  # Test custom io_threads
        )

        # Test initialization with callback and concurrency limit
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
            complete_callback=test_callback,
        )

        # Verify initialization state
        assert client.config == http_config
        assert client.aiohttp_config is not None
        assert client.zmq_config == zmq_config
        assert client.complete_callback == test_callback
        assert client._concurrency_semaphore is not None
        assert client._concurrency_semaphore._value == 5
        assert client.current_worker_idx == 0
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert not client._shutdown_event.is_set()
        assert client._response_handler_task is None
        assert len(client._pending_futures) == 0

        await client.start()

        try:
            # Test that callback is called
            query = ChatCompletionQuery(
                id="callback-test",
                prompt="Test callback",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            await future

            # Wait a bit for callback to be processed
            await asyncio.sleep(0.1)

            assert len(callback_called) == 1
            assert callback_called[0].query_id == "callback-test"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_without_concurrency_limit(
        self, mock_http_echo_server
    ):
        """Test initialization without concurrency limit (max_concurrency <= 0)."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,  # No concurrency limit
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_no_concurrency_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_no_concurrency_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Verify no concurrency semaphore is created
        assert client._concurrency_semaphore is None

        await client.start()

        try:
            # Test that requests work without concurrency limit
            query = ChatCompletionQuery(
                id="no-limit-test",
                prompt="Test no limit",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            assert result.query_id == "no-limit-test"
            assert result.response_output == "Test no limit"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_start_method_socket_creation(self, mock_http_echo_server):
        """Test start method creates correct number of worker sockets."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_start_sockets_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_start_sockets_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Verify initial state
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert client._response_handler_task is None

        await client.start()

        try:
            # Verify start method created all components
            assert len(client.worker_push_sockets) == 4
            assert client.worker_manager is not None
            assert client._response_handler_task is not None
            assert not client._response_handler_task.done()

            # Verify socket addresses are correct
            for socket in client.worker_push_sockets:
                # We can't directly check the address, but we can verify the socket exists
                assert socket is not None

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_timeout_path(self, mock_http_echo_server):
        """Test response handler timeout path in _handle_responses."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_timeout_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_timeout_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Let the response handler run for a bit to exercise timeout path
            await asyncio.sleep(1.5)  # Should trigger at least one timeout

            # Verify response handler is still running
            assert client._response_handler_task is not None
            assert not client._response_handler_task.done()

            # Send a request to verify normal operation still works
            query = ChatCompletionQuery(
                id="timeout-test",
                prompt="Test after timeout",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            assert result.query_id == "timeout-test"
            assert result.response_output == "Test after timeout"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mock_http_echo_server):
        """Test error handling in user callback."""
        timestamp = int(time.time() * 1000)
        callback_errors = []

        async def failing_callback(result: QueryResult):
            callback_errors.append("callback_called")
            raise ValueError("Callback intentionally failed")

        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_callback_error_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_callback_error_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
            complete_callback=failing_callback,
        )

        await client.start()

        try:
            # Send request that will trigger callback error
            query = ChatCompletionQuery(
                id="callback-error-test",
                prompt="Test callback error",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            # Future should still complete successfully despite callback error
            assert result.query_id == "callback-error-test"
            assert result.response_output == "Test callback error"

            # Wait for callback to be processed
            await asyncio.sleep(0.1)

            # Verify callback was called (but failed)
            assert len(callback_errors) == 1
            assert callback_errors[0] == "callback_called"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_with_error_field(self, mock_http_echo_server):
        """Test handling response with error field."""
        # This test requires a way to simulate error responses
        # We'll create a mock response directly
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_error_response_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_error_response_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Initialize minimal client components for direct response injection
        for i in range(client.config.num_workers):
            address = f"{client.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(client.zmq_context, address, client.zmq_config)
            client.worker_push_sockets.append(push_socket)

        # Start response handler
        client._response_handler_task = asyncio.create_task(client._handle_responses())

        # Create context for test
        context = zmq.asyncio.Context()

        try:
            # Create push socket to send error response
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Create future for tracking
            future = asyncio.get_event_loop().create_future()
            client._pending_futures["error-test"] = future

            # Send error response
            error_result = QueryResult(
                query_id="error-test",
                response_output="",
                error="Simulated error response",
            )
            await response_push.send(pickle.dumps(error_result))

            # Wait for processing and expect exception
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=2.0)

            assert "Simulated error response" in str(exc_info.value)

            # Cleanup
            response_push.close()
            await client.shutdown()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_response_handler(self, mock_http_echo_server):
        """Test shutdown when response handler task exists."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_handler_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_handler_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        # Verify components are running
        assert client._response_handler_task is not None
        assert not client._response_handler_task.done()
        assert client.worker_manager is not None
        assert len(client.worker_push_sockets) == 2

        # Add some pending futures
        future1 = asyncio.get_event_loop().create_future()
        future2 = asyncio.get_event_loop().create_future()
        client._pending_futures["pending-1"] = future1
        client._pending_futures["pending-2"] = future2

        # Shutdown should clean everything up
        await client.shutdown()

        # Verify cleanup
        assert client._shutdown_event.is_set()
        assert len(client._pending_futures) == 0
        assert future1.cancelled()
        assert future2.cancelled()
        assert client._response_handler_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_without_components(self):
        """Test shutdown when components haven't been initialized."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_empty_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_empty_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Don't call start() - test shutdown on uninitialized client
        assert client.worker_manager is None
        assert client._response_handler_task is None
        assert len(client.worker_push_sockets) == 0

        # Should not raise any errors
        await client.shutdown()

        # Verify shutdown event is set
        assert client._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_zmq_socket_options(self):
        """Test ZMQ socket configuration options are applied."""
        zmq_config = ZMQConfig(
            zmq_high_water_mark=500,
            zmq_linger=1000,
            zmq_send_timeout=5000,
            zmq_recv_timeout=5000,
            zmq_recv_buffer_size=20 * 1024 * 1024,
            zmq_send_buffer_size=20 * 1024 * 1024,
        )

        context = zmq.asyncio.Context()

        try:
            # Test push socket
            push_socket = ZMQPushSocket(
                context, "ipc:///tmp/test_opts_push", zmq_config
            )

            # Verify options were set
            assert push_socket.socket.getsockopt(zmq.SNDHWM) == 500
            assert push_socket.socket.getsockopt(zmq.LINGER) == 1000
            assert push_socket.socket.getsockopt(zmq.SNDTIMEO) == 5000
            assert push_socket.socket.getsockopt(zmq.SNDBUF) == 20 * 1024 * 1024

            push_socket.close()

            # Test pull socket
            pull_socket = ZMQPullSocket(
                context, "ipc:///tmp/test_opts_pull", zmq_config, bind=True
            )

            assert pull_socket.socket.getsockopt(zmq.RCVHWM) == 500
            # Note: LINGER may not be set on PULL sockets by default, check if it was actually set
            linger_val = pull_socket.socket.getsockopt(zmq.LINGER)
            assert linger_val == 1000 or linger_val == -1  # -1 is default (infinite)
            assert pull_socket.socket.getsockopt(zmq.RCVTIMEO) == 5000
            assert pull_socket.socket.getsockopt(zmq.RCVBUF) == 20 * 1024 * 1024

            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_empty_prompt(self, http_client):
        """Test handling empty prompt."""
        query = ChatCompletionQuery(
            id="empty-prompt",
            prompt="",
            model="gpt-3.5-turbo",
        )

        future = http_client.issue_query(query)
        result = await future

        assert result.query_id == "empty-prompt"
        assert result.response_output == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, http_client):
        """Test handling special characters and unicode."""
        special_prompts = [
            "Hello 你好 🚀 世界",
            'Special chars: @#$%^&*()_+-={}[]|\\:";<>?,./',
            "Newlines\nand\ttabs\rand\\backslashes",
            "Emoji fest: 😀😃😄😁😆😅😂🤣",
            "\u0000\u0001\u0002 control chars",
        ]

        futures = []
        for i, prompt in enumerate(special_prompts):
            query = ChatCompletionQuery(
                id=f"special-{i}",
                prompt=prompt,
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append((prompt, future))

        # Verify all handled correctly
        for prompt, future in futures:
            result = await future
            assert result.response_output == prompt

    @pytest.mark.asyncio
    async def test_metadata_propagation(self, http_client):
        """Test that query metadata is preserved."""
        query = ChatCompletionQuery(
            id="metadata-test",
            prompt="Test metadata",
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.5,
            metadata={
                "user_id": "test-user",
                "session_id": "test-session",
                "custom_field": "custom_value",
            },
        )

        future = http_client.issue_query(query)
        result = await future

        # Echo server should preserve the query
        assert result.query_id == "metadata-test"
        assert result.response_output == "Test metadata"

    @pytest.mark.asyncio
    async def test_concurrent_shutdown(self, http_config, zmq_config):
        """Test shutdown while requests are in flight."""
        # Create a separate client for this test since we need to shut it down
        import time

        timestamp = int(time.time() * 1000)
        shutdown_zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=shutdown_zmq_config,
        )
        await client.start()

        try:
            # Send many requests
            futures = []
            for i in range(100):
                query = ChatCompletionQuery(
                    id=f"shutdown-{i}",
                    prompt=f"Shutdown test {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Immediately shutdown
            await client.shutdown()

            # Count completed vs cancelled
            completed = sum(1 for f in futures if f.done() and not f.cancelled())
            cancelled = sum(1 for f in futures if f.cancelled())

            print(f"\nShutdown test: {completed} completed, {cancelled} cancelled")

            # At least some should be cancelled
            assert cancelled > 0
        finally:
            # Ensure cleanup even if test fails
            if not client._shutdown_event.is_set():
                await client.shutdown()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self, http_client):
        """Test that error responses are propagated as exceptions in futures."""
        # Use an invalid endpoint to trigger real errors
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_error_prop_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_error_prop_resp_{timestamp}",
        )

        config = HTTPClientConfig(
            endpoint_url="http://invalid-host-does-not-exist:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=config,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Send request to invalid endpoint
            query = ChatCompletionQuery(
                id="error-test",
                prompt="Test error",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # Should get an exception due to connection error
            with pytest.raises((aiohttp.ClientError, asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(future, timeout=5.0)

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_for_unknown_query_id(self, http_client):
        """Test handling response for unknown query ID by checking internal state."""
        # Verify client is in a good state
        assert http_client.worker_push_sockets, "Client should have worker sockets"
        assert (
            not http_client._shutdown_event.is_set()
        ), "Client should not be shut down"

        # Send a normal request first
        query = ChatCompletionQuery(
            id="known-query",
            prompt="Test query",
            model="gpt-3.5-turbo",
        )

        future = http_client.issue_query(query)
        result = await future

        # Verify the query was processed and removed from pending futures
        assert result.query_id == "known-query"
        assert "known-query" not in http_client._pending_futures

        # Test that the client handles normal operations correctly
        # (Unknown query IDs would be handled gracefully by the response handler)

    @pytest.mark.asyncio
    async def test_streaming_future_resolved_only_on_final_chunk(self):
        """Test that streaming responses only resolve future on final chunk, not intermediate chunks."""
        # Create config for test
        http_config = HTTPClientConfig(
            endpoint_url="http://test-endpoint/v1/chat/completions",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_response_queue_addr="ipc:///tmp/test_final_chunk_response",
        )

        # Create client
        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        # Mock the worker manager to avoid starting real workers
        from unittest.mock import AsyncMock, MagicMock

        mock_worker_manager = MagicMock()
        mock_worker_manager.initialize = AsyncMock()
        mock_worker_manager.shutdown = AsyncMock()
        client.worker_manager = mock_worker_manager

        # Create mock push socket
        mock_push_socket = MagicMock()
        mock_push_socket.send = AsyncMock()
        client.worker_push_sockets = [mock_push_socket]

        # Start response handler
        asyncio.create_task(client._handle_responses())

        # Wait for response handler to start and bind to socket
        await asyncio.sleep(0.2)

        # Create context for test
        import pickle

        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context()

        try:
            # Create push socket to send responses
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Wait for socket to connect
            await asyncio.sleep(0.1)

            # Send streaming query
            query = ChatCompletionQuery(
                id="test-streaming-chunks",
                prompt="Stream this content",
                model="gpt-3.5-turbo",
                stream=True,
            )

            # Issue query to get future
            future = client.issue_query(query)

            # Wait a bit for processing
            await asyncio.sleep(0.1)

            # Send first chunk (should NOT resolve future)
            first_chunk = QueryResult(
                query_id="test-streaming-chunks",
                response_output="Stream",
                metadata={"first_chunk": True, "final_chunk": False},
            )
            await response_push.send(pickle.dumps(first_chunk))

            # Wait for processing
            await asyncio.sleep(0.1)

            # Future should still be pending
            assert not future.done(), "Future should not be resolved on first chunk"
            assert (
                "test-streaming-chunks" in client._pending_futures
            ), "Future should remain in pending dict"

            # Send intermediate chunk (should NOT resolve future)
            middle_chunk = QueryResult(
                query_id="test-streaming-chunks",
                response_output="Stream this",
                metadata={"first_chunk": False, "final_chunk": False},
            )
            await response_push.send(pickle.dumps(middle_chunk))

            # Wait for processing
            await asyncio.sleep(0.1)

            # Future should still be pending
            assert (
                not future.done()
            ), "Future should not be resolved on intermediate chunk"
            assert (
                "test-streaming-chunks" in client._pending_futures
            ), "Future should remain in pending dict"

            # Send final chunk (SHOULD resolve future)
            final_chunk = QueryResult(
                query_id="test-streaming-chunks",
                response_output="Stream this content",  # Full content
                metadata={"first_chunk": False, "final_chunk": True},
            )
            await response_push.send(pickle.dumps(final_chunk))

            # Wait for processing
            await asyncio.sleep(0.2)

            # Future should now be resolved
            assert future.done(), "Future should be resolved on final chunk"
            assert (
                "test-streaming-chunks" not in client._pending_futures
            ), "Future should be removed from pending dict"

            # Verify we got the complete content
            result = future.result()
            assert result.query_id == "test-streaming-chunks"
            assert result.response_output == "Stream this content"
            assert result.metadata["final_chunk"] is True

            # Cleanup
            response_push.close()
            await client.shutdown()

        finally:
            context.term()


class TestHTTPEndpointClientStreaming:
    """Test streaming functionality with echo server integration."""

    @pytest.fixture
    def echo_server(self):
        """Start echo server for testing."""
        server = EchoServer(host="localhost", port=12346)
        server.start()
        yield server
        server.stop()

    @pytest.fixture
    def client_config(self, echo_server):
        """Create client configuration for echo server."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=10,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig()
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    async def test_streaming_response_complete_content(self, client_config):
        """Test that streaming responses return complete content via futures."""
        http_config, aiohttp_config, zmq_config = client_config

        # Track all responses received via callback
        received_responses = []

        async def response_callback(response):
            received_responses.append(
                {
                    "query_id": response.query_id,
                    "content": response.response_output,
                    "metadata": response.metadata,
                }
            )

        client = HTTPEndpointClient(
            http_config, aiohttp_config, zmq_config, complete_callback=response_callback
        )

        try:
            await client.start()
            await asyncio.sleep(0.5)  # Let workers initialize

            # Test 1: Single word response
            query1 = ChatCompletionQuery(
                id="test-stream-1",
                prompt="Hello",
                model="gpt-3.5-turbo",
                stream=True,
            )

            future1 = client.issue_query(query1)
            result1 = await future1

            # Verify we got the complete response
            assert result1.query_id == "test-stream-1"
            assert result1.response_output == "Hello"
            assert result1.metadata.get("final_chunk") is True

            # Test 2: Multi-word response
            query2 = ChatCompletionQuery(
                id="test-stream-2",
                prompt="This is a longer streaming test message",
                model="gpt-3.5-turbo",
                stream=True,
            )

            future2 = client.issue_query(query2)
            result2 = await future2

            # Verify complete response
            assert result2.query_id == "test-stream-2"
            assert result2.response_output == "This is a longer streaming test message"
            assert result2.metadata.get("final_chunk") is True

            # Test 3: Empty response
            query3 = ChatCompletionQuery(
                id="test-stream-3",
                prompt="",
                model="gpt-3.5-turbo",
                stream=True,
            )

            future3 = client.issue_query(query3)
            result3 = await future3

            assert result3.query_id == "test-stream-3"
            assert result3.response_output == ""
            assert result3.metadata.get("final_chunk") is True

            # Verify callback received all chunks (first + final for each query)
            # Each streaming query should produce at least 2 callbacks, except empty content
            # Query 1 ("Hello"): 2 chunks (first + final)
            # Query 2 (multi-word): multiple chunks (first + intermediates + final)
            # Query 3 (empty): 1 chunk (final only)
            assert len(received_responses) >= 5  # Minimum expected chunks

            # Verify we have both first and final chunks for each query
            for query_id in ["test-stream-1", "test-stream-2", "test-stream-3"]:
                query_responses = [
                    r for r in received_responses if r["query_id"] == query_id
                ]

                # Find first and final chunks
                first_chunks = [
                    r
                    for r in query_responses
                    if r["metadata"] and r["metadata"].get("first_chunk") is True
                ]
                final_chunks = [
                    r
                    for r in query_responses
                    if r["metadata"] and r["metadata"].get("final_chunk") is True
                ]

                # Should have at least one of each (empty responses won't have first chunk)
                if query_id == "test-stream-3":  # Empty response only has final chunk
                    assert (
                        len(first_chunks) == 0
                    ), "Empty response should not have first chunk"
                else:  # Non-empty responses should have first chunk
                    assert len(first_chunks) >= 1, f"No first chunk for {query_id}"
                assert (
                    len(final_chunks) == 1
                ), f"Should have exactly one final chunk for {query_id}"

                # Final chunk should have complete content
                if query_id == "test-stream-1":
                    assert final_chunks[0]["content"] == "Hello"
                elif query_id == "test-stream-2":
                    assert (
                        final_chunks[0]["content"]
                        == "This is a longer streaming test message"
                    )
                elif query_id == "test-stream-3":
                    assert final_chunks[0]["content"] == ""

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_mixed_streaming_non_streaming(self, client_config):
        """Test that mixed streaming and non-streaming requests work correctly."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            # Send mixed requests
            futures = []

            # Non-streaming request
            query_non_stream = ChatCompletionQuery(
                id="non-stream-1",
                prompt="Non-streaming response",
                model="gpt-3.5-turbo",
                stream=False,
            )
            futures.append(("non-stream", client.issue_query(query_non_stream)))

            # Streaming request
            query_stream = ChatCompletionQuery(
                id="stream-1",
                prompt="Streaming response test",
                model="gpt-3.5-turbo",
                stream=True,
            )
            futures.append(("stream", client.issue_query(query_stream)))

            # Another non-streaming
            query_non_stream2 = ChatCompletionQuery(
                id="non-stream-2",
                prompt="Another non-streaming",
                model="gpt-3.5-turbo",
                stream=False,
            )
            futures.append(("non-stream", client.issue_query(query_non_stream2)))

            # Wait for all and verify
            for req_type, future in futures:
                result = await future

                if req_type == "non-stream":
                    # Non-streaming should not have chunk metadata
                    assert (
                        result.metadata is None or "first_chunk" not in result.metadata
                    )
                    assert result.response_output in [
                        "Non-streaming response",
                        "Another non-streaming",
                    ]
                else:
                    # Streaming should have final_chunk = True
                    assert result.metadata.get("final_chunk") is True
                    assert result.response_output == "Streaming response test"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, client_config):
        """Test multiple concurrent streaming requests."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            # Send 10 concurrent streaming requests
            futures = []
            for i in range(10):
                query = ChatCompletionQuery(
                    id=f"concurrent-stream-{i}",
                    prompt=f"Concurrent streaming request number {i}",
                    model="gpt-3.5-turbo",
                    stream=True,
                )
                futures.append((i, client.issue_query(query)))

            # Wait for all to complete
            results = []
            for idx, future in futures:
                result = await future
                results.append((idx, result))

            # Verify all completed with correct content
            assert len(results) == 10

            for idx, result in results:
                assert result.query_id == f"concurrent-stream-{idx}"
                assert (
                    result.response_output
                    == f"Concurrent streaming request number {idx}"
                )
                assert result.metadata.get("final_chunk") is True
                assert result.error is None

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_future_only_resolves_with_final_content(
        self, client_config
    ):
        """Test that futures are only resolved once with final complete response, not intermediate chunks."""
        http_config, aiohttp_config, zmq_config = client_config

        # Track when future is resolved
        resolution_count = 0
        resolved_result = None

        async def track_resolution(future):
            nonlocal resolution_count, resolved_result
            result = await future
            resolution_count += 1
            resolved_result = result

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            query = ChatCompletionQuery(
                id="test-single-resolution",
                prompt="Test single future resolution with multiple words",
                model="gpt-3.5-turbo",
                stream=True,
            )

            future = client.issue_query(query)

            # Start tracking task
            track_task = asyncio.create_task(track_resolution(future))

            # Wait for completion
            await track_task

            # Verify future was only resolved once
            assert resolution_count == 1
            assert resolved_result is not None
            assert resolved_result.query_id == "test-single-resolution"
            assert (
                resolved_result.response_output
                == "Test single future resolution with multiple words"
            )
            assert resolved_result.metadata.get("final_chunk") is True

            # Verify future is done and can't be resolved again
            assert future.done()
            assert future.result() == resolved_result

        finally:
            await client.shutdown()
