"""Integration tests for the Worker module using real HTTP requests."""

import asyncio
import pickle
import signal
import time

import orjson
import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import Worker
from inference_endpoint.endpoint_client.zmq_utils import ZMQPushSocket


class TestWorkerBasicFunctionality:
    """Test basic Worker functionality for request/response handling."""

    @pytest.fixture
    def zmq_config(self):
        """Create unique ZMQ configuration for each test."""
        timestamp = int(time.time() * 1000)
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_worker_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_worker_resp_{timestamp}",
            zmq_high_water_mark=100,
        )

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )
        aiohttp_config = AioHttpConfig(
            client_timeout_total=10.0,
            tcp_connector_limit=10,
        )
        return http_config, aiohttp_config

    @pytest.mark.asyncio
    async def test_worker_non_streaming_request(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling non-streaming requests with real HTTP calls."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Create ZMQ context and sockets
        context = zmq.asyncio.Context()

        try:
            # Create request push socket - connect (worker binds the PULL side)
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            # Create response pull socket - bind (worker connects with PUSH)
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Set socket options
            request_push.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            response_pull.setsockopt(zmq.RCVHWM, zmq_config.zmq_high_water_mark)

            # Start worker in background
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to initialize
            await asyncio.sleep(0.5)

            # Send test query
            query = ChatCompletionQuery(
                id="test-non-streaming",
                prompt="Hello, echo server!",
                model="gpt-3.5-turbo",
                stream=False,
            )

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-non-streaming"
            assert response.response_output == "Hello, echo server!"
            assert response.error is None

            # Shutdown worker
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            # Cleanup
            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_request(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling streaming requests with real HTTP calls."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Create ZMQ context and sockets
        context = zmq.asyncio.Context()

        try:
            # Create sockets - request socket connects (worker binds), response socket binds (worker connects)
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Set socket options
            request_push.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            response_pull.setsockopt(zmq.RCVHWM, zmq_config.zmq_high_water_mark)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send streaming query
            query = ChatCompletionQuery(
                id="test-streaming",
                prompt="Stream this response please",
                model="gpt-3.5-turbo",
                stream=True,
            )

            await request_push.send(pickle.dumps(query))

            # Collect streaming responses
            responses = []
            final_content = ""

            while True:
                try:
                    response_data = await response_pull.recv()
                    response = pickle.loads(response_data)
                    responses.append(response)

                    # Check if this is the final chunk
                    if response.metadata.get("final_chunk", False):
                        final_content = response.response_output
                        break

                except TimeoutError:
                    break

            # Verify we got multiple chunks
            assert len(responses) >= 2  # At least first chunk and final chunk

            # Verify first chunk
            assert responses[0].metadata.get("first_chunk") is True
            assert responses[0].query_id == "test-streaming"

            # Verify final response
            assert final_content == "Stream this response please"

            # Shutdown
            worker._shutdown = True
            await worker_task

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_multiple_requests(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling multiple concurrent requests."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send multiple queries
            num_queries = 5
            queries = []
            for i in range(num_queries):
                query = ChatCompletionQuery(
                    id=f"test-multi-{i}",
                    prompt=f"Request number {i}",
                    model="gpt-3.5-turbo",
                    stream=False,
                )
                queries.append(query)
                await request_push.send(pickle.dumps(query))

            # Collect responses
            responses = {}
            for _ in range(num_queries):
                response_data = await asyncio.wait_for(
                    response_pull.recv(), timeout=2.0
                )
                response = pickle.loads(response_data)
                responses[response.query_id] = response

            # Verify all responses
            assert len(responses) == num_queries
            for i in range(num_queries):
                query_id = f"test-multi-{i}"
                assert query_id in responses
                assert responses[query_id].response_output == f"Request number {i}"
                assert responses[query_id].error is None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_signal_handling(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker responds to signals correctly."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create minimal sockets
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Verify worker is running
            assert not worker._shutdown

            # Send signal
            worker._handle_signal(signal.SIGTERM, None)

            # Verify shutdown flag is set
            assert worker._shutdown

            # Worker should exit gracefully
            await asyncio.wait_for(worker_task, timeout=3.0)

            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_first_last_token_metadata(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker correctly sets first_chunk and final_chunk metadata in streaming responses."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send streaming query with multi-word response to ensure multiple chunks
            query = ChatCompletionQuery(
                id="test-first-last-token",
                prompt="Hello world this is a test",  # Multi-word to get multiple chunks
                model="gpt-3.5-turbo",
                stream=True,
            )

            await request_push.send(pickle.dumps(query))

            # Collect all streaming responses
            responses = []

            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=2.0
                    )
                    response = pickle.loads(response_data)
                    responses.append(response)

                    # Check if this is the final chunk
                    if response.metadata.get("final_chunk", False):
                        break

                except TimeoutError:
                    break

            # Verify we got at least 2 responses (first chunk + final chunk)
            assert len(responses) >= 2

            # Verify first chunk metadata
            first_response = responses[0]
            assert first_response.metadata.get("first_chunk") is True
            assert first_response.metadata.get("final_chunk") is False
            assert first_response.query_id == "test-first-last-token"
            assert first_response.response_output  # Should have content

            # Verify final chunk metadata
            final_response = responses[-1]
            assert final_response.metadata.get("first_chunk") is False
            assert final_response.metadata.get("final_chunk") is True
            assert final_response.query_id == "test-first-last-token"
            assert final_response.response_output == "Hello world this is a test"

            # Verify intermediate chunks (if any) don't have first_chunk or final_chunk set to True
            for response in responses[1:-1]:
                assert response.metadata.get("first_chunk") is not True
                assert response.metadata.get("final_chunk") is not True

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_empty_chunks(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling streaming responses with empty chunks."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send streaming query with empty prompt (should still get response structure)
            query = ChatCompletionQuery(
                id="test-empty-chunks",
                prompt="",  # Empty prompt
                model="gpt-3.5-turbo",
                stream=True,
            )

            await request_push.send(pickle.dumps(query))

            # Collect responses
            responses = []

            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=2.0
                    )
                    response = pickle.loads(response_data)
                    responses.append(response)

                    if response.metadata.get("final_chunk", False):
                        break

                except TimeoutError:
                    break

            # Should get at least the final chunk even with empty content
            assert len(responses) >= 1

            # Verify final response
            final_response = responses[-1]
            assert final_response.metadata.get("final_chunk") is True
            assert final_response.query_id == "test-empty-chunks"
            assert final_response.response_output == ""  # Empty content

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_cleanup_with_resources(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker cleanup when resources are properly initialized."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Start worker to initialize resources
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.5)

        # Verify resources are initialized
        assert worker._session is not None
        assert worker._zmq_context is not None
        assert worker._request_socket is not None
        assert worker._response_socket is not None

        # Trigger shutdown
        worker._shutdown = True
        await asyncio.wait_for(worker_task, timeout=2.0)

        # Verify cleanup occurred (resources should be closed/terminated)
        # Note: We can't directly verify closure state, but the test ensures
        # the cleanup code path is executed without errors

    @pytest.mark.asyncio
    async def test_worker_cleanup_with_partial_resources(
        self, worker_config, zmq_config
    ):
        """Test worker cleanup when only some resources are initialized."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Manually set some resources to None to test partial cleanup
        worker._session = None
        worker._request_socket = None

        # Call cleanup directly
        await worker._cleanup()

        # Should complete without errors even with None resources

    @pytest.mark.asyncio
    async def test_worker_different_signal_types(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling different signal types."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Test SIGTERM
        worker._handle_signal(signal.SIGTERM, None)
        assert worker._shutdown is True

        # Reset and test SIGINT
        worker._shutdown = False
        worker._handle_signal(signal.SIGINT, None)
        assert worker._shutdown is True

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_binding_error(self, worker_config, zmq_config):
        """Test worker handling ZMQ socket binding errors."""
        http_config, aiohttp_config = worker_config

        # Create worker with invalid socket address
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr="invalid://socket/address",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Worker run should handle the error gracefully
        with pytest.raises(
            zmq.ZMQError
        ):  # ZMQ will raise an exception for invalid address
            await worker.run()

    @pytest.mark.asyncio
    async def test_worker_aiohttp_session_creation(self, worker_config, zmq_config):
        """Test worker aiohttp session creation with custom timeouts."""
        http_config, aiohttp_config = worker_config

        # Set custom timeout values
        aiohttp_config.client_timeout_total = 30.0
        aiohttp_config.client_timeout_connect = 10.0
        aiohttp_config.client_timeout_sock_read = 5.0

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Start worker briefly to initialize session
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.5)

        # Verify session was created with correct timeout
        assert worker._session is not None
        assert worker._session.timeout.total == 30.0

        # Shutdown
        worker._shutdown = True
        await asyncio.wait_for(worker_task, timeout=2.0)


class TestWorkerErrorHandling:
    """Test Worker error handling for various failure scenarios."""

    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for error handling tests."""
        # Use invalid port to trigger connection errors
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig(
            client_timeout_total=1.0,
            client_timeout_connect=1.0,
        )
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_error_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_error_resp_{timestamp}",
            zmq_send_timeout=500,
            zmq_recv_timeout=500,
        )
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, basic_config):
        """Test worker error handling with invalid endpoint."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Modify config to use invalid endpoint (localhost with invalid port for fast failure)
        http_config.endpoint_url = "http://localhost:99999/v1/chat/completions"
        aiohttp_config.client_timeout_total = 2.0  # Short timeout
        aiohttp_config.client_timeout_connect = 1.0  # Connect timeout

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send query
            query = ChatCompletionQuery(
                id="test-error",
                prompt="This should fail",
                model="gpt-3.5-turbo",
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=2.0)
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-error"
            assert response.error is not None
            assert (
                (
                    "connection" in response.error.lower()
                    and "refused" in response.error.lower()
                )
                or "cannot connect" in response.error.lower()
                or "99999" in response.error
            )

            # Verify error response metadata (may be empty for error responses)
            # Just check that we got an error response with the right query ID
            assert response.query_id == "test-error"
            assert response.error is not None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_http_error_handling(self, basic_config):
        """Test worker handling HTTP errors in streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint to trigger connection error
        http_config.endpoint_url = "http://localhost:99999/invalid"
        aiohttp_config.client_timeout_total = 1.0  # Short timeout

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send streaming query
            query = ChatCompletionQuery(
                id="test-streaming-error",
                prompt="This should fail",
                model="gpt-3.5-turbo",
                stream=True,
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-streaming-error"
            assert response.error is not None
            # Should get url back as error
            assert "http://localhost:99999/invalid" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_non_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for non-streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use ZMQPushSocket to send request
        context = zmq.asyncio.Context()
        request_socket = ZMQPushSocket(
            context, f"{zmq_config.zmq_request_queue_prefix}_0_requests", zmq_config
        )

        # Use raw socket to receive response (bind first before worker connects)
        response_pull = context.socket(zmq.PULL)
        response_pull.bind(zmq_config.zmq_response_queue_addr)

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        worker_task = None
        try:
            # Mock _handle_non_streaming_request to raise an exception immediately
            exception_raised = asyncio.Event()

            async def mock_handle_request(query):
                exception_raised.set()
                raise RuntimeError("Simulated processing error")

            worker._handle_non_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to be ready
            await asyncio.sleep(0.5)

            # Send non-streaming query
            query = ChatCompletionQuery(
                id="test-exception-non-streaming",
                prompt="Test exception handling",
                model="gpt-3.5-turbo",
                stream=False,
            )
            await request_socket.send(query)

            # Wait for exception to be raised
            try:
                await asyncio.wait_for(exception_raised.wait(), timeout=3.0)
            except TimeoutError:
                pytest.fail("Exception was not raised within timeout")

            # Receive error response with a reasonable timeout
            # The response should be sent almost immediately after the exception
            response_data = await asyncio.wait_for(
                response_pull.recv(),
                timeout=3.0,
            )
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-exception-non-streaming"
            assert response.error is not None
            assert "Simulated processing error" in response.error
            assert response.response_output is None

        finally:
            # Proper cleanup
            if worker_task and not worker_task.done():
                # Signal worker to shutdown
                worker._shutdown = True

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(worker_task, timeout=2.0)
                except TimeoutError:
                    # Force cancel if graceful shutdown fails
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                except Exception:
                    # Ignore other exceptions during shutdown
                    pass

            # Close sockets
            request_socket.close()
            response_pull.close()

            # Terminate context
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        context = zmq.asyncio.Context()

        # Use raw socket to receive response (bind first before worker connects)
        response_pull = context.socket(zmq.PULL)
        response_pull.bind(zmq_config.zmq_response_queue_addr)

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        worker_task = None
        request_push = None
        try:
            # Mock _handle_streaming_request to raise an exception
            async def mock_handle_request(query):
                raise RuntimeError("Simulated streaming processing error")

            worker._handle_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to be ready
            await asyncio.sleep(0.5)

            # Create the request socket after worker has bound its socket
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            # Send streaming query
            query = ChatCompletionQuery(
                id="test-exception-streaming",
                prompt="Test streaming exception handling",
                model="gpt-3.5-turbo",
                stream=True,
            )
            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await asyncio.wait_for(
                response_pull.recv(),
                timeout=2.0,
            )
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-exception-streaming"
            assert response.error is not None
            assert "Simulated streaming processing error" in response.error
            assert response.response_output is None

        finally:
            # Proper cleanup
            if worker_task and not worker_task.done():
                # Signal worker to shutdown
                worker._shutdown = True

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(worker_task, timeout=2.0)
                except TimeoutError:
                    # Force cancel if graceful shutdown fails
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                except Exception:
                    # Ignore other exceptions during shutdown
                    pass

            # Close sockets
            if request_push:
                request_push.close()
            response_pull.close()

            # Terminate context
            context.term()

    @pytest.mark.asyncio
    async def test_worker_non_streaming_connection_error(self, basic_config):
        """Test worker handles connection errors in non-streaming responses."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Already uses invalid URL from basic_config
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send query
            query = ChatCompletionQuery(
                id="test-connection-error",
                prompt="Test connection error",
                model="gpt-3.5-turbo",
                stream=False,
            )

            await request_push.send(pickle.dumps(query))

            # Should receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.query_id == "test-connection-error"
            assert response.error is not None
            assert "99999" in response.error or "Cannot connect" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_http_404_error(
        self, mock_http_echo_server, basic_config
    ):
        """Test worker handling HTTP 404 error in streaming request."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use echo server but with invalid endpoint to get 404
        http_config.endpoint_url = f"{mock_http_echo_server.url}/nonexistent"

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send streaming query
            query = ChatCompletionQuery(
                id="test-streaming-404",
                prompt="This should get 404",
                model="gpt-3.5-turbo",
                stream=True,
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify HTTP error response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-streaming-404"
            assert response.error is not None
            assert "HTTP 404" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_non_streaming_http_error_early_return(self, basic_config):
        """Test non-streaming request with HTTP error status."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Initialize components
        worker._zmq_context = zmq.asyncio.Context()
        worker._response_socket = ZMQPushSocket(
            worker._zmq_context, zmq_config.zmq_response_queue_addr, zmq_config
        )

        # Mock session to return various HTTP errors
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        # Create a proper async context manager mock
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Create session mock with regular MagicMock so post doesn't return a coroutine
        mock_session = MagicMock()
        mock_session.post.return_value = mock_context_manager
        worker._session = mock_session

        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            await asyncio.sleep(0.1)

            # Send query that will get HTTP error
            query = ChatCompletionQuery(
                id="test-http-500",
                prompt="This will fail",
                model="gpt-3.5-turbo",
            )

            await worker._handle_non_streaming_request(query)

            # Verify error response was sent
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=1.0)
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.query_id == "test-http-500"
            assert "HTTP 500" in response.error
            assert "Internal Server Error" in response.error

            response_pull.close()
            worker._response_socket.close()

        finally:
            context.term()
            worker._zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_malformed_json(self, basic_config):
        """Test worker handling malformed JSON in streaming response."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a mock server that returns malformed JSON
        from aiohttp import web

        async def malformed_json_handler(request):
            """Handler that returns malformed JSON in streaming format."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/plain"
            await response.prepare(request)

            # Send malformed JSON chunks
            await response.write(b'data: {"invalid": json}\n\n')
            await response.write(b"data: [DONE]\n\n")

            return response

        app = web.Application()
        app.router.add_post("/streaming", malformed_json_handler)

        # Start test server
        from aiohttp.test_utils import TestServer

        server = TestServer(app)

        try:
            await server.start_server()

            # Update config with test server URL
            http_config.endpoint_url = f"http://localhost:{server.port}/streaming"

            worker = Worker(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
                request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
                response_socket_addr=zmq_config.zmq_response_queue_addr,
            )

            context = zmq.asyncio.Context()

            try:
                # Create sockets
                request_push = context.socket(zmq.PUSH)
                request_push.connect(
                    f"{zmq_config.zmq_request_queue_prefix}_0_requests"
                )

                response_pull = context.socket(zmq.PULL)
                response_pull.bind(zmq_config.zmq_response_queue_addr)

                # Start worker
                worker_task = asyncio.create_task(worker.run())
                await asyncio.sleep(0.5)

                # Send streaming query
                query = ChatCompletionQuery(
                    id="test-malformed-json",
                    prompt="Test malformed JSON",
                    model="gpt-3.5-turbo",
                    stream=True,
                )

                await request_push.send(pickle.dumps(query))

                # Should still get final response even with malformed chunks
                response_data = await response_pull.recv()
                response = pickle.loads(response_data)

                # Verify we get a response (malformed JSON chunks are skipped)
                assert isinstance(response, QueryResult)
                assert response.query_id == "test-malformed-json"
                assert response.metadata.get("final_chunk") is True
                assert response.response_output == ""  # No valid content parsed

                # Shutdown
                worker._shutdown = True
                await asyncio.wait_for(worker_task, timeout=2.0)

                request_push.close()
                response_pull.close()

            finally:
                context.term()

        finally:
            await server.close()


class TestWorkerEdgeCases:
    """Test edge cases and error conditions in Worker."""

    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for edge case tests."""
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/invalid",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig(
            client_timeout_total=1.0,
            client_timeout_connect=1.0,
        )
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_edge_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_edge_resp_{timestamp}",
        )
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    async def test_worker_headers_passthrough(
        self, mock_http_echo_server, basic_config
    ):
        """Test worker properly passes through custom headers."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Override with echo server URL
        http_config.endpoint_url = f"{mock_http_echo_server.url}/v1/chat/completions"

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Create query with custom headers
            query = ChatCompletionQuery(
                id="test-headers",
                prompt="Test with headers",
                model="gpt-3.5-turbo",
                stream=False,
            )
            # Add headers attribute
            query.headers = {
                "X-Custom-Header": "test-value",
                "Authorization": "Bearer test-token",
            }

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Echo server should return our prompt
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-headers"
            assert response.response_output == "Test with headers"
            assert response.error is None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_large_response_handling(
        self, mock_http_echo_server, basic_config
    ):
        """Test worker handling very large responses."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Override with echo server URL
        http_config.endpoint_url = f"{mock_http_echo_server.url}/v1/chat/completions"

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.5)

            # Send query with large content (100KB of text with spaces)
            large_content = "Hello world! " * (100 * 1024 // 13)  # ~100KB
            query = ChatCompletionQuery(
                id="test-large-response",
                prompt=large_content,
                model="gpt-3.5-turbo",
                stream=False,
            )

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify response
            assert isinstance(response, QueryResult)
            assert response.query_id == "test-large-response"

            # Check if there's an error first
            if response.error:
                print(f"Response error: {response.error}")
                raise AssertionError(f"Unexpected error: {response.error}")

            # Verify content
            assert response.response_output == large_content
            assert response.error is None
            assert len(response.response_output) == len(large_content)

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            request_push.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_error(self, basic_config):
        """Test worker handling ZMQ socket errors."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Initialize worker resources manually
            worker._zmq_context = context
            worker._response_socket = ZMQPushSocket(
                context, zmq_config.zmq_response_queue_addr, zmq_config
            )

            # Create query
            query = ChatCompletionQuery(
                id="test-zmq-error",
                prompt="Test ZMQ error",
                model="gpt-3.5-turbo",
                stream=False,
            )

            # Close the response socket to simulate error
            worker._response_socket.socket.close()

            # Try to send response - should handle the error gracefully
            response = QueryResult(
                query_id=query.id,
                response_output="test",
                error="ZMQ socket closed",
            )

            # This should not raise an exception
            try:
                await worker._response_socket.send(pickle.dumps(response))
            except Exception:
                # Expected - socket is closed
                pass

            # Worker should handle this gracefully without crashing

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_concurrent_error_handling(self, basic_config):
        """Test multiple workers handling errors concurrently."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint (localhost with invalid port for fast failure)
        http_config.endpoint_url = "http://localhost:99999/api"
        http_config.num_workers = 3  # Test with multiple workers
        aiohttp_config.client_timeout_total = 2.0
        aiohttp_config.client_timeout_connect = 1.0

        workers = []
        worker_tasks = []
        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start multiple workers
            for i in range(3):
                worker = Worker(
                    worker_id=i,
                    http_config=http_config,
                    aiohttp_config=aiohttp_config,
                    zmq_config=zmq_config,
                    request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_{i}_requests",
                    response_socket_addr=zmq_config.zmq_response_queue_addr,
                )
                workers.append(worker)
                worker_tasks.append(asyncio.create_task(worker.run()))

            await asyncio.sleep(0.5)

            # Send queries to all workers
            request_sockets = []
            for i in range(3):
                request_push = context.socket(zmq.PUSH)
                request_push.connect(
                    f"{zmq_config.zmq_request_queue_prefix}_{i}_requests"
                )
                request_sockets.append(request_push)

                # Send a query to this worker
                query = ChatCompletionQuery(
                    id=f"test-concurrent-error-{i}",
                    prompt=f"Worker {i} error test",
                    model="gpt-3.5-turbo",
                    stream=False,
                )
                await request_push.send(pickle.dumps(query))

            # Collect error responses from all workers
            responses = {}
            for _ in range(3):
                response_data = await response_pull.recv()
                response = pickle.loads(response_data)
                responses[response.query_id] = response

            # Verify all workers handled errors
            assert len(responses) == 3
            for i in range(3):
                query_id = f"test-concurrent-error-{i}"
                assert query_id in responses
                assert responses[query_id].error is not None
                assert (
                    (
                        "connection" in responses[query_id].error.lower()
                        and "refused" in responses[query_id].error.lower()
                    )
                    or "cannot connect" in responses[query_id].error.lower()
                    or "99999" in responses[query_id].error
                )

            # Shutdown all workers
            for worker in workers:
                worker._shutdown = True

            await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Close sockets
            for sock in request_sockets:
                sock.close()
            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_receive_timeout_continue(self, basic_config):
        """Test worker handling receive timeout in main loop."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create response socket for worker to connect to
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to initialize
            await asyncio.sleep(0.5)

            # Let worker experience multiple timeouts
            # Should trigger at least 3 timeout cycles
            await asyncio.sleep(3.5)

            # Verify worker is still running after timeouts
            assert not worker._shutdown
            assert not worker_task.done()

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_parse_json_response_error_handling(self, basic_config):
        """Test _parse_json_response with invalid JSON."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Mock _send_error_response to capture calls
        error_calls = []

        async def mock_send_error(query_id: str, error_msg: str):
            error_calls.append((query_id, error_msg))

        worker._send_error_response = mock_send_error

        # Test with actually invalid JSON that will fail parsing
        invalid_json_cases = [
            ("{invalid json}", "Invalid JSON syntax"),
            ('{"key": undefined}', "Undefined value"),
            ("", "Empty string"),
            ("{", "Unclosed brace"),
            ('{"key": "value",}', "Trailing comma"),
        ]

        # Test valid JSON that parses successfully
        valid_json_cases = [
            ("null", None),  # Valid JSON, returns Python None
            ("[1, 2, 3]", [1, 2, 3]),  # Valid JSON array
            ('{"key": "value"}', {"key": "value"}),  # Valid JSON object
            ("123", 123),  # Valid JSON number
            ('"string"', "string"),  # Valid JSON string
        ]

        # Test invalid JSON cases
        for invalid_json, description in invalid_json_cases:
            error_calls.clear()
            result = await worker._parse_json_response(
                invalid_json, f"test-{description}"
            )

            # Should return None for invalid JSON
            assert result is None

            # Should have called _send_error_response
            assert len(error_calls) == 1
            query_id, error_msg = error_calls[0]
            assert query_id == f"test-{description}"
            assert "Failed to parse response" in error_msg

        # Test valid JSON cases
        for valid_json, expected in valid_json_cases:
            error_calls.clear()
            result = await worker._parse_json_response(valid_json, "test-valid")

            # Should return the parsed value
            assert result == expected

            # Should NOT have called _send_error_response
            assert len(error_calls) == 0

    @pytest.mark.asyncio
    async def test_non_streaming_invalid_json_early_return(self, basic_config):
        """Test non-streaming request with invalid JSON response."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Initialize components
        worker._zmq_context = zmq.asyncio.Context()
        worker._response_socket = ZMQPushSocket(
            worker._zmq_context, zmq_config.zmq_response_queue_addr, zmq_config
        )

        # Mock session to return invalid JSON
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="not valid json at all")

        # Create a proper async context manager mock
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Create session mock with regular MagicMock so post doesn't return a coroutine
        mock_session = MagicMock()
        mock_session.post.return_value = mock_context_manager
        worker._session = mock_session

        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            await asyncio.sleep(0.1)

            # Send query that will get invalid JSON
            query = ChatCompletionQuery(
                id="test-bad-json",
                prompt="Will get bad JSON",
                model="gpt-3.5-turbo",
            )

            await worker._handle_non_streaming_request(query)

            # Verify error response was sent
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=1.0)
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.query_id == "test-bad-json"
            assert "Failed to parse response" in response.error

            response_pull.close()
            worker._response_socket.close()

        finally:
            context.term()
            worker._zmq_context.term()

    @pytest.mark.asyncio
    async def test_non_streaming_missing_id_field(self, basic_config):
        """Test non-streaming response missing ID field."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
        )

        # Initialize components
        worker._zmq_context = zmq.asyncio.Context()
        worker._response_socket = ZMQPushSocket(
            worker._zmq_context, zmq_config.zmq_response_queue_addr, zmq_config
        )

        # Mock session to return response without ID field
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.status = 200
        # Response missing the "id" field
        mock_response.text = AsyncMock(
            return_value=orjson.dumps(
                {
                    "response_output": "Test output without ID",
                    "metadata": {"some": "data"},
                    # Note: no "id" field
                }
            ).decode()
        )

        # Create a proper async context manager mock
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Create session mock with regular MagicMock so post doesn't return a coroutine
        mock_session = MagicMock()
        mock_session.post.return_value = mock_context_manager
        worker._session = mock_session

        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            await asyncio.sleep(0.1)

            # Send query
            query = ChatCompletionQuery(
                id="test-missing-id",
                prompt="Response will miss ID",
                model="gpt-3.5-turbo",
            )

            await worker._handle_non_streaming_request(query)

            # Verify response was sent with ID added
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=1.0)
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert (
                response.query_id == "test-missing-id"
            )  # ID should be added from query
            assert response.response_output == "Test output without ID"
            assert response.error is None

            response_pull.close()
            worker._response_socket.close()

        finally:
            context.term()
            worker._zmq_context.term()
