"""Tests for ZMQ components used in endpoint client."""

import asyncio
import os
import time
from dataclasses import dataclass
from unittest import mock

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult
from inference_endpoint.endpoint_client.zmq_utils import (
    ZMQConfig,
    ZMQPullSocket,
    ZMQPushSocket,
)


@dataclass
class SampleData:
    """Simple test data class for serialization tests."""

    id: str
    value: str
    timestamp: float


class TestZMQConfig:
    """Tests for ZMQConfig class."""

    def test_default_path_generation_unix(self):
        """Test that default paths are generated correctly on Unix systems."""
        with mock.patch("os.name", "posix"):
            with mock.patch("os.getpid", return_value=12345):
                config = ZMQConfig()

                assert (
                    config.zmq_request_queue_prefix
                    == "ipc://mlperf_endpoint_http_worker_12345"
                )
                assert (
                    config.zmq_response_queue_addr
                    == "ipc://mlperf_endpoint_http_responses_12345"
                )

    def test_custom_paths_preserved(self):
        """Test that custom paths are preserved when provided."""
        custom_request = "ipc:///custom/request/path"
        custom_response = "ipc:///custom/response/path"

        config = ZMQConfig(
            zmq_request_queue_prefix=custom_request,
            zmq_response_queue_addr=custom_response,
        )

        assert config.zmq_request_queue_prefix == custom_request
        assert config.zmq_response_queue_addr == custom_response

    def test_partial_custom_paths(self):
        """Test that only unspecified paths are auto-generated."""
        custom_request = "ipc:///custom/request/path"

        with mock.patch("os.name", "posix"):
            with mock.patch("os.getpid", return_value=12345):
                config = ZMQConfig(zmq_request_queue_prefix=custom_request)

                assert config.zmq_request_queue_prefix == custom_request
                assert (
                    config.zmq_response_queue_addr
                    == "ipc://mlperf_endpoint_http_responses_12345"
                )

    def test_path_generation_includes_pid(self):
        """Test that path generation includes the process ID."""
        with mock.patch("os.name", "posix"):
            config = ZMQConfig()

            # Should include PID in the path
            pid = os.getpid()
            assert (
                f"ipc://mlperf_endpoint_http_worker_{pid}"
                == config.zmq_request_queue_prefix
            )
            assert (
                f"ipc://mlperf_endpoint_http_responses_{pid}"
                == config.zmq_response_queue_addr
            )

    def test_different_processes_get_different_paths(self):
        """Test that different processes get different socket paths."""
        with mock.patch("os.name", "posix"):
            # Simulate different PIDs
            with mock.patch("os.getpid", return_value=111):
                config1 = ZMQConfig()

            with mock.patch("os.getpid", return_value=222):
                config2 = ZMQConfig()

            # Paths should be different due to PID
            assert config1.zmq_request_queue_prefix != config2.zmq_request_queue_prefix
            assert config1.zmq_response_queue_addr != config2.zmq_response_queue_addr

    def test_zmq_config_other_attributes(self):
        """Test that other ZMQConfig attributes work correctly."""
        config = ZMQConfig(
            zmq_io_threads=8,
            zmq_high_water_mark=20000,
            zmq_linger=100,
            zmq_send_timeout=5000,
            zmq_recv_timeout=10000,
            zmq_recv_buffer_size=20 * 1024 * 1024,
            zmq_send_buffer_size=20 * 1024 * 1024,
        )

        assert config.zmq_io_threads == 8
        assert config.zmq_high_water_mark == 20000
        assert config.zmq_linger == 100
        assert config.zmq_send_timeout == 5000
        assert config.zmq_recv_timeout == 10000
        assert config.zmq_recv_buffer_size == 20 * 1024 * 1024
        assert config.zmq_send_buffer_size == 20 * 1024 * 1024


class TestZMQPushPullIntegration:
    """Integration tests for ZMQ Push/Pull sockets."""

    @pytest.fixture
    def zmq_config(self):
        """Create a ZMQ config for testing."""
        return ZMQConfig()

    @pytest.mark.asyncio
    async def test_basic_push_pull_communication(self, zmq_config):
        """Test basic push/pull communication."""
        # Create unique address for this test
        address = f"ipc:///tmp/test_basic_{int(time.time() * 1000)}"

        # Create context
        context = zmq.asyncio.Context()

        try:
            # Create pull socket first (bind)
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)

            # Create push socket (connect)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            # Allow time for connection
            await asyncio.sleep(0.1)

            # Send a query
            test_query = ChatCompletionQuery(
                id="test-123",
                prompt="Hello, world!",
                model="gpt-3.5-turbo",
                max_tokens=50,
                temperature=0.7,
            )

            await push_socket.send(test_query)

            # Receive the query
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, ChatCompletionQuery)
            assert received.id == "test-123"
            assert received.prompt == "Hello, world!"
            assert received.model == "gpt-3.5-turbo"
            assert received.max_tokens == 50
            assert received.temperature == 0.7

            # Cleanup
            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_query_result_communication(self, zmq_config):
        """Test sending and receiving QueryResult objects."""
        address = f"ipc:///tmp/test_result_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send a QueryResult
            test_result = QueryResult(
                query_id="test-456",
                response_output="This is the generated response",
                metadata={"model": "gpt-3.5-turbo", "tokens_used": 25, "latency": 1.5},
            )

            await push_socket.send(test_result)
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, QueryResult)
            assert received.query_id == "test-456"
            assert received.response_output == "This is the generated response"
            assert received.metadata["model"] == "gpt-3.5-turbo"
            assert received.metadata["tokens_used"] == 25
            assert received.metadata["latency"] == 1.5

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_streaming_response_communication(self, zmq_config):
        """Test streaming response pattern with first/final chunks."""
        address = f"ipc:///tmp/test_streaming_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send first chunk
            first_chunk = QueryResult(
                query_id="stream-123",
                response_output="Once",
                metadata={"first_chunk": True, "final_chunk": False},
            )

            await push_socket.send(first_chunk)
            received_first = await pull_socket.receive()

            assert received_first.query_id == "stream-123"
            assert received_first.response_output == "Once"
            assert received_first.metadata["first_chunk"] is True
            assert received_first.metadata["final_chunk"] is False

            # Send final chunk
            final_chunk = QueryResult(
                query_id="stream-123",
                response_output="Once upon a time in a land far away...",
                metadata={"first_chunk": False, "final_chunk": True},
            )

            await push_socket.send(final_chunk)
            received_final = await pull_socket.receive()

            assert received_final.query_id == "stream-123"
            assert (
                received_final.response_output
                == "Once upon a time in a land far away..."
            )
            assert received_final.metadata["first_chunk"] is False
            assert received_final.metadata["final_chunk"] is True

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_multiple_push_single_pull(self, zmq_config):
        """Test multiple push sockets sending to single pull socket."""
        address = f"ipc:///tmp/test_multi_push_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            # Create pull socket
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)

            # Create multiple push sockets
            push_sockets = []
            for _ in range(3):
                push_socket = ZMQPushSocket(context, address, zmq_config)
                push_sockets.append(push_socket)

            await asyncio.sleep(0.1)

            # Send from each push socket
            sent_queries = []
            for i, push_socket in enumerate(push_sockets):
                query = ChatCompletionQuery(
                    id=f"multi-{i}",
                    prompt=f"Query from socket {i}",
                    model="gpt-3.5-turbo",
                )
                sent_queries.append(query)
                await push_socket.send(query)

            # Receive all messages
            received_queries = []
            for _ in range(3):
                received = await pull_socket.receive()
                received_queries.append(received)

            # Verify all messages received (order may vary)
            received_ids = {q.id for q in received_queries}
            assert received_ids == {"multi-0", "multi-1", "multi-2"}

            # Cleanup
            for socket in push_sockets:
                socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_large_payload_communication(self, zmq_config):
        """Test sending large payloads."""
        address = f"ipc:///tmp/test_large_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Create large query
            large_prompt = "x" * 10000  # 10KB prompt
            large_query = ChatCompletionQuery(
                id="large-123",
                prompt=large_prompt,
                model="gpt-3.5-turbo",
                max_tokens=1000,
                metadata={"test": "large_payload"},
            )

            await push_socket.send(large_query)
            received = await pull_socket.receive()

            assert received.id == "large-123"
            assert len(received.prompt) == 10000
            assert received.prompt == large_prompt
            assert received.metadata["test"] == "large_payload"

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_error_response_handling(self, zmq_config):
        """Test sending error responses."""
        address = f"ipc:///tmp/test_error_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send error response
            error_result = QueryResult(
                query_id="error-123",
                response_output="",
                error="HTTP 500: Internal Server Error",
                metadata={"error_code": 500},
            )

            await push_socket.send(error_result)
            received = await pull_socket.receive()

            assert received.query_id == "error-123"
            assert received.response_output == ""
            assert received.error == "HTTP 500: Internal Server Error"
            assert received.metadata["error_code"] == 500

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_custom_data_serialization(self, zmq_config):
        """Test sending custom data types."""
        address = f"ipc:///tmp/test_custom_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send custom data
            custom_data = SampleData(
                id="custom-001",
                value="test value with special chars: 你好 🚀",
                timestamp=time.time(),
            )

            await push_socket.send(custom_data)
            received = await pull_socket.receive()

            assert isinstance(received, SampleData)
            assert received.id == "custom-001"
            assert received.value == "test value with special chars: 你好 🚀"
            assert isinstance(received.timestamp, float)

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_concurrent_send_receive(self, zmq_config):
        """Test concurrent sending and receiving."""
        address = f"ipc:///tmp/test_concurrent_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=True)
            push_socket = ZMQPushSocket(context, address, zmq_config)

            await asyncio.sleep(0.1)

            # Send multiple messages concurrently
            async def send_messages():
                tasks = []
                for i in range(10):
                    query = ChatCompletionQuery(
                        id=f"concurrent-{i}",
                        prompt=f"Concurrent query {i}",
                        model="gpt-3.5-turbo",
                    )
                    tasks.append(push_socket.send(query))
                await asyncio.gather(*tasks)

            # Receive messages concurrently
            async def receive_messages():
                messages = []
                for _ in range(10):
                    msg = await pull_socket.receive()
                    messages.append(msg)
                return messages

            # Start both operations
            send_task = asyncio.create_task(send_messages())
            receive_task = asyncio.create_task(receive_messages())

            await send_task
            received = await receive_task

            # Verify all messages received
            assert len(received) == 10
            received_ids = {msg.id for msg in received}
            expected_ids = {f"concurrent-{i}" for i in range(10)}
            assert received_ids == expected_ids

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_pull_socket_connect_mode(self, zmq_config):
        """Test ZMQPullSocket in connect mode (bind=False)."""
        address = f"ipc:///tmp/test_pull_connect_{int(time.time() * 1000)}"

        context = zmq.asyncio.Context()

        try:
            # First create a push socket that binds
            push_socket = ZMQPushSocket(context, address, zmq_config)
            # Manually bind the push socket's underlying socket
            push_socket.socket.close()  # Close the connected socket
            push_socket.socket = context.socket(zmq.PUSH)
            push_socket.socket.bind(address)
            push_socket.socket.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            push_socket.socket.setsockopt(zmq.LINGER, zmq_config.zmq_linger)
            push_socket.socket.setsockopt(zmq.SNDBUF, zmq_config.zmq_send_buffer_size)
            push_socket.socket.setsockopt(zmq.SNDTIMEO, zmq_config.zmq_send_timeout)

            # Create pull socket in connect mode (bind=False)
            pull_socket = ZMQPullSocket(context, address, zmq_config, bind=False)

            await asyncio.sleep(0.1)

            # Send a message
            test_data = SampleData(
                id="connect-test",
                value="Testing connect mode",
                timestamp=time.time(),
            )

            await push_socket.send(test_data)

            # Receive the message
            received = await pull_socket.receive()

            # Verify
            assert isinstance(received, SampleData)
            assert received.id == "connect-test"
            assert received.value == "Testing connect mode"

            push_socket.close()
            pull_socket.close()

        finally:
            context.term()
