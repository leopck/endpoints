"""ZMQ utilities for endpoint client communication."""

import pickle
from typing import Any

import zmq
import zmq.asyncio

from inference_endpoint.endpoint_client.configs import ZMQConfig


class ZMQPushSocket:
    """Async wrapper for ZMQ PUSH socket."""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig):
        """
        Initialize ZMQ push socket.

        Args:
            context: ZMQ async context
            address: Socket address to connect to
            config: ZMQ configuration
        """
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect(address)
        self.socket.setsockopt(zmq.SNDHWM, config.zmq_high_water_mark)
        self.socket.setsockopt(zmq.LINGER, config.zmq_linger)
        self.socket.setsockopt(zmq.SNDBUF, config.zmq_send_buffer_size)
        # Non-blocking send
        self.socket.setsockopt(zmq.SNDTIMEO, config.zmq_send_timeout)

    async def send(self, data: Any) -> None:
        """
        Serialize and send data through push socket (non-blocking).

        Args:
            data: Any pickleable Python object to send
        """
        serialized = pickle.dumps(data)
        await self.socket.send(serialized, zmq.NOBLOCK)

    def close(self) -> None:
        """Close socket cleanly."""
        self.socket.close()


class ZMQPullSocket:
    """Async wrapper for ZMQ PULL socket."""

    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        config: ZMQConfig,
        bind: bool = False,
    ):
        """
        Initialize ZMQ pull socket.

        Args:
            context: ZMQ async context
            address: Socket address to bind/connect to
            config: ZMQ configuration
            bind: If True, bind to address; if False, connect to address
        """
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
        """
        Receive and deserialize data from pull socket (blocking).

        Returns:
            Deserialized Python object
        """
        serialized = await self.socket.recv()
        return pickle.loads(serialized)

    def close(self) -> None:
        """Close socket cleanly."""
        self.socket.close()
