"""Configuration classes for HTTP endpoint client."""

from dataclasses import dataclass


@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""

    endpoint_url: str
    num_workers: int = 4
    # -1 means unlimited, otherwise limits concurrent requests via semaphore
    max_concurrency: int = -1


@dataclass
class AioHttpConfig:
    """Configuration for aiohttp client session and connectors."""

    # ClientSession configs
    client_session_connector_owner: bool = True

    # TCPConnector configs
    # 0 means unlimited (Operating Systems have TCP client port limits in practice)
    tcp_connector_limit: int = 0
    tcp_connector_ttl_dns_cache: int = 300
    tcp_connector_enable_cleanup_closed: bool = True
    tcp_connector_force_close: bool = False  # Keep connections alive
    tcp_connector_keepalive_timeout: int = 30
    tcp_connector_use_dns_cache: bool = True
    tcp_connector_enable_tcp_nodelay: bool = True  # Disable Nagle's algorithm

    # ClientTimeout configs
    client_timeout_total: float | None = None  # None means no timeout
    client_timeout_connect: float = 10.0
    client_timeout_sock_read: float | None = None  # None means no timeout

    # Streaming configs
    streaming_buffer_size: int = 64 * 1024  # 64KB buffer for streaming


@dataclass
class ZMQConfig:
    """Configuration for ZMQ sockets and communication."""

    zmq_io_threads: int = 4  # Number of ZMQ IO threads
    zmq_request_queue_prefix: str = "ipc:///tmp/http_worker"
    zmq_response_queue_addr: str = "ipc:///tmp/http_responses"
    zmq_high_water_mark: int = 1000  # max msg queue size
    zmq_linger: int = 0  # Don't block on close
    zmq_send_timeout: int = -1  # Non-blocking send
    zmq_recv_timeout: int = -1  # Blocking receive
    zmq_recv_buffer_size: int = 10 * 1024 * 1024  # 10MB receive buffer
    zmq_send_buffer_size: int = 10 * 1024 * 1024  # 10MB send buffer


__all__ = ["HTTPClientConfig", "AioHttpConfig", "ZMQConfig"]
