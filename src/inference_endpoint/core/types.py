"""
Core type definitions for the MLPerf Inference Endpoint Benchmarking System.

This module defines the basic data structures used throughout the system.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryStatus(Enum):
    """Status of a query in the system."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Query:
    """Represents a single query to be processed."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    model: str = ""
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    status: QueryStatus = QueryStatus.PENDING
    created_at: float | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            import time

            self.created_at = time.time()


@dataclass
class QueryResult:
    """Result of a completed query."""

    query_id: str
    content: str = ""
    tokens: int = 0
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: float | None = None

    def __post_init__(self) -> None:
        if self.completed_at is None:
            import time

            self.completed_at = time.time()


@dataclass
class StreamChunk:
    """A chunk of streaming response."""

    query_id: str
    content: str = ""
    tokens: int = 0
    is_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
