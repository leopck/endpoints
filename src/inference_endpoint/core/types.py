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
    model: str = ""
    max_tokens: int = (
        100  # TODO: This is a token count - should we have text count instead?
    )
    temperature: float = 0.7
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    status: QueryStatus = QueryStatus.PENDING
    created_at: float | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            import time

            self.created_at = time.time()

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("to_json is not implemented for Query")

    @classmethod
    def from_json(cls, json_str: dict[str, Any]) -> "Query":
        raise NotImplementedError("from_json is not implemented for Query")


@dataclass
class ChatCompletionQuery(Query):
    """Represents a single query to be processed."""

    prompt: str = (
        ""  # TODO for now a single prompt, but we can replace wiht a list of messages
    )

    headers: dict[str, str] = field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",  # TODO(vir): expose this config via __post_init__
        }
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt},
            ],
        }

    @classmethod
    def from_json(cls, json_value: dict[str, Any]) -> "ChatCompletionQuery":
        return ChatCompletionQuery(
            id=json_value["id"],
            model=json_value["model"],
            prompt=json_value["messages"][1]["content"],
        )


@dataclass
class QueryResult:
    """Result of a completed query."""

    query_id: str
    response_output: str = ""
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: float | None = None

    def __post_init__(self) -> None:
        if self.completed_at is None:
            import time

            self.completed_at = time.time()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.query_id,
            "choices": [
                {"message": {"role": "assistant", "content": self.response_output}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    @classmethod
    def from_json(cls, json_value: dict[str, Any]) -> "QueryResult":
        return QueryResult(
            query_id=json_value["id"],
            response_output=json_value["choices"][0]["message"]["content"],
        )


@dataclass
class StreamChunk:
    """A chunk of streaming response."""

    query_id: str
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
