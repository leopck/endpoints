"""
Unit tests for core types.

These tests verify the basic data structures work correctly.
"""

from inference_endpoint.core.types import Query, QueryResult, QueryStatus, StreamChunk


class TestQuery:
    """Test the Query dataclass."""

    def test_query_creation(self) -> None:
        """Test creating a basic query."""
        query = Query(prompt="Test prompt", model="test-model", max_tokens=100)

        assert query.prompt == "Test prompt"
        assert query.model == "test-model"
        assert query.max_tokens == 100
        assert query.temperature == 0.7  # default value
        assert query.status == QueryStatus.PENDING
        assert query.id is not None
        assert query.created_at is not None

    def test_query_defaults(self) -> None:
        """Test query with minimal parameters."""
        query = Query()

        assert query.prompt == ""
        assert query.model == ""
        assert query.max_tokens == 100
        assert query.temperature == 0.7
        assert query.stream is False
        assert query.metadata == {}
        assert query.status == QueryStatus.PENDING


class TestQueryResult:
    """Test the QueryResult dataclass."""

    def test_query_result_creation(self) -> None:
        """Test creating a query result."""
        result = QueryResult(
            query_id="test-123", content="Test response", tokens=50, latency=0.1
        )

        assert result.query_id == "test-123"
        assert result.content == "Test response"
        assert result.tokens == 50
        assert result.latency == 0.1
        assert result.error is None
        assert result.completed_at is not None


class TestStreamChunk:
    """Test the StreamChunk dataclass."""

    def test_stream_chunk_creation(self) -> None:
        """Test creating a stream chunk."""
        chunk = StreamChunk(
            query_id="test-123", content="partial", tokens=10, is_complete=False
        )

        assert chunk.query_id == "test-123"
        assert chunk.content == "partial"
        assert chunk.tokens == 10
        assert chunk.is_complete is False
        assert chunk.metadata == {}


class TestQueryStatus:
    """Test the QueryStatus enum."""

    def test_status_values(self) -> None:
        """Test that all expected status values exist."""
        assert QueryStatus.PENDING.value == "pending"
        assert QueryStatus.RUNNING.value == "running"
        assert QueryStatus.COMPLETED.value == "completed"
        assert QueryStatus.FAILED.value == "failed"
        assert QueryStatus.CANCELLED.value == "cancelled"
