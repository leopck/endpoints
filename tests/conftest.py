"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import asyncio

# Add src to path for imports
import sys
from pathlib import Path
from typing import Any

import pytest

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
