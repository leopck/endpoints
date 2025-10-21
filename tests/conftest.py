"""
Pytest configuration and common fixtures for the MLPerf Inference Endpoint
Benchmarking System.
This file provides shared fixtures and configuration for all tests.
"""

import logging
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.dataset_manager.dataloader import (
    DataLoader,
    DeepSeekR1ChatCompletionDataLoader,
    HFDataLoader,
    PickleReader,
)
from inference_endpoint.testing.echo_server import EchoServer

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Register the profiling plugin
pytest_plugins = ["src.inference_endpoint.profiling.pytest_profiling_plugin"]


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


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture(scope="function")
def mock_http_echo_server():
    """
    Mock HTTP server that echoes back the request payload in the appropriate format.

    This fixture creates a real HTTP server running on localhost that captures
    any HTTP request and returns the request payload as the response. Useful for
    testing HTTP clients with real network calls but controlled responses.

    Returns:
        A server instance with URL.

    Example:
        def test_my_http_client(mock_http_echo_server):
            server = mock_http_echo_server
            # Make real HTTP requests to server.url
            # The response will contain the exact payload you sent
    """

    # Create and start the server with dynamic port allocation (port=0)
    server = EchoServer(port=0)
    server.start()

    try:
        yield server
    except Exception as e:
        raise RuntimeError(f"Mock Echo Server error: {e}") from e
    finally:
        server.stop()


@pytest.fixture
def dummy_dataloader():
    """
    Returns a DummyDataLoader object which just returns the sample index.
    """

    class DummyDataLoader(DataLoader):
        def __init__(self, n_samples: int = 100):
            """
            Initialize the DummyDataLoader.

            Args:
                n_samples (int): The number of samples to load.
            """
            super().__init__(None)
            self.n_samples = n_samples

        def load_sample(self, sample_index: int) -> int:
            """
            Load a sample from the dataset.

            Args:
                sample_index (int): The index of the sample to load.
            """
            assert sample_index >= 0 and sample_index < self.n_samples
            return sample_index

        def num_samples(self) -> int:
            """
            Returns the number of samples in the dataset.
            """
            return self.n_samples

    return DummyDataLoader()


@pytest.fixture
def ds_pickle_dataset_path():
    """
    Returns the path to the ds_samples.pkl file.
    """
    return "tests/datasets/ds_samples.pkl"


@pytest.fixture
def ds_pickle_reader(ds_pickle_dataset_path):
    """
    Returns a PickleReader object for the ds_samples.pkl file.
    """

    def parser(row):
        ret = {}
        for column in row.index.to_list():
            ret[column] = row[column]
        return ret

    return PickleReader(ds_pickle_dataset_path, parser=parser)


@pytest.fixture
def hf_squad_dataset_path():
    """
    Returns the path to the squad dataset.
    """
    return "tests/datasets/squad_pruned"


@pytest.fixture
def hf_squad_dataset(hf_squad_dataset_path):
    """
    Returns a HFDataLoader object for the squad dataset.
    """
    return HFDataLoader(hf_squad_dataset_path, format="arrow")


@pytest.fixture
def events_db(tmp_path):
    """Returns a sample in-memory sqlite database for events.
    This database contains events for 3 sent queries, but only 2 are completed. The 3rd query has no 'received' events.
    """
    test_db = str(tmp_path / f"test_events_{uuid.uuid4().hex}.db")
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS events (sample_uuid INTEGER, event_type TEXT, timestamp_ns INTEGER)"
    )

    events = [
        (1, "request_sent", 10000),
        (2, "request_sent", 10003),
        (1, "first_chunk_received", 10010),
        (2, "first_chunk_received", 10190),
        (1, "non_first_chunk_received", 10201),
        (3, "request_sent", 10202),
        (1, "non_first_chunk_received", 10203),
        (2, "non_first_chunk_received", 10210),
        (1, "non_first_chunk_received", 10211),
        (1, "complete", 10211),
        (2, "non_first_chunk_received", 10214),
        (2, "non_first_chunk_received", 10217),
        (2, "non_first_chunk_received", 10219),
        (2, "complete", 10219),
    ]
    cur.executemany(
        "INSERT INTO events (sample_uuid, event_type, timestamp_ns) VALUES (?, ?, ?)",
        events,
    )
    conn.commit()
    yield test_db

    cur.close()
    conn.close()
    Path(test_db).unlink()


class OracleServer(EchoServer):
    def __init__(self, file_path):
        """
        Initialize the Oracle server with a dataset and load predefined prompt-response mappings.

        The server loads chat completion samples from the specified file path using a custom parser.
        Each sample is mapped from its input prompt to its reference output, allowing subsequent
        retrieval of responses based on exact prompt matching.

        Args:
            file_path (str): Path to the dataset file containing chat completion samples
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path

        def parser(x):
            """
            Extract the prompt and reference output from a dataset sample object.

            Converts a dataset sample into a dictionary with 'prompt' and 'output' keys,
            using the sample's text input as the prompt and reference output as the response.

            Returns:
                dict: A dictionary with 'prompt' and 'output' keys derived from the input sample.
            """
            return {"prompt": x.text_input, "output": x.ref_output}

        self.parser = parser
        data_loader = DeepSeekR1ChatCompletionDataLoader(
            self.file_path, parser=self.parser
        )
        data_loader.load()
        self.data = {}
        for i in range(data_loader.num_samples()):
            sample = data_loader.load_sample(i)
            self.data[sample["prompt"]] = sample["output"]

    def get_response(self, request: str) -> str:
        """
        Retrieve a predefined response for a given request from the loaded dataset.

        Returns the stored output corresponding to the input request. If no matching
        response is found, returns a default "No response found" message.

        Args:
            request (str): The input prompt to look up in the dataset.

        Returns:
            str: The matching output for the request, or a default message if not found.
        """
        return self.data.get(request, "No response found")


@pytest.fixture
def mock_http_oracle_server(ds_pickle_dataset_path):
    """
    Pytest fixture that creates and manages a mock HTTP oracle server for dataset-driven testing.

    Creates an OracleServer instance from a specified dataset pickle file, starts the server
    on a dynamically allocated port, and manages its lifecycle during testing.

    Args:
        ds_pickle_dataset_path (str): Path to the dataset pickle file containing chat completion samples

    Yields:
        OracleServer: A running mock HTTP server serving predefined responses from the dataset

    Raises:
        RuntimeError: If any errors occur during server setup or execution
    """
    # Create and start the server with dynamic port allocation (port=0)
    server = OracleServer(ds_pickle_dataset_path)
    server.start()

    try:
        yield server
    except Exception as e:
        raise RuntimeError(f"Mock Oracle Server error: {e}") from e
    finally:
        server.stop()
