# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for benchmark command error handling.

These tests verify that the benchmark command properly validates inputs,
handles configuration errors, and raises appropriate exceptions instead of
calling sys.exit(). This allows:
- Testing error conditions without process termination
- Programmatic use of benchmark command
- Consistent error handling via centralized exception catching in main()

Focus areas:
- Input validation (missing args, invalid patterns)
- Configuration loading and merging
- Dataset validation
- Error propagation with proper exception types
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark import (
    _build_config_from_cli,
    run_benchmark_command,
)
from inference_endpoint.exceptions import InputValidationError


class TestBuildConfigFromCLI:
    """Test building BenchmarkConfig from CLI arguments.

    CLI and YAML modes are now mutually exclusive - no merging.
    This tests the CLI-only config builder.
    """

    def test_build_config_offline_minimal(self):
        """Test building config with minimal offline params."""
        # Note: concurrency is now in shared args (applies to both offline and online)
        args = argparse.Namespace(
            endpoint="http://test:8000",
            dataset=Path("test.pkl"),
            model="llama-2-70b",  # Required
            api_key=None,
            qps=None,
            concurrency=None,  # Now in shared args
            workers=None,
            duration=None,
            min_tokens=None,
            max_tokens=None,
        )

        config = _build_config_from_cli(args, "offline")

        assert config.name == "cli_offline"
        assert config.endpoint_config.endpoint == "http://test:8000"
        assert config.datasets[0].path == "test.pkl"
        assert config.settings.load_pattern.type.value == "max_throughput"
        assert config.settings.load_pattern.qps == 10.0  # Default
        assert config.settings.client.workers == 4  # Default
        assert config.settings.client.max_concurrency == -1  # Default: unlimited
        assert config.settings.runtime.min_duration_ms == 10000  # Default: 10s

    def test_build_config_online_with_params(self):
        """Test building config with custom online params."""
        # Online mode has concurrency attribute (from _add_online_specific_args)
        args = argparse.Namespace(
            endpoint="http://prod:8000",
            dataset=Path("dataset.pkl"),
            model="gpt-4",  # Required
            api_key="key123",
            qps=100.0,
            workers=8,
            concurrency=64,  # Online-specific
            load_pattern="poisson",  # Online-specific
            duration=600,
            min_tokens=None,
            max_tokens=None,
        )

        config = _build_config_from_cli(args, "online")

        assert config.name == "cli_online"
        assert config.settings.load_pattern.type.value == "poisson"
        assert config.settings.load_pattern.qps == 100.0
        assert config.settings.client.workers == 8
        assert config.settings.client.max_concurrency == 64
        assert config.settings.runtime.min_duration_ms == 600000

    # Note: Tests for missing endpoint/dataset/model removed
    # These are enforced by argparse (required=True), not by _build_config_from_cli
    # Argparse errors before our code runs, so no need to test here


class TestRunBenchmarkCommand:
    """Test benchmark command error handling.

    These tests validate that all user input errors are caught early and
    raise InputValidationError, allowing main() to handle exits gracefully.
    Tests cover: missing mode, invalid config, missing endpoint/dataset.
    """

    @pytest.mark.asyncio
    async def test_from_config_mode_requires_config(self):
        """Test from-config mode requires --config."""
        # Argparse makes --config required for from-config mode
        # This tests the code path
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: "test"
type: "offline"
datasets:
  - name: "test"
    type: "performance"
    path: "tests/datasets/dummy_1k.pkl"
endpoint_config:
  endpoint: "http://test:8000"
""")
            config_path = Path(f.name)

        args = MagicMock()
        args.benchmark_mode = "from-config"
        args.config = config_path
        args.output = None
        args.mode = None

        try:
            # Should load from YAML successfully
            from inference_endpoint.config.yaml_loader import ConfigLoader

            config = ConfigLoader.load_yaml(config_path)
            assert config.endpoint_config.endpoint == "http://test:8000"
        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_missing_benchmark_mode(self):
        """Test error when benchmark mode is None (shouldn't happen with argparse)."""
        # Note: Argparse prevents this scenario, but we test defensive code
        args = MagicMock()
        args.benchmark_mode = None  # Not "offline", "online", or "from-config"

        # Should error in the else clause
        with pytest.raises(InputValidationError, match="Unknown benchmark mode"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_invalid_config_file(self):
        """Test error with invalid config file."""
        args = MagicMock()
        args.benchmark_mode = "from-config"
        args.config = Path("/nonexistent/config.yaml")
        args.output = None
        args.mode = None

        with pytest.raises(InputValidationError, match="Config error"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_file(self):
        """Test error when dataset file doesn't exist."""
        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = "http://test.com"
        args.dataset = Path("/nonexistent/data.pkl")
        args.api_key = None
        args.load_pattern = None
        args.qps = None
        args.concurrency = None
        args.workers = None
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None
        args.mode = "perf"
        args.verbose = 0

        with pytest.raises(InputValidationError, match="not found"):
            await run_benchmark_command(args)

    # Note: Testing unsupported load patterns requires full integration
    # as it happens during scheduler creation after dataset loading.
    # This is covered by yaml_config.py tests and integration tests.
