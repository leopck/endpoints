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

"""
TODO: PoC only, subject to change!

YAML configuration loading and merging with CLI arguments."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import (
    BenchmarkConfig,
    ClientSettings,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    Metrics,
    ModelParams,
    RuntimeConfig,
    Settings,
    TestType,
)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration error."""

    pass


class ConfigLoader:
    """Load and validate YAML configuration files."""

    @staticmethod
    def load_yaml(path: Path) -> BenchmarkConfig:
        """Load and validate YAML config file.

        Args:
            path: Path to YAML config file

        Returns:
            Validated BenchmarkConfig

        Raises:
            ConfigError: If file not found or validation fails

        Note: Delegates to BenchmarkConfig.from_yaml_file() to avoid duplication.
        """
        try:
            config = BenchmarkConfig.from_yaml_file(path)
            logger.info(f"Loaded config: {config.name} (type: {config.type})")
            return config
        except FileNotFoundError as e:
            raise ConfigError(str(e)) from e
        except (yaml.YAMLError, ValidationError) as e:
            raise ConfigError(f"Config validation failed: {e}") from e

    @staticmethod
    def validate_config(config: BenchmarkConfig, benchmark_mode=None) -> None:
        """Validate configuration consistency.

        This method validates the BenchmarkConfig but does NOT modify it.
        Immutable configs should not be changed - any issues should raise errors.

        Args:
            config: BenchmarkConfig to validate
            benchmark_mode: BenchmarkMode enum (OFFLINE or ONLINE), or string, or None

        Raises:
            ConfigError: If configuration is invalid

        Note: Uses BenchmarkConfig.validate_all() for comprehensive validation.
        This method adds additional logging and warning messages.
        """
        # Convert string to enum if needed
        if isinstance(benchmark_mode, str):
            benchmark_mode = TestType(benchmark_mode)

        # Use BenchmarkConfig's comprehensive validation
        try:
            config.validate_all(benchmark_mode)
        except ValueError as e:
            raise ConfigError(str(e)) from e

        # Additional warnings (not errors)
        load_pattern_type = config.settings.load_pattern.type
        if (
            benchmark_mode == TestType.ONLINE
            and load_pattern_type == LoadPatternType.MAX_THROUGHPUT
        ):
            logger.warning(
                "Online benchmark with 'max_throughput' pattern - consider using 'poisson' for sustained QPS"
            )
        elif load_pattern_type == LoadPatternType.CONCURRENCY:
            logger.info(
                "Concurrency-based pattern selected (will maintain fixed concurrent requests)"
            )

    @staticmethod
    def create_default_config(test_type: TestType) -> BenchmarkConfig:
        """Create default BenchmarkConfig for testing.

        Args:
            test_type: TestType enum (OFFLINE or ONLINE)

        Returns:
            Default BenchmarkConfig (immutable Pydantic model)

        Raises:
            ConfigError: If test_type is unsupported

        Note: This is primarily for testing. Production code uses _build_config_from_cli().
        """
        if test_type == TestType.OFFLINE:
            return BenchmarkConfig(
                name="default_offline",
                version="1.0",
                type=TestType.OFFLINE,
                datasets=[],
                settings=Settings(
                    load_pattern=LoadPattern(
                        type=LoadPatternType.MAX_THROUGHPUT, qps=10.0
                    ),
                    runtime=RuntimeConfig(
                        min_duration_ms=600000, max_duration_ms=1800000, random_seed=42
                    ),
                    client=ClientSettings(workers=4, max_concurrency=32),
                ),
                model_params=ModelParams(temperature=0.7, max_new_tokens=1024),
                metrics=Metrics(),
                endpoint_config=EndpointConfig(),
            )
        elif test_type == TestType.ONLINE:
            return BenchmarkConfig(
                name="default_online",
                version="1.0",
                type=TestType.ONLINE,
                datasets=[],
                settings=Settings(
                    load_pattern=LoadPattern(type=LoadPatternType.POISSON, qps=10.0),
                    runtime=RuntimeConfig(
                        min_duration_ms=600000, max_duration_ms=1800000, random_seed=42
                    ),
                    client=ClientSettings(workers=4, max_concurrency=32),
                ),
                model_params=ModelParams(temperature=0.7, max_new_tokens=1024),
                metrics=Metrics(),
                endpoint_config=EndpointConfig(),
            )
        else:
            raise ConfigError(f"Unknown test type: {test_type}")
