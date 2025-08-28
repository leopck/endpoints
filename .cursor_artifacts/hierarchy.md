# MLPerf Inference Endpoint Benchmarking System - Project Hierarchy

## Current Design Status

**Architecture**: Modular, event-driven with Load Generator as central orchestrator
**Load Generator API**: Defined and ready for implementation
**Endpoint Client**: Abstract interface ready, HTTP implementation TBD by teammates
**Configuration System**: TBD - to be designed and implemented by teammates
**Dataset Management**: Ready for implementation
**Metrics Collection**: Ready for implementation

## Project Structure Overview

```
inference-endpoint/
├── src/                           # Source code
│   ├── inference_endpoint/        # Main package
│   │   ├── __init__.py
│   │   ├── main.py               # Entry point
│   │   ├── cli.py                # Command-line interface
│   │   ├── core/                 # Core components
│   │   │   ├── __init__.py
│   │   │   ├── benchmark.py      # Main benchmark orchestrator
│   │   │   ├── exceptions.py     # Custom exceptions
│   │   │   └── types.py          # Type definitions and dataclasses
│   │   ├── dataset_manager/      # Dataset management
│   │   │   ├── __init__.py
│   │   │   ├── interface.py      # Abstract dataset interface
│   │   │   ├── manager.py        # Dataset manager implementation
│   │   │   ├── loaders/          # Dataset format loaders
│   │   │   │   ├── __init__.py
│   │   │   │   ├── json_loader.py
│   │   │   │   ├── csv_loader.py
│   │   │   │   ├── mlperf_loader.py
│   │   │   │   └── custom_loader.py
│   │   │   ├── tokenizers/       # Tokenization support
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py       # Base tokenizer interface
│   │   │   │   ├── tiktoken.py   # OpenAI tiktoken integration
│   │   │   │   └── custom.py     # Custom tokenizer support
│   │   │   └── validators/       # Dataset validation
│   │   │       ├── __init__.py
│   │   │       ├── schema.py     # Schema validation
│   │   │       └── content.py    # Content validation
│   │   ├── load_generator/       # Load generation
│   │   │   ├── __init__.py
│   │   │   ├── generator.py      # Main load generator with query lifecycle
│   │   │   ├── patterns/         # Load pattern implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py       # Base pattern interface
│   │   │   │   ├── poisson.py    # Poisson distribution
│   │   │   │   ├── uniform.py    # Uniform distribution
│   │   │   │   ├── burst.py      # Burst pattern
│   │   │   │   ├── step.py       # Step pattern
│   │   │   │   └── custom.py     # Custom pattern support
│   │   │   ├── query_manager.py  # Query lifecycle management
│   │   │   └── load_controller.py # Load throttling and control
│   │   ├── endpoint_client/      # Endpoint client management
│   │   │   ├── __init__.py
│   │   │   ├── interface.py      # Abstract endpoint client interface
│   │   │   ├── http_client.py    # HTTP implementation (API TBD)
│   │   │   ├── session.py        # Session management
│   │   │   ├── connection_pool.py # Connection pooling
│   │   │   ├── retry.py          # Retry logic
│   │   │   ├── rate_limiter.py   # Rate limiting
│   │   │   ├── streaming.py      # Streaming response handling
│   │   │   └── auth.py           # Authentication handling
│   │   ├── metrics/              # Metrics collection and analysis
│   │   │   ├── __init__.py
│   │   │   ├── collector.py      # Metrics collection
│   │   │   ├── aggregator.py     # Statistical aggregation
│   │   │   ├── analyzer.py       # Performance analysis
│   │   │   ├── storage/          # Metrics storage backends
│   │   │   │   ├── __init__.py
│   │   │   │   ├── memory.py     # In-memory storage
│   │   │   │   ├── file.py       # File-based storage
│   │   │   │   └── database.py   # Database storage
│   │   │   ├── exporters/        # Results export
│   │   │   │   ├── __init__.py
│   │   │   │   ├── csv.py        # CSV export
│   │   │   │   ├── json.py       # JSON export
│   │   │   │   ├── mlperf.py     # MLPerf format export
│   │   │   │   └── prometheus.py # Prometheus metrics
│   │   │   └── visualizers/      # Data visualization
│   │   │       ├── __init__.py
│   │   │       ├── charts.py     # Chart generation
│   │   │       └── dashboard.py  # Real-time dashboard
│   │   ├── config/               # Configuration management (TBD)
│   │   │   ├── __init__.py
│   │   │   ├── manager.py        # Configuration manager (TBD)
│   │   │   ├── validator.py      # Configuration validation (TBD)
│   │   │   └── loader.py         # Configuration loading (TBD)
│   │   ├── utils/                # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── timing.py         # Timing utilities
│   │   │   ├── memory.py         # Memory management
│   │   │   ├── network.py        # Network utilities
│   │   │   ├── logging.py        # Logging configuration
│   │   │   └── performance.py    # Performance utilities
│   │   └── plugins/              # Plugin system
│   │       ├── __init__.py
│   │       ├── base.py           # Plugin base class
│   │       ├── endpoint/         # Endpoint plugins
│   │       │   ├── __init__.py
│   │       │   ├── openai.py     # OpenAI API plugin
│   │       │   ├── vllm.py       # vLLM plugin
│   │       │   └── custom.py     # Custom endpoint plugin
│   │       └── dataset/          # Dataset plugins
│   │           ├── __init__.py
│   │           └── custom.py     # Custom dataset plugin
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   ├── unit/                    # Unit tests
│   │   ├── __init__.py
│   │   ├── test_dataset_manager/
│   │   ├── test_load_generator/
│   │   ├── test_endpoint_client/
│   │   ├── test_metrics/
│   │   ├── test_config/
│   │   └── test_utils/
│   ├── integration/             # Integration tests
│   │   ├── __init__.py
│   │   ├── test_end_to_end/
│   │   ├── test_performance/
│   │   └── test_scalability/
│   ├── performance/             # Performance tests
│   │   ├── __init__.py
│   │   ├── test_qps_limits/
│   │   ├── test_memory_usage/
│   │   └── test_network_io/
│   └── fixtures/                # Test data and fixtures
│       ├── __init__.py
│       ├── datasets/            # Test datasets
│       ├── configs/             # Test configurations
│       └── responses/           # Mock responses
├── docs/                        # Documentation
│   ├── README.md               # Project overview
│   ├── INSTALL.md              # Installation guide
│   ├── USAGE.md                # Usage guide
│   ├── API.md                  # API reference
│   ├── PERFORMANCE.md          # Performance tuning
│   ├── TROUBLESHOOTING.md      # Troubleshooting guide
│   └── examples/               # Usage examples
│       ├── basic_benchmark.py
│       ├── streaming_benchmark.py
│       ├── custom_load_pattern.py
│       └── distributed_benchmark.py
├── configs/                     # Configuration files (TBD)
│   ├── README.md               # Configuration documentation (TBD)
│   └── examples/               # Example configurations (TBD)
│       └── placeholder.md      # To be populated by teammates
├── scripts/                     # Utility scripts
│   ├── setup.sh                # Setup script
│   ├── benchmark.sh            # Benchmark execution script
│   ├── analyze_results.py      # Results analysis script
│   └── generate_report.py      # Report generation script
├── requirements/                # Dependency management
│   ├── base.txt                # Base dependencies
│   ├── dev.txt                 # Development dependencies
│   ├── test.txt                # Testing dependencies
│   └── performance.txt         # Performance testing dependencies
├── .github/                     # GitHub workflows
│   └── workflows/
│       ├── ci.yml              # Continuous integration
│       ├── test.yml            # Testing workflow
│       └── release.yml         # Release workflow
├── .cursor/                     # Cursor IDE configuration
│   └── rules/
│       └── endpoint-rules.mdc  # Project-specific rules
├── cursor_artifacts/            # Development artifacts
│   ├── requirements.md          # Functional/non-functional requirements
│   ├── design.md               # System architecture and interfaces
│   ├── hierarchy.md            # Project structure (this file)
│   ├── testing-strategy.md     # Testing approach
│   ├── progress.md             # Development progress tracking
│   ├── deployment.md           # Deployment considerations
│   └── refactoring-log.md      # Refactoring activities
├── pyproject.toml              # Project configuration
├── setup.py                    # Package setup
├── README.md                   # Project README
├── LICENSE                     # License file
└── .gitignore                  # Git ignore file
```

## Module Organization Principles

### 1. Separation of Concerns

- **Core**: Central orchestration and common interfaces
- **Dataset Manager**: Dataset handling and preprocessing
- **Load Generator**: Load pattern generation and query lifecycle management
- **Endpoint Client**: Abstract interface for endpoint communication (HTTP implementation TBD)
- **Metrics**: Performance measurement and analysis
- **Config**: Configuration management and validation (TBD)
- **Utils**: Common utilities and helpers
- **Plugins**: Extensible plugin system

### 2. Dependency Direction

- **Core** depends on all major components
- **Components** depend on **Utils** and **Config** (when implemented)
- **Load Generator** orchestrates **Dataset Manager** and **Endpoint Client**
- **Plugins** depend on **Core** interfaces
- **Tests** depend on all components for comprehensive coverage

### 3. Interface Design

- **Abstract Base Classes**: Define contracts for all major components
- **Protocol Classes**: Type-safe interfaces for Python 3.11+
- **Dependency Injection**: Loose coupling between components
- **Plugin Architecture**: Extensible design for new features

### 4. Performance Considerations

- **Async-First**: All I/O operations are async
- **Memory Efficiency**: Object pooling and efficient data structures
- **Concurrency**: Lock-free operations where possible
- **Resource Management**: Proper cleanup and resource pooling

## Package Structure Details

### 1. Main Package (`inference_endpoint`)

- **Entry Point**: `main.py` provides the main application entry
- **CLI**: `cli.py` handles command-line interface
- **Core**: `core/` contains the main benchmark orchestrator
- **Types**: `types.py` defines common data structures

### 2. Dataset Manager (`dataset_manager`)

- **Interface**: Abstract base classes for dataset operations
- **Loaders**: Format-specific dataset loading implementations
- **Tokenizers**: Tokenization support for different models
- **Validators**: Dataset validation and quality checks

### 3. Load Generator (`load_generator`)

- **Generator**: Main load generation logic with query lifecycle management
- **Patterns**: Different load pattern implementations (Poisson, uniform, burst, step)
- **Query Manager**: Query lifecycle tracking (issue_query, query_complete, token_complete)
- **Load Controller**: Load throttling and rate limiting
- **API**: start_test(), issue_query(), query_complete(), token_complete()

### 4. Endpoint Client (`endpoint_client`)

- **Interface**: Abstract endpoint client interface (ABC)
- **HTTP Implementation**: HTTP client implementation (API TBD by teammates)
- **Session**: Session management and connection pooling
- **Streaming**: Streaming response handling
- **Auth**: Authentication and security
- **Purpose**: Pluggable component for different endpoint types

### 5. Metrics (`metrics`)

- **Collector**: Real-time metrics collection
- **Aggregator**: Statistical aggregation and analysis
- **Storage**: Multiple storage backends
- **Exporters**: Results export in various formats

### 6. Configuration (`config`)

- **Manager**: Configuration loading and management (TBD)
- **Validator**: Configuration validation and error checking (TBD)
- **Loader**: Configuration loading from various sources (TBD)
- **Status**: To be designed and implemented by teammates
- **Note**: This component will be implemented after the core system is functional

## File Naming Conventions

### 1. Python Files

- **Modules**: Lowercase with underscores (`dataset_manager.py`)
- **Classes**: PascalCase (`DatasetManager`)
- **Functions**: Lowercase with underscores (`load_dataset`)
- **Constants**: Uppercase with underscores (`MAX_QPS`)

### 2. Configuration Files

- **YAML**: Lowercase with underscores (TBD - to be designed by teammates)
- **Environment**: Uppercase with underscores (`.env`)
- **Status**: Configuration format and structure TBD

### 3. Test Files

- **Unit Tests**: `test_<module_name>.py`
- **Integration Tests**: `test_<feature>_integration.py`
- **Performance Tests**: `test_<metric>_performance.py`

### 4. Documentation Files

- **Markdown**: PascalCase with descriptive names (`API.md`)
- **Examples**: Lowercase with underscores (`basic_benchmark.py`)

## Import Organization

### 1. Standard Library Imports

```python
import asyncio
import json
import logging
from typing import Dict, List, Optional
```

### 2. Third-Party Imports

```python
import aiohttp
import numpy as np
import yaml
```

### 3. Local Imports

```python
from .core.types import Query, QueryResult, QueryId
from .utils.timing import Timer
```

### 4. Import Order

1. Standard library imports
2. Third-party imports
3. Local imports
4. Each group separated by blank line

## Testing Structure

### 1. Unit Tests

- **Location**: `tests/unit/`
- **Coverage**: Individual component testing
- **Mocking**: External dependencies mocked
- **Isolation**: Tests run independently

### 2. Integration Tests

- **Location**: `tests/integration/`
- **Coverage**: Component interaction testing
- **End-to-End**: Full workflow testing
- **Performance**: Performance regression testing

### 3. Performance Tests

- **Location**: `tests/performance/`
- **Coverage**: Performance benchmarks
- **Metrics**: QPS, latency, memory usage
- **Baselines**: Performance regression detection

## Development Workflow

### 1. Feature Development

1. Create feature branch from `main`
2. Implement feature with tests
3. Run all tests and checks
4. Create pull request
5. Code review and approval
6. Merge to `main`

### 2. Testing Workflow

1. Unit tests on every commit
2. Integration tests on pull requests
3. Performance tests on main branch
4. Continuous integration automation

### 3. Documentation Updates

1. Update relevant documentation
2. Update `cursor_artifacts/` files
3. Update API documentation
4. Update usage examples

This hierarchy provides a clean, maintainable structure that supports the high-performance requirements while maintaining code quality and extensibility.
