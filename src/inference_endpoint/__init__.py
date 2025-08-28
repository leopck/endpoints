"""
MLPerf Inference Endpoint Benchmarking System

A high-performance benchmarking tool for LLM endpoints with 50k QPS capability.
"""

__version__ = "0.1.0"
__author__ = "MLPerf Inference Endpoint Team"
__description__ = "High-performance LLM endpoint benchmarking system"

from .core.benchmark import Benchmark

# Core imports - these will be implemented as components are developed
from .core.types import Query, QueryId, QueryResult

__all__ = [
    "Query",
    "QueryResult",
    "QueryId",
    "Benchmark",
]
