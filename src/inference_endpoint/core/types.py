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
Core type definitions for the MLPerf Inference Endpoint Benchmarking System.

This module defines the basic data structures used throughout the system.
"""

import time
import uuid
from enum import Enum
from typing import Any

import msgspec


class QueryStatus(Enum):
    """Status of a query in the system."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Query(msgspec.Struct, kw_only=True):
    """Represents a single query to be processed."""

    id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))
    data: dict[str, Any] = msgspec.field(default_factory=dict)
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    created_at: float = msgspec.field(default_factory=time.time)


class QueryResult(msgspec.Struct, tag="query_result", kw_only=True, frozen=True):
    """Result of a completed query."""

    id: str = ""
    response_output: str | None = None
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    error: str | None = None
    completed_at: float = msgspec.UNSET

    def __post_init__(self):
        # Disallow user setting completed_at time to prevent cheating.
        # Timestamp must be generated internally
        msgspec.structs.force_setattr(self, "completed_at", time.monotonic_ns())


class StreamChunk(msgspec.Struct, tag="stream_chunk", kw_only=True):
    """A chunk of streaming response."""

    id: str = ""
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
