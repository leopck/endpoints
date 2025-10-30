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

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from ..config.runtime_settings import RuntimeSettings
from ..config.schema import LoadPatternType


class SampleOrder(ABC):
    def __init__(
        self, total_samples_to_issue: int, n_samples_in_dataset: int, rng=random
    ):
        """
        Args:
            total_samples_to_issue (int): The total number of samples to issue.
            max_sample_index (int): The maximum sample index.
        """
        self.total_samples_to_issue = total_samples_to_issue
        self.n_samples_in_dataset = n_samples_in_dataset
        self.rng = rng

        self._issued_samples = 0

    def __iter__(self) -> Iterator[int]:
        while self._issued_samples < self.total_samples_to_issue:
            yield self.next_sample_index()
            self._issued_samples += 1

    @abstractmethod
    def next_sample_index(self) -> int:
        raise NotImplementedError


class WithoutReplacementSampleOrder(SampleOrder):
    """Sample order where a sample index cannot repeat until all samples in a dataset have been issued."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_order = list(range(self.n_samples_in_dataset))
        self._curr_idx = (
            self.n_samples_in_dataset + 1
        )  # Ensure we start at an invalid index to force shuffle

    def _reset(self):
        self.rng.shuffle(self.index_order)
        self._curr_idx = 0

    def next_sample_index(self) -> int:
        if self._curr_idx >= len(self.index_order):
            self._reset()
        retval = self.index_order[self._curr_idx]
        self._curr_idx += 1
        return retval


class WithReplacementSampleOrder(SampleOrder):
    """Sample order where a sample index can repeat, even if the dataset has not been exhausted, i.e. sampling with replacement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_sample_index(self) -> int:
        return self.rng.randint(0, self.n_samples_in_dataset - 1)


def uniform_delay_fn(
    max_delay_ns: int = 0, rng: random.Random = random
) -> Callable[[], float]:
    def _fn():
        if max_delay_ns == 0:
            return 0
        return rng.uniform(0, max_delay_ns)

    return _fn


def poisson_delay_fn(
    expected_queries_per_second: float, rng: random.Random = random
) -> Callable[[], float]:
    queries_per_ns = expected_queries_per_second / 1e9

    def _fn():
        if queries_per_ns == 0:
            return 0
        return rng.expovariate(lambd=queries_per_ns)  # lambd=1/mean, where mean=latency

    return _fn


class Scheduler:
    """Schedulers are responsible for building queries and determining when they should be issued to the SUT.

    Subclasses auto-register via __init_subclass__ by specifying load_pattern parameter.
    """

    # Registry for scheduler implementations (populated via __init_subclass__)
    _IMPL_MAP: dict[LoadPatternType, type["Scheduler"]] = {}

    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        sample_order_cls: type[SampleOrder],
    ):
        self.runtime_settings = runtime_settings

        self.total_samples_to_issue = runtime_settings.total_samples_to_issue()
        self.n_unique_samples = runtime_settings.n_samples_from_dataset
        self.sample_order = iter(
            sample_order_cls(
                self.total_samples_to_issue,
                self.n_unique_samples,
                rng=self.runtime_settings.rng_sample_index,
            )
        )
        self.delay_fn = None  # Subclasses must set this

    def __iter__(self):
        for s_idx in self.sample_order:
            yield s_idx, self.delay_fn()

    def __init_subclass__(cls, load_pattern: LoadPatternType | None = None, **kwargs):
        """Auto-register scheduler implementations.

        Args:
            load_pattern: LoadPatternType to bind this scheduler to

        Raises:
            ValueError: If load_pattern already registered
        """
        super().__init_subclass__(**kwargs)

        if load_pattern is not None:
            if load_pattern in Scheduler._IMPL_MAP:
                raise ValueError(
                    f"Cannot bind {cls.__name__} to {load_pattern} - "
                    f"Already bound to {Scheduler._IMPL_MAP[load_pattern].__name__}"
                )
            Scheduler._IMPL_MAP[load_pattern] = cls

    @classmethod
    def get_implementation(cls, load_pattern: LoadPatternType) -> type["Scheduler"]:
        """Get scheduler implementation for load pattern.

        Args:
            load_pattern: LoadPatternType enum

        Returns:
            Scheduler subclass

        Raises:
            NotImplementedError: If no implementation registered
            KeyError: If load_pattern invalid
        """
        if load_pattern not in cls._IMPL_MAP:
            available_str = ", ".join(p.value for p in cls._IMPL_MAP.keys())
            raise KeyError(
                f"No scheduler registered for '{load_pattern.value}'. "
                f"Available: {available_str}"
            )
        return cls._IMPL_MAP[load_pattern]


class MaxThroughputScheduler(Scheduler, load_pattern=LoadPatternType.MAX_THROUGHPUT):
    """Offline max throughput scheduler (all queries at t=0).

    Auto-registers for LoadPatternType.MAX_THROUGHPUT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = uniform_delay_fn(rng=self.runtime_settings.rng_sched)


class PoissonDistributionScheduler(Scheduler, load_pattern=LoadPatternType.POISSON):
    """Poisson-distributed query scheduler for online benchmarking.

    Simulates realistic client-server network usage by using a Poisson process
    to issue queries. The delay between each sample is sampled from an exponential
    distribution, centered around the expected latency based on target QPS.

    Use this scheduler for online latency testing with sustained QPS.

    Auto-registers for LoadPatternType.POISSON.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = poisson_delay_fn(
            expected_queries_per_second=self.runtime_settings.metric_target.target,
            rng=self.runtime_settings.rng_sched,
        )
