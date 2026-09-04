# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and registration of the fleet scorer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from inference_endpoint.config.schema import EndpointConfig, ModelParams, ScorerMethod
from inference_endpoint.evaluation import swe_bench_fleet_scorer as fleet_scorer_module
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.evaluation.swe_bench_fleet_scorer import SWEBenchFleetScorer
from inference_endpoint.exceptions import SetupError

pytestmark = pytest.mark.unit

URLS = ["http://svc-a:18080", "http://svc-b:18080"]


class TestRegistration:
    def test_the_scorer_is_registered(self):
        assert Scorer.get("swe_bench_fleet") is SWEBenchFleetScorer

    def test_the_scorer_method_enum_is_in_sync(self):
        assert ScorerMethod.SWE_BENCH_FLEET.value in Scorer.available_scorers()

    def test_it_skips_the_endpoint_phase(self):
        # Like the single-service scorer, this one drives the run itself rather
        # than consuming responses collected by the load generator.
        assert SWEBenchFleetScorer.SKIP_ENDPOINT_PHASE
        assert not SWEBenchFleetScorer.REQUIRES_EXTRACTOR


class TestOptions:
    def test_service_urls_are_normalised(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": ["http://svc-a:18080/"]}
        )
        assert options["service_urls"] == ["http://svc-a:18080/"]

    def test_a_comma_separated_string_is_accepted(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": "http://svc-a:18080, http://svc-b:18080"}
        )
        assert len(options["service_urls"]) == 2

    def test_the_single_service_key_still_works(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_url": "http://svc-a:18080"}
        )
        assert options["service_urls"] == ["http://svc-a:18080/"]

    def test_no_service_urls_is_a_setup_error(self):
        with pytest.raises(SetupError, match="swebench_service_urls is required"):
            SWEBenchFleetScorer._resolve_options({})

    def test_duplicate_service_urls_are_refused(self):
        # Two entries for one host is not extra capacity; it is two concurrent
        # runs contending for the same container runtime.
        with pytest.raises(SetupError, match="duplicate"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": ["http://svc-a:18080", "http://svc-a:18080/"]}
            )

    def test_defaults_are_sane(self):
        options = SWEBenchFleetScorer._resolve_options(
            {"swebench_service_urls": URLS}
        )
        assert options["shard_size"] == 10
        assert options["max_attempts"] == 3
        # The tool-call gate's floor must stay at SWE-bench prompt scale.
        assert options["min_prompt_tokens"] == 2000

    def test_a_bad_shard_size_is_rejected(self):
        with pytest.raises(SetupError, match="shard_size"):
            SWEBenchFleetScorer._resolve_options(
                {"swebench_service_urls": URLS, "shard_size": 0}
            )


def test_runtime_endpoint_secret_wins_over_redacted_report(monkeypatch, tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "sample_idx_map.json").write_text(
        '{"swe_bench":{"sample-uuid":0}}'
    )
    (report_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_params": {"name": "persisted-model"},
                "endpoint_config": {
                    "endpoints": ["http://persisted:8000"],
                    "api_key": "<redacted>",
                },
            }
        )
    )
    dataset = MagicMock()
    dataset.dataframe = pd.DataFrame(
        {"instance_id": ["repo__repo-1"], "prompt": ["prompt"]}
    )
    captured = {}

    def build_gates(**kwargs):
        captured.update(kwargs)
        return [], MagicMock()

    def refuse(_gates, _urls):
        raise fleet_scorer_module.GateFailure("stop after credential capture")

    monkeypatch.setattr(fleet_scorer_module, "build_gates", build_gates)
    monkeypatch.setattr(fleet_scorer_module, "run_gates", refuse)
    scorer = SWEBenchFleetScorer(
        dataset_name="swe_bench",
        dataset=dataset,
        report_dir=report_dir,
        swebench_service_urls=URLS,
        num_instances=1,
        model_params=ModelParams(name="runtime-model"),
        endpoint_config=EndpointConfig(
            endpoints=["http://runtime:8000"], api_key="runtime-secret"
        ),
    )

    with pytest.raises(SetupError, match="credential capture"):
        scorer.score()

    assert captured["tool_call_model"] == "runtime-model"
    assert captured["api_key"] == "runtime-secret"
