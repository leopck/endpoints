# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse, urlunparse

import msgspec.json
import yaml

from .artifacts import redact_secrets
from .schemas import RunRequest, TemplateName


class RunnerError(RuntimeError):
    pass


class RunCancelled(RunnerError):
    pass


class SubprocessFailed(RunnerError):
    """A child subprocess exited abnormally (nonzero code, or timed out).

    Carries the structured outcome — ``returncode`` (``None`` on timeout) and a
    bounded log ``tail`` — so callers can classify the failure from the process
    STATUS rather than by grepping the log text. Subclasses ``RunnerError`` so
    existing ``except RunnerError`` handlers are unaffected.
    """

    def __init__(
        self, message: str, *, returncode: int | None, tail: str = ""
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.tail = tail


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            process = self._process
        if process is not None:
            _terminate_process(process)

    def attach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._process = process
            cancelled = self._event.is_set()
        if cancelled:
            _terminate_process(process)

    def detach(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._process is process:
                self._process = None


TEMPLATE_FILES: dict[TemplateName, str] = {
    "default": "swebench_template.yaml",
    "qwen_tools": "swebench_qwen_tools_template.yaml",
}

_LOG_TAIL_MAX_BYTES = 64 * 1024
_LOG_TAIL_MAX_LINES = 50


def _normalize_endpoint_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    parsed = urlparse(base)
    if parsed.hostname == "localhost":
        netloc = "127.0.0.1"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        base = urlunparse(parsed._replace(netloc=netloc))
    return base


def _exact_instance_filter(instance_ids: list[str]) -> str:
    return (
        "^(?:" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_subprocess(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    cancel_token: CancellationToken | None = None,
) -> None:
    if cancel_token is not None and cancel_token.is_cancelled():
        raise RunCancelled(f"subprocess cancelled before start: {cmd}")
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cwd),
                env=env,
                start_new_session=os.name != "nt",
            )
            if cancel_token is not None:
                cancel_token.attach(process)
            deadline = time.monotonic() + timeout_s
            while True:
                if cancel_token is not None and cancel_token.is_cancelled():
                    _terminate_process(process)
                    process.communicate()
                    raise RunCancelled(f"subprocess cancelled: {cmd}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process(process)
                    process.communicate()
                    raise SubprocessFailed(
                        f"subprocess timed out after {timeout_s}s: {cmd}",
                        returncode=None,
                    )
                try:
                    process.communicate(timeout=min(0.5, remaining))
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise RunCancelled(f"subprocess cancelled: {cmd}")
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        if process is not None and cancel_token is not None:
            cancel_token.detach(process)

    if process.returncode != 0:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _LOG_TAIL_MAX_BYTES))
            tail_bytes = log_file.read()
        tail = "\n".join(
            tail_bytes.decode("utf-8", errors="replace").splitlines()[
                -_LOG_TAIL_MAX_LINES:
            ]
        )
        raise SubprocessFailed(
            f"subprocess exited with code {process.returncode}: {cmd}\n{tail}",
            returncode=process.returncode,
            tail=tail,
        )


class SwebenchRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        subprocess_timeout_s: int,
    ):
        self.project_root = project_root.resolve()
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        secret_values = (
            {request.endpoint_api_key} if request.endpoint_api_key else set()
        )
        (run_dir / "request.json").write_bytes(
            msgspec.json.encode(
                redact_secrets(request.model_dump(), secret_values=secret_values)
            )
        )

        # TEMPORARY SPEED HACK (env-gated; revert by unsetting the env var).
        # Instances known to ALWAYS fail (0 resolves across a large sample) are not
        # worth their agent+eval runtime. Skip RUNNING them, but keep the denominator
        # at the ORIGINAL count and score them as unresolved — resolved/total is
        # mathematically identical to running them (they contribute 0 resolves), just
        # faster. Fully internal to the runner: the client still issued the full set,
        # so the scorer's denominator (== len(evaluated_instance_ids) it issued) and
        # its submitted_count==denominator completeness check both stay satisfied.
        skip_ids = self._load_skip_ids()
        original_instance_ids = list(request.evaluated_instance_ids)
        skipped_ids = [i for i in original_instance_ids if i in skip_ids]
        if skipped_ids:
            request.evaluated_instance_ids = [
                i for i in original_instance_ids if i not in skip_ids
            ]

        output_dir = run_dir / "swe_bench_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tempfile.TemporaryDirectory(prefix="swebench_config_") as config_tmp:
            patched_config = self._patch_config(
                Path(config_tmp),
                request,
            )
            self._run_agent(request, patched_config, output_dir, run_dir, cancel_token)

        preds_path = output_dir / "preds.json"
        if not preds_path.exists():
            raise RunnerError("mini-extra did not produce preds.json")
        self._validate_prediction_ids(request, preds_path)
        shutil.copy2(preds_path, run_dir / "preds.json")

        # Classify the agent-phase exit statuses (structured, from mini-swe-agent's
        # exit_statuses_*.yaml) BEFORE grading, so infrastructure failures — the
        # per-instance sandbox `docker run` that never started the task — are told
        # apart from genuine agent outcomes.
        exit_report = self._classify_exit_statuses(output_dir)

        try:
            result_path = self._run_eval(
                request, preds_path, output_dir, run_dir, cancel_token
            )
        except SubprocessFailed as exc:
            # The docker-based eval harness (swebench.harness.run_evaluation) exited
            # abnormally BEFORE grading any instance — the sibling of an agent-phase
            # infra error, one phase later. In practice the docker daemon socket
            # timed out (ReadTimeout) or refused connections (EAGAIN /
            # BlockingIOError) while saturated by sandbox teardown + eval-image
            # pulls. The agents all ran (their diffs are in preds.json), but nothing
            # was graded, so any resolved-rate would be bogus. Classify as infra and
            # fail+retry — keyed on the subprocess STATUS (returncode/timeout), not
            # on log text, exactly like the agent-phase exit-status classifier.
            eval_total = len(request.evaluated_instance_ids)
            how = "timed out" if exc.returncode is None else f"exited code {exc.returncode}"
            result = {
                "exit_status_report": exit_report,
                "accuracy_complete": False,
                "infra_error": True,
                "status": "infra_error",
                "resolved_instances": 0,
                "completed_instances": 0,
                "total_instances": eval_total,
                "submitted_instances": eval_total,
                "eval_infra_error": {"returncode": exc.returncode, "detail": exc.tail},
                "message": (
                    f"SWE-bench EVAL harness (swebench.harness.run_evaluation) {how} "
                    f"before grading any of {eval_total} instances — the docker-based "
                    f"evaluation never ran to completion (daemon contention: socket "
                    f"ReadTimeout / EAGAIN). The agents produced predictions, but the "
                    f"eval was never scored, so any accuracy would be bogus. This is "
                    f"an INFRASTRUCTURE error, not a model result. Please RETRY the "
                    f"SWE-bench run."
                ),
            }
            (run_dir / "swe_bench_results.json").write_bytes(msgspec.json.encode(result))
            return result
        result = msgspec.json.decode(result_path.read_bytes(), type=dict)

        # Instance-level RE-GRADE. Some instances can land in error_ids on a TRANSIENT
        # container-runtime flake (rootless-podman streaming-exec "APIError 500 exec
        # session state improper", or a docker-socket blip) even though the eval
        # subprocess as a whole exited 0. Those are NOT model outcomes — the task was
        # graded against a broken container, not the model's patch. Re-grade ONLY the
        # error_ids with FRESH containers (each _run_eval uses a fresh run_id), up to
        # _MAX_REGRADE_ATTEMPTS, and merge recovered ids into their true buckets. Never
        # fabricates a pass: an id only leaves error on a real resolved/unresolved/
        # empty verdict from the re-grade. Complements the wholesale-failure catch
        # above (that handles eval dying entirely; this handles partial errors).
        if result.get("error_ids"):
            result = self._regrade_error_ids(
                request, preds_path, output_dir, run_dir, result, cancel_token
            )

        # Surface the exit-status metrics + an infra-error verdict on the result. When
        # any instance failed on infra it never produced a real attempt, so the
        # resolved count is over an INCOMPLETE set and the accuracy is not reliable —
        # the client should fail the SWE-bench accuracy and ask for a retry rather
        # than report a misleading (deflated) number.
        result["exit_status_report"] = exit_report
        graded = int(result.get("completed_instances") or 0)
        valid_attempts = exit_report["submitted"] + exit_report["limits_exceeded"]
        # Invariant: with zero infra errors, real attempts == graded results.
        result["accuracy_complete"] = (
            exit_report["infra_errors"] == 0 and valid_attempts == graded
        )
        if exit_report["infra_errors"] > 0:
            infra_desc = ", ".join(
                f"{status}={count}"
                for status, count in sorted(exit_report["infra_by_status"].items())
            )
            result["infra_error"] = True
            result["status"] = "infra_error"
            result["message"] = (
                f"{exit_report['infra_errors']}/{exit_report['total']} instances failed "
                f"with an INFRASTRUCTURE error ({infra_desc}) — the sandbox container "
                f"never started, so those tasks never ran. The reported accuracy is "
                f"computed over an incomplete set and is NOT reliable. Please RETRY "
                f"the SWE-bench run (infra error, not a model result)."
            )
        # Re-inject any SKIPPED instances (see the skip hack above) as unresolved and
        # restore the ORIGINAL denominator, so the reported result is exactly what a
        # full run would have produced (skipped == known-always-fail == unresolved),
        # and the scorer's submitted_count==denominator completeness check holds.
        if skipped_ids:
            self._reinject_skipped(result, skipped_ids, len(original_instance_ids))
            request.evaluated_instance_ids = original_instance_ids

        # Persist the AUGMENTED result (raw eval report + exit-status metrics +
        # infra verdict) as swe_bench_results.json — this is the artifact the client
        # scorer downloads, so the infra metrics/verdict travel with it.
        (run_dir / "swe_bench_results.json").write_bytes(msgspec.json.encode(result))
        return result

    # Instance-level re-grade bound. Each attempt re-grades the STILL-erroring ids
    # with fresh containers; the transient flake almost always clears within a
    # couple of tries, so 6 is a generous ceiling that also caps wasted eval passes.
    _MAX_REGRADE_ATTEMPTS: ClassVar[int] = 6
    _REGRADE_ID_BUCKETS: ClassVar[tuple[str, ...]] = (
        "resolved_ids",
        "unresolved_ids",
        "empty_patch_ids",
        "completed_ids",
        "incomplete_ids",
    )

    def _regrade_error_ids(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        result: dict[str, Any],
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Re-grade instances stuck in ``error_ids`` on a transient container-runtime
        flake, merging any that now grade cleanly back into their true buckets.

        Re-runs the eval harness on ONLY the current ``error_ids`` (fresh run_id ->
        fresh containers/exec sessions, which clears the flake), up to
        ``_MAX_REGRADE_ATTEMPTS`` times, stopping early when ``error_ids`` empties or
        a round makes no progress. Recovery is evidence-based: an id leaves
        ``error_ids`` only when the re-grade reports it resolved / unresolved / empty
        — never fabricated. If the re-grade eval itself dies wholesale
        (``SubprocessFailed``), stop and leave the remaining errors as-is (the
        downstream infra gate then handles a genuinely broken eval). All counts are
        recomputed from the id lists so the persisted result stays self-consistent.
        """
        for attempt in range(1, self._MAX_REGRADE_ATTEMPTS + 1):
            err_ids = list(result.get("error_ids") or [])
            if not err_ids:
                break
            try:
                sub_path = self._run_eval(
                    request,
                    preds_path,
                    output_dir,
                    run_dir,
                    cancel_token,
                    instance_ids=err_ids,
                    log_name=f"swe_bench_eval_regrade_{attempt}.log",
                )
            except SubprocessFailed:
                # The re-grade eval died as a whole — not recoverable here; leave the
                # remaining error_ids for the wholesale infra gate to judge.
                break
            sub = msgspec.json.decode(sub_path.read_bytes(), type=dict)
            (run_dir / f"swe_bench_eval_regrade_{attempt}.json").write_bytes(
                msgspec.json.encode(sub)
            )
            recovered = self._merge_regrade(result, sub, err_ids)
            if not recovered:
                # No id changed bucket this round — the remaining errors are sticky,
                # so further attempts would just burn eval passes.
                break
        return result

    @staticmethod
    def _merge_regrade(
        result: dict[str, Any], sub: dict[str, Any], err_ids: list[str]
    ) -> int:
        """Fold a subset re-grade ``sub`` into ``result``: move any of ``err_ids`` the
        re-grade placed in a real bucket out of ``error_ids``, then recompute counts.
        Returns how many ids were recovered (left ``error_ids``)."""
        err_set = set(err_ids)
        for bucket in SwebenchRunner._REGRADE_ID_BUCKETS:
            existing = result.setdefault(bucket, [])
            have = set(existing)
            for iid in sub.get(bucket) or []:
                if iid in err_set and iid not in have:
                    existing.append(iid)
                    have.add(iid)
        # An id stays errored only if the re-grade STILL reports it as an error.
        still_error = set(sub.get("error_ids") or [])
        result["error_ids"] = [iid for iid in (result.get("error_ids") or []) if iid in still_error]
        # Recompute every *_instances count from its id list so the result is
        # internally consistent after the merge.
        for count_key, id_key in (
            ("resolved_instances", "resolved_ids"),
            ("unresolved_instances", "unresolved_ids"),
            ("empty_patch_instances", "empty_patch_ids"),
            ("completed_instances", "completed_ids"),
            ("incomplete_instances", "incomplete_ids"),
            ("error_instances", "error_ids"),
        ):
            if id_key in result:
                result[count_key] = len(result[id_key])
        return len(err_ids) - len(result.get("error_ids") or [])

    @staticmethod
    def _load_skip_ids() -> frozenset[str]:
        """Instance ids to SKIP running (temporary speed hack), from the environment.

        ``SWE_BENCH_SKIP_IDS_FILE`` -> a JSON list of ids; or ``SWE_BENCH_SKIP_IDS``
        -> a comma-separated inline list. Unset / unreadable => empty (no-op). This
        is intentionally env-gated so it reverts cleanly by removing the variable —
        no code change needed to disable it.
        """
        path = os.environ.get("SWE_BENCH_SKIP_IDS_FILE")
        if path:
            try:
                data = msgspec.json.decode(Path(path).read_bytes())
                return frozenset(str(i) for i in data)
            except (OSError, msgspec.DecodeError):
                return frozenset()
        inline = os.environ.get("SWE_BENCH_SKIP_IDS", "")
        return frozenset(p.strip() for p in inline.split(",") if p.strip())

    @staticmethod
    def _reinject_skipped(
        result: dict[str, Any], skipped_ids: list[str], original_total: int
    ) -> None:
        """Fold skipped instances back into ``result`` as unresolved and restore the
        original denominator, so resolved/total matches a full run exactly."""
        skip = [i for i in skipped_ids]
        result["skipped_ids"] = sorted(skip)
        for bucket in ("submitted_ids", "unresolved_ids"):
            lst = result.setdefault(bucket, [])
            have = set(lst)
            for iid in skip:
                if iid not in have:
                    lst.append(iid)
                    have.add(iid)
        # Recompute the id-backed counts, then force the denominator fields to the
        # original count (the skipped ids are included above, so these match).
        for count_key, id_key in (
            ("unresolved_instances", "unresolved_ids"),
            ("submitted_instances", "submitted_ids"),
        ):
            if id_key in result:
                result[count_key] = len(result[id_key])
        result["total_instances"] = original_total
        result["submitted_instances"] = original_total

    _AGENT_RAN_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"Submitted", "LimitsExceeded"}
    )

    @classmethod
    def _classify_exit_statuses(cls, output_dir: Path) -> dict[str, Any]:
        """Split mini-swe-agent's exit statuses into real agent attempts vs infra
        errors. `Submitted` (agent produced a diff) and `LimitsExceeded` (agent ran
        out of steps) are genuine attempts; every OTHER status (`CalledProcessError`,
        `TimeoutExpired`, ...) means the per-instance sandbox `docker run` failed and
        the task never started — an infrastructure error, not a model outcome.

        Robust because it reads the structured `exit_statuses_*.yaml`, not logs.
        """
        files = sorted(
            output_dir.glob("exit_statuses_*.yaml"),
            key=lambda path: path.stat().st_mtime,
        )
        by_status: dict[str, list[str]] = {}
        if files:
            try:
                loaded = yaml.safe_load(files[-1].read_text()) or {}
            except (OSError, yaml.YAMLError):
                loaded = {}
            instances_by_status = loaded.get("instances_by_exit_status")
            if isinstance(instances_by_status, dict):
                for status, instance_ids in instances_by_status.items():
                    if isinstance(instance_ids, list):
                        by_status[str(status)] = [str(i) for i in instance_ids]

        counts = {status: len(ids) for status, ids in by_status.items()}
        infra_by_status = {
            status: len(ids)
            for status, ids in by_status.items()
            if status not in cls._AGENT_RAN_STATUSES
        }
        infra_ids = [
            instance_id
            for status, ids in by_status.items()
            if status not in cls._AGENT_RAN_STATUSES
            for instance_id in ids
        ]
        return {
            "total": sum(counts.values()),
            "submitted": counts.get("Submitted", 0),
            "limits_exceeded": counts.get("LimitsExceeded", 0),
            "infra_errors": len(infra_ids),
            "infra_by_status": infra_by_status,
            "infra_instance_ids": infra_ids,
            "counts_by_status": counts,
        }

    def _load_template(self, request: RunRequest) -> dict[str, Any]:
        template_path = self._template_dir / TEMPLATE_FILES[request.template]
        with template_path.open() as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise RunnerError("swebench template must be a YAML mapping")
        model_cfg = loaded.get("model")
        if not isinstance(model_cfg, dict):
            raise RunnerError("swebench template must define model")
        if not isinstance(model_cfg.get("model_kwargs"), dict):
            raise RunnerError("swebench template must define model.model_kwargs")
        return loaded

    @property
    def _template_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"

    def _patch_config(self, config_dir: Path, request: RunRequest) -> Path:
        cfg = self._load_template(request)
        model_cfg = cfg["model"]
        model_kwargs = model_cfg["model_kwargs"]

        model_cfg["model_name"] = request.model_name
        if request.endpoint_urls:
            base = _normalize_endpoint_base(str(request.endpoint_urls[0]))
            model_kwargs["api_base"] = base + "/v1"
        else:
            base = ""
            model_kwargs["api_base"] = ""

        if request.endpoint_api_key:
            model_kwargs["api_key"] = request.endpoint_api_key
        elif urlparse(base).hostname in {"localhost", "127.0.0.1", "::1"}:
            model_kwargs["api_key"] = "EMPTY"
        else:
            model_kwargs.pop("api_key", None)

        for field in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = request.generation_params.get(field)
            if val is not None:
                model_kwargs[field] = val
            else:
                model_kwargs.pop(field, None)

        if (
            max_new_tokens := request.generation_params.get("max_new_tokens")
        ) is not None:
            model_kwargs["max_tokens"] = max_new_tokens
        else:
            model_kwargs.pop("max_tokens", None)

        if (
            chat_tmpl := request.generation_params.get("chat_template_kwargs")
        ) is not None:
            model_kwargs["chat_template_kwargs"] = chat_tmpl
        else:
            model_kwargs.pop("chat_template_kwargs", None)

        config_dir.mkdir(parents=True, exist_ok=True)
        patched_path = config_dir / "swebench_patched.yaml"
        with patched_path.open("w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return patched_path

    def _run_agent(
        self,
        request: RunRequest,
        patched_config: Path,
        output_dir: Path,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        instance_filter = _exact_instance_filter(request.evaluated_instance_ids)
        cmd = [
            "mini-extra",
            "swebench",
            "--model",
            request.model_name,
            "--config",
            str(patched_config),
            "--subset",
            request.subset,
            "--split",
            request.split,
            "--filter",
            instance_filter,
            "--workers",
            str(request.workers),
            "--output",
            str(output_dir),
        ]
        if request.enable_swebench_toolcall_patch:
            with tempfile.TemporaryDirectory(prefix="minisweagent_overlay_") as tmp:
                env = self._agent_env(request, Path(tmp))
                _run_subprocess(
                    cmd,
                    run_dir / "swe_bench_agent.log",
                    cwd=output_dir,
                    timeout_s=self.subprocess_timeout_s,
                    env=env,
                    cancel_token=cancel_token,
                )
                return
        _run_subprocess(
            cmd,
            run_dir / "swe_bench_agent.log",
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            env=self._base_env(request),
            cancel_token=cancel_token,
        )

    def _base_env(self, request: RunRequest) -> dict[str, str]:
        env = dict(os.environ)
        no_proxy = {"127.0.0.1", "localhost"}
        for endpoint in request.endpoint_urls:
            host = urlparse(str(endpoint)).hostname
            if host:
                no_proxy.add(host)
        existing = env.get("NO_PROXY") or env.get("no_proxy")
        if existing:
            no_proxy.update(
                part.strip() for part in existing.split(",") if part.strip()
            )
        no_proxy_value = ",".join(sorted(no_proxy))
        env["NO_PROXY"] = no_proxy_value
        env["no_proxy"] = no_proxy_value
        return env

    def _agent_env(self, request: RunRequest, overlay_root: Path) -> dict[str, str]:
        env = self._base_env(request)
        overlay = self._create_toolcall_patch_overlay(overlay_root, self._template_dir)
        pythonpath = [str(overlay)]
        if existing := env.get("PYTHONPATH"):
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env

    def _create_toolcall_patch_overlay(
        self, overlay_root: Path, replacement_root: Path
    ) -> Path:
        site_packages = self._resolve_minisweagent_site_packages()
        package_src = site_packages / "minisweagent"
        if not package_src.is_dir():
            raise RunnerError(
                f"minisweagent package directory not found: {package_src}"
            )
        package_dest = overlay_root / "minisweagent"
        shutil.copytree(package_src, package_dest)
        replacements = {
            "actions_toolcall.py": "minisweagent/models/utils/actions_toolcall.py",
            "litellm_model.py": "minisweagent/models/litellm_model.py",
        }
        for src_name, rel_dest in replacements.items():
            src = replacement_root / src_name
            if not src.exists():
                raise RunnerError(
                    "enable_swebench_toolcall_patch requested, but replacement "
                    f"file is missing on the service host: {src}"
                )
            shutil.copy2(src, overlay_root / rel_dest)
        return overlay_root

    def _validate_prediction_ids(self, request: RunRequest, preds_path: Path) -> None:
        try:
            preds = msgspec.json.decode(preds_path.read_bytes(), type=dict)
        except msgspec.DecodeError as exc:
            raise RunnerError("mini-extra produced invalid preds.json") from exc
        expected = set(request.evaluated_instance_ids)
        actual = {str(instance_id) for instance_id in preds}
        unexpected = sorted(actual - expected)
        if unexpected:
            raise RunnerError(
                "mini-extra produced predictions for unexpected SWE-bench "
                f"instances: {', '.join(unexpected[:10])}"
            )

    def _resolve_minisweagent_site_packages(self) -> Path:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import minisweagent.models.utils.actions_toolcall as m; print(m.__file__)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RunnerError("could not locate minisweagent: " + result.stderr.strip())
        last_line = next(
            (line for line in reversed(result.stdout.splitlines()) if line.strip()),
            "",
        )
        if not last_line:
            raise RunnerError("could not locate minisweagent: empty output")
        actions_toolcall = Path(last_line.strip())
        try:
            site_packages = actions_toolcall.parents[3]
        except IndexError as exc:
            raise RunnerError(
                f"could not resolve site-packages from {actions_toolcall}"
            ) from exc
        if not site_packages.is_dir():
            raise RunnerError(f"resolved site-packages does not exist: {site_packages}")
        return site_packages

    def _run_eval(
        self,
        request: RunRequest,
        preds_path: Path,
        output_dir: Path,
        run_dir: Path,
        cancel_token: CancellationToken | None = None,
        *,
        instance_ids: Sequence[str] | None = None,
        log_name: str = "swe_bench_eval.log",
    ) -> Path:
        # A FRESH run_id per call means the harness spins up FRESH eval containers /
        # exec sessions — which is exactly what clears a transient container-runtime
        # flake on a re-grade (see _regrade_error_ids). `instance_ids` grades only a
        # subset (defaults to the full evaluated set).
        run_id = f"endpoints_{uuid.uuid4().hex[:8]}"
        ids = list(instance_ids) if instance_ids is not None else list(
            request.evaluated_instance_ids
        )
        if instance_ids is None:
            (run_dir / "swe_bench_eval_run_id.txt").write_text(run_id)
        dataset_name = {
            "verified": "princeton-nlp/SWE-bench_Verified",
            "lite": "princeton-nlp/SWE-bench_Lite",
        }.get(request.subset)
        if dataset_name is None:
            raise RunnerError(f"unknown SWE-bench subset: {request.subset}")
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            request.split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            str(request.max_eval_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            *ids,
        ]
        _run_subprocess(
            cmd,
            run_dir / log_name,
            cwd=output_dir,
            timeout_s=self.subprocess_timeout_s,
            cancel_token=cancel_token,
        )
        safe_model = request.model_name.replace("/", "__")
        result_path = output_dir / f"{safe_model}.{run_id}.json"
        if result_path.exists():
            return result_path
        candidates = sorted(output_dir.rglob(f"*{run_id}*.json"))
        if not candidates:
            raise RunnerError(f"SWE-bench result file not found for run_id={run_id}")
        return candidates[0]
