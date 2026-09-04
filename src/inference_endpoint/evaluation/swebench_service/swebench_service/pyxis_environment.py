# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from .runner import RunnerError

logger = logging.getLogger(__name__)

_SAFE_SRUN_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    # Proxy policy must reach enroot, which performs the registry pull inside
    # the step. Clusters that pin a container-cache proxy system-wide 403 any
    # registry outside its allow-list, and without no_proxy every per-instance
    # image import fails with "CONNECT tunnel failed, response 403".
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    # srun locates its own configuration through SLURM_CONF. Dropping it makes
    # the child fall back to /etc/slurm/slurm.conf, which on a configless or
    # multi-cluster site is either absent ("Could not establish a configuration
    # source") or a different file whose plugins are not installed
    # ("failed to initialize cli_filter plugin"). Every step then fails before
    # the container is ever created. The remaining SLURM_* variables stay out:
    # inheriting SLURM_JOB_ID / SLURM_STEP_ID is what breaks a nested srun.
    "SLURM_CONF",
    # Enroot reads these when Pyxis creates the container, which happens inside
    # the step. Dropping them silently discards the operator's override, so the
    # ~2.5 GB create-time temp lands back on whichever device holds the unpacked
    # rootfs -- exactly the device the override existed to protect.
    "ENROOT_TEMP_PATH",
    "ENROOT_CONFIG_PATH",
)
_STEP_STATUS = "/tmp/.mlperf_srun_status"
#: In-band marker the step script prints alongside its own return code. It is
#: the primary result channel: it travels back on srun's stdout and so needs no
#: readable shared filesystem. The status file remains the fallback.
_STEP_SENTINEL = "__MLPERF_STEP_RC__"
#: The status file contents before the step script runs its very first line.
_STEP_STATUS_PENDING = "pending"
_STEP_SCRIPT = r"""set +e
status_path=$1
timeout_s=$2
nonce=$3
shift 3
printf 'started\n' > "$status_path" 2>/dev/null
timeout_log="${status_path}.timeout.${nonce}"
: > "$timeout_log"
lc_all_was_set=0
[ "${LC_ALL+x}" = x ] && lc_all_was_set=1
original_lc_all=${LC_ALL-}
unshare --pid --fork --mount-proc \
    env LC_ALL=C timeout --verbose -k 5 "$timeout_s" \
    bash -c '
        if [ "$1" -eq 1 ]; then export LC_ALL=$2; else unset LC_ALL; fi
        shift 2
        exec "$@" 2>&3
    ' pyxis-command "$lc_all_was_set" "$original_lc_all" "$@" \
    3>&2 2>"$timeout_log"
returncode=$?
cat "$timeout_log" >&2
timed_out=0
grep -q '^timeout: sending signal ' "$timeout_log" && timed_out=1
rm -f "$timeout_log"
printf 'finished:%s:%s\n' "$returncode" "$timed_out" > "$status_path" 2>/dev/null
printf '\n__MLPERF_STEP_RC__ %s %s %s\n' "$nonce" "$returncode" "$timed_out"
exit "$returncode"
"""

# Host-level inspection commands such as ``enroot list`` must run in the
# node's existing namespaces.  Some sites prohibit a nested PID namespace for
# those commands even though the same isolation is required for commands run
# inside a writable SWE-bench container.  Keep the status and sentinel
# protocol identical so opting out of PID isolation does not bypass timeout or
# non-launch classification.
_HOST_STEP_SCRIPT = r"""set +e
status_path=$1
timeout_s=$2
nonce=$3
shift 3
printf 'started\n' > "$status_path" 2>/dev/null
timeout_log="${status_path}.timeout.${nonce}"
: > "$timeout_log"
lc_all_was_set=0
[ "${LC_ALL+x}" = x ] && lc_all_was_set=1
original_lc_all=${LC_ALL-}
env LC_ALL=C timeout --verbose -k 5 "$timeout_s" \
    bash -c '
        if [ "$1" -eq 1 ]; then export LC_ALL=$2; else unset LC_ALL; fi
        shift 2
        exec "$@" 2>&3
    ' pyxis-command "$lc_all_was_set" "$original_lc_all" "$@" \
    3>&2 2>"$timeout_log"
returncode=$?
cat "$timeout_log" >&2
timed_out=0
grep -q '^timeout: sending signal ' "$timeout_log" && timed_out=1
rm -f "$timeout_log"
printf 'finished:%s:%s\n' "$returncode" "$timed_out" > "$status_path" 2>/dev/null
printf '\n__MLPERF_STEP_RC__ %s %s %s\n' "$nonce" "$returncode" "$timed_out"
exit "$returncode"
"""

_PERSISTENT_EXEC_ENV = "SWEBENCH_PYXIS_PERSISTENT_EXEC"
_PERSISTENT_STATS_ENV = "SWEBENCH_PYXIS_PERSISTENT_STATS_PATH"
_PERSISTENT_ROOT = "/tmp/.mlperf_persistent_exec"
_PERSISTENT_POLL_S = 0.05
_PERSISTENT_STATS_LOCK = threading.Lock()
_PERSISTENT_SERVER_SCRIPT = r"""set -u
root=$1
generation=$2
secret=$3
shift 3
interpreter=("$@")

atomic_write() {
    path=$1
    value=$2
    temporary="${path}.tmp.$$"
    printf '%s\n' "$value" > "$temporary" || exit 70
    mv -f -- "$temporary" "$path" || exit 70
}

mkdir -p "$root/requests" || exit 70
atomic_write "$root/server_status" started
atomic_write "$root/ready" "$generation"

while :; do
    if [ -f "$root/stop" ]; then
        atomic_write "$root/server_status" stopped
        exit 0
    fi
    handled=0
    for request in "$root"/requests/*; do
        [ -d "$request" ] || continue
        mkdir "$request/claim" 2>/dev/null || continue
        status=$(cat "$request/status" 2>/dev/null || printf unknown)
        if [ "$status" != pending ]; then
            rmdir "$request/claim" 2>/dev/null || true
            continue
        fi
        handled=1
        atomic_write "$request/status" started
        timeout_s=$(cat "$request/timeout" 2>/dev/null || printf invalid)
        separate=$(cat "$request/separate" 2>/dev/null || printf 0)
        case "$timeout_s" in
            ''|*[!0-9]*) returncode=125; timed_out=0 ;;
            *)
                cwd=$(cat "$request/cwd" 2>/dev/null || printf /testbed)
                command=$(cat "$request/command" 2>/dev/null || printf '')
                if [ "$separate" = 1 ]; then
                    (
                        cd -- "$cwd" || exit 125
                        unshare --pid --fork --mount-proc \
                            timeout -k 5 "$timeout_s" "${interpreter[@]}" "$command"
                    ) > "$request/stdout.tmp" 2> "$request/stderr.tmp"
                else
                    (
                        cd -- "$cwd" || exit 125
                        unshare --pid --fork --mount-proc \
                            timeout -k 5 "$timeout_s" "${interpreter[@]}" "$command"
                    ) > "$request/stdout.tmp" 2>&1
                    : > "$request/stderr.tmp"
                fi
                returncode=$?
                case "$returncode" in 124|137) timed_out=1 ;; *) timed_out=0 ;; esac
                ;;
        esac
        [ -f "$request/stdout.tmp" ] || : > "$request/stdout.tmp"
        [ -f "$request/stderr.tmp" ] || : > "$request/stderr.tmp"
        mv -f -- "$request/stdout.tmp" "$request/stdout" || exit 70
        mv -f -- "$request/stderr.tmp" "$request/stderr" || exit 70
        stdout_size=$(wc -c < "$request/stdout") || exit 70
        stderr_size=$(wc -c < "$request/stderr") || exit 70
        nonce=${request##*/}
        digest=$(
            {
                printf '%s\0%s\0%s\0%s\0%s\0%s\0' \
                    "$secret" "$nonce" "$returncode" "$stdout_size" \
                    "$stderr_size" "$timed_out"
                cat "$request/stdout"
                printf '\0'
                cat "$request/stderr"
                printf '\0%s' "$secret"
            } | sha256sum
        ) || exit 70
        digest=${digest%% *}
        atomic_write "$request/status" "finished:$returncode"
        atomic_write "$request/complete" \
            "$returncode $stdout_size $stderr_size $timed_out $digest"
    done
    [ "$handled" -eq 1 ] || sleep 0.05
done
"""


class StepNotLaunched(RunnerError):
    """An `srun` step that reported through neither result channel.

    Subclasses :class:`RunnerError` so every existing ``except RunnerError``
    keeps working, and records the facts a caller needs to reason about the
    failure rather than only read about it:

    ``srun_rc``
        `srun`'s own exit status.
    ``status``
        The bytes actually observed in the step status file.
    ``provable_non_execution``
        True only when the status file was still ``pending`` and no in-band
        sentinel arrived -- the step script did not run even its first line, so
        the command definitely did not execute. Anything else leaves open that
        it did, which is the distinction anyone deciding whether a re-run is
        safe has to make.
    """

    def __init__(
        self,
        message: str,
        *,
        provable_non_execution: bool,
        srun_rc: int | None,
        status: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.provable_non_execution = provable_non_execution
        self.srun_rc = srun_rc
        self.status = status
        self.stdout = stdout
        self.stderr = stderr


def read_step_sentinel(text: str, nonce: str) -> tuple[int | None, str]:
    """Return ``(returncode, output_without_the_sentinel)`` if the step reported.

    ``(None, text)`` when the step did not report in band. The nonce makes the
    marker unforgeable by the command's own output.
    """
    reported, _timed_out, cleaned = _read_step_sentinel_details(text, nonce)
    return reported, cleaned


def _read_step_sentinel_details(text: str, nonce: str) -> tuple[int | None, bool, str]:
    tag = f"{_STEP_SENTINEL} {nonce} "
    for line in reversed((text or "").splitlines()):
        if not line.startswith(tag):
            continue
        fields = line[len(tag) :].split()
        if (
            fields
            and fields[0].lstrip("-").isdigit()
            and (len(fields) == 1 or fields[1] in {"0", "1"})
        ):
            return (
                int(fields[0]),
                len(fields) > 1 and fields[1] == "1",
                text[: text.rindex(line)].rstrip("\n"),
            )
    return None, False, text


def _output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _read_step_status(status_path: Path) -> str:
    try:
        return status_path.read_text().strip()
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _read_sized_file(
    path: Path,
    size: int,
    deadline: float,
    *,
    read_bytes: Callable[[Path], bytes] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Read an atomically published response, retrying stale partial views."""
    reader = read_bytes or (lambda target: target.read_bytes())
    while True:
        try:
            data = reader(path)
        except OSError:
            data = b""
        if len(data) == size:
            return data
        if len(data) > size:
            raise RunnerError(
                f"persistent Pyxis response {path} grew past manifest size "
                f"({len(data)} > {size})"
            )
        if monotonic() >= deadline:
            raise RunnerError(
                f"persistent Pyxis response {path} remained partial "
                f"({len(data)} of {size} bytes)"
            )
        sleep(_PERSISTENT_POLL_S)


def _persistent_response_digest(
    *,
    secret: str,
    nonce: str,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    timed_out: bool,
) -> str:
    prefix = "\0".join(
        (
            secret,
            nonce,
            str(returncode),
            str(len(stdout)),
            str(len(stderr)),
            "1" if timed_out else "0",
        )
    ).encode()
    payload = prefix + b"\0" + stdout + b"\0" + stderr + b"\0" + secret.encode()
    return hashlib.sha256(payload).hexdigest()


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RunnerError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off)")


class _PersistentExecChannel:
    """One long-lived scheduler step serving atomic file-based requests."""

    def __init__(
        self,
        *,
        protocol_dir: Path,
        command_factory: Callable[[str, str], list[str]],
        admission_factory: Callable[[], AbstractContextManager[None]] = nullcontext,
        failure_path: Path | None = None,
        launch_timeout_s: float = 30.0,
        driver_grace_s: float = 30.0,
        shutdown_grace_s: float = 3.0,
        capture_stderr_separately: bool = False,
        retry_target: str = "persistent-pyxis-server",
    ) -> None:
        self.protocol_dir = protocol_dir
        self._requests_dir = protocol_dir / "requests"
        self._command_factory = command_factory
        self._admission_factory = admission_factory
        self._failure_path = failure_path
        self._launch_timeout_s = launch_timeout_s
        self._driver_grace_s = driver_grace_s
        self._shutdown_grace_s = shutdown_grace_s
        self._capture_stderr_separately = capture_stderr_separately
        self._retry_target = retry_target
        self._secret = secrets.token_hex(32)
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_handle: Any = None
        self._stderr_handle: Any = None
        self._lock = threading.Lock()
        self._closed = False
        self.stats: dict[str, int] = {
            "server_starts": 0,
            "commands": 0,
            "command_failures": 0,
            "nonzero_commands": 0,
            "safe_restarts": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        }
        self._requests_dir.mkdir(parents=True, exist_ok=True)

    def _log_tail(self) -> tuple[str, str]:
        def tail(path: Path) -> str:
            try:
                return path.read_text(errors="replace")[-2000:]
            except OSError:
                return ""

        return tail(self.protocol_dir / "server.stdout"), tail(
            self.protocol_dir / "server.stderr"
        )

    def _close_handles(self) -> None:
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.close()
                finally:
                    setattr(self, handle_name, None)

    def _discard_process(self) -> None:
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self._close_handles()

    def _launch_once(self) -> None:
        generation = uuid.uuid4().hex
        (self.protocol_dir / "ready").unlink(missing_ok=True)
        (self.protocol_dir / "stop").unlink(missing_ok=True)
        _atomic_write_text(self.protocol_dir / "server_status", "pending\n")
        stdout_path = self.protocol_dir / "server.stdout"
        stderr_path = self.protocol_dir / "server.stderr"
        self._stdout_handle = open(stdout_path, "ab", buffering=0)
        self._stderr_handle = open(stderr_path, "ab", buffering=0)
        try:
            with self._admission_factory():
                self._process = subprocess.Popen(
                    self._command_factory(generation, self._secret),
                    stdin=subprocess.DEVNULL,
                    stdout=self._stdout_handle,
                    stderr=self._stderr_handle,
                    env=safe_srun_env(),
                )
                self.stats["server_starts"] += 1
                deadline = time.monotonic() + self._launch_timeout_s
                while True:
                    try:
                        ready = (self.protocol_dir / "ready").read_text().strip()
                    except OSError:
                        ready = ""
                    if ready == generation:
                        return
                    returncode = self._process.poll()
                    if returncode is not None:
                        status = _read_step_status(self.protocol_dir / "server_status")
                        stdout, stderr = self._log_tail()
                        raise StepNotLaunched(
                            "persistent Pyxis server died before readiness "
                            f"(rc={returncode}, status={status!r})"
                            + _srun_evidence(stdout, stderr),
                            provable_non_execution=status == _STEP_STATUS_PENDING,
                            srun_rc=returncode,
                            status=status,
                            stdout=stdout,
                            stderr=stderr,
                        )
                    if time.monotonic() >= deadline:
                        status = _read_step_status(self.protocol_dir / "server_status")
                        stdout, stderr = self._log_tail()
                        raise StepNotLaunched(
                            "persistent Pyxis server did not become ready within "
                            f"{self._launch_timeout_s}s (status={status!r})"
                            + _srun_evidence(stdout, stderr),
                            provable_non_execution=status == _STEP_STATUS_PENDING,
                            srun_rc=None,
                            status=status,
                            stdout=stdout,
                            stderr=stderr,
                        )
                    time.sleep(_PERSISTENT_POLL_S)
        except Exception:
            self._terminate_process()
            raise

    def start(self) -> None:
        attempts = _step_retry_attempts()
        target = self._retry_target
        for attempt in range(1, attempts + 1):
            try:
                self._launch_once()
            except StepNotLaunched as exc:
                if not exc.provable_non_execution or attempt == attempts:
                    if self._failure_path is not None:
                        self._failure_path.touch()
                    _record_step_retry(
                        target=target,
                        attempt=attempt,
                        outcome=(
                            "not_retryable"
                            if not exc.provable_non_execution
                            else "exhausted"
                        ),
                        detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
                    )
                    raise
                _record_step_retry(
                    target=target,
                    attempt=attempt,
                    outcome="retrying",
                    detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
                )
                time.sleep(min(30.0, 2.0 * attempt))
                continue
            if attempt > 1:
                _record_step_retry(target=target, attempt=attempt, outcome="recovered")
            return
        raise AssertionError("unreachable")  # pragma: no cover

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            self._close_handles()
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._shutdown_grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._shutdown_grace_s)
        else:
            process.wait()
        self._discard_process()

    def _read_completion(
        self, request_dir: Path, deadline: float
    ) -> subprocess.CompletedProcess[str] | None:
        complete_path = request_dir / "complete"
        try:
            manifest = complete_path.read_text().strip().split()
        except OSError:
            return None
        if (
            len(manifest) != 5
            or not all(value.lstrip("-").isdigit() for value in manifest[:4])
            or re.fullmatch(r"[0-9a-f]{64}", manifest[4]) is None
        ):
            if time.monotonic() >= deadline:
                raise RunnerError(
                    f"persistent Pyxis completion marker is invalid: {manifest!r}"
                )
            return None
        returncode, stdout_size, stderr_size, timed_out = map(int, manifest[:4])
        digest = manifest[4]
        if stdout_size < 0 or stderr_size < 0 or timed_out not in {0, 1}:
            raise RunnerError("persistent Pyxis completion marker has invalid fields")
        stdout = _read_sized_file(request_dir / "stdout", stdout_size, deadline)
        stderr = _read_sized_file(request_dir / "stderr", stderr_size, deadline)
        expected_digest = _persistent_response_digest(
            secret=self._secret,
            nonce=request_dir.name,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=bool(timed_out),
        )
        if not hmac.compare_digest(digest, expected_digest):
            raise RunnerError("persistent Pyxis completion digest did not verify")
        self.stats["stdout_bytes"] += len(stdout)
        self.stats["stderr_bytes"] += len(stderr)
        if returncode != 0:
            self.stats["nonzero_commands"] += 1
        result = subprocess.CompletedProcess(
            ["persistent-pyxis-exec"],
            returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )
        result.__dict__["timed_out"] = bool(timed_out)
        return result

    def execute(
        self, *, command: str, cwd: str, timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        with self._lock:
            if self._closed:
                raise RunnerError("persistent Pyxis channel is closed")
            if self._process is None or self._process.poll() is not None:
                self.stats["command_failures"] += 1
                if self._failure_path is not None:
                    self._failure_path.touch()
                raise RunnerError("persistent Pyxis server is not running")
            nonce = uuid.uuid4().hex
            temporary = self._requests_dir / f".{nonce}.{os.getpid()}.tmp"
            request_dir = self._requests_dir / nonce
            temporary.mkdir()
            (temporary / "command").write_text(command)
            (temporary / "cwd").write_text(cwd)
            (temporary / "timeout").write_text(str(timeout_s))
            (temporary / "separate").write_text(
                "1\n" if self._capture_stderr_separately else "0\n"
            )
            (temporary / "status").write_text(f"{_STEP_STATUS_PENDING}\n")
            os.replace(temporary, request_dir)
            self.stats["commands"] += 1
            deadline = time.monotonic() + timeout_s + self._driver_grace_s
            while True:
                try:
                    completed = self._read_completion(request_dir, deadline)
                except RunnerError:
                    self.stats["command_failures"] += 1
                    if self._failure_path is not None:
                        self._failure_path.touch()
                    raise
                if completed is not None:
                    try:
                        shutil.rmtree(request_dir)
                    except OSError:
                        # Finished requests are ignored by the server and
                        # nonces are never reused. The private temp tree is
                        # removed with the environment, so response cleanup
                        # cannot turn a completed command into a failure.
                        logger.debug(
                            "could not remove completed persistent request %s",
                            request_dir,
                            exc_info=True,
                        )
                    return completed
                returncode = self._process.poll() if self._process is not None else None
                if returncode is not None:
                    status = _read_step_status(request_dir / "status")
                    stdout, stderr = self._log_tail()
                    self.stats["command_failures"] += 1
                    if self._failure_path is not None:
                        self._failure_path.touch()
                    raise StepNotLaunched(
                        "persistent Pyxis server died while a request was active "
                        f"(rc={returncode}, status={status!r})"
                        + _srun_evidence(stdout, stderr),
                        # Local srun-client death cannot prove that its remote
                        # step also died. Never start a second server while the
                        # first could still claim this request.
                        provable_non_execution=False,
                        srun_rc=returncode,
                        status=status,
                        stdout=stdout,
                        stderr=stderr,
                    )
                if time.monotonic() >= deadline:
                    status = _read_step_status(request_dir / "status")
                    self.stats["command_failures"] += 1
                    if self._failure_path is not None:
                        self._failure_path.touch()
                    raise StepNotLaunched(
                        "persistent Pyxis request exceeded its driver deadline "
                        f"(status={status!r})",
                        # A live server can cross pending->started immediately
                        # after this observation, so deadline expiry is never a
                        # safe replay decision.
                        provable_non_execution=False,
                        srun_rc=None,
                        status=status,
                    )
                time.sleep(_PERSISTENT_POLL_S)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            _atomic_write_text(self.protocol_dir / "stop", "stop\n")
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=self._shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    self._terminate_process()
                else:
                    self._discard_process()
            else:
                self._discard_process()


def safe_srun_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _SAFE_SRUN_ENV if name in os.environ}
    # The caller may deliberately put tempfile.TemporaryDirectory on shared
    # storage so a container routed to another node can mount it.  Passing
    # that same TMPDIR to slurmstepd is a separate concern: the remote node may
    # not have the path yet, and its fallback warning is merged into the
    # agent's command observation.  Resolve the host-side mount first, then
    # keep only the child step's temporary files node-local.
    env["TMPDIR"] = "/tmp"
    return env


_STEP_LAUNCH_GRACE_ENV = "SWEBENCH_PYXIS_STEP_LAUNCH_GRACE_S"


def _step_launch_grace_s() -> float:
    raw = os.environ.get(_STEP_LAUNCH_GRACE_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        grace = float(raw)
    except ValueError as exc:
        raise RunnerError(
            f"{_STEP_LAUNCH_GRACE_ENV} must be a finite number at least zero"
        ) from exc
    if not math.isfinite(grace) or grace < 0:
        raise RunnerError(
            f"{_STEP_LAUNCH_GRACE_ENV} must be a finite number at least zero"
        )
    return grace


def build_srun_command(
    *,
    argv: list[str],
    image: str | Path | None = None,
    name: str | None = None,
    mounts: list[tuple[Path, str]] | None = None,
    workdir: str | None = None,
    node: str | None = None,
) -> list[str]:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RunnerError("Pyxis runtime requires SLURM_JOB_ID")
    target_node = (node or os.environ.get("SLURMD_NODENAME") or "").strip()
    if not target_node:
        raise RunnerError("Pyxis runtime requires SLURMD_NODENAME")
    command = [
        "srun",
        "--overlap",
        f"--jobid={job_id}",
        "-N1",
        "-n1",
        f"--nodelist={target_node}",
    ]
    if image is not None:
        image_ref = str(image.resolve()) if isinstance(image, Path) else image
        command.append(f"--container-image={image_ref}")
    if name is not None:
        command.append(f"--container-name={name}")
    if image is not None or name is not None:
        command.extend(
            [
                "--container-writable",
                "--container-remap-root",
                "--no-container-mount-home",
            ]
        )
        if mounts:
            specs = []
            for source, destination in mounts:
                source_text = str(source.resolve())
                if "," in source_text or "," in destination:
                    raise RunnerError("Pyxis mount paths cannot contain commas")
                specs.append(f"{source_text}:{destination}")
            command.append("--container-mounts=" + ",".join(specs))
        if workdir is not None:
            command.append(f"--container-workdir={workdir}")
    command.extend(argv)
    return command


def _run_srun_step_once(
    *,
    argv: list[str],
    status_path: Path,
    timeout_s: int,
    failure_path: Path | None = None,
    image: str | Path | None = None,
    name: str | None = None,
    mounts: list[tuple[Path, str]] | None = None,
    workdir: str | None = None,
    node: str | None = None,
    stderr: int = subprocess.STDOUT,
    isolate_pid_namespace: bool = True,
) -> subprocess.CompletedProcess[str]:
    nonce = uuid.uuid4().hex
    status_path.write_text(f"{_STEP_STATUS_PENDING}\n")
    status_path.chmod(0o666)
    command = build_srun_command(
        image=image,
        name=name,
        mounts=mounts,
        workdir=workdir,
        node=node,
        argv=[
            "bash",
            "-c",
            _STEP_SCRIPT if isolate_pid_namespace else _HOST_STEP_SCRIPT,
            "pyxis-step",
            (_STEP_STATUS if isolate_pid_namespace else str(status_path.resolve())),
            str(timeout_s),
            nonce,
            *argv,
        ],
    )
    outer_timeout_s = timeout_s + 30 + _step_launch_grace_s()
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=stderr,
            timeout=outer_timeout_s,
            env=safe_srun_env(),
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run() has already killed and reaped its Popen child before
        # re-raising TimeoutExpired. Preserve the final communicate() output,
        # then consult the same two completion channels as the normal path.
        # A scheduler client can outlive a command that finished, or time out
        # before the step script ran at all; those outcomes are not equivalent.
        stdout = _output_text(exc.output)
        captured_stderr = _output_text(exc.stderr)
        reported, timed_out, cleaned = _read_step_sentinel_details(stdout, nonce)
        if reported is not None:
            completed = subprocess.CompletedProcess(
                command, reported, stdout=cleaned, stderr=captured_stderr
            )
            completed.__dict__["timed_out"] = timed_out
            return completed
        status = _read_step_status(status_path)
        finished = re.fullmatch(r"finished:(-?\d+):([01])", status)
        if finished is not None:
            completed = subprocess.CompletedProcess(
                command,
                int(finished.group(1)),
                stdout=stdout,
                stderr=captured_stderr,
            )
            completed.__dict__["timed_out"] = finished.group(2) == "1"
            return completed
        raise StepNotLaunched(
            f"Pyxis step exceeded its {outer_timeout_s}s outer deadline "
            f"(status={status!r})" + _srun_evidence(stdout, captured_stderr),
            provable_non_execution=status == _STEP_STATUS_PENDING,
            srun_rc=None,
            status=status,
            stdout=stdout,
            stderr=captured_stderr,
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        if failure_path is not None:
            failure_path.touch()
        raise RunnerError(
            "Pyxis infrastructure failure before the command completed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Primary channel: the step reported its own return code in band.
    reported, timed_out, cleaned = _read_step_sentinel_details(result.stdout, nonce)
    if reported is not None:
        result.stdout = cleaned
        result.returncode = reported
        result.__dict__["timed_out"] = timed_out
        return result

    # Fallback channel: the status file the step script wrote into the mount.
    status = _read_step_status(status_path)
    if status in {
        f"finished:{result.returncode}:0",
        f"finished:{result.returncode}:1",
    }:
        result.__dict__["timed_out"] = status.endswith(":1")
        return result

    raise StepNotLaunched(
        "Pyxis infrastructure failure before the command completed "
        f"(srun exited {result.returncode}, status={status!r})"
        + _srun_evidence(result.stdout, result.stderr),
        provable_non_execution=status == _STEP_STATUS_PENDING,
        srun_rc=result.returncode,
        status=status,
        stdout=_output_text(result.stdout),
        stderr=_output_text(result.stderr),
    )


def enroot_container_name(job_id: str, container_name: str) -> str:
    """The Enroot container name Pyxis derives from ``--container-name``.

    Pyxis namespaces every named container by the allocation it belongs to, so
    ``--container-name=X`` inside job ``N`` becomes the Enroot container
    ``pyxis_N_X``. Anything that later addresses the container by name --
    ``enroot list``, ``enroot remove`` -- has to use the same form.
    """
    return f"pyxis_{job_id}_{container_name}"


#: Opt-in JSONL sink for container-create durations. Off unless set, so this
#: adds nothing to a normal run. Creation is the step whose cost was invisible
#: -- it was only ever observable as a uniform block of SIGKILLs in `sacct`,
#: after the run was already lost -- so measuring it has to be possible without
#: re-deriving it from step accounting.
_CREATE_TIMING_ENV = "SWEBENCH_PYXIS_CREATE_TIMING_PATH"


def _record_create_timing(image: str | Path, seconds: float, *, ok: bool) -> None:
    path = os.environ.get(_CREATE_TIMING_ENV)
    if not path:
        return
    record = {
        "ts": time.time(),
        "image": str(image),
        "secs": round(seconds, 2),
        "ok": ok,
        "pid": os.getpid(),
    }
    try:
        # One short line per create, O_APPEND from many concurrent workers.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        # Observability must never be able to fail a run. A create that
        # succeeded and could not be logged is still a create that succeeded.
        logger.debug("could not record Pyxis create timing", exc_info=True)


def _srun_evidence(*outputs: str | bytes | None, limit: int = 2000) -> str:
    """Attach srun's own words to a Pyxis failure.

    srun/pyxis/enroot report the actual cause -- image import failure, no space
    left, a step that never got resources -- on the stream this function
    captures. Dropping it turns every distinct infrastructure failure into one
    indistinguishable message, which is exactly what made a 200-instance run's
    17 lost units undiagnosable from its artifacts.
    """
    parts: list[str] = []
    for output in outputs:
        if not output:
            continue
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output = output.strip()
        if output:
            parts.append(output)
    text = "\n".join(parts)
    if not text:
        return ""
    if len(text) > limit:
        text = "..." + text[-limit:]
    return f"\n--- srun output ---\n{text}"


#: Bounded re-attempts for a step that provably never launched. Set to 1 to
#: disable. A retry here is only ever reached when the step script did not run
#: its first line, so it cannot double-apply work -- see run_srun_step.
_STEP_RETRIES_ENV = "SWEBENCH_PYXIS_STEP_RETRIES"
_DEFAULT_STEP_RETRIES = 3
#: Optional JSONL sink recording every retry and its outcome. The schema matches
#: `swe_bench_distributed.infra_retry.RetryRecord`, which reads it back to
#: publish infra_retries_total / instances_saved_by_retry / run_quality. The two
#: sides cannot share code: this is an isolated subproject that must not import
#: the benchmark client, so they share a file format instead.
_STEP_RETRY_LOG_ENV = "SWEBENCH_PYXIS_INFRA_RETRY_LOG"
_RETRY_LOG_LOCK = threading.Lock()
_RETRY_LOG_ERRORS: dict[str, list[str]] = {}

#: Optional process-wide admission rate for new Slurm steps. One Pyxis agent
#: command is one ``srun --overlap`` step, and a step costs several controller
#: RPCs. At large worker counts, limiting concurrent steps is insufficient:
#: sustained creation can exhaust slurmctld's RPC bucket and leave srun
#: backing off until its job credential expires. The deployment must set this
#: below its controller-specific safe rate; an unset value preserves the
#: package's existing unpaced behavior.
_STEP_RATE_ENV = "SWEBENCH_PYXIS_STEP_RATE_PER_S"
_STEP_RATE_STATE_ENV = "SWEBENCH_PYXIS_STEP_RATE_STATE_PATH"
_STEP_RATE_STATS_ENV = "SWEBENCH_PYXIS_STEP_RATE_STATS_PATH"
_STEP_PACER_UNSET = object()


class _StepPacer:
    """Thread-safe fixed-rate admission for step creation.

    Slots are reserved under the lock and slept outside it. This avoids both a
    burst after a quiet period and holding the lock while one caller waits.
    """

    def __init__(
        self,
        rate_per_s: float,
        *,
        state_path: Path | None = None,
        monotonic: Any = time.monotonic,
        wall_time: Any = time.time,
        sleep: Any = time.sleep,
    ) -> None:
        self.rate_per_s = rate_per_s
        self._interval_s = 1.0 / rate_per_s
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._sleep = sleep
        self._state_path = state_path
        self._next_slot = 0.0
        self._started_at = monotonic()
        self._last_report_at = 0.0
        self._issued = 0
        self._waited_s = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._state_path is None:
                now = self._monotonic()
                slot = max(now, self._next_slot)
                self._next_slot = slot + self._interval_s
            else:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with self._state_path.open("a+", encoding="utf-8") as handle:
                        fcntl.flock(handle, fcntl.LOCK_EX)
                        handle.seek(0)
                        raw = handle.read().strip()
                        now = self._wall_time()
                        try:
                            next_slot = float(raw) if raw else 0.0
                        except ValueError:
                            next_slot = 0.0
                        slot = max(now, next_slot)
                        handle.seek(0)
                        handle.truncate()
                        handle.write(f"{slot + self._interval_s:.9f}\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError as exc:
                    raise RunnerError(
                        f"could not reserve shared Pyxis step-rate slot in "
                        f"{self._state_path}: {exc}"
                    ) from exc
            self._issued += 1
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        self._record_stats(delay)

    def _record_stats(self, delay: float) -> None:
        path_text = os.environ.get(_STEP_RATE_STATS_ENV, "").strip()
        with self._lock:
            self._waited_s += max(0.0, delay)
            now = self._monotonic()
            if not path_text or now - self._last_report_at < 60.0:
                return
            self._last_report_at = now
            elapsed = max(1e-6, now - self._started_at)
            payload = {
                "pid": os.getpid(),
                "budget_steps_per_s": self.rate_per_s,
                "steps_issued": self._issued,
                "observed_steps_per_s": round(self._issued / elapsed, 3),
                "time_spent_throttled_s": round(self._waited_s, 1),
                "elapsed_s": round(elapsed, 1),
            }
        path = Path(path_text.replace("{pid}", str(os.getpid())))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=1) + "\n")
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            logger.debug("could not write Pyxis step-rate stats", exc_info=True)


_STEP_PACER_CONFIG_LOCK = threading.Lock()
_STEP_PACER_SETTING: object | tuple[str, str] = _STEP_PACER_UNSET
_STEP_PACER: _StepPacer | None = None

_STEP_CONCURRENCY_ENV = "SWEBENCH_PYXIS_STEP_CONCURRENCY"
_CREATE_CONCURRENCY_ENV = "SWEBENCH_PYXIS_CREATE_CONCURRENCY"
_LIMITERS_CONFIG_LOCK = threading.Lock()
_LIMITERS_SETTING: tuple[str, str] | None = None
_STEP_LIMITER: threading.BoundedSemaphore | None = None
_CREATE_LIMITER: threading.BoundedSemaphore | None = None


def _optional_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RunnerError(f"{name} must be an integer greater than zero") from exc
    if value <= 0:
        raise RunnerError(f"{name} must be an integer greater than zero")
    return value


def _configured_limiters() -> tuple[
    threading.BoundedSemaphore | None, threading.BoundedSemaphore | None
]:
    global _LIMITERS_SETTING, _STEP_LIMITER, _CREATE_LIMITER

    setting = (
        os.environ.get(_STEP_CONCURRENCY_ENV, "").strip(),
        os.environ.get(_CREATE_CONCURRENCY_ENV, "").strip(),
    )
    with _LIMITERS_CONFIG_LOCK:
        if setting != _LIMITERS_SETTING:
            step_limit = _optional_positive_int(_STEP_CONCURRENCY_ENV)
            create_limit = _optional_positive_int(_CREATE_CONCURRENCY_ENV)
            _STEP_LIMITER = (
                threading.BoundedSemaphore(step_limit) if step_limit else None
            )
            _CREATE_LIMITER = (
                threading.BoundedSemaphore(create_limit) if create_limit else None
            )
            _LIMITERS_SETTING = setting
        return _STEP_LIMITER, _CREATE_LIMITER


@contextmanager
def srun_step_admission(*, is_create: bool = False) -> Iterator[None]:
    """Apply rate and concurrency controls to one scheduler step."""
    step_limiter, create_limiter = _configured_limiters()
    with ExitStack() as stack:
        # One order everywhere prevents create and general admission deadlocks.
        if is_create and create_limiter is not None:
            stack.enter_context(create_limiter)
        if step_limiter is not None:
            stack.enter_context(step_limiter)
        # Take the rate slot only once the concurrency permit is held. Pacing
        # before the semaphore lets already-paced callers accumulate behind a
        # long image create and burst together when permits are released.
        _pace_srun_step()
        yield


def _pace_srun_step() -> None:
    """Wait for the process-wide step-admission slot when configured."""
    global _STEP_PACER_SETTING, _STEP_PACER

    setting = (
        os.environ.get(_STEP_RATE_ENV, "").strip(),
        os.environ.get(_STEP_RATE_STATE_ENV, "").strip(),
    )
    with _STEP_PACER_CONFIG_LOCK:
        if setting != _STEP_PACER_SETTING:
            rate_setting, state_setting = setting
            if not rate_setting:
                pacer = None
            else:
                try:
                    rate_per_s = float(rate_setting)
                except ValueError as exc:
                    raise RunnerError(
                        f"{_STEP_RATE_ENV} must be a finite number greater than zero"
                    ) from exc
                if not math.isfinite(rate_per_s) or rate_per_s <= 0:
                    raise RunnerError(
                        f"{_STEP_RATE_ENV} must be a finite number greater than zero"
                    )
                pacer = _StepPacer(
                    rate_per_s,
                    state_path=Path(state_setting) if state_setting else None,
                )
                logger.info(
                    "limiting Pyxis step creation to %.3f/s%s",
                    rate_per_s,
                    (f" using shared state {state_setting}" if state_setting else ""),
                )
            _STEP_PACER = pacer
            _STEP_PACER_SETTING = setting
        pacer = _STEP_PACER
    if pacer is not None:
        pacer.wait()


def _step_retry_attempts() -> int:
    raw = os.environ.get(_STEP_RETRIES_ENV, "").strip()
    if not raw:
        return _DEFAULT_STEP_RETRIES
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", _STEP_RETRIES_ENV, raw)
        return _DEFAULT_STEP_RETRIES


def _record_step_retry(
    *, target: str, attempt: int, outcome: str, detail: str | None = None
) -> None:
    path = os.environ.get(_STEP_RETRY_LOG_ENV)
    if not path:
        return
    record = {
        "target": target,
        "attempt": attempt,
        "outcome": outcome,
        "detail": detail,
        "at": time.time(),
    }
    try:
        # Accounting must never be able to take a run down.
        with _RETRY_LOG_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        key = str(Path(path).resolve())
        with _RETRY_LOG_LOCK:
            _RETRY_LOG_ERRORS.setdefault(key, []).append(
                f"could not append retry record for {target} attempt {attempt}"
            )
        logger.debug("could not append to the infra retry log", exc_info=True)


def read_step_retry_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read retry JSONL without letting corrupt accounting look like success."""
    with _RETRY_LOG_LOCK:
        accounting_errors = list(_RETRY_LOG_ERRORS.get(str(path.resolve()), ()))
    if not path.exists():
        return [], accounting_errors
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return [], [*accounting_errors, f"could not read {path}: {exc}"]
    records: list[dict[str, Any]] = []
    errors: list[str] = accounting_errors
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict) or not isinstance(record.get("outcome"), str):
            errors.append(f"line {line_number}: expected an object with string outcome")
            continue
        records.append(record)
    return records, errors


def run_srun_step(**kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run one `srun` step, re-attempting only a *provable* non-launch.

    Retrying is a correctness decision, not a convenience: re-running a command
    that may already have run can apply an edit twice, delete twice, or double a
    test run, and none of those announce themselves. So the only failure retried
    here is :class:`StepNotLaunched` with ``provable_non_execution`` -- the status
    file still ``pending`` and no in-band sentinel, meaning the step script did
    not execute even its first line. Every other failure, including a
    ``StepNotLaunched`` that reached ``started``, is raised immediately.

    Measured signature, from an isolated probe with no model and no GPU (20
    nodes, 200 workers, 6273 ordinary shell steps): 63 steps failed and in all 63
    the status file still read ``pending``.

    Every attempt and outcome is appended to ``SWEBENCH_PYXIS_INFRA_RETRY_LOG``
    when set. A retry loop that quietly absorbs the defect it compensates for
    turns a broken cluster into an invisible one.
    """
    attempts = _step_retry_attempts()
    target = str(kwargs.get("name") or kwargs.get("image") or "pyxis-step")
    is_create = kwargs.get("image") is not None
    for attempt in range(1, attempts + 1):
        try:
            # Every actual attempt, including a retry, is a new scheduler step.
            with srun_step_admission(is_create=is_create):
                result = _run_srun_step_once(**kwargs)
        except StepNotLaunched as exc:
            if not exc.provable_non_execution:
                # The command may have run. Another attempt could double it.
                failure_path = kwargs.get("failure_path")
                if failure_path is not None:
                    Path(failure_path).touch()
                _record_step_retry(
                    target=target,
                    attempt=attempt,
                    outcome="not_retryable",
                    detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
                )
                raise
            outcome = "exhausted" if attempt == attempts else "retrying"
            _record_step_retry(
                target=target,
                attempt=attempt,
                outcome=outcome,
                detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
            )
            if attempt == attempts:
                failure_path = kwargs.get("failure_path")
                if failure_path is not None:
                    Path(failure_path).touch()
                raise
            logger.warning(
                "Pyxis step provably never launched (attempt %d/%d, srun rc=%s, "
                "status=%r); retrying",
                attempt,
                attempts,
                exc.srun_rc,
                exc.status,
            )
            time.sleep(min(30.0, 2.0 * attempt))
            continue
        if attempt > 1:
            _record_step_retry(target=target, attempt=attempt, outcome="recovered")
        result.__dict__["srun_attempts"] = attempt
        return result
    raise AssertionError("unreachable")  # pragma: no cover


def _validate_instance_id(instance_id: str) -> None:
    if Path(instance_id).name != instance_id or instance_id in {".", ".."}:
        raise RunnerError(f"invalid SWE-bench instance ID: {instance_id}")


def resolve_image(
    image_registry: str | None,
    instance_id: str,
    *,
    image_dir: Path | None = None,
) -> str | Path:
    """Resolve one SWE-bench image from a registry or a staged local store."""
    _validate_instance_id(instance_id)
    if image_dir is not None:
        return image_dir / f"{instance_id}.sqsh"
    if image_registry is None:
        raise RunnerError("Pyxis runtime requires an image registry or image directory")
    image_registry = image_registry.rstrip("/")
    if "#" not in image_registry:
        host, separator, repository = image_registry.partition("/")
        if not separator:
            raise RunnerError("Pyxis image registry must include a repository")
        image_registry = f"{host}#{repository}"
    return f"{image_registry}/sweb.eval.arm64.{instance_id.lower()}:v4.1.0-arm64"


def load_node_map(path: Path | None) -> dict[str, str]:
    """Load an ``instance_id node`` routing file used by multi-node Pyxis."""
    if path is None:
        return {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise RunnerError(f"could not read Pyxis node map {path}: {exc}") from exc
    assignments: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise RunnerError(
                f"invalid Pyxis node map line {line_number}: expected instance_id node"
            )
        instance_id, node = fields
        _validate_instance_id(instance_id)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node) is None:
            raise RunnerError(
                f"invalid Pyxis node name on line {line_number}: {node!r}"
            )
        previous = assignments.setdefault(instance_id, node)
        if previous != node:
            raise RunnerError(
                f"conflicting Pyxis node assignments for {instance_id}: "
                f"{previous} and {node}"
            )
    return assignments


class PyxisEnvironmentConfig(BaseModel):
    image: str | Path
    run_id: str
    node: str | None = None
    cwd: str = "/testbed"
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: int = Field(
        default=30,
        validation_alias=AliasChoices("timeout_s", "timeout"),
        serialization_alias="timeout",
    )
    #: Deadline for *creating* the container, which under Pyxis includes the
    #: enroot import of a multi-GB SWE-bench image from a remote registry.
    #: Deliberately separate from ``timeout_s``: that is a per-*command*
    #: budget, sized for `pytest`-scale work inside an already-running
    #: container. Charging an image import against it made every agent whose
    #: image was not already in the enroot cache fail once the registry was
    #: shared by enough concurrent workers to push a single import past ~5
    #: minutes. Defaults to, and accepts, mini-swe-agent's ``pull_timeout``.
    create_timeout_s: int = Field(
        default=3600,
        validation_alias=AliasChoices("create_timeout_s", "pull_timeout"),
        serialization_alias="pull_timeout",
    )
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-c"])
    infrastructure_failure_path: Path | None = None
    capture_stderr_separately: bool = False
    persistent_exec: bool = False


class PyxisEnvironment:
    def __init__(self, **kwargs: Any):
        self.config = PyxisEnvironmentConfig(**kwargs)
        self._persistent_enabled = self.config.persistent_exec or _env_flag(
            _PERSISTENT_EXEC_ENV
        )
        self._persistent_channel: _PersistentExecChannel | None = None
        self._scheduler_steps = 0
        self._direct_command_steps = 0
        self._cleanup_steps = 0
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", self.config.run_id)[:24]
        self.name = f"mswe_{safe_run_id}_{uuid.uuid4().hex[:8]}"
        self._tmp = tempfile.TemporaryDirectory(prefix=f"pyxis_{self.name}_")
        self._tmp_dir = Path(self._tmp.name)
        # Pyxis remaps container root to the submitting host uid, so the
        # environment's private mount does not need world access. Keeping the
        # TemporaryDirectory's owner-only mode prevents other host users from
        # reading tool commands, responses, and the persistent-channel secret.
        self._tmp_dir.chmod(0o700)
        self._lock = threading.Lock()
        self._cleaned = False
        started = time.monotonic()
        try:
            # A no-op initializes and validates the named persistent container.
            create_result = run_srun_step(
                image=self.config.image,
                name=self.name,
                mounts=[(self._tmp_dir, "/tmp")],
                workdir=self.config.cwd,
                argv=["true"],
                status_path=self._tmp_dir / Path(_STEP_STATUS).name,
                timeout_s=self.config.create_timeout_s,
                failure_path=self.config.infrastructure_failure_path,
                node=self.config.node,
            )
            self._scheduler_steps += getattr(create_result, "srun_attempts", 1)
            if self._persistent_enabled:
                protocol_dir = self._tmp_dir / Path(_PERSISTENT_ROOT).name
                protocol_dir.mkdir()
                self._persistent_channel = _PersistentExecChannel(
                    protocol_dir=protocol_dir,
                    command_factory=self._persistent_server_command,
                    admission_factory=lambda: srun_step_admission(),
                    failure_path=self.config.infrastructure_failure_path,
                    launch_timeout_s=30.0 + _step_launch_grace_s(),
                    capture_stderr_separately=self.config.capture_stderr_separately,
                    retry_target=f"persistent-pyxis-server:{self.name}",
                )
                self._persistent_channel.start()
        except Exception as exc:
            _record_create_timing(
                self.config.image, time.monotonic() - started, ok=False
            )
            self.cleanup()
            if not isinstance(exc, RunnerError):
                exc = RunnerError(
                    "persistent Pyxis initialization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise RunnerError(
                f"failed to start Pyxis container for {self.config.image}: {exc}"
            ) from exc
        _record_create_timing(self.config.image, time.monotonic() - started, ok=True)

    def _persistent_server_command(self, generation: str, secret: str) -> list[str]:
        argv = ["env"]
        argv.extend(f"{key}={value}" for key, value in self.config.env.items())
        argv.extend(
            [
                "bash",
                "-c",
                _PERSISTENT_SERVER_SCRIPT,
                "pyxis-persistent-server",
                _PERSISTENT_ROOT,
                generation,
                secret,
                *self.config.interpreter,
            ]
        )
        return build_srun_command(
            argv=argv,
            name=self.name,
            mounts=[(self._tmp_dir, "/tmp")],
            workdir=self.config.cwd,
            node=self.config.node,
        )

    def _persistent_stats(self) -> dict[str, Any]:
        channel_stats = (
            dict(self._persistent_channel.stats)
            if self._persistent_channel is not None
            else {}
        )
        server_steps = channel_stats.get("server_starts", 0)
        commands = channel_stats.get("commands", 0)
        scheduler_steps = self._scheduler_steps + server_steps
        return {
            "enabled": self._persistent_enabled,
            "scheduler_steps": scheduler_steps,
            "container_create_steps": self._scheduler_steps
            - self._direct_command_steps
            - self._cleanup_steps,
            "server_start_steps": server_steps,
            "direct_command_steps": self._direct_command_steps,
            "cleanup_steps": self._cleanup_steps,
            "persistent_commands": commands,
            "scheduler_steps_per_persistent_command": (
                round(scheduler_steps / commands, 6) if commands else None
            ),
            **channel_stats,
        }

    def _write_persistent_stats(self) -> None:
        path = os.environ.get(_PERSISTENT_STATS_ENV, "").strip()
        if not path:
            return
        payload = {
            "at": time.time(),
            "pid": os.getpid(),
            "environment": self.name,
            **self._persistent_stats(),
        }
        try:
            with _PERSISTENT_STATS_LOCK, open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            logger.debug("could not append persistent Pyxis stats", exc_info=True)

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        logger.debug("Executing Pyxis command: %s", command)
        capture_stderr_separately = getattr(
            self.config, "capture_stderr_separately", False
        )
        timeout_s = timeout or self.config.timeout_s
        persistent_channel = getattr(self, "_persistent_channel", None)
        if persistent_channel is not None:
            result = persistent_channel.execute(
                command=command,
                cwd=cwd or self.config.cwd,
                timeout_s=timeout_s,
            )
        else:
            argv = ["env"]
            argv.extend(f"{key}={value}" for key, value in self.config.env.items())
            argv.extend([*self.config.interpreter, command])
            result = run_srun_step(
                argv=argv,
                status_path=self._tmp_dir / Path(_STEP_STATUS).name,
                timeout_s=timeout_s,
                failure_path=self.config.infrastructure_failure_path,
                name=self.name,
                mounts=[(self._tmp_dir, "/tmp")],
                workdir=cwd or self.config.cwd,
                node=self.config.node,
                stderr=(
                    subprocess.PIPE if capture_stderr_separately else subprocess.STDOUT
                ),
            )
            attempts = getattr(result, "srun_attempts", 1)
            if hasattr(self, "_scheduler_steps"):
                self._scheduler_steps += attempts
                self._direct_command_steps += attempts
        output: dict[str, Any]
        if result.returncode == 124 or getattr(result, "timed_out", False):
            output = {
                "output": result.stdout,
                "returncode": -1,
                "exception_info": "The command timed out",
                "extra": {
                    "exception_type": "TimeoutExpired",
                    "exception": (f"command timed out after {timeout_s}s"),
                },
            }
        else:
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        if capture_stderr_separately or persistent_channel is not None:
            output.setdefault("extra", {})["stderr"] = result.stderr or ""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            # mini-swe-agent is installed only in the SWE-bench service subproject.
            from minisweagent.exceptions import Submitted

            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **self.config.model_dump(by_alias=True),
            **platform.uname()._asdict(),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json", by_alias=True),
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                    "persistent_exec_stats": self._persistent_stats(),
                }
            }
        }

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
        try:
            persistent_channel = getattr(self, "_persistent_channel", None)
            if persistent_channel is not None:
                try:
                    persistent_channel.close()
                except (OSError, subprocess.SubprocessError):
                    # A failed poison/TERM/KILL must not prevent the exact
                    # named-container removal below. That cleanup is what
                    # reclaims the writable rootfs and any surviving step.
                    logger.warning(
                        "Could not stop persistent Pyxis command server %s",
                        self.name,
                        exc_info=True,
                    )
            job_id = os.environ.get("SLURM_JOB_ID", "").strip()
            if job_id:
                container = enroot_container_name(job_id, self.name)
                try:
                    # Cleanup creates a Slurm step too; leaving it outside the
                    # process-wide admission path recreates the same RPC burst
                    # when many workers finish together.
                    with srun_step_admission():
                        if hasattr(self, "_scheduler_steps"):
                            self._scheduler_steps += 1
                            self._cleanup_steps += 1
                        completed = subprocess.run(
                            build_srun_command(
                                argv=["enroot", "remove", "-f", container],
                                node=self.config.node,
                            ),
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            env=safe_srun_env(),
                        )
                except (OSError, RunnerError, subprocess.SubprocessError):
                    logger.warning(
                        "Could not remove Pyxis container %s",
                        container,
                        exc_info=True,
                    )
                else:
                    if completed.returncode != 0:
                        # Never silent: an unreclaimed rootfs is ~2.5 GB and
                        # they accumulate for the whole allocation.
                        logger.warning(
                            "enroot remove %s exited %s: %s",
                            container,
                            completed.returncode,
                            (completed.stderr or completed.stdout or "").strip()[-500:],
                        )
        finally:
            if hasattr(self, "_persistent_enabled"):
                self._write_persistent_stats()
            self._tmp.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            logger.warning(
                "Could not clean up Pyxis environment",
                exc_info=True,
            )
