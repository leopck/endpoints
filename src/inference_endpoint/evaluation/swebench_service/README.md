# SWE-bench Service

Runs mini-swe-agent and the SWE-bench harness on a host with Docker or Pyxis. The
benchmark client only needs this service URL, but the service is trusted
infrastructure: it receives one endpoint URL and optional endpoint credentials, runs
container-backed evaluations, and serves run artifacts.

The isolated service subproject commits its own `uv.lock` so deployments use a
reproducible dependency set.

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080 \
  --auth-token "$SWEBENCH_SERVICE_AUTH_TOKEN"
```

The endpoint URL in the benchmark config must be reachable from the service host.
Service mode supports exactly one endpoint URL and follows the LiveCodeBench-style
external-service convention for heavyweight evaluation work.

### Endpoint credentials

`accuracy_config.extras.swebench_service_auth_token` authenticates the _client to
this service_. The credential the agent presents to the _model endpoint_ is
separate and comes from the run's endpoint configuration.

When no endpoint credential is configured, the agent subprocess is given
`OPENAI_API_KEY=EMPTY`, which is what an unauthenticated OpenAI-compatible server
expects. An `OPENAI_API_KEY` inherited from the service host's environment is never
forwarded to the endpoint; it is replaced by the placeholder. The variable is
always set, regardless of whether the endpoint is on loopback or on another host,
because the client library refuses to issue a request with no credential at all and
retries that refusal indefinitely.

## Runtime workflow

### Common workflow

The benchmark client sends the selected SWE-bench instances, model configuration,
and endpoint URL to the service. The service first runs mini-swe-agent to generate
one patch per instance and writes the patches to `preds.json`. It then evaluates
those predictions with the SWE-bench harness and returns the aggregate result and
retained run artifacts. The selected runtime changes where and how the task
containers execute; it does not change the benchmark client configuration or the
model endpoint request path.

### Docker runtime

Docker is the default runtime and is required only on the service host. During
generation, the service runs `mini-extra swebench` unchanged. mini-swe-agent selects
the official per-instance x86_64 SWE-bench image, starts a writable Docker container
for the trajectory, and executes every model tool call in that container so its
filesystem changes persist across turns.

After generation, the service passes `preds.json` to
`swebench.harness.run_evaluation`. The standard SWE-bench evaluator starts a fresh
Docker container for each prediction, applies the generated patch, runs the task's
evaluation script, captures its output, and grades it. The service collects the
result file produced by the harness and removes containers belonging to the run.

### Pyxis runtime

Select Pyxis with `--runtime pyxis --image-registry REGISTRY`, for example
`registry.example.com/group/project`. The current path uses ARM64 images named
`sweb.eval.arm64.<instance_id>:v4.1.0-arm64`. Pyxis pulls and caches them through
Enroot; configure registry credentials in `~/.config/enroot/.credentials` when the
registry requires authentication. Launch the service on the compute node inside an
active one-node Slurm allocation. The runtime requires `SLURM_JOB_ID` and
`SLURMD_NODENAME` and assumes the node is exclusive to the user.

For a staged multi-node allocation, use `--image-dir /node/local/images
--node-map /shared/instance_nodes.txt` instead of `--image-registry`. The map
contains one `instance_id node` pair per line. Each environment and eval step is
routed to that node, and terminal container cleanup fans out over every mapped
node. The image directory name must be valid on every target node and contain
`<instance_id>.sqsh` for each assignment. Run the worker allocation with Slurm
step manager enabled (`#SBATCH --stepmgr`) when the site supports it.

Each `srun` step is given an explicit allow-list of environment variables rather
than the service's whole environment, so that inherited `SLURM_JOB_ID` /
`SLURM_STEP_ID` cannot corrupt a nested `srun`. Several entries on that list are
load bearing on real clusters:

| Variable                                              | Why it must reach the step                                                                                                                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SLURM_CONF`                                          | Without it the child `srun` falls back to `/etc/slurm/slurm.conf` and aborts on a configless or multi-cluster site.                                                                         |
| `http_proxy`, `https_proxy`, `no_proxy` (+ uppercase) | Enroot performs the registry pull inside the step and needs the caller's proxy policy.                                                                                                      |
| `ENROOT_TEMP_PATH`, `ENROOT_CONFIG_PATH`              | Enroot creates the container inside the step. Dropping these discards the operator's override, so the multi-gigabyte create-time temp lands back on the device holding the unpacked rootfs. |

Credentials such as `OPENAI_API_KEY`, `HF_TOKEN` and the service auth token are
never forwarded, and no other `SLURM_*` variable is.

During generation, the service still uses mini-swe-agent for the agent loop and
model requests, but replaces its Docker environment with `PyxisEnvironment`. Every
trajectory receives a named, writable Pyxis container. By default, each tool call
becomes an overlapping `srun` step in that container, preserving filesystem
changes across turns. Tool commands run in private PID namespaces so one
trajectory cannot signal processes belonging to another trajectory.

An experimental persistent execution mode is available by setting
`SWEBENCH_PYXIS_PERSISTENT_EXEC=1`. It starts one long-lived, paced `srun` command
server after creating each named container. Later `execute()` calls use atomic,
per-nonce request and response files in the environment's private `/tmp` mount and
create no additional Slurm steps. The Bash server preserves separate stdout and
stderr, applies the normal command timeout in a private PID namespace, and
publishes `pending` -> `started` -> `finished` state before an atomic completion
manifest. It does not require Python in the task image.

The mode never replays an active request after its local `srun` client dies:
even a locally observed `pending` file cannot prove the remote step is gone and
will not claim the request later. Cleanup sends a file-based poison pill, then
performs bounded TERM/KILL/reap before the normal exact Enroot-container
removal. Keep this mode opt-in until it passes a CPU-only A/B with the site's
real staged images and Slurm configuration.

Each step reports its outcome through two channels. The primary one is in band:
the step script prints `__MLPERF_STEP_RC__ <nonce> <rc> <timed-out>` on `srun`'s
stdout, which needs no readable shared filesystem and is stripped from the
command output before it is returned. The fallback is the status file written
into the container's `/tmp` mount.

When a step reports through neither, `StepNotLaunched` (a `RunnerError`) is raised.
Besides `srun`'s own output it carries `srun_rc`, the observed `status` bytes, and
`provable_non_execution` -- true only when the status file is still `pending` and no
sentinel arrived, meaning the step script did not run even its first line and the
command definitely did not execute. Anything else leaves open that it did. Callers
deciding whether re-running is safe must use that flag rather than the message text.
An outer `srun` timeout is classified through the same channels after Python has
killed and reaped the scheduler client: `pending` remains safely retryable,
`started` is never replayed, and a sentinel or `finished:<rc>` is accepted as a
completed command. Partial scheduler output is retained in every failure.

Cluster note: on a busy controller these failures cluster around slurmctld RPC rate
limiting (`Job credential expired`). Pacing step creation below the controller's
`rl_refill_rate` is deployment-specific. Set
`SWEBENCH_PYXIS_STEP_RATE_PER_S` to a measured safe global step rate for the
service process. Set `SWEBENCH_PYXIS_STEP_RATE_STATE_PATH` to one shared file
to coordinate that budget across several service processes. The limiter admits
execution, retry, and cleanup steps at that fixed rate and is disabled when the
rate variable is unset. On OCI AGA, whose
controller refills 10 RPC tokens/s and where one step consumes several RPCs, a
3,240-step validation completed without a scheduler failure at 1.5 steps/s.

The reusable deployment controls are environment variables so they propagate to
the isolated agent and eval subprocesses:

| Variable                               | Purpose                                                                                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SWEBENCH_PYXIS_STEP_RATE_PER_S`       | Fixed scheduler-step admission rate; unset disables pacing.                                                                                      |
| `SWEBENCH_PYXIS_STEP_RATE_STATE_PATH`  | Optional shared lock/state file that makes the configured rate global across service processes.                                                  |
| `SWEBENCH_PYXIS_STEP_RATE_STATS_PATH`  | Optional atomically replaced JSON snapshot written only by a process that issued steps.                                                          |
| `SWEBENCH_PYXIS_STEP_CONCURRENCY`      | Optional bound on all in-flight scheduler steps.                                                                                                 |
| `SWEBENCH_PYXIS_CREATE_CONCURRENCY`    | Optional additional bound on concurrent image/container creation.                                                                                |
| `SWEBENCH_PYXIS_STEP_LAUNCH_GRACE_S`   | Extra outer deadline for scheduler launch; it does not change the command's inner timeout.                                                       |
| `SWEBENCH_PYXIS_STEP_RETRIES`          | Attempts for provable non-execution only; defaults to 3.                                                                                         |
| `SWEBENCH_PYXIS_INFRA_RETRY_LOG`       | Optional JSONL retry accounting sink.                                                                                                            |
| `SWEBENCH_PYXIS_PERSISTENT_EXEC`       | Opt in to one long-lived command-server step per environment; unset/false preserves one `srun` per command. Experimental pending a site CPU A/B. |
| `SWEBENCH_PYXIS_PERSISTENT_STATS_PATH` | Optional JSONL sink with per-environment command, failure, byte, and scheduler-step counters.                                                    |

After generation, the Pyxis worker evaluates each prediction in a fresh `srun`
container step because the Docker-based SWE-bench evaluator cannot run on the
compute node. It mounts the patch, SWE-bench evaluation script, and output file into
the task image. It preserves SWE-bench 4.1.0's patch-application order, test timeout,
captured output, and `get_eval_report` grading. A patch failure or test timeout is an
unresolved task; an `srun`, Enroot, or container-start failure is an infrastructure
loss, recorded per instance and reported rather than allowed to fail the whole run
(see _Eval-phase failures_). The service then aggregates the per-instance reports
and removes its named Pyxis containers.

Pyxis namespaces a named container by its allocation: `--container-name=X` inside
job `N` is the Enroot container `pyxis_N_X`. `PyxisEnvironment.cleanup()` removes
that name and logs a warning if the removal does not succeed. Each trajectory's
rootfs is on the order of gigabytes and is only reclaimed by this call, so a
removal that quietly fails fills the node's Enroot data path for the rest of the
allocation. `scancel` does not reclaim them either -- ending the job does not
remove Enroot containers -- which is why the removal has to be both correctly
named and audible.

The parameterized Slurm/Pyxis service recipe is in
`examples/10_Agentic_Inference/swebench_pyxis/`. It keeps cluster allocation,
storage, image staging, and scheduler-control values in the deployment
environment rather than embedding site policy in the service.

### Agent-phase failures

The agent phase runs `--workers` trajectories concurrently. If it fails after
some workers have already written their patches, the service still evaluates the
predictions that reached `preds.json` rather than discarding them: the failure is
recorded in the `agent_phase_error.txt` artifact and logged at ERROR, and the run
continues into the eval phase. Instances with no prediction are simply absent from
the results, which the harness reports as unresolved.

A run whose agent phase produced no predictions at all still fails, with the agent
error chained as the cause. Cancellation is never tolerated this way: it
propagates immediately and no eval phase runs.

### Eval-phase failures

The same rule applies one phase later. The eval phase grades each prediction in
its own container, concurrently. If some of those containers fail, the rest are
still graded and the run report is still produced: an instance with no
`report.json` is counted as an error by the harness, which is the correct and
visible outcome. The instances that were lost are listed in the
`eval_infra_failures.txt` artifact, one `instance_id<TAB>error` per line.

A run in which _no_ instance could be evaluated still fails — but only after the
report has been written, so the run can be diagnosed from its own artifacts.

### Telling a degraded run apart from a bad one

Both artifacts are machine-readable on purpose. `agent_phase_error.txt` means
some instances may be missing from `swe_bench_results.json`; `eval_infra_failures.txt`
names the instances the harness lost during grading. Instances lost this way are
_infrastructure_ losses, not model failures, and a consumer that cannot separate
the two will read attrition as an accuracy regression. Neither file exists on a
clean run.

The benchmark client submits a run to this service only in `ACC` or `BOTH`
mode; the default `PERF` mode skips external evaluation.

The service requires `--auth-token TOKEN` by default. Configure the client with:

```yaml
accuracy_config:
  extras:
    swebench_service_url: http://swebench-host:18080
    swebench_service_auth_token: TOKEN
```

For isolated local development only, pass `--allow-unauthenticated` explicitly.
`/health` is intentionally public for liveness probes; every run and artifact route
requires the bearer token.

The service selects templates from its packaged allowlist. Use
`accuracy_config.extras.swebench_template: qwen_tools` to select both the Qwen
template and packaged `QwenToolsModel`; otherwise omit the template option.
Completed run metadata and artifacts are retained up to `--max-stored-runs`
runs.
