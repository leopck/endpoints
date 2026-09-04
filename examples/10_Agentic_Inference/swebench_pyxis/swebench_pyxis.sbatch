#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=swebench-pyxis
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --stepmgr

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to this repository checkout}"
: "${RUN_ROOT:?set RUN_ROOT to durable shared storage}"
: "${SWEBENCH_IMAGE_DIR:?set SWEBENCH_IMAGE_DIR to staged node-local images}"
: "${SWEBENCH_NODE_MAP:?set SWEBENCH_NODE_MAP to the shared instance/node map}"
: "${SWEBENCH_SERVICE_AUTH_TOKEN_FILE:?set a protected token-file path}"
: "${SWEBENCH_PYXIS_STEP_RATE_PER_S:?set the measured scheduler admission rate}"
: "${SWEBENCH_PYXIS_STEP_RATE_STATE_PATH:?set a shared pacing-state path}"
: "${SWEBENCH_PYXIS_STEP_CONCURRENCY:?set the scheduler-step concurrency}"
: "${SWEBENCH_PYXIS_CREATE_CONCURRENCY:?set the create-step concurrency}"
: "${SWEBENCH_PYXIS_STEP_LAUNCH_GRACE_S:?set the scheduler launch grace}"
: "${SWEBENCH_PYXIS_STEP_RETRIES:?set the safe non-execution attempt count}"

token_mode=$(stat -c '%a' "$SWEBENCH_SERVICE_AUTH_TOKEN_FILE")
if [[ "$token_mode" != 600 ]]; then
    echo "token file must have mode 600, found ${token_mode}" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TMPDIR"
export SLURMD_NODENAME=$(hostname -s)

# Nested agent steps bind to the allocation, not this service step.
unset SLURM_STEP_ID SLURM_STEPID SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES \
    SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_STEP_LAUNCHER_PORT \
    SLURM_STEP_RESV_PORTS SLURM_SRUN_COMM_HOST SLURM_SRUN_COMM_PORT \
    SLURM_TASK_PID SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_GTIDS \
    SLURM_NTASKS SLURM_NPROCS SLURM_NNODES SLURM_JOB_NUM_NODES \
    SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_DISTRIBUTION \
    SLURM_LAUNCH_NODE_IPADDR SLURM_JOBID SLURM_CPUS_PER_TASK \
    SLURM_JOB_CPUS_PER_NODE

export SWEBENCH_PYXIS_PERSISTENT_EXEC=${SWEBENCH_PYXIS_PERSISTENT_EXEC:-1}
export SWEBENCH_PYXIS_INFRA_RETRY_LOG=${SWEBENCH_PYXIS_INFRA_RETRY_LOG:-$RUN_ROOT/infra_retries.jsonl}
export SWEBENCH_PYXIS_PERSISTENT_STATS_PATH=${SWEBENCH_PYXIS_PERSISTENT_STATS_PATH:-$RUN_ROOT/persistent_exec.jsonl}
export SWEBENCH_PYXIS_STEP_RATE_STATS_PATH=${SWEBENCH_PYXIS_STEP_RATE_STATS_PATH:-$RUN_ROOT/step_rate.json}

exec uv run \
    --project "$REPO_ROOT/src/inference_endpoint/evaluation/swebench_service" \
    python -m swebench_service \
    --host 0.0.0.0 \
    --port "${SWEBENCH_SERVICE_PORT:-18080}" \
    --artifact-root "$RUN_ROOT/service_artifacts" \
    --subprocess-timeout-s "${SWEBENCH_SUBPROCESS_TIMEOUT_S:-86400}" \
    --runtime pyxis \
    --image-dir "$SWEBENCH_IMAGE_DIR" \
    --node-map "$SWEBENCH_NODE_MAP" \
    --auth-token-file "$SWEBENCH_SERVICE_AUTH_TOKEN_FILE"
