# SWE-bench accuracy primitives

Self-contained building blocks for a SWE-bench Verified accuracy pass against
the Qwen3.6 server brought up by [`../server.sflow.yaml`](../server.sflow.yaml).
No transport assumptions (no ssh, no slurm, no pyxis) — just the agent, the
config, and the scoring helpers.

## Files

| File                       | What it is                                                              |
| -------------------------- | ----------------------------------------------------------------------- |
| `agent_inner.py`           | Per-instance agent loop. Reads `/work/agent_input.json`, talks to vLLM over HTTP, executes bash tool-calls in `/testbed`, writes `model_patch.diff` + `preds_entry.json`. Python 3.5-compatible (runs inside swebench's pre-baked images). |
| `swebench_local.yaml`      | mini-swe-agent config: prompts, sampling, `api_base: http://localhost:30000/v1`. |
| `score_inline_accuracy.py` | Inline scorer for endpoints `events.jsonl` (multi-turn benchmark, not SWE-bench). |
| `rebuild_preds.py`         | Rebuild `preds.json` from per-instance `preds_entry.json` files.        |

## How `agent_inner.py` is used

Run one copy per SWE-bench instance inside that instance's swebench Docker image.
The orchestration layer (whatever wires N parallel containers up) is left to
the caller. Each container needs:

- `/work/agent_input.json` with `{instance_id, problem_statement, model, api_base, sampling, tools, step_limit, cost_limit}`
- `/testbed/` as the cwd (the swebench image provides this)
- network access to the vLLM/SMG endpoint set in `api_base`

Outputs land in `/work/`:
- `agent_output.json` — full trajectory
- `model_patch.diff` — git diff of the agent's edits
- `preds_entry.json` — `{instance_id, model_name_or_path, model_patch}` row

Aggregate the `preds_entry.json` files into a single `preds.json` with
`rebuild_preds.py <output_dir>`, then feed it to
`python -m swebench.harness.run_evaluation --predictions_path preds.json ...`
for scoring.

## Hooking up to your own orchestrator

```python
# Pseudocode — orchestrator agnostic.
for instance in swebench_verified:
    # Spin up the instance's container (Docker, podman, k8s, whatever):
    #   image = swebench/sweb.eval.x86_64.<instance_id>:latest
    #   mounts: ./work:/work
    #   stage agent_input.json + agent_inner.py into /work/
    # Then run: python3 /work/agent_inner.py
    ...
# After all instances:
#   python3 rebuild_preds.py output_dir/
#   python -m swebench.harness.run_evaluation \
#     --predictions_path output_dir/preds.json --run_id <name> ...
```

For the inline accuracy scorer (multi-turn benchmark, scores `events.jsonl`
produced by the `inference-endpoint` client when `multi_turn.inline_accuracy:
true`):

```bash
python3 score_inline_accuracy.py \
  --gt <gt.jsonl> --domain coding \
  --report-dir <client report_dir>
```
