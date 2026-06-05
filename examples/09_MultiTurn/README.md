# Multi-Turn Agentic Benchmark — Qwen3.6 + SMG

Three-piece flow for benchmarking Qwen3.6-35B-A3B served by 8 vLLM workers
behind the SMG cache-aware gateway, replaying multi-turn agentic conversations
through the mlcommons/endpoints `inference-endpoint` client.

| Step | File                            | What it does                                                 |
| ---- | ------------------------------- | ------------------------------------------------------------ |
| 1    | `server.sflow.yaml`             | nv-sflow workflow: 8× `vllm serve` + 1× SMG gateway on :30000. |
| 2    | `qwen_agentic_benchmark.yaml`   | endpoints client config: multi-turn replay against :30000.   |
| 3    | `accuracy/`                     | SWE-bench Verified accuracy pass against the same server.    |

The `kimi_agentic_benchmark.yaml` reference config alongside this README is the
upstream Kimi-K2 example. The schema doc for the client YAML format is at the
bottom of this file.

## 1. Launch the server

Fill in `backends.slurm_cluster.{account,partition}` in `server.sflow.yaml`
for your site (or swap in a `local` backend for a single-node dry-run), then:

```bash
sflow run -f server.sflow.yaml --tui
```

Brings up:
- `system_capture` — host/CUDA/NCCL/Torch snapshot for reproducibility
- `hf_prefetch`    — pre-downloads `Qwen/Qwen3.6-35B-A3B` into the shared cache
- `vllm_workers`   — 8 replicas on ports 8100..8107, one per GPU
- `smg_gateway`    — SMG `--policy cache_aware` on :30000 in front of all 8

Probes block until each worker prints `Application startup complete` and SMG's
`/v1/models` returns 200. Override variables with `--set MODEL_NAME=... --set
VLLM_IMAGE=...@sha256:...` (pin the digest for repeatable runs).

## 2. Run the client

Update `datasets[0].path` in `qwen_agentic_benchmark.yaml` to point at your
agentic JSONL, then:

```bash
uv run inference-endpoint benchmark from-config \
  --config qwen_agentic_benchmark.yaml
```

Endpoint is wired to `http://localhost:30000` (the SMG gateway). The client
runs `target_concurrency=128` multi-turn trajectories; inline accuracy scores
events.jsonl at finalize-time.

Outputs land in the config's `report_dir`:
- `events.jsonl`           — per-turn trace
- `final_snapshot.json`    — primary metrics source (TTFT/TPOT percentiles, throughput, OSL)
- `report.txt`             — human-readable summary
- `scores.json`            — inline accuracy (when `multi_turn.inline_accuracy: true`)

## 3. Run accuracy

See [`accuracy/README.md`](accuracy/README.md). The folder ships the building
blocks (per-instance agent loop, mini-swe-agent config pointing at :30000, the
inline-accuracy scorer, the preds rebuilder) — orchestrate parallel
instances with whatever you have on hand (Docker, podman, k8s, ray, GNU
parallel).

---

## Reference: client YAML schema

The client YAML (used by `from-config`) maps every field to a Pydantic model in
`endpoints/src/inference_endpoint/config/schema.py`. Key fields:

- `name`: human-readable run name written to reports and logs.
- `type: "online"`: online scheduler. Keep `"online"`.
- `model_params.name`: model name sent in each OpenAI request. Match the
  server's `--served-model-name`.
- `model_params.{temperature,top_p,max_new_tokens}`: sampling. For Qwen3.6
  agentic: `1.0 / 0.95 / 20000`.
- `model_params.chat_template_kwargs.{thinking,preserve_thinking}`: enable
  Qwen-style reasoning preservation across turns.
- `datasets[0].type: performance`: multi-turn replay.
- `datasets[0].path`: JSONL with one row per turn (see Dataset section below).
- `datasets[0].multi_turn.enable_salt: true`: deterministic salt markers so
  repeat iterations don't reuse KV cache by accident.
- `datasets[0].multi_turn.inject_tool_delay: true`: honors positive
  `delay_seconds` in the dataset before issuing user/tool turns.
- `datasets[0].multi_turn.inline_accuracy: true`: scores `events.jsonl` at the
  end of the run; writes `scores.json` under `report_dir`.
- `datasets[0].multi_turn.num_trajectories_to_issue`: total trajectories to
  start. Use an integer multiple of the dataset trajectory count.
- `datasets[0].multi_turn.stop_issuing_on_first_user_complete: false`: keeps
  replaying already-started trajectories to completion for accuracy. Set
  `true` during optimization for shorter tails; keep `false` for final.
- `settings.load_pattern.type: multi_turn`: conversation-aware issuing.
- `settings.load_pattern.target_concurrency`: max active conversations (one
  in-flight request per active conversation).
- `endpoint_config.endpoints`: server URL list. `http://localhost:30000` for
  the SMG gateway in step 1.
- `endpoint_config.api_type: openai`: routes to `/v1/chat/completions`.
- `report_dir`: output directory for events, snapshots, scores, reports.

### Dataset

Flat JSONL, one row per message. Rows for each `conversation_id` must be
contiguous and ordered by increasing `turn`:

```jsonl
{"conversation_id":"c1","turn":1,"role":"user","system":"...","content":"...","tools":[...],"delay_seconds":0.4}
{"conversation_id":"c1","turn":2,"role":"assistant","tool_calls":[...]}
{"conversation_id":"c1","turn":3,"role":"tool","tool_results":[...],"delay_seconds":1.2}
{"conversation_id":"c1","turn":4,"role":"assistant","content":"..."}
```

Required: `conversation_id`, `turn`, `role`. User rows normally include
`content`; agentic rows can also carry `system`, `tools`, `tool_calls`,
`tool_results`, `reasoning_content`, `delay_seconds`.

### Salting

`multi_turn.enable_salt: true` injects a short deterministic `[salt: ...]`
marker before and after the system prompt:

1. Fully allowed within a trajectory.
2. System prompt allowed within the same iteration of the dataset.
3. Disallowed across multiple iterations of the dataset.

### Tail management

Multi-turn benchmarks have long tails (turn counts vary widely). The client
emits `STOP_PERFORMANCE_TRACKING` when the first active user finishes its
final assigned trajectory; turns issued *before* that point stay in the
performance window, turns *after* don't.

For final submissions keep `stop_issuing_on_first_user_complete: false` so the
client drains already-started trajectories for accuracy coverage. For
optimization/debug, set `true` to cut the tail.
