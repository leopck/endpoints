# Distributed SWE-bench with Slurm and Pyxis

This example starts one authenticated SWE-bench service inside an existing
multi-node Slurm allocation. Instance images must already be staged on the
nodes named by the shared node map.

The repository does not choose cluster policy. Supply the account, partition,
node count, time limit, storage paths, and scheduler-control values when the
job is submitted. A typical submission is:

```bash
sbatch \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --nodes="${SWEBENCH_NODES}" \
  --time="${SLURM_TIME}" \
  --export=ALL,REPO_ROOT,RUN_ROOT,SWEBENCH_IMAGE_DIR,SWEBENCH_NODE_MAP,SWEBENCH_SERVICE_AUTH_TOKEN_FILE,SWEBENCH_PYXIS_STEP_RATE_PER_S,SWEBENCH_PYXIS_STEP_RATE_STATE_PATH,SWEBENCH_PYXIS_STEP_CONCURRENCY,SWEBENCH_PYXIS_CREATE_CONCURRENCY,SWEBENCH_PYXIS_STEP_LAUNCH_GRACE_S,SWEBENCH_PYXIS_STEP_RETRIES \
  examples/10_Agentic_Inference/swebench_pyxis/swebench_pyxis.sbatch
```

`SWEBENCH_IMAGE_DIR` must resolve to a node-local directory with one
`<instance_id>.sqsh` file for each assignment. `SWEBENCH_NODE_MAP` is a shared
file containing one `instance_id node` pair per line. All nodes must resolve
the image directory to their own staged copy.

Set `SWEBENCH_PYXIS_PERSISTENT_EXEC=1` to keep one command-server step alive
inside each writable environment. The pacing and concurrency settings are
deployment controls; measure values that remain below the target Slurm
controller's step-creation capacity rather than copying another site's values.

The service prints its URL after readiness. Start additional services on
different allocation nodes when using `swe_bench_fleet`, and provide their
unique URLs through `swebench_service_urls`.
