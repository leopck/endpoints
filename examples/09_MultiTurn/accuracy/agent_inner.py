#!/usr/bin/env python3
"""Per-instance agent loop. Runs INSIDE the swebench Docker image's container.

The swebench prebaked images (Princeton/Stanford) ship a /opt/miniconda3/envs/
testbed Python that may be as old as 3.5 (django images). To run inside those,
this script avoids:
  - f-strings (3.6+)
  - PEP 585 / 604 type hints (3.9+ / 3.10+)
  - subprocess.run(capture_output=, text=)  (3.7+)
  - __future__ annotations (3.7+)

Reads /work/agent_input.json. Talks to vLLM over HTTP (api_base via env).
Executes the model's bash tool-calls (cwd /testbed). Loops up to step_limit.
Writes /work/agent_output.json, /work/model_patch.diff, /work/preds_entry.json.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_TOOL_OUTPUT_CHARS = 10000


def chat_completion(api_base, model, messages, tools, sampling, timeout=600):
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
    }
    body.update(sampling)
    data = json.dumps(body).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    last_err = None
    # 10 attempts × exponential backoff capped at 60s.
    # Total max wait: 1+2+4+8+16+32+60+60+60+60 = 303s ≈ 5 min.
    # Tolerates short vLLM hiccups but bails on persistent dead endpoints.
    for attempt in range(10):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            try:
                return json.loads(resp.read().decode("utf-8"))
            finally:
                resp.close()
        except (urllib.error.URLError, ConnectionError) as e:
            last_err = e
            backoff = min(2 ** attempt, 60)
            sys.stderr.write(
                "[agent-retry] attempt {0}/10 failed: {1}; sleeping {2}s\n".format(
                    attempt + 1, e, backoff
                )
            )
            sys.stderr.flush()
            time.sleep(backoff)
    raise RuntimeError("chat_completion failed after 10 retries: {0}".format(last_err))


def truncate_tool_output(stdout, stderr, rc):
    combined = stdout
    if stderr:
        combined += ("\n" + stderr) if stdout else stderr
    if len(combined) <= MAX_TOOL_OUTPUT_CHARS:
        return (
            "<returncode>{0}</returncode>\n"
            "<output>\n{1}\n</output>"
        ).format(rc, combined)
    half = MAX_TOOL_OUTPUT_CHARS // 2
    elided = len(combined) - MAX_TOOL_OUTPUT_CHARS
    return (
        "<returncode>{0}</returncode>\n"
        "<warning>\nThe output of your last command was too long.\n"
        "</warning>\n"
        "<output_head>\n{1}\n</output_head>\n"
        "<elided_chars>\n{2} characters elided\n</elided_chars>\n"
        "<output_tail>\n{3}\n</output_tail>"
    ).format(rc, combined[:half], elided, combined[-half:])


def run_bash(command, cwd, timeout=3600):
    env = dict(os.environ)
    env.update({
        "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
        "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
    })
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            universal_newlines=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return stdout or "", stderr or "", proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return stdout or "", stderr or "", 124
    except OSError as e:
        return "", "bash exec failed: {0}".format(e), 127


def extract_submitted_patch(messages):
    for m in reversed(messages):
        content = m.get("content") or ""
        if isinstance(content, str) and "diff --git" in content:
            start = content.find("diff --git")
            return content[start:]
    return None


def main():
    work = os.environ.get("AGENT_WORK_DIR", "/work")
    api_base = os.environ["AGENT_API_BASE"]
    model = os.environ["AGENT_MODEL"]
    step_limit = int(os.environ.get("AGENT_STEP_LIMIT", "100"))
    bash_timeout = int(os.environ.get("AGENT_BASH_TIMEOUT", "3600"))
    cwd = os.environ.get("AGENT_CWD", "/testbed")

    f = open(os.path.join(work, "agent_input.json"))
    try:
        inp = json.load(f)
    finally:
        f.close()

    instance_id = inp["instance_id"]
    system_prompt = inp["system_prompt"]
    instance_prompt = inp["instance_prompt"]
    sampling = inp.get("sampling", {})

    bash_tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command and return stdout/stderr/returncode. "
                "Commands run in a fresh subshell each time; cd persists only "
                "within the same command (chain with &&)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    trajectory = []
    submitted_patch = None
    final_reason = "step_limit"

    print("[{0}] agent start cwd={1} step_limit={2}".format(instance_id, cwd, step_limit))
    sys.stdout.flush()
    t_start = time.time()

    for step in range(step_limit):
        try:
            resp = chat_completion(api_base, model, messages, [bash_tool], sampling)
        except Exception as e:
            print("[{0}] step {1}: chat error: {2}".format(instance_id, step, e))
            sys.stdout.flush()
            final_reason = "chat_error: {0}".format(e)
            break

        choice = resp["choices"][0]
        msg = choice["message"]

        assistant_msg = {
            "role": "assistant",
            "content": msg.get("content"),
        }
        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_msg)
        trajectory.append({"step": step, "assistant": msg})

        finish = choice.get("finish_reason")
        content_str = msg.get("content") or ""

        if SUBMIT_SENTINEL in content_str:
            print("[{0}] step {1}: submit sentinel in content".format(instance_id, step))
            sys.stdout.flush()
            submitted_patch = extract_submitted_patch(messages)
            final_reason = "submitted_sentinel"
            break

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            if finish == "stop":
                print("[{0}] step {1}: no tool call, finish=stop".format(instance_id, step))
                sys.stdout.flush()
                final_reason = "no_tool_stop"
                break
            messages.append({
                "role": "user",
                "content": (
                    "You must include a bash tool call. Issue at least one "
                    "`bash` command per response."
                ),
            })
            continue

        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != "bash":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": "<error>unknown tool: {0!r}</error>".format(fn.get("name")),
                })
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
                command = args.get("command") or ""
            except (ValueError, TypeError):
                command = fn.get("arguments") or ""
            stdout, stderr, rc = run_bash(command, cwd=cwd, timeout=bash_timeout)

            if SUBMIT_SENTINEL in stdout:
                tool_content = truncate_tool_output(stdout, stderr, rc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_content,
                })
                submitted_patch = extract_submitted_patch(messages) or stdout
                print("[{0}] step {1}: submit sentinel in bash output".format(instance_id, step))
                sys.stdout.flush()
                final_reason = "submitted_sentinel"
                break

            tool_content = truncate_tool_output(stdout, stderr, rc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": tool_content,
            })
            trajectory.append({
                "step": step,
                "tool_call_id": tc.get("id"),
                "command": command,
                "returncode": rc,
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
            })

        if submitted_patch is not None:
            break

    elapsed = time.time() - t_start
    print("[{0}] agent done in {1:.1f}s; reason={2}; steps_taken={3}".format(
        instance_id, elapsed, final_reason, len(trajectory)))
    sys.stdout.flush()

    diff_out, diff_err, diff_rc = run_bash("git diff", cwd=cwd, timeout=120)
    f = open(os.path.join(work, "model_patch.diff"), "w")
    try:
        f.write(diff_out)
    finally:
        f.close()
    if diff_rc != 0:
        print("[{0}] WARN: git diff rc={1} stderr={2}".format(instance_id, diff_rc, diff_err[:200]))
        sys.stdout.flush()

    patch_to_score = submitted_patch or diff_out

    f = open(os.path.join(work, "agent_output.json"), "w")
    try:
        json.dump({
            "instance_id": instance_id,
            "final_reason": final_reason,
            "steps_taken": len(trajectory),
            "elapsed_seconds": elapsed,
            "submitted_patch_present": submitted_patch is not None,
            "model_patch_len": len(patch_to_score or ""),
            "trajectory": trajectory,
            "messages": messages,
        }, f, indent=2)
    finally:
        f.close()

    f = open(os.path.join(work, "preds_entry.json"), "w")
    try:
        json.dump({
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": patch_to_score or "",
        }, f, indent=2)
    finally:
        f.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
