"""Invoke a pi-py agent for one iteration and observe the result.

Deterministic plumbing only. We spawn `pi-py --mode json -p` rooted at the
target agent's dir and parse the newline-delimited JSON event stream, extracting:
  - `final_text`: the assistant's final message (the agent's narration), and
  - `tool_events`: factual tool_execution_end outputs (the observation seam the
    judges read instead of trusting narration).

Event shapes mirror the authoritative parser in pi_coding_agent's print/json
mode. This module is agent-agnostic — it never assumes a particular agent or
tool schema; tool outputs are captured opaquely and left for the judges to read.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from .models import IterationOutput

_TOOL_EVENT_LIMIT = 12
_TOOL_DETAIL_CHARS = 4000


def _resolve_pi_py_invocation() -> list[str]:
    """Prefer the explicit `pi-py` binary; never the Node `pi` on PATH. Fall back
    to the current interpreter running the module (works under `uv run`/venv)."""
    pi_py = shutil.which("pi-py")
    if pi_py:
        return [pi_py]
    return [sys.executable, "-m", "pi_coding_agent.main"]


def _detail_for(event: dict) -> dict | None:
    raw = event.get("result")
    details = None
    if isinstance(raw, dict):
        details = raw.get("details")
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            details = parsed.get("details") if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            details = None
    if isinstance(details, dict):
        return details
    if isinstance(raw, dict):
        content = raw.get("content")
        if isinstance(content, list):
            return {"content": content}
    return None


def _truncate(obj: object) -> object:
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) <= _TOOL_DETAIL_CHARS:
        return obj
    return {"_truncated": text[:_TOOL_DETAIL_CHARS] + "…"}


def run_agent(
    prompt: str,
    agent_dir: str,
    *,
    project_root: str | None = None,
    timeout: int | None = None,
) -> IterationOutput:
    """Run one agent turn headlessly and return what the driver observed."""
    argv = _resolve_pi_py_invocation() + ["--mode", "json", "-p", f"Task: {prompt}"]

    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = agent_dir
    env["PI_PROJECT_ROOT"] = os.path.abspath(project_root or agent_dir)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            argv,
            cwd=agent_dir,
            env=env,
            stdin=subprocess.DEVNULL,  # leave the parent's stdin free for steering input
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return IterationOutput(exit_code=124, error=f"agent timed out after {timeout}s: {e}")

    final_text = ""
    tool_events: list[dict] = []

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")

        if etype == "tool_execution_end":
            name = event.get("toolName") or event.get("tool_name") or ""
            if event.get("isError"):
                continue
            details = _detail_for(event)
            if details is not None and len(tool_events) < _TOOL_EVENT_LIMIT:
                tool_events.append({"tool": name, "details": _truncate(details)})
            continue

        if etype == "message_end":
            msg = event.get("message")
            if not isinstance(msg, dict):
                msg = {"role": event.get("role"), "content": event.get("content")}
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    final_text = block["text"]

    error = None
    if proc.returncode != 0:
        error = (proc.stderr or "").strip()[:1000] or f"exit {proc.returncode}"

    return IterationOutput(
        final_text=final_text,
        tool_events=tool_events,
        exit_code=proc.returncode,
        error=error,
    )
