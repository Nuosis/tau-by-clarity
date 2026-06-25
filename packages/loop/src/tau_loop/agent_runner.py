"""Invoke a tau agent for one iteration and observe the result.

Deterministic plumbing only. We spawn `tau --mode json -p` rooted at the
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
import signal
import subprocess
import sys

from .models import IterationOutput

_TOOL_EVENT_LIMIT = 12
_TOOL_DETAIL_CHARS = 4000
_SIGTERM_GRACE_SECONDS = 5


def _resolve_tau_invocation() -> list[str]:
    """Prefer the explicit `tau` binary; never the Node `pi` on PATH. Fall back
    to the current interpreter running the module (works under `uv run`/venv)."""
    tau_bin = shutil.which("tau")
    if tau_bin:
        return [tau_bin]
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
    argv = _resolve_tau_invocation() + ["--mode", "json", "-p", f"Task: {prompt}"]

    env = dict(os.environ)
    # Canonical PI_ name (back-compat) so the child resolves it via get_agent_dir().
    env["PI_CODING_AGENT_DIR"] = agent_dir
    env["PI_PROJECT_ROOT"] = os.path.abspath(project_root or agent_dir)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            argv,
            cwd=agent_dir,
            env=env,
            stdin=subprocess.DEVNULL,  # leave the parent's stdin free for steering input
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        return IterationOutput(exit_code=1, error=f"agent launch failed: {e}")

    def _kill_process_group(sig: int) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                return

    previous_handlers: dict[int, object] = {}

    def _handle_parent_signal(signum: int, frame: object) -> None:
        del frame
        _kill_process_group(signal.SIGTERM)
        try:
            proc.wait(timeout=_SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(signal.SIGKILL)
            proc.wait()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_parent_signal)
        except ValueError:
            previous_handlers.clear()
            break

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        _kill_process_group(signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=_SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(signal.SIGKILL)
            stdout, stderr = proc.communicate()
        return IterationOutput(exit_code=124, error=f"agent timed out after {timeout}s: {e}")
    except KeyboardInterrupt:
        _kill_process_group(signal.SIGTERM)
        try:
            proc.wait(timeout=_SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(signal.SIGKILL)
            proc.wait()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    final_text = ""
    tool_events: list[dict] = []

    for line in stdout.splitlines():
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
        error = (stderr or "").strip()[:1000] or f"exit {proc.returncode}"

    return IterationOutput(
        final_text=final_text,
        tool_events=tool_events,
        exit_code=proc.returncode,
        error=error,
    )
