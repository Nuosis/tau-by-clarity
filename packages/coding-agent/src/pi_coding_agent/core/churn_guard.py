"""Churn guard: every 12 turns Jev judges the last 12; churn stops the session and reports it.

USER-AUTHORIZED — Marcus, November 4, 2026: churn wastes money and tokens; it is stopped
immediately without asking, superseding other requirements. Experiment
~/outputs/churn-experiment-20261104: at 12 turns, P(churn) > 0.5 separated every turn-based
churn case (0.53-0.98) from normal sessions (0.10-0.43).

A turn is one assistant step (including a failed model call) with the inputs since the
previous one. Turns are counted from the session journal, so a restarted process resumes
the count. A stopped session refuses further prompts until TAU_CHURN_RESUME=1.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Callable

import aiohttp

INTERVAL = 12
THRESHOLD = 0.5
JEV_MODEL = "typesafe/jev-1.13"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
STOP_ENTRY = "tau.churn_stop"
RESUME_ENTRY = "tau.churn_resume"
CHECK_ENTRY = "tau.churn_check"
QUESTION = {
    "type": "noul",
    "instructions": "These are the most recent consecutive turns of an AI agent or automated process, "
                    "oldest first. Is it churning: repeating the same or equivalent actions, retries, "
                    "failures, waits or reviews without making progress toward its goal?",
    "criteria": {
        "true": "The turns show repetition without progress: the same failure, retry, blocked action, "
                "wait or review recurs with no new result.",
        "false": "The turns show progress, or not enough repetition to call it churn.",
    },
}


class ChurnStopped(RuntimeError):
    """The session was stopped for churn; resuming needs TAU_CHURN_RESUME=1."""


def _clip(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif item.get("type") == "toolCall":
            parts.append(f"[tool call {item.get('name')}({_clip(json.dumps(item.get('arguments')), 300)})]")
    return " ".join(parts)


def _raw(entries) -> list[dict]:
    return [getattr(e, "data", e) for e in entries]


def turns(entries) -> list[str]:
    """Render the journal's assistant steps, each with the inputs that preceded it."""
    rendered, pending = [], []
    for entry in _raw(entries):
        if entry.get("type") != "message":
            continue
        message = entry.get("message") or {}
        role = message.get("role")
        if role in ("user", "toolResult"):
            label = "TOOL RESULT" if role == "toolResult" else "INPUT"
            pending.append(f"{label}: {_clip(_content(message.get('content')), 600)}")
        elif role == "assistant":
            out = _clip(_content(message.get("content")), 700)
            if (message.get("stopReason") or message.get("stop_reason")) == "error":
                error = message.get("errorMessage") or message.get("error_message") or ""
                out += f" [MODEL CALL FAILED: {_clip(error, 300)}]"
            rendered.append("\n".join(pending[-2:] + [f"ASSISTANT: {out}"]))
            pending = []
    return rendered


def stopped(entries) -> dict | None:
    """The unresolved churn stop recorded in this session, if any."""
    stop = None
    for entry in _raw(entries):
        if entry.get("type") != "custom":
            continue
        if entry.get("customType") == STOP_ENTRY:
            stop = entry.get("data") or {}
        elif entry.get("customType") == RESUME_ENTRY:
            stop = None
    return stop


async def judge(window: list[str], get_api_key: Callable) -> float:
    """Jev's probability that these turns are churn."""
    document = "\n\n".join(f"--- Turn {i + 1} of {len(window)} ---\n{t}" for i, t in enumerate(window))
    endpoint = os.environ.get("TAU_JEV_ENDPOINT") or JEV_ENDPOINT
    headers = {"Content-Type": "application/json"}
    if not os.environ.get("TAU_JEV_ENDPOINT"):
        # A configured endpoint (for example a vault proxy) supplies its own credential.
        key = await get_api_key("openrouter")
        if not key:
            raise RuntimeError("OpenRouter API key is required for churn detection")
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": JEV_MODEL, "state": {"document": {"text": document}},
               "questions": {"is_churn": QUESTION}}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90, connect=15)) as client:
        async with client.post(endpoint, json=payload, headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
    probability = result["answers"]["is_churn"]["noul"]
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("Jev returned an invalid churn probability")
    return float(probability)


def report(details: dict) -> None:
    """Tell the operator: stderr always, plus TAU_CHURN_ALERT_COMMAND with the details on stdin."""
    print(f"[tau] CHURN STOPPED: {json.dumps(details)}", file=sys.stderr, flush=True)
    command = os.environ.get("TAU_CHURN_ALERT_COMMAND")
    if command:
        try:
            subprocess.run(command, shell=True, input=json.dumps(details), text=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[tau] churn alert command failed: {type(exc).__name__}", file=sys.stderr, flush=True)


class ChurnGuard:
    """Counts assistant turns in the journal and judges every INTERVAL-th one."""

    def __init__(self, sessions: Callable[[], Any], get_api_key: Callable, stop: Callable, emit: Callable):
        self._sessions = sessions  # current session manager (sessions can be switched)
        self._get_api_key = get_api_key
        self._stop = stop
        self._emit = emit
        self._tasks: set[asyncio.Task] = set()

    def check_resumable(self) -> None:
        """Refuse to start model work in a session stopped for churn."""
        stop = stopped(self._sessions().get_entries())
        if stop is None:
            return
        if os.environ.get("TAU_CHURN_RESUME") == "1":
            self._sessions().append_custom_entry(RESUME_ENTRY, {"at": time.time(), "stop": stop})
            return
        raise ChurnStopped(f"Session stopped for churn (P={stop.get('p_churn')}); "
                           "set TAU_CHURN_RESUME=1 to resume it.")

    def is_stopped(self) -> bool:
        return stopped(self._sessions().get_entries()) is not None

    def on_assistant_turn(self) -> None:
        sessions = self._sessions()  # the check stays bound to the session it judged
        all_turns = turns(sessions.get_entries())
        if not all_turns or len(all_turns) % INTERVAL:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._check(sessions, all_turns[-INTERVAL:], len(all_turns)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for churn checks already started."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _check(self, sessions, window: list[str], turn: int) -> None:
        try:
            probability = await judge(window, self._get_api_key)
        except Exception as exc:
            # A failed judgment is recorded and reported; it never counts as churn.
            sessions.append_custom_entry(CHECK_ENTRY, {"turn": turn, "error": type(exc).__name__})
            print(f"[tau] churn check failed at turn {turn}: {type(exc).__name__}", file=sys.stderr, flush=True)
            return
        sessions.append_custom_entry(CHECK_ENTRY, {"turn": turn, "p_churn": probability})
        if probability <= THRESHOLD:
            return
        details = {"session_id": sessions.get_session_id(),
                   "session_file": str(sessions.get_session_file() or ""),
                   "turn": turn, "p_churn": probability, "threshold": THRESHOLD, "at": time.time(),
                   "last_turn": window[-1][-600:], "resume": "set TAU_CHURN_RESUME=1 and prompt the session"}
        sessions.append_custom_entry(STOP_ENTRY, details)
        # Stop first; only the session that was judged is aborted.
        if self._sessions() is sessions:
            await self._stop()
        self._emit({"type": "churn_stopped", **details})
        await asyncio.to_thread(report, details)
