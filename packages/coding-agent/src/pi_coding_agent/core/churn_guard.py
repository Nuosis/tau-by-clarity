"""Churn guard: every 12 turns Jev judges the last 12; churn stops the session and reports it.

USER-AUTHORIZED — Marcus, November 4, 2026: churn wastes money and tokens; it is stopped
immediately without asking, superseding other requirements. Experiment
~/outputs/churn-experiment-20261104: at 12 turns, P(churn) > 0.5 separated every turn-based
churn case (0.53-0.98) from normal sessions (0.10-0.43).

A turn is one assistant step (a failed model call or an output-limit retry included) with
the inputs since the previous one, counted on the active branch of the session journal so
a restarted process keeps the count. The judgment runs before the model call that would
follow the 12th, 24th, ... turn; churn means that call is never made. A stopped session
refuses prompts until TAU_CHURN_RESUME=1.
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
# Output-limit recovery retries a generation without an assistant message; each attempt is a turn.
OUTPUT_LIMIT_ENTRY = "tau.output_limit_attempt"
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


def active_branch(sessions) -> list[dict]:
    """Entries on the path from the root to the current leaf (abandoned branches excluded)."""
    path, current = [], sessions.get_leaf_entry()
    while current is not None:
        path.append(getattr(current, "data", current))
        parent = getattr(current, "parent_id", None)
        current = sessions.get_entry(parent) if parent else None
    path.reverse()
    return path


def _is_turn(entry: dict) -> bool:
    if entry.get("type") == "custom":
        return entry.get("customType") == OUTPUT_LIMIT_ENTRY
    return entry.get("type") == "message" and (entry.get("message") or {}).get("role") == "assistant"


def turns(entries: list[dict]) -> list[str]:
    """Render each turn with the inputs that preceded it."""
    rendered, pending = [], []
    for entry in entries:
        if entry.get("type") == "custom" and entry.get("customType") == OUTPUT_LIMIT_ENTRY:
            detail = _clip(json.dumps(entry.get("data") or {}), 300)
            rendered.append("\n".join(pending[-2:] + [f"ASSISTANT: [OUTPUT LIMIT REACHED; GENERATION RETRIED: {detail}]"]))
            pending = []
            continue
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


def tail(entries: list[dict], count: int) -> tuple[int, list[str]]:
    """Total turns and the last `count` rendered; only the tail is rendered."""
    positions = [i for i, e in enumerate(entries) if _is_turn(e)]
    start = positions[-count - 1] + 1 if len(positions) > count else 0
    return len(positions), turns(entries[start:])[-count:]


def stopped(entries: list[dict]) -> dict | None:
    """The unresolved churn stop on this branch, if any."""
    stop = None
    for entry in entries:
        if entry.get("type") != "custom":
            continue
        if entry.get("customType") == STOP_ENTRY:
            stop = entry.get("data") or {}
        elif entry.get("customType") == RESUME_ENTRY:
            stop = None
    return stop


def _protect(text: str) -> str:
    """Apply the registered PII filter, as ordinary model calls do."""
    from pi_ai import pii

    factory = getattr(pii, "_factory", None)
    return factory()[0](text) if factory else text


async def judge(window: list[str], get_api_key: Callable, route: Callable | None = None) -> float:
    """Jev's probability that these turns are churn."""
    document = "\n\n".join(f"--- Turn {i + 1} of {len(window)} ---\n{t}" for i, t in enumerate(window))
    # The established OpenRouter route (proxy baseUrl, headers) unless TAU_JEV_ENDPOINT overrides it.
    endpoint, route_headers = route() if route else (JEV_ENDPOINT, {})
    endpoint = os.environ.get("TAU_JEV_ENDPOINT") or endpoint
    headers = {**(route_headers or {}), "Content-Type": "application/json"}
    if not os.environ.get("TAU_JEV_ENDPOINT") and not any(k.lower() == "authorization" for k in headers):
        # A configured endpoint or a route with its own Authorization header supplies the credential.
        key = await get_api_key("openrouter")
        if not key:
            raise RuntimeError("OpenRouter API key is required for churn detection")
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": JEV_MODEL, "state": {"document": {"text": _protect(document)}},
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
    """Judges the active branch before model calls; churn raises ChurnStopped instead of calling."""

    def __init__(self, sessions: Callable[[], Any], get_api_key: Callable, emit: Callable,
                 route: Callable | None = None):
        self._sessions = sessions  # current session manager (sessions can be switched)
        self._get_api_key = get_api_key
        self._emit = emit
        self._route = route

    def _refusal(self, stop: dict) -> ChurnStopped:
        return ChurnStopped(f"Session stopped for churn (P={stop.get('p_churn')}); "
                            "set TAU_CHURN_RESUME=1 to resume it.")

    def check_resumable(self) -> None:
        """Refuse to start work in a session stopped for churn."""
        stop = stopped(active_branch(self._sessions()))
        if stop is None:
            return
        if os.environ.get("TAU_CHURN_RESUME") == "1":
            self._sessions().append_custom_entry(RESUME_ENTRY, {"at": time.time(), "stop": stop})
            return
        raise self._refusal(stop)

    def is_stopped(self) -> bool:
        return stopped(active_branch(self._sessions())) is not None

    async def before_model_call(self) -> None:
        """Raise ChurnStopped rather than let a churning session make its next model call."""
        sessions = self._sessions()
        entries = active_branch(sessions)
        stop = stopped(entries)
        if stop is not None:
            raise self._refusal(stop)
        total, window = tail(entries, INTERVAL)
        if not total or total % INTERVAL or any(
                e.get("type") == "custom" and e.get("customType") == CHECK_ENTRY
                and (e.get("data") or {}).get("turn") == total for e in entries):
            return
        try:
            probability = await judge(window, self._get_api_key, self._route)
        except Exception as exc:
            # A failed judgment is recorded and reported; it never counts as churn.
            sessions.append_custom_entry(CHECK_ENTRY, {"turn": total, "error": type(exc).__name__})
            print(f"[tau] churn check failed at turn {total}: {type(exc).__name__}", file=sys.stderr, flush=True)
            return
        sessions.append_custom_entry(CHECK_ENTRY, {"turn": total, "p_churn": probability})
        if probability <= THRESHOLD:
            return
        details = {"session_id": sessions.get_session_id(),
                   "session_file": str(sessions.get_session_file() or ""),
                   "turn": total, "p_churn": probability, "threshold": THRESHOLD, "at": time.time(),
                   "last_turn": window[-1][-600:], "resume": "set TAU_CHURN_RESUME=1 and prompt the session"}
        sessions.append_custom_entry(STOP_ENTRY, details)
        self._emit({"type": "churn_stopped", **details})
        await asyncio.to_thread(report, details)
        raise self._refusal(details)
