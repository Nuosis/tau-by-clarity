"""Append-only CLI debug logging.

This log is intentionally outside the session transcript. It is for diagnosing
cases where the process exits, crashes, hangs, or loses the session write path
before the JSONL session can explain what happened.
"""
from __future__ import annotations

import atexit
import faulthandler
import json
import os
import platform
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import Any, Sequence

from pi_coding_agent.config import VERSION, get_debug_log_path

_LOCK = threading.Lock()
_RUN_ID = uuid.uuid4().hex[:12]
_LOG_PATH: str | None = None
_FAULTHANDLER_FILE = None
_CONFIGURED = False
_EXIT_LOGGED = False

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "key",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
}
IMAGE_BASE64_REDACT_THRESHOLD_BYTES = 1024
IMAGE_BASE64_REPLACEMENT_TEMPLATE = "<image:base64-redacted bytes={n}>"
_IMAGE_BEARING_FIELD_NAMES = frozenset({"data", "url", "image_url", "image"})
_DATA_IMAGE_URL_PREFIX = "data:image/"


def configure_cli_debug_logging(cwd: str | None = None, argv: Sequence[str] | None = None) -> str | None:
    """Initialize process-wide debug logging.

    Enabled by default. Set ``PI_CLI_DEBUG_LOG=0`` to disable. Override the path
    with ``PI_CLI_DEBUG_LOG_PATH``.
    """
    global _CONFIGURED, _LOG_PATH, _FAULTHANDLER_FILE
    if _CONFIGURED:
        return _LOG_PATH
    _CONFIGURED = True

    if os.environ.get("PI_CLI_DEBUG_LOG", "1").lower() in {"0", "false", "no", "off"}:
        return None

    log_cwd = cwd or os.getcwd()
    _LOG_PATH = os.environ.get("PI_CLI_DEBUG_LOG_PATH") or get_debug_log_path(log_cwd)
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), mode=0o700, exist_ok=True)
    except Exception:
        _LOG_PATH = None
        return None

    try:
        _FAULTHANDLER_FILE = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_FAULTHANDLER_FILE, all_threads=True)
    except Exception:
        _FAULTHANDLER_FILE = None

    log_event(
        "process_start",
        cwd=log_cwd,
        argv=_sanitize_argv(list(argv if argv is not None else sys.argv[1:])),
        pid=os.getpid(),
        ppid=os.getppid(),
        python=sys.version.split()[0],
        executable=sys.executable,
        platform=platform.platform(),
        version=VERSION,
    )
    atexit.register(_log_process_exit)
    return _LOG_PATH


def log_event(event: str, **fields: Any) -> None:
    """Append a single JSON line to the debug log."""
    if not _LOG_PATH:
        return
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "run_id": _RUN_ID,
        "pid": os.getpid(),
        "event": event,
    }
    record.update(redact_image_base64(_sanitize(fields)))
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        with _LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
    except Exception:
        # Logging must never affect the CLI.
        pass


def log_exception(event: str, exc: BaseException, **fields: Any) -> None:
    log_event(
        event,
        exception_type=type(exc).__name__,
        exception=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **fields,
    )


def attach_session_event_logging(session: Any) -> Any | None:
    """Subscribe to AgentSession events and log compact metadata only."""
    subscribe = getattr(session, "subscribe", None)
    if not callable(subscribe):
        return None

    def on_event(event: Any) -> None:
        try:
            log_event("session_event", **_event_summary(event))
        except Exception:
            pass

    try:
        return subscribe(on_event)
    except Exception as exc:
        log_exception("session_event_logger_attach_failed", exc)
        return None


def log_session_snapshot(label: str, session: Any) -> None:
    try:
        state = getattr(session, "state", None)
        model = getattr(state, "model", None)
        messages = getattr(state, "messages", None) or []
        sm = getattr(session, "session_manager", None)
        get_active_tool_names = getattr(session, "get_active_tool_names", None)
        log_event(
            "session_snapshot",
            label=label,
            session_id=getattr(sm, "session_id", None),
            cwd=getattr(sm, "cwd", None) or getattr(session, "cwd", None),
            model_id=getattr(model, "id", None),
            provider=getattr(model, "provider", None),
            context_window=getattr(model, "context_window", None),
            message_count=len(messages),
            active_tools=get_active_tool_names() if callable(get_active_tool_names) else None,
        )
    except Exception:
        pass


def _event_summary(event: Any) -> dict[str, Any]:
    etype = getattr(event, "type", None)
    data: dict[str, Any] = {"type": etype or type(event).__name__}
    message = getattr(event, "message", None)
    if message is not None:
        data.update(_message_summary(message))
    for src, dst in (
        ("tool_name", "tool_name"),
        ("toolName", "tool_name"),
        ("tool_call_id", "tool_call_id"),
        ("toolCallId", "tool_call_id"),
        ("is_error", "is_error"),
        ("isError", "is_error"),
    ):
        val = getattr(event, src, None)
        if val is not None:
            data[dst] = val
    return data


def _message_summary(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None) or []
    tool_calls = []
    text_chars = 0
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "toolCall":
            tool_calls.append(getattr(block, "name", None))
        text = getattr(block, "text", None) or getattr(block, "thinking", None)
        if isinstance(text, str):
            text_chars += len(text)
    usage = getattr(message, "usage", None)
    return {
        "role": getattr(message, "role", None),
        "stop_reason": getattr(message, "stop_reason", None),
        "error_message": getattr(message, "error_message", None),
        "provider": getattr(message, "provider", None),
        "model": getattr(message, "model", None),
        "content_blocks": len(content),
        "text_chars": text_chars,
        "tool_calls": tool_calls,
        "usage": _usage_summary(usage),
    }


def _usage_summary(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input": getattr(usage, "input", None),
        "output": getattr(usage, "output", None),
        "cache_read": getattr(usage, "cache_read", None),
        "cache_write": getattr(usage, "cache_write", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_s = str(key)
            if any(secret in key_s.lower().replace("-", "_") for secret in _SECRET_KEYS):
                out[key_s] = "<redacted>"
            else:
                out[key_s] = _sanitize(item)
        return out
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith(("sk-", "sk_")):
        return "<redacted>"
    return redact_image_base64(value)


def _redact_image_base64_value(value: Any, *, in_image_path: bool = False) -> Any:
    if isinstance(value, str):
        should_redact = (
            len(value) >= IMAGE_BASE64_REDACT_THRESHOLD_BYTES
            and (value.startswith(_DATA_IMAGE_URL_PREFIX) or in_image_path)
        )
        if should_redact:
            return IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(value.encode("utf-8")))
        return value
    if isinstance(value, Mapping):
        return {
            key: _redact_image_base64_value(
                item,
                in_image_path=str(key) in _IMAGE_BEARING_FIELD_NAMES,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_image_base64_value(item, in_image_path=in_image_path) for item in value]
    if isinstance(value, tuple):
        return [_redact_image_base64_value(item, in_image_path=in_image_path) for item in value]
    return value


def redact_image_base64(payload: Any) -> Any:
    """Redact large image payloads from diagnostic logs without touching non-image blobs."""
    return _redact_image_base64_value(payload)


def _sanitize_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        lower = arg.lower()
        if lower in {"--api-key", "--token", "--password"}:
            redacted.append(arg)
            redact_next = True
        elif lower.startswith(("--api-key=", "--token=", "--password=")):
            redacted.append(arg.split("=", 1)[0] + "=<redacted>")
        else:
            redacted.append(arg)
    return redacted


def _log_process_exit() -> None:
    global _EXIT_LOGGED
    if _EXIT_LOGGED:
        return
    _EXIT_LOGGED = True
    log_event("process_atexit")
    try:
        if _FAULTHANDLER_FILE is not None:
            _FAULTHANDLER_FILE.flush()
    except Exception:
        pass
