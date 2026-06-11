"""HTTP idle timeout settings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000


@dataclass(frozen=True)
class HttpIdleTimeoutChoice:
    label: str
    timeout_ms: int


HTTP_IDLE_TIMEOUT_CHOICES = [
    HttpIdleTimeoutChoice("30 sec", 30_000),
    HttpIdleTimeoutChoice("1 min", 60_000),
    HttpIdleTimeoutChoice("2 min", 120_000),
    HttpIdleTimeoutChoice("5 min", 300_000),
    HttpIdleTimeoutChoice("disabled", 0),
]

_configured_http_idle_timeout_ms = DEFAULT_HTTP_IDLE_TIMEOUT_MS


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if not trimmed:
            return None
        try:
            return parse_http_idle_timeout_ms(float(trimmed))
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return math.floor(value)


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for choice in HTTP_IDLE_TIMEOUT_CHOICES:
        if choice.timeout_ms == timeout_ms:
            return choice.label
    return f"{timeout_ms / 1000:g} sec"


def configure_http_dispatcher(timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS) -> int:
    """Validate/store the idle timeout.

    Python providers use their own clients instead of Node undici, so this
    function records the normalized setting as a runtime contract.
    """
    global _configured_http_idle_timeout_ms
    normalized = parse_http_idle_timeout_ms(timeout_ms)
    if normalized is None:
        raise ValueError(f"Invalid HTTP idle timeout: {timeout_ms}")
    _configured_http_idle_timeout_ms = normalized
    return normalized


def get_configured_http_idle_timeout_ms() -> int:
    return _configured_http_idle_timeout_ms


__all__ = [
    "DEFAULT_HTTP_IDLE_TIMEOUT_MS",
    "HTTP_IDLE_TIMEOUT_CHOICES",
    "HttpIdleTimeoutChoice",
    "configure_http_dispatcher",
    "format_http_idle_timeout_ms",
    "get_configured_http_idle_timeout_ms",
    "parse_http_idle_timeout_ms",
]
