"""
Footer component helpers.

Mirrors the testable behavior of TypeScript components/footer.ts.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pi_coding_agent.config import VERSION

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def visible_width(text: str) -> int:
    return len(_ANSI.sub("", text))


def truncate_to_width(text: str, width: int, suffix: str = "...") -> str:
    if width <= 0:
        return ""
    if visible_width(text) <= width:
        return text
    suffix_width = visible_width(suffix)
    keep = max(0, width - suffix_width)
    plain_count = 0
    result: list[str] = []
    i = 0
    while i < len(text) and plain_count < keep:
        if text[i] == "\x1b":
            match = _ANSI.match(text, i)
            if match:
                result.append(match.group(0))
                i = match.end()
                continue
        result.append(text[i])
        plain_count += 1
        i += 1
    return "".join(result) + suffix


def sanitize_status_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def format_tokens(count: int | float | None) -> str:
    value = int(count or 0)
    if value < 1000:
        return str(value)
    if value < 10000:
        return f"{value / 1000:.1f}k"
    if value < 1_000_000:
        return f"{round(value / 1000)}k"
    if value < 10_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{round(value / 1_000_000)}M"


def format_cwd_for_footer(cwd: str, home: str | None = None) -> str:
    if not home:
        return cwd
    resolved_cwd = Path(cwd).expanduser().resolve()
    resolved_home = Path(home).expanduser().resolve()
    try:
        relative = resolved_cwd.relative_to(resolved_home)
    except ValueError:
        return cwd
    rel = str(relative)
    return "~" if rel == "." else os.path.join("~", rel)


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class FooterComponent:
    """Render pwd/session/status and usage/model footer lines."""

    def __init__(self, session: Any, footer_data: Any) -> None:
        self.session = session
        self.footer_data = footer_data
        self.auto_compact_enabled = True

    def set_session(self, session: Any) -> None:
        self.session = session

    def set_auto_compact_enabled(self, enabled: bool) -> None:
        self.auto_compact_enabled = bool(enabled)

    def invalidate(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    def _usage_totals(self) -> tuple[int, int, int, int, float, float | None]:
        total_input = total_output = total_cache_read = total_cache_write = 0
        total_cost = 0.0
        latest_cache_hit_rate: float | None = None
        entries = self.session.session_manager.get_entries()
        for entry in entries:
            entry_type = _get_attr_or_key(entry, "type")
            message = _get_attr_or_key(entry, "message") or _get_attr_or_key(entry, "data", {}).get("message", {})
            if entry_type != "message" or _get_attr_or_key(message, "role") != "assistant":
                continue
            usage = _get_attr_or_key(message, "usage", {}) or {}
            input_tokens = int(_get_attr_or_key(usage, "input", 0) or 0)
            output_tokens = int(_get_attr_or_key(usage, "output", 0) or 0)
            cache_read = int(_get_attr_or_key(usage, "cache_read", _get_attr_or_key(usage, "cacheRead", 0)) or 0)
            cache_write = int(_get_attr_or_key(usage, "cache_write", _get_attr_or_key(usage, "cacheWrite", 0)) or 0)
            cost = _get_attr_or_key(usage, "cost", {}) or {}
            total = _get_attr_or_key(cost, "total", 0) if not isinstance(cost, (int, float)) else cost
            total_input += input_tokens
            total_output += output_tokens
            total_cache_read += cache_read
            total_cache_write += cache_write
            total_cost += float(total or 0)
            prompt_tokens = input_tokens + cache_read + cache_write
            latest_cache_hit_rate = (cache_read / prompt_tokens * 100) if prompt_tokens > 0 else None
        return total_input, total_output, total_cache_read, total_cache_write, total_cost, latest_cache_hit_rate

    def render(self, width: int) -> list[str]:
        state = self.session.state
        model = _get_attr_or_key(state, "model")
        context_usage = self.session.get_context_usage()
        context_window = (
            _get_attr_or_key(context_usage, "contextWindow", None)
            or _get_attr_or_key(context_usage, "context_window", None)
            or _get_attr_or_key(model, "context_window", 0)
            or 0
        )
        percent = _get_attr_or_key(context_usage, "percent", None) if context_usage is not None else None

        cwd = self.session.session_manager.get_cwd()
        pwd = format_cwd_for_footer(cwd, os.environ.get("HOME") or os.environ.get("USERPROFILE"))
        branch = self.footer_data.get_git_branch()
        if branch:
            pwd = f"{pwd} ({branch})"
        session_name = self.session.session_manager.get_session_name()
        if session_name:
            pwd = f"{pwd} • {session_name}"

        total_input, total_output, cache_read, cache_write, total_cost, latest_hit = self._usage_totals()
        stats_parts: list[str] = []
        if total_input:
            stats_parts.append(f"↑{format_tokens(total_input)}")
        if total_output:
            stats_parts.append(f"↓{format_tokens(total_output)}")
        if cache_read:
            stats_parts.append(f"R{format_tokens(cache_read)}")
        if cache_write:
            stats_parts.append(f"W{format_tokens(cache_write)}")
        if (cache_read or cache_write) and latest_hit is not None:
            stats_parts.append(f"CH{latest_hit:.1f}%")
        using_subscription = bool(model and self.session.model_registry.is_using_oauth(model))
        if total_cost or using_subscription:
            stats_parts.append(f"${total_cost:.3f}{' (sub)' if using_subscription else ''}")
        auto_indicator = " (auto)" if self.auto_compact_enabled else ""
        if percent is None:
            stats_parts.append(f"?/{format_tokens(context_window)}{auto_indicator}")
        else:
            stats_parts.append(f"{float(percent):.1f}%/{format_tokens(context_window)}{auto_indicator}")
        stats_parts.append(f"v{VERSION}")

        stats_left = " ".join(stats_parts)
        model_name = _get_attr_or_key(model, "id", "no-model") if model else "no-model"
        reasoning = bool(_get_attr_or_key(model, "reasoning", False)) if model else False
        right = model_name
        if reasoning:
            thinking = _get_attr_or_key(state, "thinking_level", _get_attr_or_key(state, "thinkingLevel", "off"))
            right = f"{model_name} • thinking off" if thinking == "off" else f"{model_name} • {thinking}"
        if self.footer_data.get_available_provider_count() > 1 and model:
            with_provider = f"({_get_attr_or_key(model, 'provider')}) {right}"
            if visible_width(stats_left) + 2 + visible_width(with_provider) <= width:
                right = with_provider

        left_width = visible_width(stats_left)
        right_width = visible_width(right)
        if left_width + 2 + right_width <= width:
            stats_line = stats_left + " " * (width - left_width - right_width) + right
        else:
            available = max(0, width - left_width - 2)
            stats_line = stats_left if available <= 0 else stats_left + "  " + truncate_to_width(right, available, "")

        lines = [truncate_to_width(pwd, width), truncate_to_width(stats_line, width)]
        statuses = self.footer_data.get_extension_statuses()
        if statuses:
            status_line = " ".join(sanitize_status_text(statuses[key]) for key in sorted(statuses))
            lines.append(truncate_to_width(status_line, width))
        return lines


__all__ = [
    "FooterComponent",
    "format_cwd_for_footer",
    "format_tokens",
    "sanitize_status_text",
    "truncate_to_width",
    "visible_width",
]
