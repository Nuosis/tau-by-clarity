"""
Message display components.

These mirror the deterministic rendering behavior of the TypeScript user,
assistant, branch-summary, compaction-summary, and custom message components.
"""
from __future__ import annotations

from typing import Any

OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        kind = _get_attr_or_key(item, "type")
        if kind == "text":
            text = _get_attr_or_key(item, "text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class UserMessageComponent:
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int | None = None) -> list[str]:
        lines = self.text.splitlines() or [""]
        if width is not None:
            lines = [line[:width] for line in lines]
        if len(lines) == 1:
            lines[0] = OSC133_ZONE_START + OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[0]
        else:
            lines[0] = OSC133_ZONE_START + lines[0]
            lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines


class AssistantMessageComponent:
    def __init__(
        self,
        message: Any | None = None,
        hide_thinking_block: bool = False,
        hidden_thinking_label: str = "Thinking...",
    ) -> None:
        self.hide_thinking_block = hide_thinking_block
        self.hidden_thinking_label = hidden_thinking_label
        self.last_message = message
        self.has_tool_calls = False

    def set_hide_thinking_block(self, hide: bool) -> None:
        self.hide_thinking_block = hide

    def set_hidden_thinking_label(self, label: str) -> None:
        self.hidden_thinking_label = label

    def update_content(self, message: Any) -> None:
        self.last_message = message

    def render(self, width: int | None = None) -> list[str]:
        message = self.last_message
        if message is None:
            return []
        content = _get_attr_or_key(message, "content", []) or []
        lines: list[str] = []
        self.has_tool_calls = False
        for item in content:
            kind = _get_attr_or_key(item, "type")
            if kind == "toolCall":
                self.has_tool_calls = True
            elif kind == "text":
                text = str(_get_attr_or_key(item, "text", "")).strip()
                if text:
                    lines.extend(text.splitlines())
            elif kind == "thinking":
                thinking = str(_get_attr_or_key(item, "thinking", "")).strip()
                if thinking:
                    rendered = self.hidden_thinking_label if self.hide_thinking_block else thinking
                    lines.extend(rendered.splitlines())

        stop_reason = _get_attr_or_key(message, "stop_reason", _get_attr_or_key(message, "stopReason"))
        if not self.has_tool_calls and stop_reason in {"aborted", "error"}:
            error_message = _get_attr_or_key(message, "error_message", _get_attr_or_key(message, "errorMessage", ""))
            if stop_reason == "aborted":
                lines.append(error_message if error_message and error_message != "Request was aborted" else "Operation aborted")
            else:
                lines.append(f"Error: {error_message or 'Unknown error'}")

        if width is not None:
            lines = [line[:width] for line in lines]
        if lines and not self.has_tool_calls:
            if len(lines) == 1:
                lines[0] = OSC133_ZONE_START + OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[0]
            else:
                lines[0] = OSC133_ZONE_START + lines[0]
                lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines


class CustomMessageComponent:
    def __init__(self, message: Any) -> None:
        self.message = message

    def render(self, width: int | None = None) -> list[str]:
        text = _content_text(_get_attr_or_key(self.message, "content", ""))
        lines = text.splitlines() or [""]
        if width is not None:
            lines = [line[:width] for line in lines]
        return lines


class BranchSummaryMessageComponent:
    def __init__(self, summary: str) -> None:
        self.summary = summary

    def render(self, width: int | None = None) -> list[str]:
        lines = ["Branch summary", *self.summary.splitlines()]
        if width is not None:
            lines = [line[:width] for line in lines]
        return lines


class CompactionSummaryMessageComponent:
    def __init__(self, summary: str, tokens_before: int = 0) -> None:
        self.summary = summary
        self.tokens_before = tokens_before

    def render(self, width: int | None = None) -> list[str]:
        header = "Compaction summary"
        if self.tokens_before:
            header += f" ({self.tokens_before} tokens before)"
        lines = [header, *self.summary.splitlines()]
        if width is not None:
            lines = [line[:width] for line in lines]
        return lines


class SkillInvocationMessageComponent:
    def __init__(self, skill_block: Any) -> None:
        self.skill_block = skill_block
        self.expanded = False

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)

    def setExpanded(self, expanded: bool) -> None:
        self.set_expanded(expanded)

    def render(self, width: int | None = None) -> list[str]:
        name = str(_get_attr_or_key(self.skill_block, "name", ""))
        content = str(_get_attr_or_key(self.skill_block, "content", ""))
        if self.expanded:
            lines = [f"[skill] {name}", *content.splitlines()]
        else:
            lines = [f"[skill] {name} (expand to expand)"]
        if width is not None:
            lines = [line[:width] for line in lines]
        return lines


__all__ = [
    "AssistantMessageComponent",
    "BranchSummaryMessageComponent",
    "CompactionSummaryMessageComponent",
    "CustomMessageComponent",
    "OSC133_ZONE_END",
    "OSC133_ZONE_FINAL",
    "OSC133_ZONE_START",
    "SkillInvocationMessageComponent",
    "UserMessageComponent",
]
