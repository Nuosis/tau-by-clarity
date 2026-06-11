"""
Bash execution display component.

Mirrors the stateful behavior of TypeScript components/bash-execution.ts.
"""
from __future__ import annotations

import re
from typing import Any

from pi_coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail

PREVIEW_LINES = 20
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


class BashExecutionComponent:
    def __init__(self, command: str, exclude_from_context: bool = False) -> None:
        self.command = command
        self.exclude_from_context = exclude_from_context
        self.output_lines: list[str] = []
        self.status = "running"
        self.exit_code: int | None = None
        self.full_output_path: str | None = None
        self.expanded = False
        self.truncated = False

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)

    def append_output(self, chunk: str) -> None:
        clean = strip_ansi(chunk).replace("\r\n", "\n").replace("\r", "\n")
        new_lines = clean.split("\n")
        if self.output_lines and new_lines:
            self.output_lines[-1] += new_lines[0]
            self.output_lines.extend(new_lines[1:])
        else:
            self.output_lines.extend(new_lines)

    def set_complete(
        self,
        exit_code: int | None,
        cancelled: bool,
        truncation_result: Any | None = None,
        full_output_path: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.status = "cancelled" if cancelled else "error" if exit_code not in (0, None) else "complete"
        self.full_output_path = full_output_path
        self.truncated = bool(getattr(truncation_result, "truncated", False) or (isinstance(truncation_result, dict) and truncation_result.get("truncated")))

    def get_output(self) -> str:
        return "\n".join(self.output_lines)

    def get_command(self) -> str:
        return self.command

    def render(self, width: int | None = None) -> list[str]:
        full_output = self.get_output()
        context = truncate_tail(full_output, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)
        if isinstance(context, str):
            content = context
            context_truncated = False
        else:
            content = getattr(context, "content", full_output)
            context_truncated = bool(getattr(context, "truncated", False))
        output_lines = content.splitlines() if content else []
        hidden = max(0, len(output_lines) - PREVIEW_LINES)
        visible = output_lines if self.expanded else output_lines[-PREVIEW_LINES:]

        lines = [f"$ {self.command}"]
        lines.extend(visible)
        if self.status == "running":
            lines.append("Running...")
        else:
            if hidden and not self.expanded:
                lines.append(f"... {hidden} more lines (to expand)")
            elif hidden and self.expanded:
                lines.append("(to collapse)")
            if self.status == "cancelled":
                lines.append("(cancelled)")
            elif self.status == "error":
                lines.append(f"(exit {self.exit_code})")
            if (self.truncated or context_truncated) and self.full_output_path:
                lines.append(f"Output truncated. Full output: {self.full_output_path}")
        if width is not None:
            lines = [line[:width] for line in lines]
        return lines


__all__ = ["BashExecutionComponent", "PREVIEW_LINES", "strip_ansi"]
