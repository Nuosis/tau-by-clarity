"""
Permission Gate Extension

Prompts for confirmation before running potentially dangerous bash commands.
Patterns checked: rm -rf, sudo, chmod/chown 777.

A tool_call handler returns {"block": True, "reason": ...} to stop execution,
or None to allow it.

Port of examples/extensions/permission-gate.ts from the Node reference.
"""

import re
from typing import Any

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+(-rf?|--recursive)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\b(chmod|chown)\b.*777", re.IGNORECASE),
]


def extension_factory(pi):
    # Handler is invoked as (event, ctx). event is a dict with toolName/tool_name + input.
    async def on_tool_call(event: dict, ctx: Any):
        if event.get("toolName") != "bash":
            return None

        command = (event.get("input") or {}).get("command", "")
        is_dangerous = any(p.search(command) for p in DANGEROUS_PATTERNS)
        if not is_dangerous:
            return None

        if not getattr(ctx, "hasUI", False):
            # Non-interactive mode: block by default, nothing can confirm.
            return {"block": True, "reason": "Dangerous command blocked (no UI for confirmation)"}

        choice = await ctx.ui.select(
            f"⚠️ Dangerous command:\n\n  {command}\n\nAllow?",
            ["Yes", "No"],
        )
        if choice != "Yes":
            return {"block": True, "reason": "Blocked by user"}

        return None

    pi.on("tool_call", on_tool_call)
