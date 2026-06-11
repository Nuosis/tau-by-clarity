"""
Tools Extension

Provides a /tools command to enable/disable tools interactively. The selection
persists across session reloads and respects branch navigation — it is stored as
a custom session entry and restored from the current branch on session_start and
session_tree.

Usage:
1. Copy this file to ~/.pi/agent/extensions/ or your project's .pi/extensions/
2. Use /tools to toggle a tool on or off.

Port of examples/extensions/tools.ts from the Node reference. The persistence and
branch-restore logic is a faithful 1:1 port built on:
    pi.get_all_tools(), pi.get_active_tools(), pi.set_active_tools(),
    pi.append_entry(), and ctx.session_manager.get_branch().

The one deviation: the Node version renders a live multi-toggle SettingsList via
ctx.ui.custom(). The Python TUI does not expose ctx.ui.custom() yet, so the picker
here uses ctx.ui.select() to toggle one tool per invocation. The state model is
identical; only the widget differs.
"""

from typing import Any

CONFIG_ENTRY_TYPE = "tools-config"


def extension_factory(pi):
    state = {"enabled": set()}  # type: dict[str, set]

    def persist() -> None:
        pi.append_entry(CONFIG_ENTRY_TYPE, {"enabledTools": sorted(state["enabled"])})

    def apply() -> None:
        pi.set_active_tools(sorted(state["enabled"]))

    def restore_from_branch(ctx: Any) -> None:
        all_tool_names = [t.get("name") for t in pi.get_all_tools()]

        # Find the most recent tools-config custom entry in the current branch.
        saved = None
        session_manager = getattr(ctx, "sessionManager", None) or getattr(ctx, "session_manager", None)
        branch = session_manager.get_branch() if session_manager else []
        for entry in branch:
            if getattr(entry, "type", None) == "custom" and entry.data.get("customType") == CONFIG_ENTRY_TYPE:
                data = entry.data.get("data") or {}
                if isinstance(data, dict) and "enabledTools" in data:
                    saved = data["enabledTools"]

        if saved is not None:
            # Restore, dropping any tools that no longer exist.
            state["enabled"] = {t for t in saved if t in all_tool_names}
            apply()
        else:
            # No saved state — sync with whatever is currently active.
            state["enabled"] = set(pi.get_active_tools())

    async def handler(args: str, ctx: Any):
        all_tools = pi.get_all_tools()
        if not all_tools:
            ctx.ui.notify("No tools registered", "info")
            return

        items = [
            f"[{'x' if t.get('name') in state['enabled'] else ' '}] {t.get('name')}"
            for t in all_tools
        ]
        selected = await ctx.ui.select("Toggle a tool", items)
        if not selected:
            return

        name = selected.split("] ", 1)[-1]
        if name in state["enabled"]:
            state["enabled"].discard(name)
            ctx.ui.notify(f"Disabled tool: {name}", "info")
        else:
            state["enabled"].add(name)
            ctx.ui.notify(f"Enabled tool: {name}", "info")

        apply()
        persist()

    pi.register_command("tools", description="Enable/disable tools", handler=handler)

    async def on_session_start(event: dict, ctx: Any):
        restore_from_branch(ctx)

    async def on_session_tree(event: dict, ctx: Any):
        restore_from_branch(ctx)

    pi.on("session_start", on_session_start)
    pi.on("session_tree", on_session_tree)
