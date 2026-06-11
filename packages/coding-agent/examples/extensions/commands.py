"""
Commands Extension

Demonstrates pi.get_commands() by providing a /commands command that lists all
available slash commands in the current session, optionally filtered by source.

Usage:
1. Copy this file to ~/.pi/agent/extensions/ or your project's .pi/extensions/
2. Use /commands to see available commands.
3. Use /commands extensions  (or prompt / skill) to filter by source.

Port of examples/extensions/commands.ts from the Node reference. Relies on
pi.get_commands(), which returns dicts of {name, description, source, sourceInfo}.
"""

from typing import Any


def extension_factory(pi):
    def get_argument_completions(prefix: str):
        sources = ["extension", "prompt", "skill"]
        matches = [s for s in sources if s.startswith(prefix)]
        return [{"value": s, "label": s} for s in matches] if matches else None

    # Command handler is invoked as (args, ctx). args is the raw argument string.
    async def handler(args: str, ctx: Any):
        commands = pi.get_commands()
        source_filter = args.strip()

        if source_filter:
            filtered = [c for c in commands if c.get("source") == source_filter]
        else:
            filtered = commands

        if not filtered:
            ctx.ui.notify(
                f"No {source_filter} commands found" if source_filter else "No commands found",
                "info",
            )
            return

        def format_command(cmd: dict) -> str:
            desc = f" - {cmd['description']}" if cmd.get("description") else ""
            return f"/{cmd.get('name', '')}{desc}"

        items: list[str] = []
        for key, label in (("extension", "Extensions"), ("prompt", "Prompts"), ("skill", "Skills")):
            cmds = [c for c in filtered if c.get("source") == key]
            if cmds:
                items.append(f"--- {label} ---")
                items.extend(format_command(c) for c in cmds)

        selected = await ctx.ui.select("Available Commands", items)

        # If the user picked a real command (not a header), offer to show its path.
        if selected and not selected.startswith("---"):
            cmd_name = selected.split(" - ")[0].lstrip("/")
            cmd = next((c for c in commands if c.get("name") == cmd_name), None)
            path = (cmd or {}).get("sourceInfo", {}) or {}
            path = path.get("path") if isinstance(path, dict) else None
            if path:
                show = await ctx.ui.confirm(cmd_name, f"View source path?\n{path}")
                if show:
                    ctx.ui.notify(path, "info")

    pi.register_command(
        "commands",
        description="List available slash commands",
        handler=handler,
        get_argument_completions=get_argument_completions,
    )
