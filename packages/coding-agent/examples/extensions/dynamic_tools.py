"""
Dynamic Tools Extension

Demonstrates registering tools after the factory has run.

- Registers one tool (`echo_session`) during the session_start event.
- Registers additional echo tools at runtime via /add-echo-tool <name>.

Port of examples/extensions/dynamic-tools.ts from the Node reference.
"""

import re
from typing import Any

ECHO_PARAMS = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "Message to echo"},
    },
    "required": ["message"],
}


def _normalize_tool_name(value: str) -> str | None:
    trimmed = value.strip().lower()
    if not trimmed:
        return None
    if not re.fullmatch(r"[a-z0-9_]+", trimmed):
        return None
    return trimmed


def extension_factory(pi):
    registered_tool_names: set[str] = set()

    def register_echo_tool(name: str, label: str, prefix: str) -> bool:
        if name in registered_tool_names:
            return False
        registered_tool_names.add(name)

        async def execute(tool_call_id: str, params: dict, signal: Any, on_update: Any, ctx: Any):
            return {
                "content": [{"type": "text", "text": f"{prefix}{params.get('message', '')}"}],
                "details": {"tool": name, "prefix": prefix},
            }

        pi.register_tool(
            name=name,
            label=label,
            description=f"Echo a message with prefix: {prefix}",
            prompt_snippet=f"Echo back user-provided text with {prefix.strip()} prefix",
            prompt_guidelines=["Use echo_session when the user asks for exact echo output."],
            parameters=ECHO_PARAMS,
            execute=execute,
        )
        return True

    # Handler is invoked as (event, ctx). event is a dict.
    async def on_session_start(event: dict, ctx: Any):
        register_echo_tool("echo_session", "Echo Session", "[session] ")
        ctx.ui.notify("Registered dynamic tool: echo_session", "info")

    pi.on("session_start", on_session_start)

    # Command handler is invoked as (args, ctx). args is the raw argument string.
    async def add_echo_tool(args: str, ctx: Any):
        tool_name = _normalize_tool_name(args)
        if not tool_name:
            ctx.ui.notify(
                "Usage: /add-echo-tool <tool_name> (lowercase, numbers, underscores)",
                "warning",
            )
            return
        created = register_echo_tool(tool_name, f"Echo {tool_name}", f"[{tool_name}] ")
        if not created:
            ctx.ui.notify(f"Tool already registered: {tool_name}", "warning")
            return
        ctx.ui.notify(f"Registered dynamic tool: {tool_name}", "info")

    pi.register_command(
        "add-echo-tool",
        description="Register a new echo tool dynamically: /add-echo-tool <tool_name>",
        handler=add_echo_tool,
    )
