"""
Hello Tool — minimal custom tool example.

Registers a single LLM-callable tool, `hello`, that greets a name.

Usage:
1. Copy this file to ~/.pi/agent/extensions/ or your project's .pi/extensions/
2. Ask the agent to "use the hello tool to greet Marcus".

Port of examples/extensions/hello.ts from the Node reference.
"""

from typing import Any


async def hello_execute(
    tool_call_id: str,
    params: dict,
    signal: Any,
    on_update: Any,
    ctx: Any,
):
    name = params.get("name", "world")
    return {
        "content": [{"type": "text", "text": f"Hello, {name}!"}],
        "details": {"greeted": name},
    }


def extension_factory(pi):
    pi.register_tool(
        name="hello",
        label="Hello",
        description="A simple greeting tool",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"},
            },
            "required": ["name"],
        },
        execute=hello_execute,
    )
