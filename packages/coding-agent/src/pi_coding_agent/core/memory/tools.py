"""Native memory retrieval tools.

These are agent-triggered read paths for durable project memory. They do not
write to CCR and they do not inject full tool outputs into passive recall.
"""
from __future__ import annotations

from typing import Any

from pi_coding_agent.core.extensions.types import Extension, ToolDefinition


def _text_response(text: str, *, is_error: bool = False, details: dict[str, Any] | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        response["isError"] = True
    if details is not None:
        response["details"] = details
    return response


def _memory_extension(store: Any) -> Extension:
    async def tool_log_lookup(tool_call_id, params, signal, on_update, ctx):
        del tool_call_id, signal, on_update, ctx
        lookup_id = str((params or {}).get("tool_call_id") or "").strip()
        if not lookup_id:
            return _text_response("tool_call_id is required.", is_error=True)
        row = store.get_tool_log(lookup_id) if hasattr(store, "get_tool_log") else None
        if row is None:
            return _text_response(f"No durable tool log found for {lookup_id!r}.", is_error=True)
        args = row.get("tool_args") or "{}"
        text = (
            f"[memory.tool_log_lookup: {lookup_id}]\n"
            f"tool: {row.get('tool_name') or 'unknown'}\n"
            f"args: {args}\n\n"
            f"{row.get('output') or ''}"
        ).rstrip()
        return _text_response(text, details={"tool_call_id": lookup_id, "tool_name": row.get("tool_name")})

    async def summarize_expand(tool_call_id, params, signal, on_update, ctx):
        del tool_call_id, signal, on_update, ctx
        summary_id = str((params or {}).get("summary_id") or "").strip()
        if not summary_id:
            return _text_response("summary_id is required.", is_error=True)
        row = store.get_summary(summary_id) if hasattr(store, "get_summary") else None
        if row is None:
            return _text_response(f"No conversation summary found for {summary_id!r}.", is_error=True)
        text = (
            f"[memory.summarize_expand: {summary_id}]\n"
            f"description: {row.get('description') or ''}\n"
            f"summary: {row.get('summary') or ''}\n\n"
            f"{row.get('full_content') or ''}"
        ).rstrip()
        return _text_response(text, details={"summary_id": summary_id})

    return Extension(
        path="memory:native",
        resolved_path="memory:native",
        tools={
            "memory.tool_log_lookup": ToolDefinition(
                name="memory.tool_log_lookup",
                label="Lookup durable tool log",
                description=(
                    "Retrieve the durable project-local output for an exact tool_call_id "
                    "from memory.tool_log_memory. Use this for cross-session tool output lookup."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_call_id": {
                            "type": "string",
                            "description": "Exact tool_call_id to retrieve.",
                        },
                    },
                    "required": ["tool_call_id"],
                },
                execute=tool_log_lookup,
            ),
            "memory.summarize_expand": ToolDefinition(
                name="memory.summarize_expand",
                label="Expand conversation summary",
                description=(
                    "Expand an exact compacted conversation summary_id from durable "
                    "project-local conversation memory."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary_id": {
                            "type": "string",
                            "description": "Exact summary_id to expand.",
                        },
                    },
                    "required": ["summary_id"],
                },
                execute=summarize_expand,
            ),
        },
    )


def register_memory_tools(runner: Any, store: Any) -> None:
    """Register native memory lookup tools on an ExtensionRunner idempotently."""
    if runner is None or store is None:
        return
    existing = set()
    get_all = getattr(runner, "get_all_registered_tools", None)
    if callable(get_all):
        existing = {getattr(tool, "name", "") for tool in get_all()}
    needed = {"memory.tool_log_lookup", "memory.summarize_expand"}
    if needed.issubset(existing):
        return
    extensions = getattr(runner, "extensions", None)
    if isinstance(extensions, list):
        extensions.append(_memory_extension(store))

