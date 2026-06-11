"""Helpers for converting extension tool definitions to runtime tools."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from pi_coding_agent.core.extensions.types import ExtensionContext, ToolDefinition


def _normalize_tool_result(result: Any) -> AgentToolResult:
    if isinstance(result, AgentToolResult):
        return result
    if isinstance(result, dict):
        return AgentToolResult(**result)
    return AgentToolResult(content=[TextContent(type="text", text=str(result))])


def _tool_attr(tool: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool, dict):
        return tool.get(name, default)
    return getattr(tool, name, default)


def _positional_arity(fn: Callable[..., Any]) -> int | None:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    positional_count = 0
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return None
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional_count += 1
    return positional_count


def _call_tool_execute(
    fn: Callable[..., Any],
    tool_call_id: str,
    params: dict[str, Any],
    cancel_event: asyncio.Event | None,
    on_update: Any,
    ctx: ExtensionContext | None,
) -> Any:
    arity = _positional_arity(fn)
    if arity is None or arity >= 5:
        return fn(tool_call_id, params, cancel_event, on_update, ctx)
    if arity >= 4:
        return fn(tool_call_id, params, cancel_event, on_update)
    if arity >= 2:
        return fn(params, ctx)
    if arity == 1:
        return fn(params)
    return fn()


def wrap_tool_definition(
    definition: ToolDefinition,
    ctx_factory: Callable[[], ExtensionContext] | None = None,
) -> AgentTool:
    """Wrap a ToolDefinition into an AgentTool for the core runtime."""
    if definition.execute is None:
        raise ValueError(f"Tool definition has no execute function: {definition.name}")

    async def _execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        result = _call_tool_execute(
            definition.execute,
            tool_call_id,
            params,
            cancel_event,
            on_update,
            ctx_factory() if ctx_factory else None,
        )
        if asyncio.iscoroutine(result):
            result = await result
        return _normalize_tool_result(result)

    tool_kwargs: dict[str, Any] = {
        "name": definition.name,
        "label": definition.label or definition.name,
        "description": definition.description or "",
        "parameters": definition.parameters or {},
        "execute": _execute,
    }
    if definition.prepare_arguments is not None:
        tool_kwargs["prepareArguments"] = definition.prepare_arguments
    if definition.execution_mode is not None:
        tool_kwargs["executionMode"] = definition.execution_mode
    if definition.execution_policy is not None:
        tool_kwargs["executionPolicy"] = definition.execution_policy
    return AgentTool(**tool_kwargs)


def wrap_tool_definitions(
    definitions: list[ToolDefinition],
    ctx_factory: Callable[[], ExtensionContext] | None = None,
) -> list[AgentTool]:
    """Wrap multiple ToolDefinitions into AgentTools."""
    return [wrap_tool_definition(definition, ctx_factory) for definition in definitions]


def create_tool_definition_from_agent_tool(tool: AgentTool) -> ToolDefinition:
    """Synthesize a minimal ToolDefinition from a plain AgentTool."""
    async def _execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Any = None,
        ctx: ExtensionContext | None = None,
    ) -> AgentToolResult:
        result = tool.execute(tool_call_id, params, cancel_event, on_update)
        if asyncio.iscoroutine(result):
            result = await result
        return _normalize_tool_result(result)

    return ToolDefinition(
        name=_tool_attr(tool, "name", ""),
        label=_tool_attr(tool, "label", ""),
        description=_tool_attr(tool, "description", ""),
        parameters=_tool_attr(tool, "parameters", {}) or {},
        prepare_arguments=_tool_attr(tool, "prepareArguments", None),
        execution_mode=_tool_attr(tool, "executionMode", None),
        execution_policy=_tool_attr(tool, "executionPolicy", None),
        execute=_execute,
    )


__all__ = [
    "create_tool_definition_from_agent_tool",
    "wrap_tool_definition",
    "wrap_tool_definitions",
]
