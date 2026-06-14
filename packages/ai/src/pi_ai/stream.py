"""
Unified streaming functions — mirrors packages/ai/src/stream.ts

Provides stream(), complete(), stream_simple(), complete_simple().
"""
from __future__ import annotations

from typing import AsyncGenerator

from .api_registry import get_api_provider
from .compression import compress_context
from .pii import detok_event, protect_context
from .env_api_keys import get_env_api_key
from .providers import register_builtins
from .types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    EventThinkingDelta,
    EventThinkingEnd,
    EventThinkingStart,
    EventToolCallDelta,
    EventToolCallEnd,
    EventToolCallStart,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)

register_builtins()


async def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream a response with unified reasoning options.
    Automatically resolves API key from environment if not provided.
    Mirrors streamSimple() from TypeScript.
    """
    opts = options or SimpleStreamOptions()

    # Auto-resolve API key from env if not set
    if not opts.api_key:
        opts = opts.model_copy(update={"api_key": get_env_api_key(model.provider)})

    provider = get_api_provider(model.api)
    if provider is None:
        raise ValueError(f"No stream function registered for API: {model.api!r}")

    # Active compression (one-way), then the PII chokepoint.
    context = compress_context(context)
    context, _detok = protect_context(context)
    async for event in provider.stream_simple(model, context, opts):
        event = _normalize_stream_event(event)
        if _detok is not None:
            event = detok_event(event, _detok)
        yield event


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """
    Get a complete (non-streaming) response.
    Mirrors completeSimple() from TypeScript.
    """
    final_message: AssistantMessage | None = None

    async for event in stream_simple(model, context, options):
        if isinstance(event, EventDone):
            final_message = event.message
        elif isinstance(event, EventError):
            final_message = event.error

    if final_message is None:
        raise RuntimeError("Stream completed without a final message")

    return final_message


async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """
    Stream with provider-specific options (no reasoning normalization).
    Mirrors stream() from TypeScript.
    """
    opts = options or StreamOptions()

    # Auto-resolve API key from env if not set (same behavior as stream_simple)
    if not opts.api_key:
        opts = opts.model_copy(update={"api_key": get_env_api_key(model.provider)})

    provider = get_api_provider(model.api)
    if provider is None:
        raise ValueError(f"No stream function registered for API: {model.api!r}")

    # Active compression (one-way), then the PII chokepoint.
    context = compress_context(context)
    context, _detok = protect_context(context)
    async for event in provider.stream(model, context, opts):
        event = _normalize_stream_event(event)
        if _detok is not None:
            event = detok_event(event, _detok)
        yield event


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """
    Get a complete response with provider-specific options.
    Mirrors complete() from TypeScript.
    """
    final_message: AssistantMessage | None = None

    async for event in stream(model, context, options):
        if isinstance(event, EventDone):
            final_message = event.message
        elif isinstance(event, EventError):
            final_message = event.error

    if final_message is None:
        raise RuntimeError("Stream completed without a final message")

    return final_message


def _normalize_stream_event(event: AssistantMessageEvent | dict) -> AssistantMessageEvent:
    """Convert legacy dict stream events into typed stream events."""
    if not isinstance(event, dict):
        return event

    event_type = event.get("type")
    data = dict(event)
    for key in ("partial", "message", "error"):
        if isinstance(data.get(key), dict):
            data[key] = _normalize_assistant_message(data[key])
    if event_type == "start":
        return EventStart(**data)
    if event_type == "text_start":
        return EventTextStart(**data)
    if event_type == "text_delta":
        return EventTextDelta(**data)
    if event_type == "text_end":
        return EventTextEnd(**data)
    if event_type == "thinking_start":
        return EventThinkingStart(**data)
    if event_type == "thinking_delta":
        return EventThinkingDelta(**data)
    if event_type == "thinking_end":
        return EventThinkingEnd(**data)
    if event_type == "toolcall_start":
        return EventToolCallStart(**data)
    if event_type == "toolcall_delta":
        return EventToolCallDelta(**data)
    if event_type == "toolcall_end":
        if isinstance(data.get("tool_call"), dict):
            data["tool_call"] = ToolCall(**data["tool_call"])
        return EventToolCallEnd(**data)
    if event_type == "done":
        return EventDone(**data)
    if event_type == "error":
        return EventError(**data)
    raise ValueError(f"Unknown stream event type: {event_type!r}")


def _normalize_assistant_message(message: dict) -> AssistantMessage:
    data = dict(message)
    data["usage"] = _normalize_usage(data.get("usage"))
    data["content"] = [_normalize_content_block(block) for block in data.get("content", [])]
    return AssistantMessage(**data)


def _normalize_usage(value: object) -> Usage:
    if isinstance(value, Usage):
        return value
    if isinstance(value, dict):
        return Usage(**value)
    return Usage()


def _normalize_content_block(block: object) -> TextContent | ThinkingContent | ToolCall:
    if isinstance(block, (TextContent, ThinkingContent, ToolCall)):
        return block
    if not isinstance(block, dict):
        return TextContent(type="text", text=str(block))
    block_type = block.get("type")
    if block_type == "text":
        return TextContent(**block)
    if block_type == "thinking":
        return ThinkingContent(**block)
    if block_type == "toolCall":
        data = dict(block)
        data.pop("partial_json", None)
        return ToolCall(**data)
    return TextContent(type="text", text=str(block))
