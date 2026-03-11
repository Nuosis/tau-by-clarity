"""
Proxy stream function — mirrors packages/agent/src/proxy.ts

Allows routing LLM calls through a server endpoint.
The server strips the partial field from delta events to reduce bandwidth.
We reconstruct the partial message client-side.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from pi_ai.types import (
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
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)
from pi_ai.utils.event_stream import EventStream
from pi_ai.utils.json_parse import parse_streaming_json


class ProxyStreamOptions(SimpleStreamOptions):
    """Options for proxy streaming. Extends SimpleStreamOptions."""
    auth_token: str = ""
    proxy_url: str = ""


class ProxyMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            is_done=lambda event: event.type in ("done", "error"),
            extract_result=lambda event: (
                event.message if event.type == "done"
                else event.error if event.type == "error"
                else None
            ),
        )


def stream_proxy(
    model: Model,
    context: Context,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """
    Stream LLM responses through a proxy server.
    Mirrors streamProxy() in TypeScript.
    """
    stream = ProxyMessageEventStream()

    async def _run() -> None:
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            payload = {
                "model": model.model_dump() if hasattr(model, "model_dump") else {},
                "context": context.model_dump() if hasattr(context, "model_dump") else {},
                "options": {
                    "temperature": options.temperature,
                    "maxTokens": options.max_tokens,
                    "reasoning": options.reasoning,
                },
            }

            headers = {
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{options.proxy_url}/api/stream",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                        try:
                            error_data = json.loads(error_text)
                            if "error" in error_data:
                                error_message = f"Proxy error: {error_data['error']}"
                        except Exception:
                            pass
                        raise RuntimeError(error_message)

                    buffer = ""
                    async for chunk in response.aiter_text():
                        if options.signal and options.signal.is_set():
                            raise RuntimeError("Request aborted by user")

                        buffer += chunk
                        lines = buffer.split("\n")
                        buffer = lines.pop()

                        for line in lines:
                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data and data != "[DONE]":
                                    proxy_event = json.loads(data)
                                    event = _process_proxy_event(proxy_event, partial)
                                    if event:
                                        stream.push(event)

            stream.end()

        except Exception as e:
            error_message = str(e)
            reason = "aborted" if (options.signal and options.signal.is_set()) else "error"
            partial.stop_reason = reason
            partial.error_message = error_message
            stream.push(EventError(type="error", reason=reason, error=partial))
            stream.end()

    asyncio.ensure_future(_run())
    return stream


def _process_proxy_event(
    proxy_event: dict[str, Any],
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type")

    if event_type == "start":
        return EventStart(type="start", partial=partial)

    elif event_type == "text_start":
        idx = proxy_event.get("contentIndex", 0)
        _ensure_content_slot(partial, idx)
        partial.content[idx] = TextContent(type="text", text="")
        return EventTextStart(type="text_start", content_index=idx, partial=partial)

    elif event_type == "text_delta":
        idx = proxy_event.get("contentIndex", 0)
        delta = proxy_event.get("delta", "")
        content = partial.content[idx] if idx < len(partial.content) else None
        if content and content.type == "text":
            content.text += delta
        return EventTextDelta(
            type="text_delta", content_index=idx, delta=delta, partial=partial
        )

    elif event_type == "text_end":
        idx = proxy_event.get("contentIndex", 0)
        content = partial.content[idx] if idx < len(partial.content) else None
        text = ""
        if content and content.type == "text":
            if "contentSignature" in proxy_event:
                content.text_signature = proxy_event["contentSignature"]
            text = content.text
        return EventTextEnd(
            type="text_end", content_index=idx, content=text, partial=partial
        )

    elif event_type == "thinking_start":
        idx = proxy_event.get("contentIndex", 0)
        _ensure_content_slot(partial, idx)
        partial.content[idx] = ThinkingContent(type="thinking", thinking="")
        return EventThinkingStart(
            type="thinking_start", content_index=idx, partial=partial
        )

    elif event_type == "thinking_delta":
        idx = proxy_event.get("contentIndex", 0)
        delta = proxy_event.get("delta", "")
        content = partial.content[idx] if idx < len(partial.content) else None
        if content and content.type == "thinking":
            content.thinking += delta
        return EventThinkingDelta(
            type="thinking_delta", content_index=idx, delta=delta, partial=partial
        )

    elif event_type == "thinking_end":
        idx = proxy_event.get("contentIndex", 0)
        content = partial.content[idx] if idx < len(partial.content) else None
        text = ""
        if content and content.type == "thinking":
            if "contentSignature" in proxy_event:
                content.thinking_signature = proxy_event["contentSignature"]
            text = content.thinking
        return EventThinkingEnd(
            type="thinking_end", content_index=idx, content=text, partial=partial
        )

    elif event_type == "toolcall_start":
        idx = proxy_event.get("contentIndex", 0)
        _ensure_content_slot(partial, idx)
        partial.content[idx] = ToolCall(
            type="toolCall",
            id=proxy_event.get("id", ""),
            name=proxy_event.get("toolName", ""),
            arguments={},
        )
        partial.content[idx]._partial_json = ""
        return EventToolCallStart(
            type="toolcall_start", content_index=idx, partial=partial
        )

    elif event_type == "toolcall_delta":
        idx = proxy_event.get("contentIndex", 0)
        delta = proxy_event.get("delta", "")
        content = partial.content[idx] if idx < len(partial.content) else None
        if content and content.type == "toolCall":
            pj = getattr(content, "_partial_json", "") + delta
            content._partial_json = pj
            parsed = parse_streaming_json(pj)
            if parsed is not None:
                content.arguments = parsed
        return EventToolCallDelta(
            type="toolcall_delta", content_index=idx, delta=delta, partial=partial
        )

    elif event_type == "toolcall_end":
        idx = proxy_event.get("contentIndex", 0)
        content = partial.content[idx] if idx < len(partial.content) else None
        if content and content.type == "toolCall":
            if hasattr(content, "_partial_json"):
                delattr(content, "_partial_json")
            return EventToolCallEnd(
                type="toolcall_end", content_index=idx, tool_call=content, partial=partial
            )
        return None

    elif event_type == "done":
        partial.stop_reason = proxy_event.get("reason", "stop")
        if "usage" in proxy_event:
            partial.usage = _parse_usage(proxy_event["usage"])
        return EventDone(type="done", reason=partial.stop_reason, message=partial)

    elif event_type == "error":
        partial.stop_reason = proxy_event.get("reason", "error")
        partial.error_message = proxy_event.get("errorMessage")
        if "usage" in proxy_event:
            partial.usage = _parse_usage(proxy_event["usage"])
        return EventError(type="error", reason=partial.stop_reason, error=partial)

    return None


def _ensure_content_slot(partial: AssistantMessage, idx: int) -> None:
    """Extend content list to ensure idx is accessible."""
    while len(partial.content) <= idx:
        partial.content.append(TextContent(type="text", text=""))


def _parse_usage(data: dict[str, Any]) -> Usage:
    """Parse usage data from proxy response."""
    return Usage(
        input=data.get("input", 0),
        output=data.get("output", 0),
        cache_read=data.get("cacheRead", 0),
        cache_write=data.get("cacheWrite", 0),
        total_tokens=data.get("totalTokens", 0),
        cost=data.get("cost", 0),
    )
