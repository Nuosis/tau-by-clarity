"""
Print mode — mirrors packages/coding-agent/src/modes/print-mode.ts

Non-interactive (single-shot) mode: sends prompts to agent, outputs result.
Used for `pi -p "prompt"` (text) and `pi --mode json "prompt"` (JSON event stream).
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from pi_agent.types import (
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventToolEnd,
    AgentEventToolStart,
)
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)

from ..core.agent_session import AgentSession


class PrintModeOptions:
    """Options for print mode."""
    def __init__(
        self,
        mode: str = "text",              # "text" | "json"
        messages: list[str] | None = None,
        initial_message: str | None = None,
        initial_images: list[Any] | None = None,  # list[ImageContent]
        initialMessage: str | None = None,
        initialImages: list[Any] | None = None,
    ) -> None:
        self.mode = mode
        self.messages = messages or []
        self.initial_message = initial_message if initial_message is not None else initialMessage
        self.initial_images = initial_images if initial_images is not None else (initialImages or [])
        self.initialMessage = self.initial_message
        self.initialImages = self.initial_images


async def run_print_mode(
    session: AgentSession | Any,
    prompt: str | None = None,
    show_thinking: bool = False,
    json_output: bool = False,
    # New parity params
    options: PrintModeOptions | None = None,
) -> int:
    """
    Run in print (single-shot) mode.
    Mirrors runPrintMode() in TypeScript.

    Supports:
    - Multiple messages (messages[] array)
    - Initial images
    - JSON mode: outputs session header + all events as newline-delimited JSON
    - Error exit: returns 1 if stop_reason == "error" or "aborted"
    - Explicit stdout flush

    Returns exit code (0 = success, 1 = error).
    """
    runtime_host = session if hasattr(session, "session") else None
    if runtime_host is not None:
        session = runtime_host.session

    # Build options object (backwards-compat with old positional API)
    if options is None:
        options = PrintModeOptions(
            mode="json" if json_output else "text",
            initial_message=prompt,
        )

    mode = options.mode

    # JSON mode: output session header first
    if mode == "json":
        try:
            sm = session.session_manager if hasattr(session, "session_manager") else None
            header = sm.get_header() if sm and hasattr(sm, "get_header") else None
            if header:
                print(json.dumps(header), flush=True)
        except Exception:
            pass

    printed_assistant_text = False

    # Subscribe to events
    def on_event(event: AgentEvent) -> None:
        nonlocal printed_assistant_text
        if mode == "json":
            try:
                obj = _event_to_dict(event)
                print(json.dumps(obj, default=str), flush=True)
            except Exception:
                pass
        else:
            if event.type == "message_end" and isinstance(event.message, AssistantMessage):
                printed_assistant_text = any(
                    isinstance(block, TextContent) and bool(block.text)
                    for block in event.message.content
                )
            _handle_print_event(event, show_thinking=show_thinking)

    unsub = session.subscribe(on_event)

    try:
        # Send initial message (with optional images)
        if options.initial_message:
            if options.initial_images:
                await session.prompt(options.initial_message, images=options.initial_images)
            else:
                await session.prompt(options.initial_message)

        # Send remaining messages in sequence
        for msg in options.messages:
            await session.prompt(msg)

        # In text mode, output final assistant response
        if mode == "text" and not printed_assistant_text:
            state = session.state
            msgs = state.messages if hasattr(state, "messages") else []
            last = msgs[-1] if msgs else None

            if last and isinstance(last, AssistantMessage):
                # Check for error/aborted
                stop = getattr(last, "stop_reason", None) or getattr(last, "stopReason", None)
                if stop in ("error", "aborted"):
                    err_msg = getattr(last, "error_message", None) or getattr(last, "errorMessage", f"Request {stop}")
                    print(err_msg or f"Request {stop}", file=sys.stderr, flush=True)
                    return 1

                # Output text content
                for block in last.content:
                    if isinstance(block, TextContent) and block.text:
                        print(block.text, flush=True)

        # Check state for errors
        if hasattr(session.state, "error") and session.state.error:
            return 1

        return 0

    finally:
        unsub()
        if runtime_host is not None and hasattr(runtime_host, "dispose"):
            maybe = runtime_host.dispose()
            if asyncio.iscoroutine(maybe):
                await maybe
        # Explicit stdout flush (mirrors TS process.stdout.write("", resolve))
        sys.stdout.flush()


def _handle_print_event(event: AgentEvent, show_thinking: bool = False) -> None:
    """Handle a single agent event in text print mode."""
    if event.type == "message_end":
        msg = event.message
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextContent) and block.text.strip():
                    print(block.text, flush=True)
                elif hasattr(block, "thinking") and show_thinking and block.thinking.strip():
                    print(f"[Thinking: {block.thinking[:200]}...]", flush=True)

    elif event.type == "tool_execution_start":
        name = getattr(event, "tool_name", "")
        args = getattr(event, "args", {})
        print(f"  → {name}({_format_args(args)})", flush=True)

    elif event.type == "tool_execution_end":
        if getattr(event, "is_error", False):
            print("  ✗ Tool error", file=sys.stderr, flush=True)


def _format_args(args: Any) -> str:
    """Format tool arguments for display."""
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for k, v in list(args.items())[:3]:
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:47] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-serializable view of any value (Pydantic-aware)."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """Serialize a message's content blocks (text, thinking, toolCall, image)."""
    out: list[dict[str, Any]] = []
    for b in content or []:
        if isinstance(b, TextContent):
            out.append({"type": "text", "text": b.text})
        elif isinstance(b, ThinkingContent):
            out.append({"type": "thinking", "thinking": b.thinking})
        elif isinstance(b, ToolCall):
            out.append({
                "type": "toolCall",
                "id": getattr(b, "id", ""),
                "name": getattr(b, "name", ""),
                "arguments": getattr(b, "arguments", None) or getattr(b, "args", None),
            })
        elif isinstance(b, ImageContent):
            out.append({"type": "image", "mimeType": b.mime_type})
        else:
            out.append(_serialize(b))
    return out


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Serialize an assistant/tool message to a JSON dict."""
    if isinstance(msg, AssistantMessage):
        d: dict[str, Any] = {"role": "assistant", "content": _content_blocks(msg.content)}
        for attr, key in (("api", "api"), ("provider", "provider"), ("model", "model")):
            val = getattr(msg, attr, None)
            if val:
                d[key] = val
        stop = getattr(msg, "stop_reason", None) or getattr(msg, "stopReason", None)
        if stop:
            d["stopReason"] = stop
        error_message = (
            getattr(msg, "error_message", None)
            or getattr(msg, "errorMessage", None)
        )
        if error_message:
            d["errorMessage"] = error_message
        usage = getattr(msg, "usage", None)
        if usage:
            d["usage"] = {
                "input": getattr(usage, "input", 0),
                "output": getattr(usage, "output", 0),
            }
        return d
    if isinstance(msg, ToolResultMessage):
        return {
            "role": "tool_result",
            "toolCallId": msg.tool_call_id,
            "content": _content_blocks(msg.content),
            "isError": msg.is_error,
        }
    role = getattr(msg, "role", "")
    return {"role": role, "content": _content_blocks(getattr(msg, "content", []))}


def _event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Convert an agent event to a JSON-serializable dict.

    Mirrors the Node print-mode JSON stream: message_start/_update/_end carry the
    (partial) message, and message_update additionally carries the streaming
    delta (assistantMessageEvent) so headless consumers see live progress.
    """
    base: dict[str, Any] = {"type": event.type}

    if event.type == "message_end":
        base.update(_message_to_dict(event.message))

    elif event.type == "message_start":
        msg = getattr(event, "message", None)
        if msg is not None:
            base["message"] = _message_to_dict(msg)
            base["role"] = base["message"].get("role", "")
        else:
            base["role"] = getattr(event, "role", "")

    elif event.type == "message_update":
        base["assistantMessageEvent"] = _serialize(
            getattr(event, "assistant_message_event", None)
        )
        msg = getattr(event, "message", None)
        if msg is not None:
            base["message"] = _message_to_dict(msg)

    elif event.type == "tool_execution_start":
        base["toolName"] = getattr(event, "tool_name", "")
        base["args"] = getattr(event, "args", {})

    elif event.type == "tool_execution_update":
        base["toolCallId"] = getattr(event, "tool_call_id", "")
        base["toolName"] = getattr(event, "tool_name", "")
        base["args"] = getattr(event, "args", None)
        base["partialResult"] = _serialize(getattr(event, "partial_result", None))

    elif event.type == "tool_execution_end":
        base["toolCallId"] = getattr(event, "tool_call_id", "")
        base["toolName"] = getattr(event, "tool_name", "")
        base["result"] = _serialize(getattr(event, "result", None))
        base["isError"] = getattr(event, "is_error", False)

    elif event.type == "run_state":
        base["state"] = getattr(event, "state", "")
        base["phase"] = getattr(event, "phase", None)
        base["reason"] = getattr(event, "reason", None)
        base["toolCallId"] = getattr(event, "tool_call_id", None)
        base["toolName"] = getattr(event, "tool_name", None)
        base["terminal"] = getattr(event, "terminal", False)
        base["details"] = _serialize(getattr(event, "details", {}))

    elif event.type == "agent_end":
        reason = getattr(event, "reason", "") or getattr(event, "stop_reason", "")
        base["reason"] = reason

    return base
