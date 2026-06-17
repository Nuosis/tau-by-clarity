"""
Agent loop — mirrors packages/agent/src/agent-loop.ts

Core loop logic: agentLoop(), agentLoopContinue(), runLoop().
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, AsyncGenerator

from pi_ai import stream_simple as _default_stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils.event_stream import EventStream
from pi_ai.utils.validation import validate_tool_arguments

from .types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventToolEnd,
    AgentEventToolStart,
    AgentEventToolUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentEventRunState,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolExecutionPolicy,
    AgentToolResult,
    StreamFn,
)


def _log_loop_exception(event: str, exc: BaseException) -> None:
    try:
        from pi_coding_agent.core.cli_debug_log import log_exception

        log_exception(event, exc)
    except Exception:
        pass


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream(
        is_done=lambda e: e.type == "agent_end",
        get_result=lambda e: e.messages if e.type == "agent_end" else [],
    )


def _tool_result_terminates(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("terminate", False))
    return bool(getattr(result, "terminate", False))


def _format_malformed_tool_arguments(tool_call: ToolCall) -> str:
    raw = getattr(tool_call, "arguments_raw", None) or ""
    error = getattr(tool_call, "arguments_parse_error", None) or "unknown parse error"
    preview_head = raw[:1000]
    preview_tail = raw[-1000:] if len(raw) > 1000 else ""
    payload = {
        "tool_name": tool_call.name,
        "tool_call_id": tool_call.id,
        "error": error,
        "raw_length": len(raw),
        "raw_head": preview_head,
        "raw_tail": preview_tail,
    }
    return (
        f'Tool "{tool_call.name}" received malformed JSON arguments and was not executed.\n'
        f"Diagnostic:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def _emit_run_state(
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
    state: str,
    *,
    phase: str | None = None,
    reason: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    terminal: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    ev_stream.push(AgentEventRunState(
        state=state,
        phase=phase,
        reason=reason,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        terminal=terminal,
        details=details or {},
    ))


def _tool_execution_policy(tool: AgentTool) -> AgentToolExecutionPolicy:
    raw = getattr(tool, "executionPolicy", None)
    if isinstance(raw, AgentToolExecutionPolicy):
        return raw
    if isinstance(raw, dict):
        try:
            return AgentToolExecutionPolicy(**raw)
        except Exception:
            return AgentToolExecutionPolicy()
    return AgentToolExecutionPolicy()


def _tool_result_details(result: Any) -> dict[str, Any]:
    details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
    return details if isinstance(details, dict) else {}


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Start an agent loop with new prompt messages.
    Mirrors agentLoop() in TypeScript.
    """
    ev_stream = _create_agent_stream()

    async def _run():
        try:
            new_messages: list[AgentMessage] = list(prompts)
            current_context = AgentContext(
                system_prompt=context.system_prompt,
                messages=list(context.messages) + list(prompts),
                tools=context.tools,
            )

            ev_stream.push(AgentEventAgentStart())
            ev_stream.push(AgentEventTurnStart())
            for prompt in prompts:
                ev_stream.push(AgentEventMessageStart(message=prompt))
                ev_stream.push(AgentEventMessageEnd(message=prompt))

            await _run_loop(current_context, new_messages, config, cancel_event, ev_stream, stream_fn)
        except Exception as e:
            _log_loop_exception("agent_loop_exception", e)
            # Ensure the stream is always terminated even if the loop crashes
            if not ev_stream._result_event.is_set():
                ev_stream.fail(e)

    asyncio.ensure_future(_run())
    return ev_stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """
    Continue from the current context without adding a new message.
    Mirrors agentLoopContinue() in TypeScript.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if hasattr(last, "role") and last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    ev_stream = _create_agent_stream()

    async def _run():
        try:
            new_messages: list[AgentMessage] = []
            current_context = AgentContext(
                system_prompt=context.system_prompt,
                messages=list(context.messages),
                tools=context.tools,
            )

            ev_stream.push(AgentEventAgentStart())
            ev_stream.push(AgentEventTurnStart())

            await _run_loop(current_context, new_messages, config, cancel_event, ev_stream, stream_fn)
        except Exception as e:
            _log_loop_exception("agent_loop_continue_exception", e)
            if not ev_stream._result_event.is_set():
                ev_stream.fail(e)

    asyncio.ensure_future(_run())
    return ev_stream


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> None:
    """
    Main loop logic — mirrors runLoop() in TypeScript.
    """
    first_turn = True
    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages:
        pending_messages = await config.get_steering_messages()
    terminal_state = "assistant_finished"
    terminal_reason = "assistant_stop_no_pending_tools"
    terminal_phase = "assistant"

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                ev_stream.push(AgentEventTurnStart())
            else:
                first_turn = False

            # Inject pending messages
            if pending_messages:
                for msg in pending_messages:
                    ev_stream.push(AgentEventMessageStart(message=msg))
                    ev_stream.push(AgentEventMessageEnd(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            # Stream assistant response
            _emit_run_state(ev_stream, "waiting_on_model", phase="model")
            message = await _stream_assistant_response(
                current_context, config, cancel_event, ev_stream, stream_fn
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                _emit_run_state(
                    ev_stream,
                    "aborted" if message.stop_reason == "aborted" else "provider_error",
                    phase="model",
                    reason=getattr(message, "error_message", None) or message.stop_reason,
                    terminal=True,
                )
                ev_stream.push(AgentEventTurnEnd(message=message, tool_results=[]))
                ev_stream.push(AgentEventAgentEnd(messages=new_messages))
                ev_stream.end(new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                execution = await _execute_tool_calls(
                    current_context.tools,
                    current_context,
                    message,
                    config,
                    cancel_event,
                    ev_stream,
                    config.get_steering_messages,
                )
                tool_results.extend(execution["tool_results"])
                has_more_tool_calls = not bool(execution.get("terminate", False))
                if execution.get("terminate"):
                    terminal_state = "tool_terminated"
                    terminal_reason = "terminal_tool_result"
                    terminal_phase = "tool"

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            ev_stream.push(AgentEventTurnEnd(message=message, tool_results=tool_results))

            turn_context = {
                "message": message,
                "tool_results": tool_results,
                "toolResults": tool_results,
                "context": current_context,
                "new_messages": new_messages,
                "newMessages": new_messages,
            }
            prepare_next_turn = config.prepare_next_turn or config.prepareNextTurn
            if prepare_next_turn:
                next_turn_snapshot = prepare_next_turn(turn_context)
                if inspect.isawaitable(next_turn_snapshot):
                    next_turn_snapshot = await next_turn_snapshot
                if next_turn_snapshot:
                    if isinstance(next_turn_snapshot, dict):
                        next_context = next_turn_snapshot.get("context")
                        next_model = next_turn_snapshot.get("model")
                        next_thinking = next_turn_snapshot.get(
                            "thinking_level",
                            next_turn_snapshot.get("thinkingLevel", None),
                        )
                    else:
                        next_context = getattr(next_turn_snapshot, "context", None)
                        next_model = getattr(next_turn_snapshot, "model", None)
                        next_thinking = getattr(
                            next_turn_snapshot,
                            "thinking_level",
                            getattr(next_turn_snapshot, "thinkingLevel", None),
                        )
                    if next_context is not None:
                        current_context = next_context
                        turn_context["context"] = current_context
                    if next_model is not None:
                        config.model = next_model
                    if next_thinking is not None:
                        config.reasoning = None if next_thinking == "off" else next_thinking

            should_stop_after_turn = config.should_stop_after_turn or config.shouldStopAfterTurn
            if should_stop_after_turn:
                should_stop = should_stop_after_turn(turn_context)
                if inspect.isawaitable(should_stop):
                    should_stop = await should_stop
                if should_stop:
                    ev_stream.push(AgentEventAgentEnd(messages=new_messages))
                    ev_stream.end(new_messages)
                    return

            pending_messages = []
            if config.get_steering_messages:
                pending_messages = await config.get_steering_messages()

        # Check for follow-up messages
        follow_up_messages: list[AgentMessage] = []
        if config.get_follow_up_messages:
            follow_up_messages = await config.get_follow_up_messages()

        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    _emit_run_state(
        ev_stream,
        terminal_state,
        phase=terminal_phase,
        reason=terminal_reason,
        terminal=True,
    )
    ev_stream.push(AgentEventAgentEnd(messages=new_messages))
    ev_stream.end(new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """
    Stream an assistant response from the LLM.
    Mirrors streamAssistantResponse() in TypeScript.
    """
    messages = context.messages

    # Apply context transform if configured
    if config.transform_context:
        messages = await config.transform_context(messages, cancel_event)

    # Convert to LLM-compatible messages
    convert = config.convert_to_llm
    if inspect.iscoroutinefunction(convert):
        llm_messages = await convert(messages)
    else:
        result = convert(messages)
        if inspect.isawaitable(result):
            llm_messages = await result
        else:
            llm_messages = result

    # Build LLM context
    llm_context = Context(
        system_prompt=context.system_prompt or None,
        messages=llm_messages,
        tools=[t for t in (context.tools or [])],
    )

    fn = stream_fn or _default_stream_simple

    # Resolve API key
    resolved_api_key = config.api_key
    if config.get_api_key:
        key_result = config.get_api_key(config.model.provider)
        if inspect.isawaitable(key_result):
            key_result = await key_result
        resolved_api_key = key_result or resolved_api_key

    from pi_ai import SimpleStreamOptions
    stream_opts = SimpleStreamOptions(
        reasoning=config.reasoning,
        thinking_budgets=config.thinking_budgets,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        signal=cancel_event,
        api_key=resolved_api_key,
        transport=config.transport,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        on_payload=config.on_payload,
        on_response=config.on_response,
        headers=config.headers,
        max_retry_delay_ms=config.max_retry_delay_ms,
        metadata=config.metadata,
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    response_stream = fn(config.model, llm_context, stream_opts)

    async for event in response_stream:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            ev_stream.push(AgentEventMessageStart(message=partial_message))

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                ev_stream.push(AgentEventMessageUpdate(
                    message=partial_message,
                    assistant_message_event=event,
                ))

        elif event.type in ("done", "error"):
            final_message = event.message if event.type == "done" else event.error
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                ev_stream.push(AgentEventMessageStart(message=final_message))
            ev_stream.push(AgentEventMessageEnd(message=final_message))
            return final_message

    # Fallback: return partial if no done/error event
    if partial_message:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Request was aborted")
        return partial_message

    raise RuntimeError("Stream ended without a final message")


async def _execute_tool_calls(
    tools: list[AgentTool] | None,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: Any | None = None,
) -> dict[str, Any]:
    """
    Execute tool calls from an assistant message.
    Mirrors executeToolCalls() in TypeScript.
    """
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    tools_by_name = {tool.name: tool for tool in (tools or [])}
    has_sequential_tool_call = any(
        tools_by_name.get(tool_call.name) is not None
        and tools_by_name[tool_call.name].executionMode == "sequential"
        for tool_call in tool_calls
    )
    if config.toolExecution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            tools,
            current_context,
            assistant_message,
            tool_calls,
            config,
            cancel_event,
            ev_stream,
            get_steering_messages,
        )
    return await _execute_tool_calls_parallel(
        tools,
        current_context,
        assistant_message,
        tool_calls,
        config,
        cancel_event,
        ev_stream,
    )


async def _execute_tool_calls_sequential(
    tools: list[AgentTool] | None,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: Any | None = None,
) -> dict[str, Any]:
    results: list[ToolResultMessage] = []
    terminate_flags: list[bool] = []

    for tool_call in tool_calls:
        ev_stream.push(AgentEventToolStart(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))
        preparation = await _prepare_tool_call(
            tools,
            current_context,
            assistant_message,
            tool_call,
            config,
            cancel_event,
        )
        if preparation["kind"] == "immediate":
            finalized = {
                "tool_call": tool_call,
                "result": preparation["result"],
                "is_error": preparation["is_error"],
            }
            if preparation["is_error"]:
                _emit_run_state(
                    ev_stream,
                    "tool_error",
                    phase="tool",
                    reason="tool_preflight_error",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    details=_tool_result_details(preparation["result"]),
                )
        else:
            finalized = await _execute_prepared_tool_call(
                current_context,
                assistant_message,
                preparation,
                config,
                cancel_event,
                ev_stream,
            )
        _emit_tool_execution_end(finalized, ev_stream)
        tool_result_msg = _create_tool_result_message(finalized)
        results.append(tool_result_msg)
        terminate_flags.append(_tool_result_terminates(finalized["result"]))
        _emit_tool_result_message(tool_result_msg, ev_stream)

    return {
        "tool_results": results,
        "terminate": bool(terminate_flags) and all(terminate_flags),
    }


async def _execute_tool_calls_parallel(
    tools: list[AgentTool] | None,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
) -> dict[str, Any]:
    preflight_entries: list[tuple[str, dict[str, Any]]] = []

    for tool_call in tool_calls:
        ev_stream.push(AgentEventToolStart(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))
        preparation = await _prepare_tool_call(
            tools,
            current_context,
            assistant_message,
            tool_call,
            config,
            cancel_event,
        )
        if preparation["kind"] == "immediate":
            finalized = {
                "tool_call": tool_call,
                "result": preparation["result"],
                "is_error": preparation["is_error"],
            }
            if preparation["is_error"]:
                _emit_run_state(
                    ev_stream,
                    "tool_error",
                    phase="tool",
                    reason="tool_preflight_error",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    details=_tool_result_details(preparation["result"]),
                )
            _emit_tool_execution_end(finalized, ev_stream)
            preflight_entries.append(("finalized", finalized))
            if cancel_event and cancel_event.is_set():
                break
            continue
        preflight_entries.append(("prepared", preparation))
        if cancel_event and cancel_event.is_set():
            break

    finalized_calls: list[dict[str, Any]] = []
    for entry in await asyncio.gather(*[
        _execute_prepared_tool_call_and_emit(
            current_context,
            assistant_message,
            value,
            config,
            cancel_event,
            ev_stream,
        ) if kind == "prepared" else _resolved(value)
        for kind, value in preflight_entries
    ]):
        finalized_calls.append(entry)

    results: list[ToolResultMessage] = []
    terminate_flags: list[bool] = []
    for finalized in finalized_calls:
        tool_result_msg = _create_tool_result_message(finalized)
        results.append(tool_result_msg)
        terminate_flags.append(_tool_result_terminates(finalized["result"]))
        _emit_tool_result_message(tool_result_msg, ev_stream)

    return {
        "tool_results": results,
        "terminate": bool(terminate_flags) and all(terminate_flags),
    }


async def _resolved(value: dict[str, Any]) -> dict[str, Any]:
    return value


async def _execute_prepared_tool_call_and_emit(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    preparation: dict[str, Any],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
) -> dict[str, Any]:
    finalized = await _execute_prepared_tool_call(
        current_context,
        assistant_message,
        preparation,
        config,
        cancel_event,
        ev_stream,
    )
    _emit_tool_execution_end(finalized, ev_stream)
    return finalized


async def _prepare_tool_call(
    tools: list[AgentTool] | None,
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
) -> dict[str, Any]:
    tool = next((t for t in (tools or []) if t.name == tool_call.name), None)

    try:
        if not tool:
            raise ValueError(f"Tool {tool_call.name} not found")

        # Build a Tool-compatible object for validation
        from pi_ai.types import Tool as AiTool
        ai_tool = AiTool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )
        prepared_tool_call = tool_call
        if tool.prepareArguments is not None:
            prepared_args = tool.prepareArguments(tool_call.arguments)
            if prepared_args is not tool_call.arguments:
                prepared_tool_call = tool_call.model_copy(update={"arguments": prepared_args})
        if getattr(prepared_tool_call, "arguments_parse_error", None):
            raise ValueError(_format_malformed_tool_arguments(prepared_tool_call))
        validated_args = validate_tool_arguments(ai_tool, prepared_tool_call)

        if config.before_tool_call:
            before_result = await config.before_tool_call(
                {
                    "assistant_message": assistant_message,
                    "assistantMessage": assistant_message,
                    "tool_call": tool_call,
                    "toolCall": tool_call,
                    "args": validated_args,
                    "context": current_context,
                },
                cancel_event,
            )
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Operation aborted")
            block = False
            reason = None
            if isinstance(before_result, dict):
                block = bool(before_result.get("block"))
                reason = before_result.get("reason")
            elif before_result is not None:
                block = bool(getattr(before_result, "block", False))
                reason = getattr(before_result, "reason", None)
            if block:
                raise RuntimeError(reason or "Tool execution was blocked")
            # Allow the hook to rewrite arguments before execution (e.g. a PII
            # filter reapplying real values for tokens the model emitted). Honor
            # an "arguments" (or "input") field returned from the hook.
            if isinstance(before_result, dict):
                rewritten = before_result.get("arguments")
                if rewritten is None:
                    rewritten = before_result.get("input")
                if isinstance(rewritten, dict):
                    validated_args = rewritten
        return {
            "kind": "prepared",
            "tool_call": tool_call,
            "tool": tool,
            "args": validated_args,
        }
    except Exception as e:
        return {
            "kind": "immediate",
            "result": AgentToolResult(
                content=[TextContent(type="text", text=str(e))],
                details={},
            ),
            "is_error": True,
        }


async def _execute_prepared_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    preparation: dict[str, Any],
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
) -> dict[str, Any]:
    tool_call = preparation["tool_call"]
    tool = preparation["tool"]
    args = preparation["args"]
    is_error = False
    policy = _tool_execution_policy(tool)

    def on_update(partial_result: AgentToolResult) -> None:
        ev_stream.push(AgentEventToolUpdate(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
            partial_result=partial_result,
        ))

    max_attempts = max(1, int(policy.max_attempts or 1))
    attempt = 0
    while True:
        attempt += 1
        _emit_run_state(
            ev_stream,
            "waiting_on_tool",
            phase="tool",
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            details={
                "timeout_ms": policy.timeout_ms,
                "retryable": policy.retryable,
                "idempotent": policy.idempotent,
                "max_attempts": policy.max_attempts,
                "attempt": attempt,
            },
        )
        try:
            coroutine = tool.execute(tool_call.id, args, cancel_event, on_update)
            if policy.timeout_ms is not None and policy.timeout_ms > 0:
                result = await asyncio.wait_for(coroutine, timeout=policy.timeout_ms / 1000)
            else:
                result = await coroutine
            break
        except asyncio.TimeoutError:
            details = {
                "kind": "tool_timeout",
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "timeout_ms": policy.timeout_ms,
                "retryable": policy.retryable,
                "idempotent": policy.idempotent,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            _emit_run_state(
                ev_stream,
                "tool_timeout",
                phase="tool",
                reason="tool_execution_timeout",
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                details=details,
            )
            if policy.retryable and policy.idempotent and attempt < max_attempts:
                _emit_run_state(
                    ev_stream,
                    "tool_retry",
                    phase="tool",
                    reason="tool_timeout_retry",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    details=details,
                )
                continue
            result = AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Tool {tool_call.name} timed out after "
                            f"{policy.timeout_ms}ms."
                        ),
                    )
                ],
                details=details,
            )
            is_error = True
            break
        except Exception as e:
            details = {
                "kind": "tool_error",
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            _emit_run_state(
                ev_stream,
                "tool_error",
                phase="tool",
                reason=str(e),
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                details=details,
            )
            if policy.retryable and policy.idempotent and attempt < max_attempts:
                _emit_run_state(
                    ev_stream,
                    "tool_retry",
                    phase="tool",
                    reason="tool_error_retry",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    details=details,
                )
                continue
            result = AgentToolResult(
                content=[TextContent(type="text", text=str(e))],
                details=details,
            )
            is_error = True
            break

    result, is_error = await _apply_after_tool_call(
        current_context,
        assistant_message,
        tool_call,
        args,
        result,
        is_error,
        config,
        cancel_event,
    )
    finalized = {"tool_call": tool_call, "result": result, "is_error": is_error}
    return finalized


async def _apply_after_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    args: dict[str, Any],
    result: AgentToolResult,
    is_error: bool,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None,
) -> tuple[AgentToolResult, bool]:
    if not config.after_tool_call:
        return result, is_error
    try:
        after_result = await config.after_tool_call(
            {
                "assistant_message": assistant_message,
                "assistantMessage": assistant_message,
                "tool_call": tool_call,
                "toolCall": tool_call,
                "args": args,
                "result": result,
                "is_error": is_error,
                "isError": is_error,
                "context": current_context,
            },
            cancel_event,
        )
        if after_result:
            if isinstance(after_result, dict):
                content = after_result.get("content", None)
                details = after_result.get("details", None)
                next_is_error = after_result.get("is_error", after_result.get("isError", None))
                terminate = after_result.get("terminate", None)
            else:
                content = getattr(after_result, "content", None)
                details = getattr(after_result, "details", None)
                next_is_error = getattr(after_result, "is_error", getattr(after_result, "isError", None))
                terminate = getattr(after_result, "terminate", None)
            if content is not None:
                result.content = content
            if details is not None:
                result.details = details
            if next_is_error is not None:
                is_error = bool(next_is_error)
            if terminate is not None:
                result.terminate = bool(terminate)
    except Exception as after_error:
        result = AgentToolResult(
            content=[TextContent(type="text", text=str(after_error))],
            details={},
        )
        is_error = True
    return result, is_error


def _create_tool_result_message(finalized: dict[str, Any]) -> ToolResultMessage:
    tool_call = finalized["tool_call"]
    result = finalized["result"]
    return ToolResultMessage(
        role="toolResult",
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=finalized["is_error"],
        timestamp=int(time.time() * 1000),
    )


def _emit_tool_execution_end(
    finalized: dict[str, Any],
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
) -> None:
    tool_call = finalized["tool_call"]
    ev_stream.push(AgentEventToolEnd(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=finalized["result"],
        is_error=finalized["is_error"],
    ))


def _emit_tool_result_message(
    tool_result_msg: ToolResultMessage,
    ev_stream: EventStream[AgentEvent, list[AgentMessage]],
) -> None:
    ev_stream.push(AgentEventMessageStart(message=tool_result_msg))
    ev_stream.push(AgentEventMessageEnd(message=tool_result_msg))
