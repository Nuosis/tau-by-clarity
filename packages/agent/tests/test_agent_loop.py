"""
Tests for agent_loop — mirrors packages/agent/test/agent-loop.test.ts
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from pi_ai.types import (
    AssistantMessage,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    agent_loop,
)
from pi_agent.types import AgentEventAgentEnd, AgentEventAgentStart, AgentEventMessageEnd


def _ts() -> int:
    return int(time.time() * 1000)


def make_user_message(text: str = "Hello") -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=_ts())


def make_assistant_message(model_id: str = "test-model", text: str = "Hi!") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model=model_id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )


async def _mock_stream_fn(model, context, options=None) -> AsyncGenerator:
    """Mock stream function that returns a simple text response."""
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)

    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)

    with_delta = partial.model_copy(update={"content": [TextContent(type="text", text="Hi!")]})
    yield EventTextDelta(type="text_delta", content_index=0, delta="Hi!", partial=with_delta)
    yield EventTextEnd(type="text_end", content_index=0, content="Hi!", partial=with_delta)

    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Hi!")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


@pytest.mark.asyncio
async def test_agent_loop_basic():
    """Test that agent_loop emits agent_start, message events, and agent_end."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    context = AgentContext(system_prompt="You are helpful", messages=[])
    prompts = [make_user_message("Hello")]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    event_types = []
    stream = agent_loop(prompts, context, config, stream_fn=_mock_stream_fn)

    async for event in stream:
        event_types.append(event.type)

    assert "agent_start" in event_types
    assert "agent_end" in event_types
    assert "message_start" in event_types
    assert "message_end" in event_types


def test_agent_loop_config_accepts_provider_specific_reasoning_level():
    """Provider-compatible models may use non-OpenAI reasoning labels."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    config = AgentLoopConfig(
        model=model,
        reasoning="adaptive",
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role")],
    )

    assert config.reasoning == "adaptive"


@pytest.mark.asyncio
async def test_agent_loop_returns_new_messages():
    """Test that agent_loop returns the new messages via result()."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    context = AgentContext(messages=[])
    prompts = [make_user_message("Hello")]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_mock_stream_fn)
    # Drain the stream
    async for _ in stream:
        pass

    result = await stream.result()
    assert len(result) >= 1  # At least the user prompt


@pytest.mark.asyncio
async def test_agent_loop_with_tool():
    """Test that tools get called when the assistant returns a tool_use block."""
    from pi_ai import get_model
    from pi_ai.types import Model, ModelCost
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    tool_executed = []

    async def execute_calculator(tool_call_id, params, cancel=None, on_update=None):
        tool_executed.append(params)
        return AgentToolResult(
            content=[TextContent(type="text", text=str(params.get("a", 0) + params.get("b", 0)))],
            details={"sum": params.get("a", 0) + params.get("b", 0)},
        )

    calculator = AgentTool(
        name="calculator",
        label="calculator",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        execute=execute_calculator,
    )

    # Mock stream that returns a tool call, then after tool result, returns text
    call_count = [0]

    async def _stream_with_tool(m, ctx, opts=None):
        call_count[0] += 1
        partial = AssistantMessage(
            role="assistant", content=[], api=m.api, provider=m.provider,
            model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count[0] == 1:
            # First call: return tool use
            tc = ToolCall(type="toolCall", id="tc1", name="calculator", arguments={"a": 2, "b": 3})
            with_tc = partial.model_copy(update={"content": [tc]})
            from pi_ai.types import EventToolCallEnd, EventToolCallStart
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant", content=[tc], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
        else:
            # Subsequent call: return text
            with_text = partial.model_copy(update={"content": [TextContent(type="text", text="5")]})
            yield EventTextStart(type="text_start", content_index=0, partial=with_text)
            yield EventTextEnd(type="text_end", content_index=0, content="5", partial=with_text)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="5")],
                api=m.api, provider=m.provider, model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

    context = AgentContext(messages=[], tools=[calculator])
    prompts = [make_user_message("What is 2+3?")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_tool)
    event_types = []
    async for event in stream:
        event_types.append(event.type)

    assert tool_executed, "Tool should have been called"
    assert tool_executed[0] == {"a": 2, "b": 3}
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_applies_prepare_arguments_before_validation_and_execution():
    """Tool prepareArguments should patch args before validation and execution."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    executed_args = []

    async def execute_prepared(tool_call_id, params, cancel=None, on_update=None):
        executed_args.append(params)
        return AgentToolResult(
            content=[TextContent(type="text", text=f"prepared:{params['prepared']}")],
            details={"params": params},
        )

    prepared_tool = AgentTool(
        name="prepared_tool",
        label="prepared_tool",
        description="Requires prepared args",
        parameters={
            "type": "object",
            "properties": {
                "raw": {"type": "string"},
                "prepared": {"type": "boolean"},
            },
            "required": ["raw", "prepared"],
        },
        prepareArguments=lambda params: {**params, "prepared": True},
        execute=execute_prepared,
    )

    call_count = 0

    async def _stream_with_prepared_tool(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count == 1:
            from pi_ai.types import EventToolCallEnd, EventToolCallStart

            tc = ToolCall(type="toolCall", id="tc-prepared", name="prepared_tool", arguments={"raw": "value"})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant",
                content=[tc],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
            return

        final = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventDone(type="done", reason="stop", message=final)

    context = AgentContext(messages=[], tools=[prepared_tool])
    prompts = [make_user_message("Use prepared tool")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_prepared_tool)
    async for _ in stream:
        pass

    assert executed_args == [{"raw": "value", "prepared": True}]


@pytest.mark.asyncio
async def test_agent_loop_reports_malformed_tool_arguments_without_executing():
    """Malformed streamed tool args should not be collapsed to {} and executed."""
    from pi_ai import get_model
    from pi_ai.types import EventToolCallEnd, EventToolCallStart

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    executed_args = []

    async def execute_tool(tool_call_id, params, cancel=None, on_update=None):
        executed_args.append(params)
        return AgentToolResult(content=[TextContent(type="text", text="should not run")])

    broken_tool = AgentTool(
        name="broken_tool",
        label="broken_tool",
        description="Requires args",
        parameters={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
        execute=execute_tool,
    )

    call_count = 0

    async def _stream_with_malformed_tool(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count > 1:
            final = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)
            return

        tc = ToolCall(
            type="toolCall",
            id="tc-malformed",
            name="broken_tool",
            arguments={},
            arguments_raw='{"status": completed',
            arguments_parse_error="Expecting value at line 1 column 12",
        )
        with_tc = partial.model_copy(update={"content": [tc]})
        yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
        yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
        final = AssistantMessage(
            role="assistant",
            content=[tc],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="toolUse",
            timestamp=_ts(),
        )
        yield EventDone(type="done", reason="toolUse", message=final)

    context = AgentContext(messages=[], tools=[broken_tool])
    prompts = [make_user_message("Use broken tool")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    tool_results = []
    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_malformed_tool)
    async for event in stream:
        if event.type == "message_end":
            msg = getattr(event, "message", None)
            if getattr(msg, "role", None) == "toolResult":
                tool_results.append(msg)

    assert executed_args == []
    assert tool_results
    text = tool_results[0].content[0].text
    assert "malformed JSON arguments" in text
    assert "tc-malformed" in text
    assert '\\"status\\": completed' in text


@pytest.mark.asyncio
async def test_agent_loop_stops_when_all_tool_results_request_termination():
    """A terminating tool result should stop the loop after emitting its tool result."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    call_count = 0

    async def execute_terminating(tool_call_id, params, cancel=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text="terminal result")],
            details={"done": True},
            terminate=True,
        )

    terminating_tool = AgentTool(
        name="terminating_tool",
        label="terminating_tool",
        description="Stops after result",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_terminating,
    )

    async def _stream_with_terminating_tool(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count == 1:
            from pi_ai.types import EventToolCallEnd, EventToolCallStart

            tc = ToolCall(type="toolCall", id="tc-term", name="terminating_tool", arguments={})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant",
                content=[tc],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
            return

        raise AssertionError("terminating tool result should stop before a second model call")

    context = AgentContext(messages=[], tools=[terminating_tool])
    prompts = [make_user_message("Use terminating tool")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_terminating_tool)
    events = []
    async for event in stream:
        events.append(event)

    assert call_count == 1
    tool_results = [event.message for event in events if event.type == "message_end" and getattr(event.message, "role", "") == "toolResult"]
    assert len(tool_results) == 1
    assert tool_results[0].content[0].text == "terminal result"
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_agent_loop_after_tool_call_can_set_termination():
    """after_tool_call terminate override should stop the loop after the tool batch."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    call_count = 0

    async def execute_tool(tool_call_id, params, cancel=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text="nonterminal result")],
            details={},
        )

    tool = AgentTool(
        name="hook_terminated_tool",
        label="hook_terminated_tool",
        description="Terminates via hook",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_tool,
    )

    async def after_tool_call(context, signal=None):
        return {"terminate": True}

    async def _stream_with_tool(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count == 1:
            from pi_ai.types import EventToolCallEnd, EventToolCallStart

            tc = ToolCall(type="toolCall", id="tc-hook-term", name="hook_terminated_tool", arguments={})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            final = AssistantMessage(
                role="assistant",
                content=[tc],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            )
            yield EventDone(type="done", reason="toolUse", message=final)
            return

        raise AssertionError("after_tool_call terminate should stop before a second model call")

    context = AgentContext(messages=[], tools=[tool])
    prompts = [make_user_message("Use hook terminating tool")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        after_tool_call=after_tool_call,
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_tool)
    async for _ in stream:
        pass

    assert call_count == 1


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_batch_in_parallel_by_default():
    """Same-batch tools should overlap by default and emit result messages in source order."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    completions: list[str] = []
    call_count = 0

    async def execute_first(tool_call_id, params, cancel=None, on_update=None):
        first_started.set()
        await asyncio.wait_for(second_started.wait(), timeout=0.2)
        await asyncio.sleep(0.02)
        completions.append("first")
        return AgentToolResult(
            content=[TextContent(type="text", text="first-overlapped")],
            details={},
            terminate=True,
        )

    async def execute_second(tool_call_id, params, cancel=None, on_update=None):
        await asyncio.wait_for(first_started.wait(), timeout=0.2)
        second_started.set()
        completions.append("second")
        return AgentToolResult(
            content=[TextContent(type="text", text="second-overlapped")],
            details={},
            terminate=True,
        )

    first_tool = AgentTool(
        name="first_tool",
        label="first_tool",
        description="First tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_first,
    )
    second_tool = AgentTool(
        name="second_tool",
        label="second_tool",
        description="Second tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_second,
    )

    async def _stream_with_two_tools(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("terminating tool batch should stop before a second model call")
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        from pi_ai.types import EventToolCallEnd, EventToolCallStart

        first_call = ToolCall(type="toolCall", id="tc-first", name="first_tool", arguments={})
        second_call = ToolCall(type="toolCall", id="tc-second", name="second_tool", arguments={})
        with_calls = partial.model_copy(update={"content": [first_call, second_call]})
        yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=first_call, partial=with_calls)
        yield EventToolCallStart(type="toolcall_start", content_index=1, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=1, tool_call=second_call, partial=with_calls)
        yield EventDone(
            type="done",
            reason="toolUse",
            message=AssistantMessage(
                role="assistant",
                content=[first_call, second_call],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            ),
        )

    context = AgentContext(messages=[], tools=[first_tool, second_tool])
    prompts = [make_user_message("Use two tools")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_with_two_tools)
    events = []
    async for event in stream:
        events.append(event)

    assert completions == ["second", "first"]
    tool_result_ends = [
        event.message for event in events
        if event.type == "message_end" and getattr(event.message, "role", "") == "toolResult"
    ]
    assert [msg.tool_name for msg in tool_result_ends] == ["first_tool", "second_tool"]
    assert [msg.content[0].text for msg in tool_result_ends] == ["first-overlapped", "second-overlapped"]


@pytest.mark.asyncio
async def test_agent_loop_parallel_mode_preflights_tool_calls_sequentially_before_execution():
    """Parallel mode should run validation/before hooks in source order before any execution starts."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    before_order: list[str] = []
    execution_events: list[str] = []
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def execute_first(tool_call_id, params, cancel=None, on_update=None):
        execution_events.append(f"execute:{params['value']}:before_count:{len(before_order)}")
        first_started.set()
        await asyncio.wait_for(second_started.wait(), timeout=0.2)
        return AgentToolResult(
            content=[TextContent(type="text", text="first")],
            details={},
            terminate=True,
        )

    async def execute_second(tool_call_id, params, cancel=None, on_update=None):
        execution_events.append(f"execute:{params['value']}:before_count:{len(before_order)}")
        await asyncio.wait_for(first_started.wait(), timeout=0.2)
        second_started.set()
        return AgentToolResult(
            content=[TextContent(type="text", text="second")],
            details={},
            terminate=True,
        )

    first_tool = AgentTool(
        name="first_tool",
        label="first_tool",
        description="First tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute_first,
    )
    second_tool = AgentTool(
        name="second_tool",
        label="second_tool",
        description="Second tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute_second,
    )

    async def before_tool_call(context, signal=None):
        value = context["args"]["value"]
        await asyncio.sleep(0.02)
        before_order.append(value)

    async def _stream_with_two_tools(m, ctx, opts=None):
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        from pi_ai.types import EventToolCallEnd, EventToolCallStart
        first_call = ToolCall(type="toolCall", id="tc-first", name="first_tool", arguments={"value": "first"})
        second_call = ToolCall(type="toolCall", id="tc-second", name="second_tool", arguments={"value": "second"})
        with_calls = partial.model_copy(update={"content": [first_call, second_call]})
        yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=first_call, partial=with_calls)
        yield EventToolCallStart(type="toolcall_start", content_index=1, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=1, tool_call=second_call, partial=with_calls)
        yield EventDone(
            type="done",
            reason="toolUse",
            message=AssistantMessage(
                role="assistant",
                content=[first_call, second_call],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            ),
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        before_tool_call=before_tool_call,
    )
    stream = agent_loop(
        [make_user_message("Use two tools")],
        AgentContext(messages=[], tools=[first_tool, second_tool]),
        config,
        stream_fn=_stream_with_two_tools,
    )
    async for _ in stream:
        pass

    assert before_order == ["first", "second"]
    assert set(execution_events) == {
        "execute:first:before_count:2",
        "execute:second:before_count:2",
    }


@pytest.mark.asyncio
async def test_agent_loop_sequential_execution_config_runs_tool_batch_in_order():
    """Global toolExecution=sequential should run same-batch tools one by one."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    execution_order: list[str] = []

    async def execute_first(tool_call_id, params, cancel=None, on_update=None):
        execution_order.append("first")
        return AgentToolResult(
            content=[TextContent(type="text", text="first")],
            details={},
            terminate=True,
        )

    async def execute_second(tool_call_id, params, cancel=None, on_update=None):
        execution_order.append("second")
        return AgentToolResult(
            content=[TextContent(type="text", text="second")],
            details={},
            terminate=True,
        )

    first_tool = AgentTool(
        name="first_tool",
        label="first_tool",
        description="First tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_first,
    )
    second_tool = AgentTool(
        name="second_tool",
        label="second_tool",
        description="Second tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_second,
    )

    async def _stream_with_two_tools(m, ctx, opts=None):
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        from pi_ai.types import EventToolCallEnd, EventToolCallStart
        first_call = ToolCall(type="toolCall", id="tc-first", name="first_tool", arguments={})
        second_call = ToolCall(type="toolCall", id="tc-second", name="second_tool", arguments={})
        with_calls = partial.model_copy(update={"content": [first_call, second_call]})
        yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=first_call, partial=with_calls)
        yield EventToolCallStart(type="toolcall_start", content_index=1, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=1, tool_call=second_call, partial=with_calls)
        yield EventDone(
            type="done",
            reason="toolUse",
            message=AssistantMessage(
                role="assistant",
                content=[first_call, second_call],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            ),
        )

    context = AgentContext(messages=[], tools=[first_tool, second_tool])
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        toolExecution="sequential",
    )

    stream = agent_loop([make_user_message("Use two tools")], context, config, stream_fn=_stream_with_two_tools)
    async for _ in stream:
        pass

    assert execution_order == ["first", "second"]


@pytest.mark.asyncio
async def test_agent_loop_sequential_tool_execution_mode_forces_batch_sequential():
    """Any called tool with executionMode=sequential should force the whole batch sequential."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    execution_order: list[str] = []

    async def execute_first(tool_call_id, params, cancel=None, on_update=None):
        execution_order.append("first")
        return AgentToolResult(
            content=[TextContent(type="text", text="first")],
            details={},
            terminate=True,
        )

    async def execute_second(tool_call_id, params, cancel=None, on_update=None):
        execution_order.append("second")
        return AgentToolResult(
            content=[TextContent(type="text", text="second")],
            details={},
            terminate=True,
        )

    first_tool = AgentTool(
        name="first_tool",
        label="first_tool",
        description="First tool",
        parameters={"type": "object", "properties": {}, "required": []},
        executionMode="sequential",
        execute=execute_first,
    )
    second_tool = AgentTool(
        name="second_tool",
        label="second_tool",
        description="Second tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_second,
    )

    async def _stream_with_two_tools(m, ctx, opts=None):
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        from pi_ai.types import EventToolCallEnd, EventToolCallStart
        first_call = ToolCall(type="toolCall", id="tc-first", name="first_tool", arguments={})
        second_call = ToolCall(type="toolCall", id="tc-second", name="second_tool", arguments={})
        with_calls = partial.model_copy(update={"content": [first_call, second_call]})
        yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=first_call, partial=with_calls)
        yield EventToolCallStart(type="toolcall_start", content_index=1, partial=with_calls)
        yield EventToolCallEnd(type="toolcall_end", content_index=1, tool_call=second_call, partial=with_calls)
        yield EventDone(
            type="done",
            reason="toolUse",
            message=AssistantMessage(
                role="assistant",
                content=[first_call, second_call],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=_ts(),
            ),
        )

    context = AgentContext(messages=[], tools=[first_tool, second_tool])
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    stream = agent_loop([make_user_message("Use two tools")], context, config, stream_fn=_stream_with_two_tools)
    async for _ in stream:
        pass

    assert execution_order == ["first", "second"]


@pytest.mark.asyncio
async def test_agent_loop_should_stop_after_turn_exits_before_follow_up_polling():
    """should_stop_after_turn should gracefully end after turn_end before follow-up queues are polled."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    call_count = 0
    follow_up_polled = False
    seen_stop_context: dict[str, object] = {}

    async def _stream_once(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("should_stop_after_turn should prevent a second model call")
        final = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventDone(type="done", reason="stop", message=final)

    async def should_stop_after_turn(context):
        seen_stop_context.update(context)
        return True

    async def get_follow_up_messages():
        nonlocal follow_up_polled
        follow_up_polled = True
        return [make_user_message("follow up")]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        should_stop_after_turn=should_stop_after_turn,
        get_follow_up_messages=get_follow_up_messages,
    )

    stream = agent_loop([make_user_message("Hello")], AgentContext(messages=[]), config, stream_fn=_stream_once)
    events = []
    async for event in stream:
        events.append(event)

    assert call_count == 1
    assert follow_up_polled is False
    assert seen_stop_context["message"].content[0].text == "done"
    assert seen_stop_context["tool_results"] == []
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_agent_loop_prepare_next_turn_replaces_context_before_next_model_call():
    """prepare_next_turn should be able to replace context before the automatic next turn."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    call_count = 0
    second_call_roles: list[str] = []

    async def execute_nonterminating(tool_call_id, params, cancel=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text="tool complete")],
            details={},
        )

    tool = AgentTool(
        name="nonterminal_tool",
        label="nonterminal_tool",
        description="Requires a second model turn",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute_nonterminating,
    )

    async def _stream_tool_then_text(m, ctx, opts=None):
        nonlocal call_count, second_call_roles
        call_count += 1
        if call_count == 2:
            second_call_roles = [getattr(msg, "role", "") for msg in ctx.messages]
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count == 1:
            from pi_ai.types import EventToolCallEnd, EventToolCallStart

            tc = ToolCall(type="toolCall", id="tc-nonterminal", name="nonterminal_tool", arguments={})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            yield EventDone(
                type="done",
                reason="toolUse",
                message=AssistantMessage(
                    role="assistant",
                    content=[tc],
                    api=m.api,
                    provider=m.provider,
                    model=m.id,
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=_ts(),
                ),
            )
            return

        yield EventDone(
            type="done",
            reason="stop",
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="replacement context observed")],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=_ts(),
            ),
        )

    async def prepare_next_turn(context):
        return {
            "context": AgentContext(
                system_prompt="replacement",
                messages=[make_user_message("replacement prompt")],
                tools=[tool],
            )
        }

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        prepare_next_turn=prepare_next_turn,
    )

    stream = agent_loop([make_user_message("Use the tool")], AgentContext(messages=[], tools=[tool]), config, stream_fn=_stream_tool_then_text)
    async for _ in stream:
        pass

    assert call_count == 2
    assert second_call_roles == ["user"]


@pytest.mark.asyncio
async def test_agent_loop_injects_steering_after_current_tool_batch_completes():
    """Queued steering should be injected after all same-batch tool calls complete."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    executed: list[str] = []
    steering_delivered = False
    saw_steering_on_second_call = False
    call_count = 0

    async def execute_echo(tool_call_id, params, cancel=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(type="text", text=f"ok:{params['value']}")],
            details={},
            terminate=False,
        )

    tool = AgentTool(
        name="echo",
        label="echo",
        description="Echo",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute_echo,
    )
    queued_message = make_user_message("interrupt")

    async def get_steering_messages():
        nonlocal steering_delivered
        if len(executed) >= 1 and not steering_delivered:
            steering_delivered = True
            return [queued_message]
        return []

    async def _stream_two_tools_then_done(m, ctx, opts=None):
        nonlocal call_count, saw_steering_on_second_call
        call_count += 1
        if call_count == 2:
            saw_steering_on_second_call = any(
                getattr(msg, "role", "") == "user" and getattr(msg, "content", "") == "interrupt"
                for msg in ctx.messages
            )
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)

        if call_count == 1:
            from pi_ai.types import EventToolCallEnd, EventToolCallStart

            first_call = ToolCall(type="toolCall", id="tc-first", name="echo", arguments={"value": "first"})
            second_call = ToolCall(type="toolCall", id="tc-second", name="echo", arguments={"value": "second"})
            with_calls = partial.model_copy(update={"content": [first_call, second_call]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_calls)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=first_call, partial=with_calls)
            yield EventToolCallStart(type="toolcall_start", content_index=1, partial=with_calls)
            yield EventToolCallEnd(type="toolcall_end", content_index=1, tool_call=second_call, partial=with_calls)
            yield EventDone(
                type="done",
                reason="toolUse",
                message=AssistantMessage(
                    role="assistant",
                    content=[first_call, second_call],
                    api=m.api,
                    provider=m.provider,
                    model=m.id,
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=_ts(),
                ),
            )
            return

        yield EventDone(
            type="done",
            reason="stop",
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                api=m.api,
                provider=m.provider,
                model=m.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=_ts(),
            ),
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        toolExecution="sequential",
        get_steering_messages=get_steering_messages,
    )

    stream = agent_loop([make_user_message("Use echo")], AgentContext(messages=[], tools=[tool]), config, stream_fn=_stream_two_tools_then_done)
    events = []
    async for event in stream:
        events.append(event)

    assert executed == ["first", "second"]
    assert saw_steering_on_second_call is True
    event_sequence = []
    for event in events:
        if event.type != "message_start":
            continue
        if getattr(event.message, "role", "") == "toolResult":
            event_sequence.append(f"tool:{event.message.tool_call_id}")
        elif getattr(event.message, "role", "") == "user" and getattr(event.message, "content", "") == "interrupt":
            event_sequence.append("user:interrupt")
    assert event_sequence == ["tool:tc-first", "tool:tc-second", "user:interrupt"]


@pytest.mark.asyncio
async def test_agent_loop_handles_error_event_payload():
    """EventError uses `error` field (not `message`) and must not crash."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def _stream_error(m, ctx, opts=None):
        partial = AssistantMessage(
            role="assistant",
            content=[],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        err_msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            api=m.api,
            provider=m.provider,
            model=m.id,
            usage=Usage(),
            stop_reason="error",
            error_message="boom",
            timestamp=_ts(),
        )
        yield EventError(type="error", reason="error", error=err_msg)

    context = AgentContext(messages=[])
    prompts = [make_user_message("Hello")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_stream_error)
    event_types = []
    async for event in stream:
        event_types.append(event.type)

    assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_emits_native_assistant_finished_run_state():
    """Assistant stop with no pending tools is the native completion condition."""
    from pi_ai import get_model
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    context = AgentContext(messages=[])
    prompts = [make_user_message("Hello")]
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs
            if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )

    stream = agent_loop(prompts, context, config, stream_fn=_mock_stream_fn)
    states = []
    async for event in stream:
        if event.type == "run_state":
            states.append(event)

    assert [state.state for state in states] == ["waiting_on_model", "assistant_finished"]
    assert states[-1].terminal is True
    assert states[-1].reason == "assistant_stop_no_pending_tools"


@pytest.mark.asyncio
async def test_agent_loop_classifies_tool_timeout_and_continues():
    """A configured tool timeout is a tool fault, not agent completion."""
    from pi_ai import get_model
    from pi_ai.types import EventToolCallEnd, EventToolCallStart
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")

    async def execute_slow(tool_call_id, params, cancel=None, on_update=None):
        await asyncio.sleep(0.05)
        return AgentToolResult(content=[TextContent(type="text", text="late")])

    slow_tool = AgentTool(
        name="slow_tool",
        label="slow_tool",
        description="Slow test tool",
        parameters={"type": "object", "properties": {}},
        executionPolicy={"timeout_ms": 1, "retryable": True, "idempotent": True},
        execute=execute_slow,
    )

    call_count = [0]

    async def _stream_tool_then_done(m, ctx, opts=None):
        call_count[0] += 1
        partial = AssistantMessage(
            role="assistant", content=[], api=m.api, provider=m.provider,
            model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        if call_count[0] == 1:
            tc = ToolCall(type="toolCall", id="tc-slow", name="slow_tool", arguments={})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            yield EventDone(
                type="done",
                reason="toolUse",
                message=partial.model_copy(update={"content": [tc], "stop_reason": "toolUse"}),
            )
            return
        final = partial.model_copy(update={
            "content": [TextContent(type="text", text="handled timeout")],
            "stop_reason": "stop",
        })
        yield EventDone(type="done", reason="stop", message=final)

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs
            if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
    )
    stream = agent_loop(
        [make_user_message("Use slow tool")],
        AgentContext(messages=[], tools=[slow_tool]),
        config,
        stream_fn=_stream_tool_then_done,
    )
    states = []
    tool_end = None
    async for event in stream:
        if event.type == "run_state":
            states.append(event.state)
        if event.type == "tool_execution_end":
            tool_end = event

    assert "tool_timeout" in states
    assert states[-1] == "assistant_finished"
    assert tool_end is not None
    assert tool_end.is_error is True
    assert tool_end.result.details["kind"] == "tool_timeout"
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_before_tool_call_can_rewrite_arguments():
    """A before_tool_call hook returning {"arguments": ...} replaces the args the
    tool executes with (e.g. a PII filter reapplying real values for tokens)."""
    from pi_ai import get_model

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    received_args: dict = {}

    async def execute_tool(tool_call_id, params, cancel=None, on_update=None):
        received_args.update(params)
        return AgentToolResult(content=[TextContent(type="text", text="ok")], details={})

    tool = AgentTool(
        name="echo_tool",
        label="echo_tool",
        description="echo",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        execute=execute_tool,
    )

    async def before_tool_call(context, signal=None):
        # Model emitted a token; reapply the real value before execution.
        assert context["args"]["command"] == "echo [PII:EMAIL:1]"
        return {"arguments": {"command": "echo real@user.com"}}

    call_count = 0

    async def _stream_with_tool(m, ctx, opts=None):
        nonlocal call_count
        call_count += 1
        from pi_ai.types import EventToolCallEnd, EventToolCallStart

        partial = AssistantMessage(
            role="assistant", content=[], api=m.api, provider=m.provider,
            model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        if call_count == 1:
            tc = ToolCall(type="toolCall", id="tc-1", name="echo_tool",
                          arguments={"command": "echo [PII:EMAIL:1]"})
            with_tc = partial.model_copy(update={"content": [tc]})
            yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
            yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
            yield EventDone(
                type="done", reason="toolUse",
                message=AssistantMessage(
                    role="assistant", content=[tc], api=m.api, provider=m.provider,
                    model=m.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
                ),
            )
            return
        # Second turn (after tool result): finish.
        final = AssistantMessage(
            role="assistant", content=[TextContent(type="text", text="done")],
            api=m.api, provider=m.provider, model=m.id, usage=Usage(),
            stop_reason="stop", timestamp=_ts(),
        )
        yield EventDone(type="done", reason="stop", message=final)

    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ],
        before_tool_call=before_tool_call,
    )
    stream = agent_loop([make_user_message("go")], context, config, stream_fn=_stream_with_tool)
    async for _ in stream:
        pass

    assert received_args == {"command": "echo real@user.com"}
