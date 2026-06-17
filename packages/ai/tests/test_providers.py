"""Tests for provider utilities."""
from __future__ import annotations

import pytest

from pi_ai.utils.json_parse import parse_partial_json, parse_streaming_json_result
from pi_ai.utils.validation import validate_tool_arguments
from pi_ai.types import Tool, ToolCall
from pi_ai.providers.transform_messages import transform_messages
from pi_ai import Context, UserMessage, AssistantMessage, TextContent, ToolCall, Usage
import time


# ── JSON parse tests ────────────────────────────────────────────────────────

def test_parse_partial_json_complete():
    result = parse_partial_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_partial_json_truncated():
    # Missing closing brace
    result = parse_partial_json('{"key": "val')
    # Should return None or partial result
    # The parser attempts to fix it
    assert result is None or isinstance(result, dict)


def test_parse_partial_json_empty():
    assert parse_partial_json("") is None
    assert parse_partial_json("   ") is None


def test_parse_partial_json_nested():
    result = parse_partial_json('{"a": {"b": 1}, "c": [1, 2]}')
    assert result == {"a": {"b": 1}, "c": [1, 2]}


def test_parse_streaming_json_result_repairs_truncated_string_and_object():
    result = parse_streaming_json_result('{"status": "completed", "summary": "Bounder')

    assert result.ok
    assert result.repair_applied
    assert result.value == {"status": "completed", "summary": "Bounder"}
    assert result.raw.endswith("Bounder")
    assert result.repaired_text is not None


def test_parse_streaming_json_result_preserves_unrepairable_raw():
    result = parse_streaming_json_result('{"status": completed')

    assert not result.ok
    assert result.value is None
    assert result.raw == '{"status": completed'
    assert result.error


# ── Validation tests ─────────────────────────────────────────────────────────

def make_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Calculates things",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )


def make_tool_call(args: dict) -> ToolCall:
    return ToolCall(type="toolCall", id="tc1", name="calculator", arguments=args)


def test_validate_tool_arguments_valid():
    tool = make_tool()
    tc = make_tool_call({"a": 1, "b": 2})
    result = validate_tool_arguments(tool, tc)
    assert result == {"a": 1, "b": 2}


def test_validate_tool_arguments_missing_required():
    tool = make_tool()
    tc = make_tool_call({"a": 1})  # missing "b"
    with pytest.raises(ValueError, match="Missing required parameter"):
        validate_tool_arguments(tool, tc)


def test_validate_tool_arguments_extra_fields_ok():
    tool = make_tool()
    tc = make_tool_call({"a": 1, "b": 2, "extra": "ok"})
    result = validate_tool_arguments(tool, tc)
    assert "a" in result


# ── Transform messages tests ──────────────────────────────────────────────────

def test_transform_messages_passthrough():
    from pi_ai.types import Model, ModelCost
    ts = int(time.time() * 1000)
    model = Model(
        id="claude-3-5-sonnet-20241022",
        name="Claude Sonnet",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        cost=ModelCost(),
        context_window=200000,
        max_tokens=8192,
    )
    messages = [UserMessage(role="user", content="Hello", timestamp=ts)]
    result = transform_messages(messages, model)
    assert len(result) == 1


def test_transform_messages_thinking_to_text():
    from pi_ai.types import ThinkingContent, Model, ModelCost
    ts = int(time.time() * 1000)
    assistant_msg = AssistantMessage(
        role="assistant",
        content=[
            ThinkingContent(type="thinking", thinking="I think..."),
            TextContent(type="text", text="Answer"),
        ],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        usage=Usage(),
        stop_reason="stop",
        timestamp=ts,
    )
    target_model = Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )
    messages = [
        UserMessage(role="user", content="Hello", timestamp=ts),
        assistant_msg,
    ]
    result = transform_messages(messages, target_model)
    # Thinking blocks should be converted to text for cross-model
    for msg in result:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                assert not isinstance(block, ThinkingContent), "Thinking block should be converted"
