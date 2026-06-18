from __future__ import annotations

import time

import pytest

from pi_ai.providers.openai_codex_responses import _raise_for_stream_failure
from pi_ai.providers.openai_responses_shared import (
    _format_response_failed_error,
    _format_responses_error,
    process_responses_stream,
)
from pi_ai.types import AssistantMessage, Model, ModelCost, Usage
from pi_ai.utils.event_stream import EventStream


def test_openai_codex_responses_preserves_stream_error_message():
    msg = AssistantMessage(
        role="assistant",
        content=[],
        api="openai-codex-responses",
        provider="openai",
        model="gpt-5.5",
        usage=Usage(),
        stop_reason="error",
        error_message="Error Code None: None",
        timestamp=int(time.time() * 1000),
    )

    with pytest.raises(RuntimeError, match="Error Code None: None"):
        _raise_for_stream_failure(msg)


def test_responses_error_uses_nested_error_payload():
    event = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "No tool output found for function call call_123.",
        },
        "request_id": "req_abc",
    }

    message = _format_responses_error(event)

    assert "invalid_request_error" in message
    assert "No tool output found for function call call_123." in message
    assert "req_abc" in message


def test_response_failed_uses_nested_response_error_payload():
    event = {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {
                "code": "server_error",
                "message": "Upstream stream ended unexpectedly.",
            },
        },
    }

    message = _format_response_failed_error(event)

    assert "server_error" in message
    assert "Upstream stream ended unexpectedly." in message


@pytest.mark.asyncio
async def test_responses_final_tool_call_marks_repaired_arguments_malformed():
    async def events():
        for event in [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "write",
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "write",
                    "arguments": '{"path": "/tmp/readme.md", "content": "truncated',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]:
            yield event

    output = AssistantMessage(
        role="assistant",
        content=[],
        api="openai-codex-responses",
        provider="openai",
        model="gpt-5.5",
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    model = Model(
        id="gpt-5.5",
        name="GPT 5.5",
        api="openai-codex-responses",
        provider="openai",
        base_url="https://chatgpt.com/backend-api",
        cost=ModelCost(),
        context_window=200000,
        max_tokens=8192,
    )

    await process_responses_stream(events(), output, EventStream(), model)

    assert output.stop_reason == "toolUse"
    tool_call = output.content[0]
    assert tool_call.arguments == {"path": "/tmp/readme.md", "content": "truncated"}
    assert tool_call.arguments_repair_applied is True
    assert tool_call.arguments_parse_error
