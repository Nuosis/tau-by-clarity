"""Provider payload contracts for per-turn hook context."""

from pi_agent.hooks import _append_to_last_user_message


def test_pregeneration_context_uses_responses_input_text_block():
    payload = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}]}

    assert _append_to_last_user_message(payload, "extra context") is True
    assert payload["input"][0]["content"] == [
        {"type": "input_text", "text": "ping"},
        {"type": "input_text", "text": "extra context"},
    ]


def test_pregeneration_context_preserves_messages_text_block_shape():
    payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]}

    assert _append_to_last_user_message(payload, "extra context") is True
    assert payload["messages"][0]["content"][-1] == {
        "type": "text", "text": "extra context",
    }
