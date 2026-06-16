from __future__ import annotations

import time

import pytest

from pi_ai.providers.openai_codex_responses import _raise_for_stream_failure
from pi_ai.types import AssistantMessage, Usage


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
