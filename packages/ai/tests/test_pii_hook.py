"""The universal PII chokepoint: every complete_simple/stream call tokenizes the
outbound context and detokenizes the reply, for ANY caller, when a filter is
registered — and is an exact no-op otherwise."""

import base64
import hashlib
import importlib
import io
import json
import os
import random
import sys
import time

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pi_ai
from pi_ai.api_registry import register_api_provider, unregister_api_providers
from pi_ai.types import (
    AssistantMessage,
    Context,
    EventDone,
    ImageContent,
    Model,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _model() -> Model:
    return Model(id="fake-1", name="Fake", api="fake-pii-api", provider="fake", base_url="http://x")


class _CapturingProvider:
    """Records the context it receives and echoes the last user text back."""

    def __init__(self):
        self.seen_context = None

    async def stream_simple(self, model, context, options):
        self.seen_context = context
        last_user = ""
        for m in context.messages:
            c = m.content
            last_user = c if isinstance(c, str) else "".join(
                getattr(b, "text", "") for b in c if getattr(b, "type", None) == "text"
            )
        msg = AssistantMessage(
            content=[TextContent(text=f"echo: {last_user}")],
            api=model.api, provider=model.provider, model=model.id, timestamp=0,
        )
        yield EventDone(message=msg, reason="stop")

    async def stream(self, model, context, options):
        async for e in self.stream_simple(model, context, options):
            yield e


@pytest.fixture
def provider():
    p = _CapturingProvider()
    register_api_provider("fake-pii-api", p, source_id="test-pii")
    yield p
    unregister_api_providers("test-pii")
    pi_ai.unregister_pii_filter()
    pi_ai.unregister_compression_observer()
    pi_ai.unregister_compressor()
    pi_ai.reset_compression_stats()
    pi_ai.unregister_request_context_manager()


def _filter_factory():
    # Trivial reversible filter: SECRET <-> [TOK].
    return (lambda s: s.replace("SECRET", "[TOK]"), lambda s: s.replace("[TOK]", "SECRET"))


def _large_token_text(prefix: str = "xxxxxxxx") -> str:
    return (f"{prefix} alpha beta gamma delta epsilon zeta eta theta iota kappa " * 30)


@pytest.mark.asyncio
async def test_outbound_tokenized_and_reply_detokenized(provider):
    pi_ai.register_pii_filter(_filter_factory)
    ctx = Context(
        system_prompt="system has SECRET too",
        messages=[UserMessage(content="my value is SECRET", timestamp=0)],
    )
    result = await pi_ai.complete_simple(_model(), ctx)

    # Provider (the wire) never saw the real value.
    sent_user = provider.seen_context.messages[0].content
    assert "SECRET" not in sent_user and "[TOK]" in sent_user
    assert "SECRET" not in provider.seen_context.system_prompt

    # Caller's reply is restored to cleartext.
    reply = "".join(b.text for b in result.content if b.type == "text")
    assert reply == "echo: my value is [TOK]".replace("[TOK]", "SECRET")

    # Caller's original context object was not mutated.
    assert ctx.messages[0].content == "my value is SECRET"


@pytest.mark.asyncio
async def test_no_filter_is_exact_noop(provider):
    pi_ai.unregister_pii_filter()
    assert pi_ai.has_pii_filter() is False
    ctx = Context(messages=[UserMessage(content="my value is SECRET", timestamp=0)])
    await pi_ai.complete_simple(_model(), ctx)
    assert provider.seen_context.messages[0].content == "my value is SECRET"


def _tool_result(tool_name: str, text: str, *, details=None) -> Context:
    return Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name=tool_name,
                content=[TextContent(type="text", text=text)],
                details=details,
                timestamp=0,
            )
        ]
    )


def test_active_compression_compresses_bash_tool_results():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _tool_result("bash", _large_token_text("xxxxxxxx"))
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "COMPRESSED:xxxxxxxx"


def test_active_compression_reuses_identical_tool_output_within_request():
    original = "identical tool output payload line\n" * 120
    calls = {"count": 0}

    def compressor(text: str) -> str:
        calls["count"] += 1
        return f"COMPRESSED:{text[:12]}"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text=original)],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="bash",
                    content=[TextContent(type="text", text=original)],
                    timestamp=1,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert calls["count"] == 1
    assert out.messages[0].content[0].text == "COMPRESSED:identical to"
    assert out.messages[1].content[0].text == "COMPRESSED:identical to"
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 2}


def test_active_compression_reuses_cached_tool_output_across_requests():
    original = "cross request repeated tool output payload line\n" * 120
    calls = {"count": 0}

    def compressor(text: str) -> str:
        calls["count"] += 1
        return f"COMPRESSED:{text[:14]}"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        first = pi_ai.compress_context(_tool_result("bash", original))
        second = pi_ai.compress_context(_tool_result("bash", original))
    finally:
        pi_ai.unregister_compressor()

    assert first.messages[0].content[0].text == "COMPRESSED:cross request "
    assert second.messages[0].content[0].text == first.messages[0].content[0].text
    assert calls["count"] == 1
    cache_stats = pi_ai.get_compression_cache_stats()
    assert cache_stats.entries == 1
    assert cache_stats.hits == 1
    assert cache_stats.misses == 1


def test_active_compression_cache_does_not_rewrite_frozen_prefix():
    original = "frozen prefix repeated tool output payload line\n" * 120
    calls = {"count": 0}

    def compressor(text: str) -> str:
        calls["count"] += 1
        return f"COMPRESSED:{text[:13]}"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        pi_ai.compress_context(_tool_result("bash", original))
        frozen_ctx = Context(
            compression_frozen_message_count=1,
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text=original)],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="bash",
                    content=[TextContent(type="text", text="live suffix payload line\n" * 120)],
                    timestamp=1,
                ),
            ],
        )
        out = pi_ai.compress_context(frozen_ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert out.messages[1].content[0].text == "COMPRESSED:live suffix p"
    assert calls["count"] == 2
    cache_stats = pi_ai.get_compression_cache_stats()
    assert cache_stats.hits == 0
    assert cache_stats.misses == 2


def test_active_compression_cache_evicts_least_recently_used_tool_output(monkeypatch):
    monkeypatch.setenv("TAU_COMPRESSION_CACHE_MAX_ENTRIES", "2")
    payload_a = "cache payload A line\n" * 120
    payload_b = "cache payload B line\n" * 120
    payload_c = "cache payload C line\n" * 120
    calls: list[str] = []

    def compressor(text: str) -> str:
        calls.append(text)
        return f"COMPRESSED:{text[:15]}"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        first_a = pi_ai.compress_context(_tool_result("bash", payload_a))
        pi_ai.compress_context(_tool_result("bash", payload_b))
        second_a = pi_ai.compress_context(_tool_result("bash", payload_a))
        pi_ai.compress_context(_tool_result("bash", payload_c))
        second_b = pi_ai.compress_context(_tool_result("bash", payload_b))
    finally:
        pi_ai.unregister_compressor()

    assert first_a.messages[0].content[0].text == "COMPRESSED:cache payload A"
    assert second_a.messages[0].content[0].text == first_a.messages[0].content[0].text
    assert second_b.messages[0].content[0].text == "COMPRESSED:cache payload B"
    assert calls == [payload_a, payload_b, payload_c, payload_b]
    cache_stats = pi_ai.get_compression_cache_stats()
    assert cache_stats.entries == 2
    assert cache_stats.hits == 1
    assert cache_stats.misses == 4


def test_active_compression_rejects_non_shrinking_text_replacement():
    original = "short words " * 80
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: text + (" expanded" * 80))
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_fails_open_when_text_compressor_raises():
    original = "short words " * 120
    pi_ai.reset_compression_stats()

    def broken_compressor(text: str) -> str:
        raise RuntimeError(f"cannot compress {len(text)} bytes")

    pi_ai.register_compressor(broken_compressor)
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_rejects_non_string_compressor_output():
    original = "short words " * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: {"compressed": text[:8]})  # type: ignore[return-value]
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_skips_text_below_default_token_floor():
    calls: list[str] = []

    def eager_compressor(text: str) -> str:
        calls.append(text)
        return "COMPRESSED"

    original = "small output " * 40
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(eager_compressor)
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert calls == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_uses_headroom_token_floor_boundary():
    calls: list[str] = []

    def eager_compressor(text: str) -> str:
        calls.append(text)
        return "COMPRESSED"

    original = "hello world " * 100
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(eager_compressor)
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "COMPRESSED"
    assert calls == [original]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_active_compression_protects_fresh_read_tool_results():
    seen: list[str] = []
    original = "fresh exact file line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("read", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_protects_fresh_grep_tool_results():
    seen: list[str] = []
    original = "\n".join(f"src/app.py:{i}:target match {i}" for i in range(1, 140))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("grep", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_observer_records_text_strategy_event():
    events = []
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    pi_ai.register_compression_observer(events.append)
    try:
        ctx = _tool_result("bash", _large_token_text("xxxxxxxx"))
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compression_observer()
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "COMPRESSED:xxxxxxxx"
    assert len(events) == 1
    event = events[0]
    assert event.strategy == "text"
    assert event.role == "toolResult"
    assert event.tool_name == "bash"
    assert event.original_tokens > event.compressed_tokens
    assert event.original_bytes > event.compressed_bytes


def test_active_compression_observer_failure_is_non_fatal():
    def failing_observer(_event):
        raise RuntimeError("metrics sink down")

    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    pi_ai.register_compression_observer(failing_observer)
    try:
        ctx = _tool_result("bash", _large_token_text("xxxxxxxx"))
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compression_observer()
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "COMPRESSED:xxxxxxxx"


def test_active_compression_circuit_breaker_opens_after_consecutive_failures(monkeypatch):
    monkeypatch.setenv("TAU_COMPRESSION_CACHE_MAX_ENTRIES", "0")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "3")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", "60")
    pi_ai.reset_compression_circuit_breaker()
    calls = 0

    def failing_compressor(_text: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("compressor down")

    original = _large_token_text("breaker")
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(failing_compressor)
    try:
        for _ in range(3):
            out = pi_ai.compress_context(_tool_result("bash", original))
            assert out.messages[0].content[0].text == original

        state = pi_ai.get_compression_circuit_breaker_state()
        assert state["open"] is True
        assert state["consecutive_failures"] == 0

        out = pi_ai.compress_context(_tool_result("bash", original))
    finally:
        pi_ai.unregister_compressor()
        pi_ai.reset_compression_circuit_breaker()

    assert out.messages[0].content[0].text == original
    assert calls == 3
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_circuit_breaker_success_resets_consecutive_failures(monkeypatch):
    monkeypatch.setenv("TAU_COMPRESSION_CACHE_MAX_ENTRIES", "0")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "3")
    pi_ai.reset_compression_circuit_breaker()
    calls = 0

    def flaky_compressor(text: str) -> str:
        nonlocal calls
        calls += 1
        if calls in {1, 2, 4, 5}:
            raise RuntimeError("temporary compressor failure")
        return f"COMPRESSED:{text[:8]}"

    original = _large_token_text("flaky")
    pi_ai.register_compressor(flaky_compressor)
    try:
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == "COMPRESSED:flaky al"
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert pi_ai.get_compression_circuit_breaker_state()["open"] is False
    finally:
        pi_ai.unregister_compressor()
        pi_ai.reset_compression_circuit_breaker()

    assert calls == 5


def test_active_compression_circuit_breaker_cooldown_expires(monkeypatch):
    monkeypatch.setenv("TAU_COMPRESSION_CACHE_MAX_ENTRIES", "0")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "1")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", "0.05")
    pi_ai.reset_compression_circuit_breaker()
    calls = 0

    def compressor(text: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("initial failure")
        return f"COMPRESSED:{text[:8]}"

    original = _large_token_text("cooldown")
    pi_ai.register_compressor(compressor)
    try:
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert pi_ai.get_compression_circuit_breaker_state()["open"] is True
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == original
        assert calls == 1
        time.sleep(0.1)
        assert pi_ai.compress_context(_tool_result("bash", original)).messages[0].content[0].text == "COMPRESSED:cooldown"
    finally:
        pi_ai.unregister_compressor()
        pi_ai.reset_compression_circuit_breaker()


def test_active_compression_circuit_breaker_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_THRESHOLD", "three")
    monkeypatch.setenv("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", "1m")
    pi_ai.reset_compression_circuit_breaker()

    state = pi_ai.get_compression_circuit_breaker_state()

    assert state["threshold"] == 3
    assert state["cooldown_s"] == 60.0


def test_active_compression_inflation_guard_reverts_bloated_context(monkeypatch):
    compression_module = importlib.import_module("pi_ai.compression")

    def bloating_inner(context, _fn, _policy):
        bloated = _tool_result("bash", "PADDING " * 600)
        return context.model_copy(update={"messages": bloated.messages})

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    monkeypatch.setattr(compression_module, "_compress_context_inner", bloating_inner)
    try:
        ctx = _tool_result("bash", _large_token_text("smallish"))
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out is ctx
    assert out.messages[0].content[0].text == ctx.messages[0].content[0].text
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_stats_aggregate_real_events_and_reset():
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            messages=[
                    ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[
                        TextContent(type="text", text=_large_token_text("xxxxxxxx")),
                        TextContent(type="text", text="cached-" + "y" * 1000, cache_zone="prefix"),
                    ],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="ccr_retrieve",
                    content=[TextContent(type="text", text="z" * 1000)],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "COMPRESSED:xxxxxxxx"
    assert out.messages[0].content[1].text == "cached-" + "y" * 1000
    assert out.messages[1].content[0].text == "z" * 1000

    stats = pi_ai.get_compression_stats()
    assert stats.total_compressions == 1
    assert stats.compressions_by_strategy == {"text": 1}
    assert stats.tokens_saved_by_strategy["text"] == stats.total_tokens_saved
    assert stats.bytes_saved_by_strategy["text"] == stats.total_bytes_saved
    assert stats.total_original_tokens > stats.total_compressed_tokens
    assert stats.total_original_bytes > stats.total_compressed_bytes

    pi_ai.reset_compression_stats()
    reset = pi_ai.get_compression_stats()
    assert reset.total_compressions == 0
    assert reset.compressions_by_strategy == {}
    assert reset.total_tokens_saved == 0


def test_active_compression_protects_user_text_by_default():
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    old_text = "old pasted log line\n" * 100
    latest_text = "new pasted log line\n" * 100
    try:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[TextContent(type="text", text=old_text)],
                    timestamp=0,
                ),
                UserMessage(
                    content=[TextContent(type="text", text=latest_text)],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == old_text
    assert out.messages[1].content[0].text == latest_text
    assert ctx.messages[1].content[0].text == latest_text
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_can_opt_in_to_user_text_compression():
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    old_text = "old pasted log line\n" * 100
    latest_text = "new pasted log line\n" * 100
    try:
        ctx = Context(
            compression_compress_user_messages=True,
            messages=[
                UserMessage(
                    content=[TextContent(type="text", text=old_text)],
                    timestamp=0,
                ),
                UserMessage(
                    content=[TextContent(type="text", text=latest_text)],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == old_text
    assert out.messages[1].content[0].text == "COMPRESSED:new past"
    assert ctx.messages[1].content[0].text == latest_text

    stats = pi_ai.get_compression_stats()
    assert stats.total_compressions == 1
    assert stats.compressions_by_strategy == {"text": 1}
    assert stats.total_tokens_saved > 0


def test_active_compression_protects_system_prompt_by_default():
    pi_ai.reset_compression_stats()
    original = "system instructions " * 120
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(system_prompt=original, messages=[])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.system_prompt == original
    assert ctx.system_prompt == original
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_can_opt_into_system_prompt_compression():
    pi_ai.reset_compression_stats()
    original = "system instructions " * 120
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            system_prompt=original,
            compression_compress_system_messages=True,
            messages=[],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.system_prompt == "COMPRESSED:system i"
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_active_compression_compacts_tool_schemas_without_stripping_property_names():
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: text)
    tool = Tool(
        name="create_field",
        description="Create   a schema\nfield descriptor.",
        parameters={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "CreateFieldParameters",
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Human title."},
                "deprecated": {"type": "boolean", "description": "Mark as deprecated."},
                "examples": {
                    "type": "array",
                    "items": {"type": "string", "title": "ExampleItem"},
                },
                "readOnly": {"type": "boolean", "description": "Whether the field is read-only."},
            },
            "required": ["title", "deprecated", "examples", "readOnly"],
        },
    )
    ctx = Context(messages=[], tools=[tool])

    try:
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert ctx.tools[0].parameters["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert ctx.tools[0].parameters["title"] == "CreateFieldParameters"

    compacted = out.tools[0]
    params = compacted.parameters
    assert compacted.description == "Create a schema field descriptor."
    assert "$schema" not in params
    assert "title" not in params
    assert set(params["required"]) == {"title", "deprecated", "examples", "readOnly"}
    assert set(params["properties"]) == {"title", "deprecated", "examples", "readOnly"}
    assert "title" not in params["properties"]["examples"]["items"]

    stats = pi_ai.get_compression_stats()
    assert stats.total_compressions == 1
    assert stats.compressions_by_strategy == {"tool_schema": 1}
    assert stats.total_tokens_saved > 0


def _assistant_tool_call(call_id: str, name: str, arguments: dict) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id=call_id, name=name, arguments=arguments)],
        api="fake-pii-api",
        provider="fake",
        model="fake-1",
        timestamp=0,
    )


def _compressor_with_ccr_marker(seen: list[str]):
    def compress(text: str) -> str:
        if len(text.encode("utf-8")) < pi_ai.compression.READ_LIFECYCLE_MIN_BYTES:
            return text
        seen.append(text)
        handle = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"[CCR:{handle}] compressed {len(text)} bytes"

    return compress


class _LooseContext:
    def __init__(
        self,
        messages,
        tools=None,
        compression_frozen_message_count=0,
        compression_compress_stale_reads=True,
        compression_compress_superseded_reads=False,
        compression_read_lifecycle_min_bytes=pi_ai.compression.READ_LIFECYCLE_MIN_BYTES,
        compression_image_optimize=True,
    ):
        self.messages = messages
        self.tools = tools
        self.compression_frozen_message_count = compression_frozen_message_count
        self.compression_compress_stale_reads = compression_compress_stale_reads
        self.compression_compress_superseded_reads = compression_compress_superseded_reads
        self.compression_read_lifecycle_min_bytes = compression_read_lifecycle_min_bytes
        self.compression_image_optimize = compression_image_optimize

    def model_copy(self, update=None):
        update = update or {}
        return _LooseContext(
            messages=update.get("messages", self.messages),
            tools=update.get("tools", self.tools),
            compression_frozen_message_count=update.get(
                "compression_frozen_message_count",
                self.compression_frozen_message_count,
            ),
            compression_compress_stale_reads=update.get(
                "compression_compress_stale_reads",
                self.compression_compress_stale_reads,
            ),
            compression_compress_superseded_reads=update.get(
                "compression_compress_superseded_reads",
                self.compression_compress_superseded_reads,
            ),
            compression_read_lifecycle_min_bytes=update.get(
                "compression_read_lifecycle_min_bytes",
                self.compression_read_lifecycle_min_bytes,
            ),
            compression_image_optimize=update.get(
                "compression_image_optimize",
                self.compression_image_optimize,
            ),
        )


def test_active_compression_marks_read_stale_after_later_edit():
    seen: list[str] = []
    original_read = "old source line\n" * 120
    handle = hashlib.sha1(original_read.encode("utf-8")).hexdigest()[:12]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    marker = out.messages[1].content[0].text
    assert marker == (
        f"[Read content stale: src/app.py was modified after this read. "
        f"Retrieve original: hash={handle}. Re-read the file for current content if needed.]"
    )
    assert original_read not in marker
    assert seen == [original_read]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"read_lifecycle:stale": 1}
    assert stats.total_original_bytes > stats.total_compressed_bytes


def test_active_compression_fails_open_when_read_lifecycle_compressor_raises():
    original_read = "old source line\n" * 120
    pi_ai.reset_compression_stats()

    def broken_compressor(text: str) -> str:
        raise RuntimeError(f"cannot store {len(text)} bytes")

    pi_ai.register_compressor(broken_compressor)
    try:
        ctx = Context(
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == original_read
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_does_not_mark_read_lifecycle_without_ccr_handle():
    original_read = "old source line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == original_read
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_can_disable_stale_read_compression():
    seen: list[str] = []
    original_read = "old source line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            compression_compress_stale_reads=False,
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == original_read
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_uses_headroom_read_lifecycle_size_floor():
    seen: list[str] = []
    original_read = "mid source line\n" * 40
    assert 512 <= len(original_read.encode("utf-8")) < 800
    handle = hashlib.sha1(original_read.encode("utf-8")).hexdigest()[:12]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    marker = out.messages[1].content[0].text
    assert marker == (
        f"[Read content stale: src/app.py was modified after this read. "
        f"Retrieve original: hash={handle}. Re-read the file for current content if needed.]"
    )
    assert seen == [original_read]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"read_lifecycle:stale": 1}


def test_active_compression_can_raise_read_lifecycle_size_floor():
    seen: list[str] = []
    original_read = "mid source line\n" * 40
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            compression_read_lifecycle_min_bytes=2000,
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
                ToolResultMessage(
                    tool_call_id="edit-1",
                    tool_name="edit",
                    content=[TextContent(text="edited src/app.py")],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == original_read
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_marks_anthropic_tool_result_read_stale():
    seen: list[str] = []
    original_read = "anthropic old source line\n" * 120
    handle = hashlib.sha1(original_read.encode("utf-8")).hexdigest()[:12]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "Read",
                            "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": original_read,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "edit-1",
                            "name": "Edit",
                            "input": {"file_path": "src/app.py", "old_string": "old", "new_string": "new"},
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    marker = out.messages[1]["content"][0]["content"]
    assert marker == (
        f"[Read content stale: src/app.py was modified after this read. "
        f"Retrieve original: hash={handle}. Re-read the file for current content if needed.]"
    )
    assert original_read not in marker
    assert seen == [original_read]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"read_lifecycle:stale": 1}


def test_active_compression_compresses_anthropic_tool_result_content():
    seen: list[str] = []
    original_read = "anthropic fresh source line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                            {
                                "type": "tool_use",
                                "id": "read-1",
                                "name": "Bash",
                                "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": original_read,
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[1]["content"][0]["content"]
    assert compressed.startswith("[CCR:")
    assert original_read not in compressed
    assert seen == [original_read]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"text": 1}


def test_active_compression_compresses_anthropic_tool_result_nested_text_list():
    seen: list[str] = []
    original_read = "anthropic nested source line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                            {
                                "type": "tool_use",
                                "id": "read-1",
                                "name": "Bash",
                                "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": [{"type": "text", "text": original_read}],
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[1]["content"][0]["content"][0]["text"]
    assert compressed.startswith("[CCR:")
    assert original_read not in compressed
    assert seen == [original_read]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"text": 1}


def test_active_compression_compresses_anthropic_tool_result_dict_content():
    seen: list[str] = []
    rows = {"rows": [{"id": i, "padding": "x" * 30} for i in range(120)]}
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                            {
                                "type": "tool_use",
                                "id": "read-1",
                                "name": "Bash",
                                "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": rows,
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[1]["content"][0]["content"]
    assert isinstance(compressed, str)
    assert compressed.startswith("[CCR:")
    assert rows["rows"][0]["padding"] not in compressed
    assert seen == [json.dumps(rows, ensure_ascii=False, separators=(",", ":"))]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"text": 1}


def test_active_compression_protects_cache_controlled_anthropic_tool_result_block():
    seen: list[str] = []
    original_read = "cache controlled anthropic payload line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "cache_control": {"type": "ephemeral"},
                            "content": original_read,
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    block = out.messages[0]["content"][0]
    assert block["content"] == original_read
    assert block["cache_control"] == {"type": "ephemeral"}
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_compresses_openai_tool_role_dict_content():
    seen: list[str] = []
    original = "openai tool role payload line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(messages=[{"role": "tool", "tool_call_id": "tool-1", "content": original}])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0]["content"]
    assert compressed.startswith("[CCR:")
    assert original not in compressed
    assert seen == [original]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"text": 1}


def test_active_compression_compresses_tool_role_dict_text_blocks():
    seen: list[str] = []
    original = "tool role text block payload line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": [{"type": "text", "text": original}],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0]["content"][0]["text"]
    assert compressed.startswith("[CCR:")
    assert original not in compressed
    assert seen == [original]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"text": 1}


def test_active_compression_protects_cache_controlled_tool_role_dict_text_block():
    seen: list[str] = []
    original = "cache controlled tool text block payload line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": [
                        {
                            "type": "text",
                            "text": original,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    block = out.messages[0]["content"][0]
    assert block["text"] == original
    assert block["cache_control"] == {"type": "ephemeral"}
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_protects_user_dict_text_blocks():
    seen: list[str] = []
    original = "user dict text block payload line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(messages=[{"role": "user", "content": [{"type": "text", "text": original}]}])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0]["content"][0]["text"] == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_records_unit_outcomes_for_applied_and_protected_units():
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            system_prompt="system prompt payload " * 120,
            messages=[
                UserMessage(
                    content=[TextContent(type="text", text="user prompt payload " * 120)],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text="tool output payload " * 120)],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="ccr_retrieve",
                    content=[TextContent(type="text", text="retrieved payload " * 120)],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.system_prompt == ctx.system_prompt
    assert out.messages[0].content[0].text == ctx.messages[0].content[0].text
    assert out.messages[1].content[0].text == "COMPRESSED:tool out"
    assert out.messages[2].content[0].text == ctx.messages[2].content[0].text
    unit_stats = pi_ai.get_unit_outcome_stats()
    assert unit_stats.outcomes_by_reason["protected_system_message"] == 1
    assert unit_stats.outcomes_by_reason["protected_user_message"] == 1
    assert unit_stats.outcomes_by_reason["excluded_tool_result"] == 1
    assert unit_stats.outcomes_by_reason["applied"] == 1
    assert unit_stats.outcomes_by_category["protected_role"] == 2
    assert unit_stats.outcomes_by_category["protected_content"] == 1
    assert unit_stats.outcomes_by_category["applied"] == 1


def test_active_compression_protects_small_typed_tool_error_output():
    seen: list[str] = []
    original = ("Traceback (most recent call last):\nValueError: bad input\n" + "exact failure context\n" * 80)
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_protects_small_anthropic_is_error_tool_result():
    seen: list[str] = []
    original = "tool failed unexpectedly\n" + "exact failure context\n" * 80
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "is_error": True,
                            "content": original,
                        }
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0]["content"][0]["content"] == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_protects_small_openai_tool_role_error_output():
    seen: list[str] = []
    original = ("Traceback (most recent call last):\nRuntimeError: crashed\n" + "exact failure context\n" * 80)
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(messages=[{"role": "tool", "tool_call_id": "tool-1", "content": original}])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0]["content"] == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_compresses_large_tool_error_log():
    seen: list[str] = []
    original = "\n".join(
        ["Traceback (most recent call last):", "RuntimeError: crashed"]
        + [f"debug context line {i}" for i in range(700)]
    )
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(messages=[{"role": "tool", "tool_call_id": "tool-1", "content": original}])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0]["content"]
    assert compressed.startswith("[CCR:")
    assert original not in compressed
    assert seen == [original]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def _large_python_code() -> str:
    lines = ["import os", "from typing import Any", ""]
    for i in range(80):
        lines.extend(
            [
                f"def process_{i}(value: Any) -> str:",
                f'    """Process value {i}."""',
                "    result = str(value)",
                "    for j in range(5):",
                "        result += str(j)",
                "    return result",
                "",
            ]
        )
    return "\n".join(lines)


def test_active_compression_protects_recent_code_tool_result():
    seen: list[str] = []
    code = _large_python_code()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("bash", code)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == code
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_can_disable_recent_code_protection():
    seen: list[str] = []
    code = _large_python_code()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            compression_protect_recent=0,
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text=code)],
                    timestamp=0,
                )
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0].content[0].text
    assert compressed.startswith("[CCR:")
    assert seen == [code]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_active_compression_protects_code_when_latest_user_requests_review():
    seen: list[str] = []
    code = _large_python_code()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            messages=[
                    ToolResultMessage(
                        tool_call_id="call-1",
                        tool_name="bash",
                    content=[TextContent(type="text", text=code)],
                    timestamp=0,
                ),
                UserMessage(content="ack", timestamp=0),
                UserMessage(content="ack", timestamp=0),
                UserMessage(content="ack", timestamp=0),
                UserMessage(content="ack", timestamp=0),
                UserMessage(content="please review this code for bugs", timestamp=0),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == code
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_compresses_recent_non_code_tool_result():
    seen: list[str] = []
    original = "plain data line\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0].content[0].text
    assert compressed.startswith("[CCR:")
    assert original not in compressed
    assert seen == [original]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_active_compression_pins_typed_tool_result_with_own_ccr_marker():
    seen: list[str] = []
    original = ("prefix\n" * 200) + "[CCR:abcdef123456] compressed previous payload\n" + ("noise\n" * 200)
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _tool_result("bash", original)
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_pins_anthropic_tool_result_with_own_ccr_marker():
    seen: list[str] = []
    original = ("prefix\n" * 200) + "[CCR:abcdef123456] compressed previous payload\n" + ("noise\n" * 200)
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": original,
                        }
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0]["content"][0]["content"] == original
    assert seen == []
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_compresses_live_tool_text_around_retrieval_marker():
    seen: list[str] = []
    marker = "Retrieve more: hash=abcdef123456"
    original = ("prefix\n" * 200) + marker + "\n" + ("noise\n" * 200)
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(messages=[{"role": "tool", "tool_call_id": "tool-1", "content": original}])
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    compressed = out.messages[0]["content"]
    assert marker in compressed
    assert compressed != original
    assert compressed.startswith("[CCR:")
    assert compressed.endswith("bytes")
    assert seen == ["prefix\n" * 200, "\n" + "noise\n" * 200]
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_active_compression_skips_superseded_reads_by_default():
    seen: list[str] = []
    first_read = "anthropic first full file snapshot\n" * 120
    second_read = "anthropic second full file snapshot\n" * 120
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = _LooseContext(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "Read",
                            "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-1",
                            "content": first_read,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read-2",
                            "name": "Read",
                            "input": {"file_path": "src/app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read-2",
                            "content": second_read,
                        }
                    ],
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    first_out = out.messages[1]["content"][0]["content"]
    second_out = out.messages[3]["content"][0]["content"]
    assert first_out == first_read
    assert second_out == second_read
    assert "Read content superseded" not in first_out
    assert seen == []
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {}


def test_active_compression_can_opt_in_to_superseded_read_compression():
    seen: list[str] = []
    first_read = "first full file snapshot\n" * 120
    second_read = "second full file snapshot\n" * 120
    first_handle = hashlib.sha1(first_read.encode("utf-8")).hexdigest()[:12]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            compression_compress_superseded_reads=True,
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=first_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("read-2", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-2",
                    tool_name="read",
                    content=[TextContent(text=second_read)],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    first_out = out.messages[1].content[0].text
    second_out = out.messages[3].content[0].text
    assert first_out == (
        f"[Read content superseded: src/app.py was re-read later. "
        f"Retrieve original: hash={first_handle}. Re-read the file for current content if needed.]"
    )
    assert second_out == second_read
    assert "Read content superseded" not in second_out
    assert seen == [first_read]
    stats = pi_ai.get_compression_stats()
    assert stats.compressions_by_strategy == {"read_lifecycle:superseded": 1}


def test_active_compression_does_not_supersede_disjoint_partial_reads():
    seen: list[str] = []
    first_read = "first page\n" * 120
    second_read = "second page\n" * 120
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py", "offset": 0, "limit": 50}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=first_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("read-2", "read", {"path": "src/app.py", "offset": 200, "limit": 50}),
                ToolResultMessage(
                    tool_call_id="read-2",
                    tool_name="read",
                    content=[TextContent(text=second_read)],
                    timestamp=0,
                ),
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == first_read
    assert out.messages[3].content[0].text == second_read
    assert "Read content superseded" not in out.messages[1].content[0].text
    assert seen == []


def test_active_compression_does_not_apply_read_lifecycle_inside_frozen_prefix():
    seen: list[str] = []
    original_read = "cached old source line\n" * 120
    pi_ai.register_compressor(_compressor_with_ccr_marker(seen))
    try:
        ctx = Context(
            compression_frozen_message_count=2,
            messages=[
                _assistant_tool_call("read-1", "read", {"path": "src/app.py"}),
                ToolResultMessage(
                    tool_call_id="read-1",
                    tool_name="read",
                    content=[TextContent(text=original_read)],
                    timestamp=0,
                ),
                _assistant_tool_call("edit-1", "edit", {"path": "src/app.py", "old": "old", "new": "new"}),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[1].content[0].text == original_read
    assert seen == []


def test_active_compression_does_not_recompress_ccr_retrieve_results():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _tool_result("ccr_retrieve", "retrieved evidence")
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "retrieved evidence"


def test_active_compression_skips_non_live_cache_zone_tool_results():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _tool_result("read", "x" * 1000, details={"cache_zone": "prefix"})
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "x" * 1000


def test_active_compression_respects_block_cache_metadata():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[
                        TextContent(type="text", text="cached-" + "a" * 1000, cache_control={"type": "ephemeral"}),
                        TextContent(type="text", text=_large_token_text("live-bbb")),
                        TextContent(type="text", text="prefix-" + "c" * 1000, cache_zone="prefix"),
                        TextContent(type="text", text="fixed-" + "d" * 1000, mutable=False),
                    ],
                    timestamp=0,
                )
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    blocks = out.messages[0].content
    assert blocks[0].text == "cached-" + "a" * 1000
    assert blocks[1].text == "COMPRESSED:live-bbb"
    assert blocks[2].text == "prefix-" + "c" * 1000
    assert blocks[3].text == "fixed-" + "d" * 1000


def test_active_compression_respects_dict_block_cache_metadata():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "bash",
                    "content": [
                        {"type": "text", "text": _large_token_text("live-aaa")},
                        {"type": "text", "text": "prefix-" + "b" * 1000, "cache_zone": "prefix"},
                        {"type": "text", "text": "cachezone-" + "c" * 1000, "cacheZone": "prefix"},
                        {"type": "text", "text": "fixed-" + "d" * 1000, "mutable": False},
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    blocks = out.messages[0]["content"]
    assert blocks[0]["text"] == "COMPRESSED:live-aaa"
    assert blocks[1]["text"] == "prefix-" + "b" * 1000
    assert blocks[2]["text"] == "cachezone-" + "c" * 1000
    assert blocks[3]["text"] == "fixed-" + "d" * 1000


def test_active_compression_respects_dict_message_cache_metadata():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "bash",
                    "cache_zone": "prefix",
                    "content": "cached-" + "a" * 1000,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-2",
                    "name": "bash",
                    "content": _large_token_text("live-bbb"),
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0]["content"] == "cached-" + "a" * 1000
    assert out.messages[1]["content"] == "COMPRESSED:live-bbb"


def _large_png_base64() -> str:
    img = Image.new("RGB", (900, 700))
    rng = random.Random(1234)
    pixels = []
    for _ in range(900 * 700):
        pixels.append((rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    assert len(data) > pi_ai.compression.IMAGE_MIN_BYTES
    return base64.b64encode(data).decode("ascii")


def _image_size(data: str) -> tuple[int, int]:
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    return img.size


def _data_url_bytes(url: str) -> bytes:
    assert url.startswith("data:image/jpeg;base64,")
    return base64.b64decode(url.split(",", 1)[1])


def test_active_compression_image_token_estimators_match_provider_tile_rules():
    assert pi_ai.compression.estimate_openai_image_tokens(1920, 1080, "low") == 85
    assert pi_ai.compression.estimate_openai_image_tokens(512, 512) == 255
    assert pi_ai.compression.estimate_openai_image_tokens(768, 768) == 765
    assert pi_ai.compression.estimate_anthropic_image_tokens(1024, 768) == (1024 * 768) // 750
    assert pi_ai.compression.estimate_anthropic_image_tokens(1568, 1568) <= 1534


def test_active_compression_downscales_live_user_images_without_rewriting_user_text():
    original_data = _large_png_base64()
    events = []
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    pi_ai.register_compression_observer(events.append)
    try:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(type="text", text="describe this screenshot exactly"),
                        ImageContent(type="image", data=original_data, mime_type="image/png"),
                    ],
                    timestamp=0,
                )
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compression_observer()
        pi_ai.unregister_compressor()

    text_block, image_block = out.messages[0].content
    assert text_block.text == "describe this screenshot exactly"
    assert image_block.mime_type == "image/jpeg"
    assert len(base64.b64decode(image_block.data)) < len(base64.b64decode(original_data))
    assert max(_image_size(image_block.data)) <= pi_ai.compression.IMAGE_MAX_DIMENSION
    assert ctx.messages[0].content[1].data == original_data
    assert len(events) == 1
    event = events[0]
    assert event.strategy == "image_resize"
    assert event.role == "user"
    assert event.tool_name is None
    assert event.original_tokens == pi_ai.compression.estimate_openai_image_tokens(900, 700)
    assert event.compressed_tokens == pi_ai.compression.estimate_openai_image_tokens(*_image_size(image_block.data))
    assert event.original_tokens > event.compressed_tokens
    assert event.original_bytes > event.compressed_bytes
    stats = pi_ai.get_compression_stats()
    assert stats.total_compressions == 1
    assert stats.compressions_by_strategy == {"image_resize": 1}
    assert stats.tokens_saved_by_strategy["image_resize"] == stats.total_tokens_saved


def test_active_compression_can_disable_typed_image_optimization_without_disabling_text_compression():
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(type="text", text="describe this screenshot exactly"),
                        ImageContent(type="image", data=original_data, mime_type="image/png"),
                    ],
                    timestamp=0,
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text=_large_token_text("live-text"))],
                    timestamp=1,
                ),
            ],
            compression_image_optimize=False,
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text_block, image_block = out.messages[0].content
    assert text_block.text == "describe this screenshot exactly"
    assert image_block.mime_type == "image/png"
    assert image_block.data == original_data
    assert out.messages[1].content[0].text == "COMPRESSED:live-tex"
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}
    unit_stats = pi_ai.get_unit_outcome_stats()
    assert unit_stats.outcomes_by_reason["image_optimize_disabled"] == 1


def test_active_compression_downscales_openai_dict_image_blocks():
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{original_data}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text_block, image_block = out.messages[0]["content"]
    assert text_block["text"] == "What is this image?"
    assert image_block["image_url"]["detail"] == "auto"
    compressed = _data_url_bytes(image_block["image_url"]["url"])
    assert len(compressed) < len(base64.b64decode(original_data))
    assert max(_image_size(image_block["image_url"]["url"].split(",", 1)[1])) <= pi_ai.compression.IMAGE_MAX_DIMENSION
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_resize": 1}


def test_active_compression_can_disable_dict_image_optimization_and_ocr(monkeypatch):
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    monkeypatch.setattr(
        pi_ai.compression,
        "_ocr_extract_image_text",
        lambda image_bytes: "Traceback line 42\nOperationalError",
    )
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the text in this screenshot"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{original_data}"},
                        },
                    ],
                }
            ],
            compression_image_optimize=False,
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text_block, image_block = out.messages[0]["content"]
    assert text_block["text"] == "Read the text in this screenshot"
    assert image_block["image_url"]["url"] == f"data:image/png;base64,{original_data}"
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}
    unit_stats = pi_ai.get_unit_outcome_stats()
    assert unit_stats.outcomes_by_reason["image_optimize_disabled"] == 1


def test_active_compression_transcodes_openai_image_to_ocr_text_for_text_intent(monkeypatch):
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    monkeypatch.setattr(
        pi_ai.compression,
        "_ocr_extract_image_text",
        lambda image_bytes: "Traceback line 42\nOperationalError",
    )
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the text in this screenshot"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{original_data}"},
                        },
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text_block, ocr_block = out.messages[0]["content"]
    assert text_block["text"] == "Read the text in this screenshot"
    assert ocr_block == {
        "type": "text",
        "text": "[OCR from image]\nTraceback line 42\nOperationalError",
    }
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_ocr": 1}


def test_active_compression_does_not_ocr_openai_image_without_text_intent(monkeypatch):
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    monkeypatch.setattr(
        pi_ai.compression,
        "_ocr_extract_image_text",
        lambda image_bytes: "Traceback line 42\nOperationalError",
    )
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{original_data}"},
                        },
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    image_block = out.messages[0]["content"][1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_resize": 1}


def test_active_compression_transcodes_typed_user_image_to_ocr_text(monkeypatch):
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    monkeypatch.setattr(
        pi_ai.compression,
        "_ocr_extract_image_text",
        lambda image_bytes: "Invoice total: $42.00",
    )
    try:
        ctx = Context(
            messages=[
                UserMessage(
                    content=[
                        TextContent(type="text", text="Read the text in this invoice"),
                        ImageContent(type="image", data=original_data, mime_type="image/png"),
                    ],
                    timestamp=0,
                )
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text_block, ocr_block = out.messages[0].content
    assert text_block.text == "Read the text in this invoice"
    assert isinstance(ocr_block, TextContent)
    assert ocr_block.text == "[OCR from image]\nInvoice total: $42.00"
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_ocr": 1}


def test_active_compression_downscales_anthropic_dict_image_blocks():
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": original_data,
                            },
                        },
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    image_block = out.messages[0]["content"][1]
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert len(base64.b64decode(image_block["source"]["data"])) < len(base64.b64decode(original_data))
    assert max(_image_size(image_block["source"]["data"])) <= pi_ai.compression.IMAGE_MAX_DIMENSION
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_resize": 1}


def test_active_compression_downscales_google_dict_image_blocks():
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {"text": "What do you see?"},
                        {"inlineData": {"mimeType": "image/png", "data": original_data}},
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    image_block = out.messages[0]["content"][1]
    assert image_block["inlineData"]["mimeType"] == "image/jpeg"
    assert len(base64.b64decode(image_block["inlineData"]["data"])) < len(base64.b64decode(original_data))
    assert max(_image_size(image_block["inlineData"]["data"])) <= pi_ai.compression.IMAGE_MAX_DIMENSION
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"image_resize": 1}


def test_active_compression_skips_cached_dict_image_blocks():
    original_data = _large_png_base64()
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "cache_zone": "prefix",
                            "image_url": {"url": f"data:image/png;base64,{original_data}"},
                        },
                        {
                            "type": "image",
                            "mutable": False,
                            "source": {"type": "base64", "media_type": "image/png", "data": original_data},
                        },
                    ],
                }
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    first, second = out.messages[0]["content"]
    assert first["image_url"]["url"] == f"data:image/png;base64,{original_data}"
    assert second["source"]["data"] == original_data
    assert pi_ai.get_compression_stats().compressions_by_strategy == {}


def test_active_compression_skips_cached_and_immutable_images():
    original_data = _large_png_base64()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read",
                    content=[
                        ImageContent(
                            type="image",
                            data=original_data,
                            mime_type="image/png",
                            cache_control={"type": "ephemeral"},
                        ),
                        ImageContent(type="image", data=original_data, mime_type="image/png", mutable=False),
                    ],
                    timestamp=0,
                )
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].data == original_data
    assert out.messages[0].content[1].data == original_data


def test_active_compression_skips_immutable_tool_results():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = _tool_result("read", "x" * 1000, details={"mutable": False})
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "x" * 1000


def test_active_compression_skips_frozen_prefix_messages():
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        ctx = Context(
            compression_frozen_message_count=1,
            messages=[
                    ToolResultMessage(
                        tool_call_id="call-1",
                        tool_name="bash",
                    content=[TextContent(type="text", text="a" * 1000)],
                    timestamp=0,
                ),
                    ToolResultMessage(
                        tool_call_id="call-2",
                        tool_name="bash",
                    content=[TextContent(type="text", text="b" * 1000)],
                    timestamp=0,
                ),
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.messages[0].content[0].text == "a" * 1000
    assert out.messages[1].content[0].text == "COMPRESSED:bbbbbbbb"


def test_active_compression_preserves_existing_ccr_markers_inside_live_user_text():
    calls = []

    def compressor(text: str) -> str:
        calls.append(text)
        return f"<{len(text)}>"

    marker = "[CCR:abcdef123456]"
    pi_ai.register_compressor(compressor)
    try:
        ctx = Context(
            compression_compress_user_messages=True,
            messages=[UserMessage(content=f"{'a' * 1000}\n{marker}\n{'b' * 800}", timestamp=0)],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    text = out.messages[0].content
    assert marker in text
    assert text.startswith("<")
    assert text.endswith(">")
    assert calls == ["a" * 1000 + "\n", "\n" + "b" * 800]
