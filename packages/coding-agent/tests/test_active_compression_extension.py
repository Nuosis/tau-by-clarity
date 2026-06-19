import json
import logging
import time

import pi_ai
import pytest
from pi_ai import Context, TextContent, ToolResultMessage
from pi_coding_agent import active_compression as active_compression_runtime
from pi_coding_agent.active_compression.ccr import CCRStore
from pi_coding_agent.active_compression.extension import extension_factory


class _FakePi:
    def __init__(self):
        self.tools = {}
        self.commands = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def register_command(self, name, command):
        self.commands[name] = command


def _tool_result(text: str) -> Context:
    return Context(
        messages=[
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="bash",
                content=[TextContent(type="text", text=text)],
                timestamp=0,
            )
        ]
    )


@pytest.mark.asyncio
async def test_compression_command_reports_real_runtime_stats():
    pi_ai.unregister_compressor()
    pi_ai.reset_compression_stats()
    pi_ai.reset_compression_learning_stats()
    pi_ai.reset_cache_alignment_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")

    try:
        out = pi_ai.compress_context(
            _tool_result("xxxxxxxx alpha beta gamma delta epsilon zeta eta theta iota kappa " * 30)
        )
        assert out.messages[0].content[0].text == "COMPRESSED:xxxxxxxx"
        subscription_ctx = _tool_result(
            "subscription alpha beta gamma delta epsilon zeta eta theta iota kappa " * 30
        ).model_copy(update={"compression_auth_mode": "subscription"})
        subscription_out = pi_ai.compress_context(subscription_ctx)
        assert subscription_out.messages[0].content[0].text == "COMPRESSED:subscrip"

        pi = _FakePi()
        extension_factory(pi)

        text = await pi.commands["compression"]["handler"]("stats")
        assert "Active compression stats:" in text
        assert "compressor registered: yes" in text
        assert "total compressions: 2" in text
        assert "by strategy: text: 2" in text
        assert "learning events: 1 (read-only skipped 1)" in text
        assert "learning by strategy: text: 1" in text
        assert "learning skipped by strategy: text: 1" in text
        assert "cache alignment scans: 0 (findings 0, policy skipped 0)" in text
        assert "volatile findings by label: none" in text
        assert "compression cache: hits 0, misses 2, entries 2" in text
        assert "unit outcomes: 2" in text
        assert "unit outcomes by category: applied: 2" in text
        assert "unit outcomes by reason: applied: 2" in text

        reset_text = await pi.commands["compression"]["handler"]("reset")
        assert reset_text == "Active compression stats reset."
        assert pi_ai.get_compression_stats().total_compressions == 0
        assert pi_ai.get_compression_learning_stats().total_events == 0
        assert pi_ai.get_compression_learning_stats().total_skipped_read_only == 0
        assert pi_ai.get_cache_alignment_stats().total_scans == 0
        assert pi_ai.get_cache_alignment_stats().skipped_by_policy == 0
        assert pi_ai.get_compression_cache_stats().entries == 0
        assert pi_ai.get_unit_outcome_stats().total_units == 0
    finally:
        pi_ai.unregister_compressor()
        pi_ai.reset_compression_stats()
        pi_ai.reset_compression_learning_stats()
        pi_ai.reset_cache_alignment_stats()


def test_extension_registers_only_canonical_ccr_retrieve_tool():
    pi = _FakePi()
    extension_factory(pi)

    assert "ccr_retrieve" in pi.tools
    assert "headroom_retrieve" not in pi.tools
    schema = pi.tools["ccr_retrieve"]["parameters"]
    assert schema["required"] == ["handle", "query"]
    assert "handle" in schema["properties"]
    assert "query" in schema["properties"]


@pytest.mark.asyncio
async def test_ccr_retrieve_requires_query_scoped_retrieval(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    handle = "abc123abc123"
    original = json.dumps([
        {"id": "a", "message": "ordinary startup"},
        {"id": "b", "message": "needle payment failure"},
        {"id": "c", "message": "ordinary shutdown"},
    ])
    store.put_with_handle(handle, original)
    try:
        pi = _FakePi()
        extension_factory(pi)
        result = await pi.tools["ccr_retrieve"]["execute"](
            "tool-1",
            {"handle": handle, "query": "needle payment"},
            None,
            None,
            None,
        )
    finally:
        active_compression_runtime._store = previous_store

    text = result["content"][0]["text"]
    assert "[CCR query 'needle payment':" in text
    assert "needle payment failure" in text
    assert "ordinary startup" not in text
    assert result["details"]["handle"] == handle
    assert result["details"]["query"] == "needle payment"
    assert store.is_expanded(handle) is False


def test_ccr_store_expires_entries_and_reports_status(tmp_path, monkeypatch):
    store = CCRStore(str(tmp_path / "ccr.db"), default_ttl=10)
    monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1000.0)
    handle = store.put_with_handle("abcabcabcabc", "payload", ttl=5)

    monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1006.0)
    status = store.get_entry_status(handle, clean_expired=True)

    assert status["status"] == "expired"
    assert status["ttl_seconds"] == 5
    assert status["expires_at"] == 1005.0
    assert store.get(handle) is None


def test_ccr_store_eviction_removes_oldest_original(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"), max_entries=2)

    first = store.put_with_handle("111111111111", "first")
    time.sleep(0.01)
    second = store.put_with_handle("222222222222", "second")
    time.sleep(0.01)
    third = store.put_with_handle("333333333333", "third")

    assert store.exists(first) is False
    assert store.get(second) == "second"
    assert store.get(third) == "third"


def test_ccr_store_tracks_metadata_stats_and_query_access(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    original = json.dumps([
        {"id": "a", "message": "ordinary startup"},
        {"id": "b", "message": "needle payment failure"},
    ])
    handle = store.put_with_handle(
        "444444444444",
        original,
        compressed_content="[]",
        original_tokens=100,
        compressed_tokens=10,
        original_item_count=2,
        compressed_item_count=0,
        tool_name="search_tool",
        tool_call_id="call-1",
        query_context="payment issue",
        compression_strategy="json_rows",
    )

    results = store.search(handle, "needle payment")
    entry = store.retrieve(handle, query="needle payment")
    metadata = store.get_metadata(handle)
    stats = store.get_stats()
    events = store.get_retrieval_events(tool_name="search_tool")

    assert results == [{"id": "b", "message": "needle payment failure"}]
    assert entry is not None
    assert entry.retrieval_count >= 2
    assert entry.search_queries == ["needle payment"]
    assert metadata is not None
    assert metadata["compressed_content"] == "[]"
    assert metadata["query_context"] == "payment issue"
    assert stats["entry_count"] == 1
    assert stats["total_original_tokens"] == 100
    assert stats["total_compressed_tokens"] == 10
    assert {event.retrieval_type for event in events} == {"full", "search"}


def test_ccr_retrieval_log_redacts_secret_payload_values(tmp_path, caplog):
    store = CCRStore(str(tmp_path / "ccr.db"))
    handle = store.put_with_handle(
        "555555555555",
        "OPENAI_API_KEY=sk-proj-secret1234567890 Authorization: Bearer token123456789",
        tool_name="bash",
    )

    with caplog.at_level(logging.INFO, logger="pi_coding_agent.active_compression.ccr"):
        entry = store.retrieve(handle)

    assert entry is not None
    log_text = caplog.text
    assert "sk-proj-secret1234567890" not in log_text
    assert "Bearer token123456789" not in log_text
    assert "OPENAI_API_KEY=[REDACTED]" in log_text
    assert "Authorization: [REDACTED]" in log_text
