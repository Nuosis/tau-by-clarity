import re
from types import SimpleNamespace

import pytest

from pi_ai import TextContent, ToolResultMessage
from pi_coding_agent import active_compression as active_compression_runtime
from pi_coding_agent.active_compression.ccr import CCRStore
from pi_coding_agent.active_compression.extension import extension_factory
from pi_coding_agent.active_compression.persisted_context import (
    apply_persisted_compressed_messages,
    compress_message_for_persistence,
    extract_ccr_handles,
    refresh_or_rebuild_session_refs,
)
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session import _message_to_dict
from pi_coding_agent.core.session_manager import SessionManager


class _FakePi:
    def __init__(self):
        self.tools = {}
        self.commands = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def register_command(self, name, command):
        self.commands[name] = command


def _large_tool_message(text: str) -> dict:
    return _message_to_dict(
        ToolResultMessage(
            tool_call_id="call-persist-1",
            tool_name="bash",
            content=[TextContent(type="text", text=text)],
            timestamp=0,
        )
    )


def _large_log() -> str:
    return "\n".join(
        (
            f"2026-06-18T12:00:{i % 60:02d} INFO worker-{i} persisted context "
            f"{'auth_middleware_unique_token' if i == 42 else f'ordinary-{i}'} processing queue"
        )
        for i in range(260)
    )


def test_extract_ccr_handles_reads_top_level_and_inline_markers():
    text = "[CCR:ABCDEF123456] summary <<ccr:111111111111 60_rows_offloaded>> [CCR:abcdef123456]"

    assert extract_ccr_handles(text) == ["abcdef123456", "111111111111"]


def test_compress_message_for_persistence_keeps_model_facing_message_compressed(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    original = _large_log()
    try:
        compressed, metadata = compress_message_for_persistence(_large_tool_message(original))
    finally:
        active_compression_runtime._store = previous_store

    assert metadata is not None
    compressed_text = compressed["content"][0]["text"]
    assert compressed_text != original
    assert "ccr_retrieve" in compressed_text
    assert original not in compressed_text
    assert compressed["tool_call_id"] == "call-persist-1"
    assert metadata["refs"][0]["toolCallId"] == "call-persist-1"
    handle = metadata["refs"][0]["handle"]
    assert re.fullmatch(r"[0-9a-f]{12}", handle)
    assert store.get(handle) == original
    assert metadata["refs"][0]["original"] == original


def test_agent_session_message_end_persists_compressed_tool_result_with_metadata(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    sm = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"), session_id="persist-hook")
    session = AgentSession(cwd=str(tmp_path), session_manager=sm)
    original = _large_log()
    try:
        session._on_agent_event(SimpleNamespace(type="message_end", message=ToolResultMessage(
            tool_call_id="call-persist-hook",
            tool_name="bash",
            content=[TextContent(type="text", text=original)],
            timestamp=0,
        )))
    finally:
        active_compression_runtime._store = previous_store

    entries = sm.get_entries()
    assert len(entries) == 1
    raw = entries[0].data
    persisted_text = raw["message"]["content"][0]["text"]
    assert persisted_text != original
    assert "ccr_retrieve" in persisted_text
    assert raw["activeCompression"]["refs"][0]["original"] == original
    assert raw["activeCompression"]["refs"][0]["toolCallId"] == "call-persist-hook"


def test_refresh_or_rebuild_session_refs_restores_expired_ccr_from_session_metadata(tmp_path, monkeypatch):
    store = CCRStore(str(tmp_path / "ccr.db"), default_ttl=5)
    monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1000.0)
    handle = store.put_with_handle("aaaaaaaaaaaa", "durable original", ttl=5)
    entry = {
        "type": "message",
        "message": {"role": "toolResult", "content": [{"type": "text", "text": f"[CCR:{handle}] compressed"}]},
        "activeCompression": {
            "version": 1,
            "refs": [{
                "handle": handle,
                "original": "durable original",
                "compressed": f"[CCR:{handle}] compressed",
                "toolName": "bash",
                "toolCallId": "call-1",
            }],
        },
    }

    monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1006.0)
    assert store.exists(handle, clean_expired=True) is False

    result = refresh_or_rebuild_session_refs([entry], store=store)

    assert result == {"refreshed": 0, "rebuilt": 1, "missing": 0}
    assert store.get(handle) == "durable original"


def test_apply_persisted_compressed_messages_replaces_live_raw_tool_result(tmp_path):
    original = _large_log()
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    try:
        raw_message = _large_tool_message(original)
        compressed, metadata = compress_message_for_persistence(raw_message)
    finally:
        active_compression_runtime._store = previous_store
    assert metadata is not None
    entries = [{"message": compressed, "activeCompression": metadata}]

    transformed = apply_persisted_compressed_messages([raw_message], entries)

    assert transformed[0]["tool_call_id"] == raw_message["tool_call_id"]
    assert transformed[0]["content"][0]["text"] == compressed["content"][0]["text"]
    assert original not in transformed[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_session_transform_refreshes_persisted_refs_during_live_loop(tmp_path, monkeypatch):
    store = CCRStore(str(tmp_path / "ccr.db"), default_ttl=10)
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1000.0)
    handle = store.put_with_handle("bbbbbbbbbbbb", "durable original", ttl=5)
    sm = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"), session_id="live-refresh")
    sm.append_message(
        {"role": "toolResult", "content": [{"type": "text", "text": f"[CCR:{handle}] compressed"}]},
        active_compression={
            "version": 1,
            "refs": [{
                "handle": handle,
                "original": "durable original",
                "compressed": f"[CCR:{handle}] compressed",
                "toolName": "bash",
                "toolCallId": "call-live",
            }],
        },
    )
    session = AgentSession(cwd=str(tmp_path), session_manager=sm)
    try:
        monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1004.0)
        await session._transform_context([])
        monkeypatch.setattr("pi_coding_agent.active_compression.ccr.time.time", lambda: 1008.0)
        assert store.get(handle) == "durable original"
    finally:
        active_compression_runtime._store = previous_store


@pytest.mark.asyncio
async def test_agent_session_transform_uses_persisted_compressed_tool_result_in_live_loop(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    sm = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"), session_id="live-compressed")
    session = AgentSession(cwd=str(tmp_path), session_manager=sm)
    original = _large_log()
    raw = ToolResultMessage(
        tool_call_id="call-live-compressed",
        tool_name="bash",
        content=[TextContent(type="text", text=original)],
        timestamp=0,
    )
    try:
        session._on_agent_event(SimpleNamespace(type="message_end", message=raw))
        transformed = await session._transform_context([raw])
    finally:
        active_compression_runtime._store = previous_store

    assert isinstance(transformed[0], dict)
    transformed_text = transformed[0]["content"][0]["text"]
    assert transformed_text != original
    assert "ccr_retrieve" in transformed_text
    assert original not in transformed_text


@pytest.mark.asyncio
async def test_reopened_session_rebuilds_ccr_and_retrieves_compressed_context(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    original = _large_log()
    try:
        compressed, metadata = compress_message_for_persistence(_large_tool_message(original))
        assert metadata is not None
        sm = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"), session_id="resume-ccr")
        sm.append_message(compressed, active_compression=metadata)
        session_file = sm.get_session_file()
        assert session_file is not None
        handle = metadata["refs"][0]["handle"]
        store.clear()

        reopened = SessionManager.open(session_file)
        context = reopened.build_context()

        assert context.messages[0]["content"][0]["text"] == compressed["content"][0]["text"]
        assert store.get(handle) == original
        pi = _FakePi()
        extension_factory(pi)
        result = await pi.tools["ccr_retrieve"]["execute"](
            "tool-1",
            {"handle": handle, "query": "auth_middleware_unique_token"},
            None,
            None,
            None,
        )
    finally:
        active_compression_runtime._store = previous_store

    text = result["content"][0]["text"]
    assert result.get("isError") is not True
    assert "auth_middleware_unique_token" in text
