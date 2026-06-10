"""P5 live-loop gate — a real AgentSession with PI_MEMORY_ENABLED=1 curates a turn into
the store and the recall hook re-injects it. Exercises the session methods directly
(curator LLM stubbed; no real model/network needed)."""
from __future__ import annotations

import json

from pi_ai import get_model
from pi_ai.types import ToolResultMessage, UserMessage
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings


class AStub:
    async def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": True})
        return json.dumps({"decisions": [{
            "title": "reconnect", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
            "memory_type": "file_api", "key": "fileapi:max_reconnect",
            "source_ids": ["m0"], "verdict": "auto_commit", "confidence": 0.9}]})


def _session(tmp_path, monkeypatch) -> AgentSession:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("PI_MEMORY_ENABLED", "1")
    monkeypatch.setenv("PI_MEMORY_EMBED", "deterministic")
    monkeypatch.chdir(tmp_path)                       # store roots at cwd
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    sm = SessionManager.create(cwd=str(tmp_path), session_dir=str(tmp_path))
    return AgentSession(cwd=str(tmp_path), model=model,
                        settings=Settings(auto_compact=False), session_manager=sm)


async def test_memory_attached_when_flag_on(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch)
    assert s._memory is not None and s._memory_store is not None


async def test_live_curation_then_recall(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch)
    s._memory.curator.allm_fn = AStub()              # stub the curator's model call

    s._agent.append_message(UserMessage(content="please read config/net.py", timestamp=0))
    s._agent.append_message(ToolResultMessage(
        tool_call_id="t1", tool_name="read",
        content=[{"type": "text", "text": "config/net.py: MAX_RECONNECT_ATTEMPTS = 7741"}],
        timestamp=0))

    await s._curate_turn()                            # live write path
    hits = s._memory_store.search("MAX_RECONNECT_ATTEMPTS")
    assert hits and "7741" in hits[0].memory.content  # curated into the store

    # recall hook re-injects it into a later turn's context
    out = await s._transform_context(
        [UserMessage(content="what is MAX_RECONNECT_ATTEMPTS?", timestamp=0)])
    assert any("7741" in str(getattr(m, "content", "")) for m in out)
