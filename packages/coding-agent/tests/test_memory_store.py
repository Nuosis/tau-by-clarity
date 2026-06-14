"""P0 gate — memory store schema + write/read round-trip, supersession, scope, recall."""
from __future__ import annotations

import os

import pytest

from pi_coding_agent.core.memory import (
    ConversationTurn,
    DeterministicEmbeddingProvider,
    MemoryStore,
    MemoryType,
    Scope,
    SemanticMemory,
)


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())


def _mem(**kw) -> SemanticMemory:
    base = dict(id="", project="", memory_type=MemoryType.DECISION.value,
                title="t", content="c", key="k", provenance="sess-1")
    base.update(kw)
    return SemanticMemory(**base)


def test_db_created_in_project_dir(tmp_path):
    s = _store(tmp_path)
    assert os.path.exists(os.path.join(str(tmp_path), ".tau", "memory", "memory.db"))
    s.close()


def test_write_read_round_trip(tmp_path):
    s = _store(tmp_path)
    mid = s.write_semantic(_mem(title="DB choice",
                                content="Use SQLite not Postgres for the store",
                                key="decision:db_choice"))
    got = s.get(mid)
    assert got is not None
    assert got.title == "DB choice" and got.key == "decision:db_choice"
    assert got.status == "active" and got.embedding is not None
    s.close()


def test_canonical_key_supersession(tmp_path):
    s = _store(tmp_path)
    s.write_semantic(_mem(content="retry limit is 3", key="fileapi:net.retries"))
    s.write_semantic(_mem(content="retry limit is 7", key="fileapi:net.retries"))
    hits = s.search("retry limit", k=10)
    actives = [h for h in hits if h.memory.key == "fileapi:net.retries"]
    assert len(actives) == 1                      # only the latest active
    assert "7" in actives[0].memory.content       # superseded by the newer value
    s.close()


def test_scope_isolation(tmp_path):
    s = _store(tmp_path)
    s.write_semantic(_mem(title="global rule", content="run tests before commit",
                          key="pref:test_first", scope_type="project"))
    s.write_semantic(_mem(title="task note", content="run tests before commit",
                          key="task:abc:note", scope_type="task", scope_id="abc"))
    # querying task xyz must NOT see task abc's note, but DOES see project-global
    hits = s.search("run tests before commit",
                    scope=Scope(project=str(tmp_path), scope_type="task", scope_id="xyz"),
                    k=10)
    keys = {h.memory.key for h in hits}
    assert "pref:test_first" in keys
    assert "task:abc:note" not in keys
    s.close()


def test_hybrid_recall_ranks_match_first(tmp_path):
    s = _store(tmp_path)
    s.write_semantic(_mem(title="auth", content="never log raw passwords",
                          key="constraint:no_pw_log"))
    s.write_semantic(_mem(title="net", content="MAX_RECONNECT_ATTEMPTS is 7741",
                          key="fileapi:max_reconnect"))
    hits = s.search("what is MAX_RECONNECT_ATTEMPTS", k=2)
    assert hits and hits[0].memory.key == "fileapi:max_reconnect"
    assert hits[0].score > 0
    s.close()


def test_conversation_log_round_trip(tmp_path):
    s = _store(tmp_path)
    s.append_turn(ConversationTurn(id="", project="", role="user", content="hi"))
    s.append_turn(ConversationTurn(id="", project="", role="assistant", content="hello"))
    turns = s.recent_turns()
    assert [t.role for t in turns] == ["user", "assistant"]
    s.close()
