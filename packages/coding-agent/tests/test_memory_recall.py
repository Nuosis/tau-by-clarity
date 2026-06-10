"""P1 gate — recall read path: live store surfaces the needle (literal + paraphrase)."""
from __future__ import annotations

import pytest

from pi_coding_agent.core.memory import (
    DeterministicEmbeddingProvider,
    MemoryStore,
    MemoryType,
    OllamaEmbeddingProvider,
    SemanticMemory,
)
from pi_coding_agent.core.memory.recall import build_recall_block, latest_user_query


def _mem(title, content, key, mtype=MemoryType.FILE_API.value) -> SemanticMemory:
    return SemanticMemory(id="", project="", memory_type=mtype, title=title,
                          content=content, key=key, provenance="sess-1")


def _seed(store: MemoryStore) -> None:
    store.write_semantic(_mem("auth", "never log raw passwords",
                              "constraint:no_pw_log", MemoryType.CONSTRAINT.value))
    store.write_semantic(_mem("db choice", "use SQLite not Postgres",
                              "decision:db", MemoryType.DECISION.value))
    store.write_semantic(_mem("reconnect limit",
                              "in config/net.py MAX_RECONNECT_ATTEMPTS is set to 7741",
                              "fileapi:max_reconnect"))


def test_recall_block_surfaces_needle_literal(tmp_path):
    s = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    _seed(s)
    block = build_recall_block(s, "what is MAX_RECONNECT_ATTEMPTS set to?")
    assert block is not None and "7741" in block
    s.close()


def test_recall_block_none_on_empty(tmp_path):
    s = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    assert build_recall_block(s, "anything") is None      # empty store
    _seed(s)
    assert build_recall_block(s, "   ") is None            # empty query
    s.close()


def test_latest_user_query():
    class M:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    msgs = [M("user", "first"), M("assistant", [{"type": "text", "text": "ans"}]),
            M("user", [{"type": "text", "text": "the real question"}])]
    assert latest_user_query(msgs) == "the real question"


def _ollama_up() -> bool:
    try:
        OllamaEmbeddingProvider().embed(["ping"])
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
def test_recall_paraphrase_semantic_live(tmp_path):
    # reproduces §9 semantic recall on the LIVE store: a paraphrase with no shared
    # terms must still surface the needle via embeddings.
    s = MemoryStore(str(tmp_path), embedder=OllamaEmbeddingProvider())
    _seed(s)
    block = build_recall_block(
        s, "how many times will the client retry a dropped connection before giving up?")
    assert block is not None and "7741" in block
    s.close()
