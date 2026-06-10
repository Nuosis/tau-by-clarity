"""P3 gate — file-fact staleness: code facts invalidate when their file changes/vanishes."""
from __future__ import annotations

import os

from pi_coding_agent.core.memory import (
    DeterministicEmbeddingProvider,
    MemoryStore,
    MemoryType,
    SemanticMemory,
)


def _store(tmp_path):
    return MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())


def _file_mem(file_path) -> SemanticMemory:
    return SemanticMemory(id="", project="", memory_type=MemoryType.FILE_API.value,
                          title="reconnect", content="MAX_RECONNECT_ATTEMPTS is 7741",
                          key="fileapi:max_reconnect", provenance="curator",
                          file_path=file_path)


def test_hash_stamped_on_write(tmp_path):
    (tmp_path / "net.py").write_text("MAX_RECONNECT_ATTEMPTS = 7741\n")
    s = _store(tmp_path)
    mid = s.write_semantic(_file_mem("net.py"))
    assert s.get(mid).content_hash is not None     # auto-stamped from disk
    s.close()


def test_changed_file_supersedes(tmp_path):
    (tmp_path / "net.py").write_text("MAX_RECONNECT_ATTEMPTS = 7741\n")
    s = _store(tmp_path)
    s.write_semantic(_file_mem("net.py"))
    assert s.search("MAX_RECONNECT_ATTEMPTS")       # recallable while fresh
    (tmp_path / "net.py").write_text("MAX_RECONNECT_ATTEMPTS = 99\n")  # file changed
    inval = s.invalidate_stale()
    assert len(inval) == 1
    assert s.search("MAX_RECONNECT_ATTEMPTS") == []  # stale fact no longer recalled
    s.close()


def test_missing_file_archives(tmp_path):
    (tmp_path / "net.py").write_text("X = 1\n")
    s = _store(tmp_path)
    mid = s.write_semantic(_file_mem("net.py"))
    os.remove(tmp_path / "net.py")
    assert s.invalidate_stale() == [mid]
    assert s.get(mid).status == "archived"
    s.close()


def test_unchanged_file_kept(tmp_path):
    (tmp_path / "net.py").write_text("MAX_RECONNECT_ATTEMPTS = 7741\n")
    s = _store(tmp_path)
    s.write_semantic(_file_mem("net.py"))
    assert s.invalidate_stale() == []               # unchanged → untouched
    assert s.search("MAX_RECONNECT_ATTEMPTS")
    s.close()


def test_non_file_memory_untouched(tmp_path):
    s = _store(tmp_path)
    s.write_semantic(SemanticMemory(id="", project="", memory_type=MemoryType.DECISION.value,
                                    title="db", content="use SQLite", key="decision:db",
                                    provenance="curator"))
    assert s.invalidate_stale() == []               # no file_path → never invalidated
    s.close()
