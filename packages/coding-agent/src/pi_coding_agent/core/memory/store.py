"""
Project-local memory store (SQLite) — P0.

Rooted at <project>/.tau/memory/memory.db, committed to git, the project root being
the poisoning boundary. Hybrid recall = max(lexical, local-semantic) per design doc §9
(settles lexical-vs-semantic: use both). Embeddings stored as JSON; cosine in Python
(fine at project scale; swap for sqlite-vec later if needed).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid

from pi_coding_agent.config import CONFIG_DIR_NAME

from .embeddings import EmbeddingProvider, cosine, embedding_provider_from_env
from .models import ConversationTurn, MemoryHit, Scope, SemanticMemory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_memory (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, memory_type TEXT NOT NULL,
    title TEXT NOT NULL, content TEXT NOT NULL, key TEXT NOT NULL,
    provenance TEXT NOT NULL, scope_type TEXT NOT NULL, scope_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', file_path TEXT, content_hash TEXT,
    embedding TEXT, metadata TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sem_scope ON semantic_memory(project, status);
CREATE INDEX IF NOT EXISTS idx_sem_key ON semantic_memory(project, key, status);
CREATE TABLE IF NOT EXISTS conversation_memory (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    scope_id TEXT NOT NULL DEFAULT '', summary_id TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS summary_memory (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, description TEXT NOT NULL,
    summary TEXT NOT NULL, full_content TEXT NOT NULL,
    source_ids TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_log_memory (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL, tool_args TEXT NOT NULL DEFAULT '{}',
    output TEXT NOT NULL, created_at REAL NOT NULL
);
"""


def _now() -> float:
    return time.time()


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) >= 3}


class MemoryStore:
    """SQLite-backed, project-local. One store per project root."""

    def __init__(self, project_root: str, embedder: EmbeddingProvider | None = None,
                 db_path: str | None = None) -> None:
        self.project_root = os.path.abspath(project_root)
        self.embedder = embedder or embedding_provider_from_env()
        if db_path is None:
            mem_dir = os.path.join(self.project_root, CONFIG_DIR_NAME, "memory")
            os.makedirs(mem_dir, exist_ok=True)
            db_path = os.path.join(mem_dir, "memory.db")
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── write path ──────────────────────────────────────────────────────────

    def write_semantic(self, mem: SemanticMemory) -> str:
        """Insert an atomic memory. Canonical-key dedup: an existing active row with the
        same (project, key) is superseded (status='superseded')."""
        if not mem.id:
            mem.id = uuid.uuid4().hex
        if not mem.created_at:
            mem.created_at = _now()
        if mem.embedding is None:
            mem.embedding = self.embedder.embed([mem.embedding_text()])[0]
        # stamp file-fact hash for staleness tracking (P3) if not provided
        if mem.file_path and mem.content_hash is None:
            mem.content_hash = self.file_content_hash(mem.file_path)
        self._conn.execute(
            "UPDATE semantic_memory SET status='superseded' "
            "WHERE project=? AND key=? AND status='active'",
            (self.project_root, mem.key))
        self._conn.execute(
            "INSERT INTO semantic_memory (id, project, memory_type, title, content, key, "
            "provenance, scope_type, scope_id, status, file_path, content_hash, embedding, "
            "metadata, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mem.id, self.project_root, mem.memory_type, mem.title, mem.content, mem.key,
             mem.provenance, mem.scope_type, mem.scope_id, mem.status, mem.file_path,
             mem.content_hash, json.dumps(mem.embedding), json.dumps(mem.metadata),
             mem.created_at))
        self._conn.commit()
        return mem.id

    # ── read path: hybrid recall (max(lexical, semantic)) ────────────────────

    def search(self, query: str, scope: Scope | None = None, k: int = 5,
               min_score: float = 0.05) -> list[MemoryHit]:
        rows = self._active_rows(scope)
        if not rows:
            return []
        qvec = self.embedder.embed([query])[0]
        qtok = _tokens(query)
        hits: list[MemoryHit] = []
        for r in rows:
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            sem = cosine(qvec, emb) if emb else 0.0
            rtok = _tokens(r["title"] + " " + r["content"])
            lex = (len(qtok & rtok) / len(qtok)) if qtok else 0.0
            score = max(lex, sem)
            if score >= min_score:
                hits.append(MemoryHit(self._row_to_mem(r), score, lex, sem))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def get(self, memory_id: str) -> SemanticMemory | None:
        r = self._conn.execute("SELECT * FROM semantic_memory WHERE id=?",
                               (memory_id,)).fetchone()
        return self._row_to_mem(r) if r else None

    # ── lifecycle + file-fact staleness (P3) ─────────────────────────────────

    def set_status(self, memory_id: str, status: str) -> None:
        self._conn.execute("UPDATE semantic_memory SET status=? WHERE id=?",
                           (status, memory_id))
        self._conn.commit()

    def file_content_hash(self, file_path: str) -> str | None:
        """SHA1 of a project-relative (or absolute) file's bytes; None if missing."""
        path = file_path if os.path.isabs(file_path) else os.path.join(
            self.project_root, file_path)
        try:
            with open(path, "rb") as fh:
                return hashlib.sha1(fh.read()).hexdigest()
        except OSError:
            return None

    def invalidate_stale(self) -> list[str]:
        """Re-check active file-scoped memories against current file content. A changed
        file → 'superseded' (the fact may be wrong); a missing file → 'archived'.
        Returns the invalidated memory ids. This is the tau-specific hazard Claire
        doesn't face: code facts rot when files change."""
        rows = self._conn.execute(
            "SELECT id, file_path, content_hash FROM semantic_memory "
            "WHERE project=? AND status='active' AND file_path IS NOT NULL",
            (self.project_root,)).fetchall()
        invalidated: list[str] = []
        for r in rows:
            current = self.file_content_hash(r["file_path"])
            if current is None:
                self.set_status(r["id"], "archived")
                invalidated.append(r["id"])
            elif r["content_hash"] is not None and current != r["content_hash"]:
                self.set_status(r["id"], "superseded")
                invalidated.append(r["id"])
        return invalidated

    def _active_rows(self, scope: Scope | None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM semantic_memory WHERE project=? AND status='active'"
        params: list = [self.project_root]
        if scope and scope.scope_type != "project" and scope.scope_id:
            # include the requested sub-scope OR project-global memories
            sql += " AND (scope_type='project' OR (scope_type=? AND scope_id=?))"
            params += [scope.scope_type, scope.scope_id]
        return self._conn.execute(sql, params).fetchall()

    @staticmethod
    def _row_to_mem(r: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=r["id"], project=r["project"], memory_type=r["memory_type"],
            title=r["title"], content=r["content"], key=r["key"],
            provenance=r["provenance"], scope_type=r["scope_type"], scope_id=r["scope_id"],
            status=r["status"], file_path=r["file_path"], content_hash=r["content_hash"],
            embedding=json.loads(r["embedding"]) if r["embedding"] else None,
            metadata=json.loads(r["metadata"]), created_at=r["created_at"])

    # ── conversation log (compaction substrate; minimal for P0) ──────────────

    def append_turn(self, turn: ConversationTurn) -> str:
        if not turn.id:
            turn.id = uuid.uuid4().hex
        if not turn.created_at:
            turn.created_at = _now()
        self._conn.execute(
            "INSERT INTO conversation_memory (id, project, role, content, scope_id, "
            "summary_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (turn.id, self.project_root, turn.role, turn.content, turn.scope_id,
             turn.summary_id, turn.created_at))
        self._conn.commit()
        return turn.id

    def recent_turns(self, limit: int = 20) -> list[ConversationTurn]:
        rows = self._conn.execute(
            "SELECT * FROM conversation_memory WHERE project=? ORDER BY created_at DESC "
            "LIMIT ?", (self.project_root, limit)).fetchall()
        return [ConversationTurn(id=r["id"], project=r["project"], role=r["role"],
                                 content=r["content"], scope_id=r["scope_id"],
                                 summary_id=r["summary_id"], created_at=r["created_at"])
                for r in reversed(rows)]
