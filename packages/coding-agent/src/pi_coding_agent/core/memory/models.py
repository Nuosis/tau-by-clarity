"""
Memory domain models — project-local atomic memory for tau.

The store is rooted in the project (cwd) and committed to git; the project root is the
poisoning boundary (an agent in another project cannot reach it). See
design/context-and-memory-management.md §8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class MemoryType(str, Enum):
    """The DECIDED curator atomic taxonomy (design doc §8)."""
    DECISION = "decision"        # e.g. "use SQLite not Postgres"
    CONSTRAINT = "constraint"    # e.g. "don't touch the auth module"
    FILE_API = "file_api"        # signatures, config values, where things live
    TASK_STATE = "task_state"    # what's done / pending
    ERROR_FIX = "error_fix"      # an error and its resolution
    PREFERENCE = "preference"    # e.g. "run tests before commit"


# Scope = the in-project sub-boundary. project is the hard poisoning boundary.
ScopeType = Literal["project", "session", "task", "file"]

MemoryStatus = Literal["active", "superseded", "archived", "deleted"]


@dataclass
class Scope:
    """Where a memory lives / is queried. `project` is the root (poisoning boundary)."""
    project: str
    scope_type: ScopeType = "project"
    scope_id: str = ""


@dataclass
class SemanticMemory:
    """One atomic, durable memory unit (the thing the curator writes and recall reads)."""
    id: str
    project: str
    memory_type: str
    title: str
    content: str
    key: str                              # canonical dedup key; supersede on collision
    provenance: str                       # session/agent id that produced it
    scope_type: ScopeType = "project"
    scope_id: str = ""
    status: MemoryStatus = "active"
    # file-fact staleness (P3): file-scoped memories invalidate when the file changes
    file_path: str | None = None
    content_hash: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def embedding_text(self) -> str:
        """Text fed to the embedder — title then content (Claire's convention)."""
        return f"{self.title}\n{self.content}"


@dataclass
class MemoryHit:
    memory: SemanticMemory
    score: float          # 0..1, hybrid max(lexical, semantic)
    lexical: float = 0.0
    semantic: float = 0.0


@dataclass
class ConversationTurn:
    """Exact ordered conversation turn (compaction substrate)."""
    id: str
    project: str
    role: str
    content: str
    scope_id: str = ""
    summary_id: str | None = None
    created_at: float = 0.0
