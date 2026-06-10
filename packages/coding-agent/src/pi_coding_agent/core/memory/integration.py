"""
Memory integration (P5) — bundle store + recall + curator + compression for a session.

Flag-gated (PI_MEMORY_ENABLED=1). When enabled, AgentSession attaches the store so the
P1 recall hook fires and (when an llm_fn is wired) the curator records turns. The live
compaction *replacement* and its parity are validated end-to-end by P6 (deep-swe); until
then memory recall augments and the flag defaults off (kill-switch retained).
"""
from __future__ import annotations

import os

from .curator import Curator, Evidence, LlmFn
from .models import Scope
from .recall import build_recall_block
from .store import MemoryStore
from .working_context import CtxBlock, WorkingContextConfig, compress_working_context, profile_for


def memory_enabled() -> bool:
    return os.environ.get("PI_MEMORY_ENABLED", "") == "1"


class MemoryIntegration:
    """One bundle per session, rooted at the project (cwd)."""

    def __init__(self, project_root: str, *, llm_fn: LlmFn | None = None,
                 model: str | None = None, scope: Scope | None = None,
                 config: WorkingContextConfig | None = None) -> None:
        self.store = MemoryStore(project_root)
        self.scope = scope or Scope(project=os.path.abspath(project_root))
        self.config = config or profile_for(model)
        self.curator = Curator(llm_fn, self.store) if llm_fn else None

    # read path
    def recall_block(self, query: str) -> str | None:
        return build_recall_block(self.store, query, self.scope,
                                  k=self.config.recall_k,
                                  token_budget=self.config.recall_budget_tokens)

    # write path (no-op without an llm_fn wired)
    def record_turn(self, evidence: list[Evidence]) -> list[str]:
        if not self.curator:
            return []
        return self.curator.curate_and_commit(evidence)

    # working-context management
    def compress(self, blocks: list[CtxBlock], query: str) -> list[CtxBlock]:
        return compress_working_context(blocks, self.store, query, self.config, self.scope)

    def close(self) -> None:
        self.store.close()
