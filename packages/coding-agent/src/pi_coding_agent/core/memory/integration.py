"""
Memory integration (P5) — bundle store + recall + curator + compression for a session.

Flag-gated: settings.json `memory_enabled` is the normal control plane; env vars
``PI_MEMORY_DISABLED=1`` / ``PI_CODING_AGENT_MEMORY_DISABLED=1`` are explicit kill
switches; ``PI_MEMORY_ENABLED=1`` / ``PI_CODING_AGENT_MEMORY_ENABLED=1`` force on
for tests/CI. When the agent folder is silent on ``memory_enabled``, memory is
**default-on** (Tau's memory system is native, not extension-based). When
enabled, AgentSession: attaches the store (recall read-path fires in
_transform_context); curates each turn's evidence into the store via the async
curator (_curate_turn in _post_turn_checks); records every tool call to
tool_log_memory and every conversation turn to conversation_memory (both
programmatic, harness-driven, see LANES.md); and curates-before-compacting so
facts survive the lossy summary and recall re-injects them. All four paths are
unit- + live-session-tested. End-to-end task-success (does it beat a simple
harness) is the separate P6 deep-swe acceptance run.
"""
from __future__ import annotations

import os

from .curator import AsyncLlmFn, Curator, Evidence, LlmFn
from .models import Scope
from .recall import build_recall_block
from .store import MemoryStore
from .working_context import CtxBlock, WorkingContextConfig, profile_for


def memory_enabled() -> bool:
    return (
        os.environ.get("PI_MEMORY_ENABLED", "") == "1"
        or os.environ.get("PI_CODING_AGENT_MEMORY_ENABLED", "") == "1"
    )


class MemoryIntegration:
    """One bundle per session, rooted at the project (cwd)."""

    def __init__(self, project_root: str, *, llm_fn: LlmFn | None = None,
                 allm_fn: AsyncLlmFn | None = None, model: str | None = None,
                 scope: Scope | None = None,
                 config: WorkingContextConfig | None = None) -> None:
        self.store = MemoryStore(project_root)
        self.scope = scope or Scope(project=os.path.abspath(project_root))
        self.config = config or profile_for(model)
        self.curator = (Curator(llm_fn, self.store, allm_fn=allm_fn)
                        if (llm_fn or allm_fn) else None)

    # read path
    def recall_block(self, query: str) -> str | None:
        return build_recall_block(self.store, query, self.scope,
                                  k=self.config.recall_k,
                                  token_budget=self.config.recall_budget_tokens)

    # write path — sync (tests) and async (live loop)
    def record_turn(self, evidence: list[Evidence]) -> list[str]:
        if not self.curator or not self.curator.llm_fn:
            return []
        return self.curator.curate_and_commit(evidence)

    async def arecord_turn(self, evidence: list[Evidence]) -> list[str]:
        if not self.curator or not self.curator.allm_fn:
            return []
        return await self.curator.acurate_and_commit(evidence)

    # NOTE: working-context positional compression was dropped (design §12): active
    # compression (Headroom/CCR) replaces it. Memory now only records + recalls.

    def close(self) -> None:
        self.store.close()
