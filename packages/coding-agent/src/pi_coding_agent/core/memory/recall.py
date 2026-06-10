"""
Recall — the read path (P1).

Turns a query into a tail-injected recall block of atomic memories via the store's
hybrid (max(lexical, semantic)) search. Breadcrumb-aware and token-budgeted. Placed at
the tail so the stable prefix stays cache-warm (design doc §7 "recall placement").
"""
from __future__ import annotations

from typing import Any

from .models import Scope
from .store import MemoryStore

RECALL_HEADER = "## Relevant retrieved project memory (recall)"


def build_recall_block(store: MemoryStore, query: str, scope: Scope | None = None,
                       k: int = 6, token_budget: int = 1500) -> str | None:
    """Search the store and format the top hits as a recall block, or None if nothing
    relevant. Each line is one atomic memory (type-tagged); budget-capped."""
    if not query.strip():
        return None
    hits = store.search(query, scope=scope, k=k)
    if not hits:
        return None
    lines: list[str] = []
    used = 0
    for h in hits:
        line = f"- [{h.memory.memory_type}] {h.memory.title}: {h.memory.content}"
        cost = max(1, len(line) // 4)
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    if not lines:
        return None
    return RECALL_HEADER + ":\n" + "\n".join(lines)


def latest_user_query(messages: list[Any]) -> str:
    """Extract the most recent user message's text from an AgentMessage list."""
    for m in reversed(messages):
        if getattr(m, "role", None) != "user":
            continue
        content = getattr(m, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                     for b in content]
            text = " ".join(p for p in parts if p)
            if text:
                return text
    return ""
