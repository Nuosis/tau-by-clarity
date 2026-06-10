#!/usr/bin/env python3
"""
Live curator eval (P2) — does the curator extract the right atoms with a REAL model?

Complements the stub unit tests (control logic) with a live extraction-quality check on
MiniMax-M3. Feeds a small realistic coding-evidence packet and prints the decisions:
expect the grounded decision + file-fact to auto_commit, and the assistant-only aside to
NOT become a grounded durable memory.

    python evals/curator_eval.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "packages/coding-agent/src")
from pi_coding_agent.core.memory import (  # noqa: E402
    Curator,
    DeterministicEmbeddingProvider,
    Evidence,
    MemoryStore,
)


def m3_llm(system: str, user: str) -> str:
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.startswith("MINIMAX_API_KEY=") and line.split("=", 1)[1].strip():
            key = line.split("=", 1)[1].strip()
    import openai
    c = openai.OpenAI(base_url="https://api.minimax.io/v1", api_key=key, timeout=300)
    r = c.chat.completions.create(model="MiniMax-M3", messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def main() -> int:
    evidence = [
        Evidence("e1", "user_turn",
                 "Let's use SQLite for the memory store, not Postgres — simpler for a local CLI."),
        Evidence("e2", "tool_result", "config/net.py: MAX_RECONNECT_ATTEMPTS = 7741"),
        Evidence("e3", "assistant_output",
                 "I'll note that we should probably add a caching layer at some point."),
    ]
    with tempfile.TemporaryDirectory() as d:
        store = MemoryStore(d, embedder=DeterministicEmbeddingProvider())
        cur = Curator(m3_llm, store)
        decisions = cur.curate(evidence)
        print(f"extracted {len(decisions)} decisions:")
        for x in decisions:
            print(f"  [{x.verdict}] {x.memory_type}:{x.key}  src={x.source_ids}  "
                  f"conf={x.confidence}  — {x.title}: {x.content[:60]}")
        written = cur.commit(decisions)
        print(f"\nauto-committed (active): {len(written)}")
        auto = [d for d in decisions if d.verdict == "auto_commit"]
        grounded_ok = all(set(d.source_ids) <= {"e1", "e2"} for d in auto)
        print("PASS: grounded-only auto-commits" if grounded_ok and written
              else "CHECK: review output above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
