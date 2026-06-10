#!/usr/bin/env python3
"""
Coding-agent recall verification — the test we actually needed.

Take a REAL coding-agent trace (Langfuse devin_agent/ATC), plant a coding-shaped fact
as its OWN atomic memory unit (the faithful curated-memory simulation), compress the
middle, and ask whether HYBRID recall (max of lexical + local semantic) surfaces it so
the model answers — vs full context (ceiling) and no-recovery (floor). Run a LITERAL
and a PARAPHRASE query to finally measure the coding paraphrase tail on real data.

  full        : whole trace verbatim + needle           -> ceiling
  no-recovery : head/tail verbatim, middle stubbed       -> floor (needle lost)
  hybrid      : + top-k hybrid-recovered atomic units    -> the candidate

Embeddings are LOCAL (Ollama nomic-embed-text). Answers via OpenAI (--model).

    python evals/coding_recall_ab.py evals/fixtures/langfuse/<id>.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from niah_recovery import cos, embed, embed_all  # noqa: E402
from recovery_eval import compressed_ids, keyword_cues, lexical_score  # noqa: E402
from tier0_context_replay import Block, lex_set, load_session  # noqa: E402

# coding-shaped needle, planted as its own atomic memory unit
NEEDLE = ("Decision logged: in config/net.py the constant MAX_RECONNECT_ATTEMPTS "
          "is set to 7741 (chosen to match the upstream gateway's session budget).")
Q_LITERAL = "What value is MAX_RECONNECT_ATTEMPTS set to in config/net.py?"
Q_PARAPHRASE = "How many times will the client retry a dropped connection before giving up?"
ANSWER = "7741"


def stub(b: Block, cue: set[str]) -> str:
    return f"[compressed {b.eid}; keywords: {' '.join(sorted(cue))}]"


def hybrid_ranked(query: str, comp: list[Block], cues, embs) -> list[tuple[float, str]]:
    qt = lex_set(query)
    qemb = embed([query])[0]
    out = []
    for b in comp:
        lex = len(qt & b.toks) / max(1, len(qt))      # normalized lexical overlap
        sem = cos(qemb, embs[b.eid])                   # cosine, ~[0,1]
        out.append((max(lex, sem), b.eid))             # hybrid = max (Claire's pattern)
    return sorted(out, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--head-tok", type=int, default=8000)
    ap.add_argument("--tail-tok", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--base-url", default="https://api.minimax.io/v1")
    args = ap.parse_args()

    var = "MINIMAX_API_KEY" if "minimax" in args.base_url else "OPENAI_API_KEY"
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.startswith(var + "=") and line.split("=", 1)[1].strip():
            key = line.split("=", 1)[1].strip()
    import openai
    client = openai.OpenAI(base_url=args.base_url, api_key=key, timeout=600)

    blocks = load_session(args.trace, per_message_turns=True)
    ni = max(1, min(len(blocks) - 2, int(len(blocks) * args.depth)))
    needle = Block(eid="needle", role="assistant", text=NEEDLE, turn=blocks[ni].turn)
    blocks = blocks[:ni] + [needle] + blocks[ni:]      # insert as its OWN atomic unit
    for i, b in enumerate(blocks):                     # renumber turns
        b.turn = i

    comp = compressed_ids(blocks, args.head_tok, args.tail_tok)
    cues = keyword_cues(blocks, 40)
    comp_blocks = [b for b in blocks if b.eid in comp]
    print(f"trace {os.path.basename(args.trace)}  units={len(blocks)}  "
          f"~{sum(b.tokens for b in blocks)} tok  compressed={len(comp_blocks)}  "
          f"needle compressed? {'YES' if 'needle' in comp else 'NO'}")
    if "needle" not in comp:
        print("needle landed in head/tail — adjust depth/budgets"); return 1

    embs = dict(zip([b.eid for b in comp_blocks],
                    embed_all([b.text for b in comp_blocks])))

    def render(mode: str, query: str):
        if mode == "full":
            # everything verbatim — the big, noisy ceiling
            return "\n".join(f"<{b.role}> {b.text}" for b in blocks), True
        if mode == "no-recovery":
            # head/tail verbatim, middle as stubs — needle absent
            parts = [f"<{b.role}> {b.text}" if b.eid not in comp
                     else stub(b, cues.get(b.eid, set())) for b in blocks]
            return "\n".join(parts), False
        # hybrid = LEAN recall context. Recovered units are ATOMIC (cap to ~60 words,
        # simulating curator-extracted facts) and placed LAST (recency). The needle is
        # its own short unit, so capping leaves it intact.
        def atomize(t: str, n: int = 60) -> str:
            w = t.split()
            return " ".join(w[:n]) + ("…" if len(w) > n else "")
        top = [eid for _, eid in hybrid_ranked(query, comp_blocks, cues, embs)][:args.topk]
        tail = [b for b in blocks if b.eid not in comp][-2:]
        rec = [b for b in comp_blocks if b.eid in top]
        parts = [f"<{b.role}> {atomize(b.text)}" for b in tail]
        parts.append("\n## Relevant retrieved memories:\n"
                     + "\n".join(f"- {atomize(b.text)}" for b in rec))
        return "\n".join(parts), ("needle" in top)

    def ask(ctx: str, q: str) -> str:
        r = client.chat.completions.create(model=args.model, messages=[
            {"role": "system", "content": "Answer the question using only the provided context. If the context does not contain the answer, say you don't know."},
            {"role": "user", "content": f"{ctx}\n\nQUESTION: {q}"}])
        return r.choices[0].message.content or ""

    print(f"\n{'query':<11} {'arm':<12} {'needle recovered':>16} {'answer correct':>15}")
    print("-" * 58)
    for label, q in (("LITERAL", Q_LITERAL), ("PARAPHRASE", Q_PARAPHRASE)):
        # needle's hybrid rank for context
        ranked = hybrid_ranked(q, comp_blocks, cues, embs)
        nrank = [eid for _, eid in ranked].index("needle") + 1
        for mode in ("full", "no-recovery", "hybrid"):
            ctx, recovered = render(mode, q)
            a = ask(ctx, q)
            ok = ANSWER in a
            rec = "(verbatim)" if mode == "full" else ("YES" if recovered else "no")
            print(f"{label:<11} {mode:<12} {rec:>16} {'YES' if ok else 'NO':>15}")
        print(f"{'':11} (needle hybrid rank among {len(comp_blocks)} compressed: {nrank})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
