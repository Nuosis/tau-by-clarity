#!/usr/bin/env python3
"""
Recovery eval — does lexical-only recovery (no embeddings) lose much?

Companion to design/context-and-memory-management.md. Tests the "active compression"
recovery question without any embedding model (none installed; embeddings would mean
network + a key + shipping data out). Two stdlib-only tests over a real session:

  Test A (controlled): plant a needle in the compressible middle, query it two ways —
    LITERAL (shares words with the fact) and PARAPHRASE (shares none) — and show
    lexical recovery catches the literal case and *provably misses* the paraphrase.
    That is the embedding-shaped gap, demonstrated without embeddings.

  Test B (real prevalence): of the cross-references that actually occur (a later turn
    re-uses terms first introduced in a now-compressed middle block), what fraction
    survive deterministic keyword compression and are lexically recoverable? This is
    how common the lexical-friendly case is in real traffic — i.e. how much we lose by
    skipping embeddings.

Compression model: position-based. Keep a head budget + tail budget verbatim; the
middle is compressed to a deterministic TF-IDF keyword cue. Recovery signal = overlap
between the query terms and a compressed block's keyword cue.

    python evals/recovery_eval.py evals/fixtures/langfuse/<id>.jsonl
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier0_context_replay import Block, approx_tokens, lex_set, load_session  # noqa: E402

NEEDLE_FACT = "The deployment passphrase is QUOKKA-77-VERMILION."
NEEDLE_LITERAL = "what was the deployment passphrase mentioned earlier"
NEEDLE_PARAPHRASE = "which secret credential do I use to ship the release to production"


# ─── deterministic keyword cue (TF-IDF top-k) ─────────────────────────────────

def keyword_cues(blocks: list[Block], top_k: int = 12) -> dict[str, set[str]]:
    n = len(blocks) or 1
    df: dict[str, int] = {}
    for b in blocks:
        for t in b.toks:
            df[t] = df.get(t, 0) + 1
    cues: dict[str, set[str]] = {}
    for b in blocks:
        tf: dict[str, int] = {}
        for w in b.text.lower().split():
            w2 = "".join(c for c in w if c.isalnum() or c in "_/.-")
            if len(w2) >= 4:
                tf[w2] = tf.get(w2, 0) + 1
        scored = sorted(
            ((t, c * math.log(n / df.get(t, 1))) for t, c in tf.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        cues[b.eid] = {t for t, _ in scored[:top_k]}
    return cues


def compressed_ids(blocks: list[Block], head_tok: int, tail_tok: int) -> set[str]:
    """Block ids in the compressible middle (outside head/tail token budgets)."""
    head: set[str] = set()
    acc = 0
    for b in blocks:
        if acc >= head_tok:
            break
        head.add(b.eid)
        acc += b.tokens
    tail: set[str] = set()
    acc = 0
    for b in reversed(blocks):
        if acc >= tail_tok:
            break
        tail.add(b.eid)
        acc += b.tokens
    protected = head | tail
    return {b.eid for b in blocks if b.eid not in protected}


def lexical_score(query_terms: set[str], cue: set[str]) -> int:
    return len(query_terms & cue)


# ─── Test A: literal vs paraphrase recovery ───────────────────────────────────

def test_a(blocks: list[Block], cues: dict[str, set[str]], comp: set[str]) -> None:
    needle = Block(eid="needle", role="assistant",
                   text=f"Status note, nothing unusual. {NEEDLE_FACT} Carry on.",
                   turn=blocks[len(blocks) // 2].turn)
    all_blocks = blocks + [needle]
    cues2 = keyword_cues(all_blocks)
    needle_cue = cues2["needle"]
    comp2 = comp | {"needle"}

    print("Test A — controlled needle (lexical recovery, no embeddings)")
    print(f"  needle keyword cue: {sorted(needle_cue)}")
    for label, q in (("LITERAL", NEEDLE_LITERAL), ("PARAPHRASE", NEEDLE_PARAPHRASE)):
        qt = lex_set(q)
        scores = sorted(
            ((eid, lexical_score(qt, cues2.get(eid, set()))) for eid in comp2),
            key=lambda kv: kv[1], reverse=True,
        )
        rank = next((i for i, (eid, _) in enumerate(scores) if eid == "needle"), -1)
        needle_pts = lexical_score(qt, needle_cue)
        top = scores[0][1] if scores else 0
        verdict = ("RECOVERED" if rank == 0 and needle_pts > 0
                   else f"rank #{rank+1}" if needle_pts > 0 else "MISSED (0 overlap)")
        print(f"  {label:<10} query={q!r}")
        print(f"             needle overlap={needle_pts}, top score={top}, "
              f"among {len(comp2)} compressed → {verdict}")


# ─── Test B: real cross-reference recoverability ──────────────────────────────

def test_b(blocks: list[Block], cues: dict[str, set[str]], comp: set[str],
           min_shared: int = 3) -> None:
    by_turn: dict[int, set[str]] = {}
    for b in blocks:
        by_turn.setdefault(b.turn, set()).update(b.toks)

    comp_blocks = [b for b in blocks if b.eid in comp]
    true_refs = 0          # later turn genuinely re-uses a compressed block's content
    recoverable = 0        # ...and its keyword cue would surface it lexically
    cue_term_retention = []

    for b in comp_blocks:
        full = b.toks
        cue = cues.get(b.eid, set())
        if full:
            cue_term_retention.append(len(cue & full) / max(1, len(full)))
        for t, tterms in by_turn.items():
            if t <= b.turn:
                continue
            shared_full = len(full & tterms)
            if shared_full >= min_shared:          # genuine cross-reference (ground truth)
                true_refs += 1
                if lexical_score(tterms, cue) >= 1:  # cue still surfaces it
                    recoverable += 1

    print("\nTest B — real cross-reference recoverability (no embeddings)")
    print(f"  compressed middle blocks: {len(comp_blocks)}")
    print(f"  genuine later-turn cross-references (>= {min_shared} shared terms): {true_refs}")
    if true_refs:
        print(f"  lexically recoverable from keyword cue: {recoverable} "
              f"({100*recoverable/true_refs:.0f}%)")
    if cue_term_retention:
        avg = sum(cue_term_retention) / len(cue_term_retention)
        print(f"  avg fraction of a block's distinctive terms kept in its cue: {avg:.0%}")
    print("  (paraphrase cross-references — zero shared terms — are invisible to this\n"
          "   ground truth by construction; that subset is what embeddings would add.)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--head-tok", type=int, default=8000)
    ap.add_argument("--tail-tok", type=int, default=8000)
    ap.add_argument("--top-k", type=int, default=12,
                    help="keyword-cue size; pass repeatedly via --sweep to compare")
    ap.add_argument("--sweep", default="",
                    help="comma-separated cue sizes to sweep, e.g. 12,30,60,120")
    args = ap.parse_args()

    blocks = load_session(args.session, per_message_turns=True)
    if not blocks:
        print("no blocks", file=sys.stderr)
        return 1
    total = sum(b.tokens for b in blocks)
    comp = compressed_ids(blocks, args.head_tok, args.tail_tok)
    print(f"session: {os.path.basename(args.session)}  blocks={len(blocks)}  "
          f"~{total} tok  compressed-middle={len(comp)} "
          f"(head {args.head_tok} / tail {args.tail_tok} tok protected)\n")

    sizes = [int(x) for x in args.sweep.split(",") if x.strip()] or [args.top_k]
    test_a(blocks, keyword_cues(blocks, sizes[0]), comp)
    for k in sizes:
        print(f"\n--- cue size top_k={k} ---")
        test_b(blocks, keyword_cues(blocks, k), comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
