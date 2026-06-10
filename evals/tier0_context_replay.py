#!/usr/bin/env python3
"""
Tier-0 offline context-replay harness.

Companion to docs/context-and-memory-management.md (§9). Proves — or kills — the
relevance-shaped context-management model WITHOUT a single LLM call.

For each user turn in a real session it reassembles the context two ways:

    baseline   = full append-only transcript (what pi-py sends today)
    candidate  = relevance-shaped fidelity gradient (the §7 proposal)

and measures, with pure string/lexical math:

    token delta            how many fewer tokens the candidate sends
    prefix stability       cache-hit proxy: fraction of the prompt that is
                           byte-identical to the previous turn's prompt
                           (append-only ≈ high; re-compression ≈ lower — this is
                           the cache cost of being clever, made visible)
    needle preservation    of the facts a LATER turn actually re-references, what
                           fraction survived the candidate's compression
                           (this is the §9 "compaction-recall-miss", inverted)

The candidate scorer is the Generative Agents formula (Park et al. 2023):
    score = w_rec*recency + w_rel*relevance + w_size*(1 - size_norm)
recency = exp-decay since last reference; relevance = lexical overlap with the
current turn's query (Tier-0 uses lexical, not embeddings, to stay zero-cost);
size penalises big rarely-referenced blocks (verbose tool dumps).

Anchor (first turn) and frontier (last --frontier turns) are always verbatim
(Lost-in-the-Middle U-shape). Demoted blocks are replaced by a reversible stub,
so nothing is destroyed — exactly the fidelity-gradient design.

Usage:
    python evals/tier0_context_replay.py .pi-py/agent/sessions/*.jsonl
    python evals/tier0_context_replay.py SESSION.jsonl --floor 0 --json out.json
    python evals/tier0_context_replay.py SESSION.jsonl --needle   # synthetic needle

Token counts are char/4 approximations: fine for *deltas*, not for billing.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# ─── tokenisation (approximate, zero-dependency) ──────────────────────────────

_WORD = re.compile(r"[A-Za-z0-9_/.\-]+")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def lex_set(text: str) -> set[str]:
    """Content tokens for lexical overlap / needle detection (len>=4, deduped)."""
    return {w.lower() for w in _WORD.findall(text) if len(w) >= 4}


# ─── session model ────────────────────────────────────────────────────────────

@dataclass
class Block:
    """One transcript entry, flattened to text plus provenance."""
    eid: str
    role: str                 # user | assistant | toolResult | summary | ...
    text: str
    turn: int                 # index of the user turn this block belongs to
    tokens: int = 0
    toks: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.tokens = approx_tokens(self.text)
        self.toks = lex_set(self.text)

    def stub(self) -> str:
        """Reversible low-fidelity reference (the 'ref code' tier)."""
        head = self.text.strip().replace("\n", " ")[:100]
        return f"[ref {self.eid} {self.role} ~{self.tokens}tok: {head}…]"


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text", ""))
        elif t == "thinking":
            parts.append(b.get("thinking", ""))
        elif t == "toolCall":
            parts.append(f"{b.get('name', '')}({json.dumps(b.get('arguments', {}))})")
        elif t == "image":
            parts.append("[image]")
    return "\n".join(p for p in parts if p)


def load_session(path: str) -> list[Block]:
    """Linearise a v3 session JSONL into ordered Blocks tagged by user turn."""
    blocks: list[Block] = []
    turn = -1
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            etype = e.get("type")
            if etype == "session":
                continue
            if etype == "message":
                m = e.get("message") or {}
                role = m.get("role", "?")
                text = _flatten_content(m.get("content"))
                if role == "user":
                    turn += 1
            elif etype in ("compaction", "branch_summary"):
                role = "summary"
                data = e.get("data") or {}
                text = data.get("summary") or json.dumps(data)[:500]
            else:
                continue
            if not text:
                continue
            blocks.append(Block(eid=e.get("id", f"e{len(blocks)}"),
                                role=role, text=text, turn=max(turn, 0)))
    return blocks


# ─── needle injection (Chroma context-rot style, optional) ────────────────────

_NEEDLE_FACT = "The deployment passphrase is QUOKKA-77-VERMILION."
_NEEDLE_QUERY = "What was the deployment passphrase mentioned earlier?"


def inject_needle(blocks: list[Block]) -> tuple[list[Block], str]:
    """Plant a fact early and a query for it at the end; returns (blocks, token)."""
    if len(blocks) < 2:
        return blocks, ""
    fact = Block(eid="needle-fact", role="user",
                 text=f"Important for later: {_NEEDLE_FACT}", turn=blocks[0].turn)
    last_turn = blocks[-1].turn + 1
    query = Block(eid="needle-query", role="user", text=_NEEDLE_QUERY, turn=last_turn)
    return [fact] + blocks + [query], "quokka-77-vermilion"


# ─── strategies ───────────────────────────────────────────────────────────────

def context_baseline(blocks: list[Block], upto_turn: int) -> list[Block]:
    """Full append-only: everything up to and including the current turn."""
    return [b for b in blocks if b.turn <= upto_turn]


def context_candidate(
    blocks: list[Block],
    upto_turn: int,
    *,
    frontier: int,
    floor_tokens: int,
    w_rec: float = 1.0,
    w_rel: float = 1.0,
    w_size: float = 1.0,
    keep_threshold: float = 0.45,
    decay: float = 0.5,
) -> list[Block]:
    """Relevance-shaped fidelity gradient. Anchor + frontier verbatim; cold,
    low-relevance, bulky middle blocks demoted to reversible stubs — but only once
    total context exceeds the break-even floor (§7)."""
    live = [b for b in blocks if b.turn <= upto_turn]
    total = sum(b.tokens for b in live)
    if total <= floor_tokens:
        return live  # below break-even: don't pay the cache cost to be clever

    query_toks: set[str] = set()
    for b in live:
        if b.turn == upto_turn:
            query_toks |= b.toks
    anchor_turn = live[0].turn
    frontier_cut = upto_turn - frontier
    max_tok = max((b.tokens for b in live), default=1)

    out: list[Block] = []
    for b in live:
        if b.turn == anchor_turn or b.turn > frontier_cut:
            out.append(b)                       # U-shape: anchor + frontier verbatim
            continue
        # last turn at which any later/earlier block re-mentions this block's tokens
        last_ref = b.turn
        for other in live:
            if other.turn > b.turn and (other.toks & b.toks):
                last_ref = max(last_ref, other.turn)
        recency = math.exp(-decay * (upto_turn - last_ref))
        overlap = len(b.toks & query_toks)
        relevance = overlap / (len(query_toks) or 1)
        size_norm = b.tokens / max_tok
        score = (w_rec * recency + w_rel * relevance + w_size * (1 - size_norm)) / (
            w_rec + w_rel + w_size
        )
        if score >= keep_threshold:
            out.append(b)
        else:
            out.append(Block(eid=b.eid, role=b.role, text=b.stub(), turn=b.turn))
    return out


# ─── metrics ──────────────────────────────────────────────────────────────────

def render(ctx: list[Block]) -> str:
    return "\n".join(f"<{b.role} {b.eid}>\n{b.text}" for b in ctx)


def prefix_stability(prev: str, cur: str) -> float:
    """Fraction of the current prompt (by char) that is an unchanged prefix of it
    vs the previous prompt. Cache-hit proxy. 1.0 = pure append."""
    if not cur:
        return 1.0
    n = min(len(prev), len(cur))
    i = 0
    while i < n and prev[i] == cur[i]:
        i += 1
    return i / len(cur)


@dataclass
class SessionReport:
    path: str
    turns: int
    base_tok_sum: int
    cand_tok_sum: int
    base_prefix: float
    cand_prefix: float
    needle_base: float | None
    needle_cand: float | None

    @property
    def savings_pct(self) -> float:
        if not self.base_tok_sum:
            return 0.0
        return 100.0 * (self.base_tok_sum - self.cand_tok_sum) / self.base_tok_sum


def run_session(path: str, args: argparse.Namespace) -> SessionReport | None:
    blocks = load_session(path)
    needle_tok = ""
    if args.needle:
        blocks, needle_tok = inject_needle(blocks)
    if not blocks:
        return None
    turns = max(b.turn for b in blocks) + 1

    base_sum = cand_sum = 0
    base_prev = cand_prev = ""
    base_prefixes: list[float] = []
    cand_prefixes: list[float] = []
    # needle: count, over turns AFTER the fact appears, how often it survives
    nb_hit = nb_tot = nc_hit = nc_tot = 0

    for t in range(turns):
        if not any(b.turn == t for b in blocks):
            continue
        base = context_baseline(blocks, t)
        cand = context_candidate(blocks, t, frontier=args.frontier,
                                 floor_tokens=args.floor)
        bs, cs = render(base), render(cand)
        base_sum += sum(b.tokens for b in base)
        cand_sum += sum(b.tokens for b in cand)
        if base_prev:
            base_prefixes.append(prefix_stability(base_prev, bs))
            cand_prefixes.append(prefix_stability(cand_prev, cs))
        base_prev, cand_prev = bs, cs

        if needle_tok and t > blocks[0].turn:
            nb_tot += 1
            nc_tot += 1
            nb_hit += int(needle_tok in bs.lower())
            nc_hit += int(needle_tok in cs.lower())

    return SessionReport(
        path=path, turns=turns,
        base_tok_sum=base_sum, cand_tok_sum=cand_sum,
        base_prefix=sum(base_prefixes) / len(base_prefixes) if base_prefixes else 1.0,
        cand_prefix=sum(cand_prefixes) / len(cand_prefixes) if cand_prefixes else 1.0,
        needle_base=(100.0 * nb_hit / nb_tot) if nb_tot else None,
        needle_cand=(100.0 * nc_hit / nc_tot) if nc_tot else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session .jsonl paths (globs ok)")
    ap.add_argument("--frontier", type=int, default=2,
                    help="recent turns kept verbatim (default 2)")
    ap.add_argument("--floor", type=int, default=128_000,
                    help="break-even floor in tokens; below this, no compression "
                         "(default 128000; use 0 to force the gradient on)")
    ap.add_argument("--needle", action="store_true",
                    help="inject a synthetic early fact + late query (controlled test)")
    ap.add_argument("--json", metavar="PATH", help="write full report as JSON")
    args = ap.parse_args()

    paths: list[str] = []
    for p in args.sessions:
        paths.extend(sorted(glob.glob(p)) or [p])

    reports = [r for r in (run_session(p, args) for p in paths) if r]
    if not reports:
        print("no sessions with content found", file=sys.stderr)
        return 1

    print(f"{'session':<40} {'turns':>5} {'base_tok':>9} {'cand_tok':>9} "
          f"{'save%':>6} {'pfx_b':>6} {'pfx_c':>6} {'ndl_b':>6} {'ndl_c':>6}")
    print("-" * 100)
    tb = tc = 0
    for r in reports:
        name = r.path.rsplit("/", 1)[-1][:40]
        nb = "-" if r.needle_base is None else f"{r.needle_base:.0f}"
        nc = "-" if r.needle_cand is None else f"{r.needle_cand:.0f}"
        print(f"{name:<40} {r.turns:>5} {r.base_tok_sum:>9} {r.cand_tok_sum:>9} "
              f"{r.savings_pct:>5.1f} {r.base_prefix:>6.2f} {r.cand_prefix:>6.2f} "
              f"{nb:>6} {nc:>6}")
        tb += r.base_tok_sum
        tc += r.cand_tok_sum
    print("-" * 100)
    agg = 100.0 * (tb - tc) / tb if tb else 0.0
    print(f"{'TOTAL':<40} {'':>5} {tb:>9} {tc:>9} {agg:>5.1f}")
    print("\npfx_b/pfx_c = mean prefix stability (cache-hit proxy); higher = more "
          "cacheable.\nndl_b/ndl_c = needle preservation %% (--needle only); "
          "candidate must stay ~100.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([r.__dict__ for r in reports], fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
