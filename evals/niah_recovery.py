#!/usr/bin/env python3
"""
Needle-in-a-haystack recovery at realistic scale — lexical vs semantic.

Tests the ACTUAL Oracle-shaped mechanism (the one running in Claire): a needle is
planted in a block that then gets position-compressed, and we ask whether recovery
surfaces that block for a query — comparing LEXICAL recovery (keyword overlap, no
embeddings) vs SEMANTIC recovery (local nomic-embed-text via Ollama). Two query
phrasings: LITERAL (shares words with the needle) and PARAPHRASE (shares none) — the
paraphrase case is exactly where lexical was predicted to fail (recovery_eval §9).

Corpus is a long public-domain text (>128k tokens, so compression actually fires).
The core test is FREE and LOCAL (no OpenAI): it measures whether the needle block
lands in the top-k recovered set. --confirm adds a few OpenAI calls to verify the
model then answers correctly from the recovered context.

    python evals/niah_recovery.py evals/fixtures/corpus/mobydick.txt --depth 0.5

Embeds full block text via Ollama (local). Retrieval recall needs no cloud at all.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery_eval import compressed_ids, lexical_score  # noqa: E402
from tier0_context_replay import Block, approx_tokens, lex_set  # noqa: E402

NEEDLE = ("Captain Ahab kept a private cipher hidden inside his locked sea-chest; "
          "the cipher reads MARLIN-NINE-AZURE.")
Q_LITERAL = "What does the private cipher hidden in Ahab's locked sea-chest read?"
Q_PARAPHRASE = "What confidential passcode did the ship's commander conceal in his trunk?"
NEEDLE_ANSWER = "marlin-nine-azure"


def embed(texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": "nomic-embed-text", "input": texts}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["embeddings"]


def embed_all(texts: list[str], batch: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(embed(texts[i:i + batch]))
    return out


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return d / (na * nb)


def chunk(text: str, words_per: int = 300) -> list[Block]:
    words = text.split()
    blocks = []
    for i in range(0, len(words), words_per):
        t = " ".join(words[i:i + words_per])
        blocks.append(Block(eid=f"b{i//words_per}", role="doc", text=t, turn=i // words_per))
    return blocks


def rank_of(needle_eid: str, scored: list[tuple[float, str]]) -> int:
    order = [eid for _, eid in sorted(scored, key=lambda x: x[0], reverse=True)]
    return order.index(needle_eid) + 1 if needle_eid in order else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--max-tokens", type=int, default=160000,
                    help="cap corpus (keeps full-context LLM arm within window)")
    ap.add_argument("--depth", type=float, default=0.5, help="needle depth (0..1)")
    ap.add_argument("--head-tok", type=int, default=20000)
    ap.add_argument("--tail-tok", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--confirm", action="store_true",
                    help="add OpenAI calls to verify the model answers from recovery")
    ap.add_argument("--model", default="gpt-5.4-mini")
    args = ap.parse_args()

    raw = open(args.corpus, encoding="utf-8", errors="replace").read()
    blocks = chunk(raw)
    # cap to budget
    acc, kept = 0, []
    for b in blocks:
        if acc >= args.max_tokens:
            break
        kept.append(b)
        acc += b.tokens
    blocks = kept

    # plant needle into the block at the requested depth
    ni = max(1, min(len(blocks) - 2, int(len(blocks) * args.depth)))
    needle_block = blocks[ni]
    needle_block.text += " " + NEEDLE
    blocks[ni] = Block(eid=needle_block.eid, role="doc", text=needle_block.text,
                       turn=needle_block.turn)
    needle_eid = blocks[ni].eid
    total_tok = sum(b.tokens for b in blocks)

    comp = compressed_ids(blocks, args.head_tok, args.tail_tok)
    comp_blocks = [b for b in blocks if b.eid in comp]
    print(f"corpus {os.path.basename(args.corpus)}  blocks={len(blocks)}  "
          f"~{total_tok} tok  needle in {needle_eid} (depth {args.depth})  "
          f"compressed-middle={len(comp_blocks)}  topk={args.topk}")
    print(f"needle compressed? {'YES' if needle_eid in comp else 'NO (in head/tail!)'}\n")
    if needle_eid not in comp:
        print("needle not in compressed region — raise depth spread or budgets")
        return 1

    # embed compressed blocks (full text) once, locally
    print("embedding compressed blocks locally (nomic-embed-text)...")
    embs = dict(zip([b.eid for b in comp_blocks],
                    embed_all([b.text for b in comp_blocks])))

    print(f"\n{'query':<11} {'lexical rank':>13} {'semantic rank':>14} "
          f"{'lex@k':>6} {'sem@k':>6}")
    print("-" * 56)
    results = {}
    for label, q in (("LITERAL", Q_LITERAL), ("PARAPHRASE", Q_PARAPHRASE)):
        qt = lex_set(q)
        lex_scored = [(float(lexical_score(qt, b.toks)), b.eid) for b in comp_blocks]
        qemb = embed([q])[0]
        sem_scored = [(cos(qemb, embs[b.eid]), b.eid) for b in comp_blocks]
        lr, sr = rank_of(needle_eid, lex_scored), rank_of(needle_eid, sem_scored)
        lk = "yes" if 0 < lr <= args.topk else "NO"
        sk = "yes" if 0 < sr <= args.topk else "NO"
        print(f"{label:<11} {lr:>13} {sr:>14} {lk:>6} {sk:>6}")
        results[label] = {"lex_rank": lr, "sem_rank": sr,
                          "sem_block": sorted(sem_scored, reverse=True)[0][1]}

    if args.confirm:
        import openai
        from pathlib import Path
        key = ""
        for line in Path(".env").read_text().splitlines():
            if line.startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
                key = line.split("=", 1)[1].strip()
        client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=key)

        def ask(context: str, q: str) -> str:
            r = client.chat.completions.create(model=args.model, messages=[
                {"role": "system", "content": "Answer the question using only the text provided."},
                {"role": "user", "content": f"{context}\n\nQUESTION: {q}"}])
            return (r.choices[0].message.content or "")

        # semantic-recovered context = head/tail + top-k semantic blocks (paraphrase query)
        qemb = embed([Q_PARAPHRASE])[0]
        topk_eids = [eid for _, eid in sorted(
            ((cos(qemb, embs[b.eid]), b.eid) for b in comp_blocks), reverse=True)][:args.topk]
        recovered = "\n\n".join(b.text for b in comp_blocks if b.eid in topk_eids)
        no_recovery = "(earlier text omitted)"
        print("\n--confirm (paraphrase query, gpt answers):")
        a_rec = ask(recovered, Q_PARAPHRASE)
        a_none = ask(no_recovery, Q_PARAPHRASE)
        print(f"  semantic-recovered: {'CORRECT' if NEEDLE_ANSWER in a_rec.lower() else 'wrong'}"
              f"  -> {a_rec[:80]!r}")
        print(f"  no-recovery       : {'CORRECT' if NEEDLE_ANSWER in a_none.lower() else 'wrong'}"
              f"  -> {a_none[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
