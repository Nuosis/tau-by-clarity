#!/usr/bin/env python3
"""
Calibration sweep — a model's reliable working-set size.

Pure needle-in-haystack on UNCOMPRESSED context, swept over size × depth: plant a
distinctive fact in `size` tokens of filler at fractional `depth`, ask for it, record
whether the model recalls it. The largest size at which recall holds across depths is
the model's **reliable working-set size** = our compression floor/target (§7).
Model-specific by design — run per model to compare (e.g. MiniMax-M3 1M vs gpt-5.4-mini).

    python evals/niah_calibrate.py evals/fixtures/corpus/mobydick.txt \
        --model MiniMax-M3 --base-url https://api.minimax.io/v1 \
        --sizes 20000,60000,120000,250000 --depths 0.1,0.5,0.9

Clean prose filler = optimistic (standard NIAH); dense agent content fails earlier
(the §9 55k dense data point is the conservative anchor).
"""
from __future__ import annotations

import argparse
from pathlib import Path

NEEDLE = "Note for the record: the hidden vault code is MARLIN-NINE-AZURE."
QUERY = "What is the hidden vault code?"
ANSWER = "marlin-nine-azure"


def load_key(base_url: str) -> tuple[str, str]:
    var = "MINIMAX_API_KEY" if "minimax" in base_url else "OPENAI_API_KEY"
    for line in Path(".env").read_text().splitlines():
        if line.startswith(var + "=") and line.split("=", 1)[1].strip():
            return line.split("=", 1)[1].strip(), var
    return "", var


def build_context(words: list[str], size_tok: int, depth: float) -> str:
    n_words = int(size_tok * 0.75)
    body = words[:n_words]
    at = max(1, min(len(body) - 1, int(len(body) * depth)))
    body = body[:at] + NEEDLE.split() + body[at:]
    return " ".join(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--base-url", default="https://api.minimax.io/v1")
    ap.add_argument("--sizes", default="20000,60000,120000,250000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles", type=int, default=1)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    depths = [float(d) for d in args.depths.split(",")]
    key, _ = load_key(args.base_url)
    import openai
    client = openai.OpenAI(base_url=args.base_url, api_key=key, timeout=600)

    words = Path(args.corpus).read_text(encoding="utf-8", errors="replace").split()
    print(f"model={args.model}  corpus={Path(args.corpus).name} ({len(words)} words)\n")
    print(f"{'size_tok':>9} | " + " ".join(f"d={d:<4}" for d in depths))
    print("-" * (12 + 8 * len(depths)))

    reliable = 0
    for size in sizes:
        if int(size * 0.75) > len(words):
            print(f"{size:>9} | (corpus too small)")
            continue
        cells, all_ok = [], True
        for depth in depths:
            ok = 0
            for _ in range(args.needles):
                ctx = build_context(words, size, depth)
                try:
                    r = client.chat.completions.create(model=args.model, messages=[
                        {"role": "system", "content": "Answer using only the provided text. If absent, say you don't know."},
                        {"role": "user", "content": ctx + "\n\nQUESTION: " + QUERY}])
                    a = (r.choices[0].message.content or "").lower()
                    ok += int(ANSWER in a)
                except Exception as e:
                    a = f"ERR {str(e)[:40]}"
            cells.append(f"{ok}/{args.needles}")
            if ok < args.needles:
                all_ok = False
        print(f"{size:>9} | " + " ".join(f"{c:<6}" for c in cells))
        if all_ok:
            reliable = size
    print(f"\nreliable working-set size (all depths recalled): "
          f"{reliable if reliable else '< smallest tested'} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
