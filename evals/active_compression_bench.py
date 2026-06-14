#!/usr/bin/env python3
"""Benchmark: does active compression deliver as advertised?

Measures, on representative payloads, the Headroom-style claims:
  - large token savings on structured tool output,
  - error/anomaly items ALWAYS preserved in the compressed output,
  - full reversibility (the original is byte-exact recoverable from the CCR), and
  - the harness rehydration trigger restores a non-kept needle when referenced.

    python evals/active_compression_bench.py [path/to/big_text.txt]

No cloud needed; pure local compressor + CCR.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages", "coding-agent", "src"))

from pi_coding_agent.active_compression.ccr import CCRStore  # noqa: E402
from pi_coding_agent.active_compression.compressor import compress  # noqa: E402
from pi_coding_agent.active_compression.extension import extension_factory  # noqa: E402


def _toks(s: str) -> int:
    return max(1, len(s) // 4)


def _row(out: dict) -> None:
    print(f"  {out['name']:<26} {out['before']:>9} → {out['after']:>7} tok   "
          f"{out['ratio']:>5.1f}% saved   {out['note']}")


def bench_json_tool_output(store: CCRStore) -> dict:
    # 1000-row API result; a planted FATAL error row and a needle in a normal row.
    rows = [{"id": i, "status": 200, "name": f"svc-{i}", "latency_ms": 12 + (i % 30)} for i in range(1000)]
    rows[137] = {"id": 137, "status": 503, "error": "FATAL: upstream timeout at shard 137"}
    rows[642] = {"id": 642, "status": 200, "name": "svc-642", "secret_token": "NEEDLE-MARLIN-9"}
    original = json.dumps(rows)
    out = compress(original, store)
    handle = out[len("[CCR:"):].split("]")[0]
    return {
        "name": "JSON tool output (1000)",
        "before": _toks(original), "after": _toks(out),
        "ratio": 100 * (1 - _toks(out) / _toks(original)),
        "note": ("error kept ✓ " if "FATAL: upstream timeout" in out else "ERROR DROPPED ✗ ")
                + ("reversible ✓" if store.get(handle) == original else "NOT REVERSIBLE ✗"),
        "_handle": handle, "_original": original, "_compressed": out,
    }


def bench_logs(store: CCRStore) -> dict:
    lines = [f"2026-06-13T10:00:{i:02d} INFO request {i} ok" for i in range(300)]
    lines[211] = "2026-06-13T10:03:31 ERROR Traceback: ValueError in handler X"
    original = "\n".join(lines)
    out = compress(original, store)
    return {
        "name": "Build/test log (300 ln)",
        "before": _toks(original), "after": _toks(out),
        "ratio": 100 * (1 - _toks(out) / _toks(original)),
        "note": "error line kept ✓" if "ValueError in handler X" in out else "ERROR LINE DROPPED ✗",
    }


def bench_big_text(store: CCRStore, path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    original = open(path, encoding="utf-8", errors="replace").read()[:60000]
    if _toks(original) < 200:
        return None
    out = compress(original, store)
    handle = out[len("[CCR:"):].split("]")[0]
    return {
        "name": f"big text ({os.path.basename(path)})",
        "before": _toks(original), "after": _toks(out),
        "ratio": 100 * (1 - _toks(out) / _toks(original)),
        "note": "reversible ✓" if store.get(handle) == original else "NOT REVERSIBLE ✗",
    }


async def bench_rehydration(store: CCRStore, json_result: dict) -> str:
    """The needle is in a NON-kept row → absent from the compressed output, but the
    harness restores it in place when the model references the handle."""
    handle, original, compressed = json_result["_handle"], json_result["_original"], json_result["_compressed"]
    needle = "NEEDLE-MARLIN-9"
    in_compressed = needle in compressed

    # Wire the extension against this store and run its context hook.
    import pi_coding_agent.active_compression as ac
    ac._store = store

    class _Pi:
        def __init__(self): self.hooks = []
        def register_tool(self, **k): pass
        def on(self, ev, h): self.hooks.append(h) if ev == "context" else None

    pi = _Pi(); extension_factory(pi)
    messages = [
        {"role": "toolResult", "content": [{"type": "text", "text": compressed}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"I need ccr_retrieve {handle}."}]},
    ]
    await pi.hooks[0]({"messages": messages}, None)
    restored = needle in messages[0]["content"][0]["text"]
    return (f"needle in compressed: {in_compressed} (expected False) | "
            f"recovered after reference: {restored} (expected True) | "
            f"exact-restore: {messages[0]['content'][0]['text'] == original}")


def main() -> int:
    big = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evals", "fixtures", "corpus", "dense_agent.txt")
    with tempfile.TemporaryDirectory() as d:
        store = CCRStore(os.path.join(d, "ccr.db"))
        print("Active compression benchmark\n" + "-" * 64)
        jr = bench_json_tool_output(store)
        _row(jr)
        _row(bench_logs(store))
        bt = bench_big_text(store, big)
        if bt:
            _row(bt)
        print("-" * 64)
        print("Rehydration (needle in a dropped row):")
        print("  " + asyncio.run(bench_rehydration(store, jr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
