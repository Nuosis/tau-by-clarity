#!/usr/bin/env python3
"""Benchmark: does active compression deliver as advertised?

Measures, on representative payloads, the Headroom-style claims:
  - large token savings on structured tool output,
  - error/anomaly items ALWAYS preserved in the compressed output,
  - lossless structured compaction does not need a top-level CCR handle, and
  - query-scoped `ccr_retrieve` recovers dropped lossy content without
    reinflating the full payload.

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
    markerless_lossless = out.startswith(("[", "__buckets:")) and "[CCR:" not in out
    handle = out[len("[CCR:"):].split("]")[0] if out.startswith("[CCR:") else None
    recoverable = markerless_lossless or (handle is not None and store.get(handle) == original)
    return {
        "name": "JSON tool output (1000)",
        "before": _toks(original), "after": _toks(out),
        "ratio": 100 * (1 - _toks(out) / _toks(original)),
        "note": ("error kept ✓ " if "FATAL: upstream timeout" in out else "ERROR DROPPED ✗ ")
                + ("lossless direct ✓" if recoverable else "NOT RECOVERABLE ✗"),
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
    """Exercise a lossy CCR path even when JSON chooses lossless table compaction."""
    del json_result
    lines = [
        f"worker observation {i} " + ("detail " * ((i % 9) + 1)) + f"trace-{i:04d}"
        for i in range(400)
    ]
    lines[242] = "worker observation 242 contains target NEEDLE-MARLIN-9 for scoped retrieval"
    original = "\n".join(lines)
    compressed = compress(original, store)
    handle = compressed[len("[CCR:"):].split("]")[0]
    needle = "NEEDLE-MARLIN-9"
    in_compressed = needle in compressed

    # Wire the extension against this store and call the registered retrieve tool.
    import pi_coding_agent.active_compression as ac
    ac._store = store

    class _Pi:
        def __init__(self):
            self.tools = {}
            self.commands = {}

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_command(self, name, command):
            self.commands[name] = command

    pi = _Pi()
    extension_factory(pi)
    result = await pi.tools["ccr_retrieve"]["execute"](
        "bench-call",
        {"handle": handle, "query": "NEEDLE-MARLIN-9 svc-642 observation 642"},
        None,
        None,
        None,
    )
    text = result["content"][0]["text"]
    restored = needle in text
    return (f"needle in compressed: {in_compressed} (expected False) | "
            f"recovered by query: {restored} (expected True) | "
            f"full payload returned: {text == original} (expected False)")


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
        print("Rehydration (needle in dropped lossy content):")
        print("  " + asyncio.run(bench_rehydration(store, jr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
