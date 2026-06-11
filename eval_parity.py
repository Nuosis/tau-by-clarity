#!/usr/bin/env python3
"""Golden-transcript functional-parity eval: Python pi -p  vs  Node pi -p.

Runs the SAME scripted prompts through both engines in `--mode json` headless
mode against the SAME provider/model (MiniMax-M3 by default), in identical
fresh sandbox dirs, then compares *functional* behavior — not prose, which is
non-deterministic with a live model.

What it compares (all reproducible despite model nondeterminism):
  - exit code / did the run complete
  - the set & ordering of top-level event `type`s
  - whether `message_update` events carry streaming payloads (Node does)
  - whether reasoning is separated from output, or <think> tags leak into text
  - which built-in TOOLS were invoked (by name)
  - GROUND TRUTH: file-system side effects in the sandbox (authoritative —
    independent of the model's wording)

Usage:
  python3 eval_parity.py                 # run all scenarios, print matrix
  python3 eval_parity.py --json out.json # also dump machine-readable results
  python3 eval_parity.py --only read     # run one scenario by id
  python3 eval_parity.py --runs 2        # repeat each scenario N times

Requires: MINIMAX_API_KEY in pi-mono-python/.env (last non-empty wins),
and the Node engine built at packages/coding-agent/dist/cli.js.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PY_ROOT = Path("/Users/marcusswift/cli/pi-mono-python")
NODE_CLI = Path("/Users/marcusswift/cli/pi-mono-node-reference/packages/coding-agent/dist/cli.js")

PROVIDER = "minimax"
MODEL = "MiniMax-M3"

# Non-deterministic / cosmetic keys stripped before any structural comparison.
VOLATILE = {"id", "timestamp", "responseId", "thinkingSignature", "usage",
            "cost", "totalTokens", "cacheRead", "cacheWrite", "cwd", "version",
            "input", "output", "index", "contentIndex"}


def minimax_key() -> str:
    """Resolve the MiniMax key the same way pi does: stored auth.json first,
    then the .env fallback. So the eval keeps working after .env is removed."""
    # 1. pi key store (~/.pi/agent/auth.json) — priority over env, as in pi
    auth = Path.home() / ".pi/agent/auth.json"
    if auth.exists():
        try:
            entry = json.loads(auth.read_text()).get(PROVIDER, {})
            key = entry.get("key") or entry.get("api_key")
            if key:
                return key.strip()
        except (json.JSONDecodeError, OSError):
            pass
    # 2. .env fallback (last non-empty MINIMAX_API_KEY wins)
    env = PY_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MINIMAX_API_KEY=") and len(line.split("=", 1)[1]) > 10:
                return line.split("=", 1)[1].strip()
    raise SystemExit("No MiniMax key in ~/.pi/agent/auth.json or .env")


# ---------------------------------------------------------------------------
# Scenario definitions. Each is deterministic-by-construction: the prompt
# strongly constrains which tool must run and what the file system should look
# like afterwards, so we can assert ground truth regardless of model wording.
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    prompt: str
    setup: Callable[[Path], None] = lambda d: None
    expect_tools: tuple[str, ...] = ()          # tool names that should appear
    check_fs: Callable[[Path], bool] = lambda d: True
    fs_desc: str = "n/a"


def _seed(d: Path, name: str, content: str) -> None:
    (d / name).write_text(content)


SCENARIOS: list[Scenario] = [
    Scenario(
        id="echo",
        prompt="Reply with exactly the word PONG and nothing else.",
        fs_desc="no fs change",
    ),
    Scenario(
        id="read",
        prompt="Use the read tool to read the file notes.txt, then reply with "
               "ONLY its exact contents.",
        setup=lambda d: _seed(d, "notes.txt", "BANANA-42"),
        expect_tools=("read",),
        fs_desc="no fs change (read only)",
    ),
    Scenario(
        id="write",
        prompt="Use the write tool to create a file named result.txt whose "
               "exact contents are the text OK123 (no trailing newline).",
        expect_tools=("write",),
        check_fs=lambda d: (d / "result.txt").exists()
        and "OK123" in (d / "result.txt").read_text(),
        fs_desc="result.txt contains OK123",
    ),
    Scenario(
        id="edit",
        prompt="Use the edit tool to change the word OLD to NEW in config.txt.",
        setup=lambda d: _seed(d, "config.txt", "value=OLD\n"),
        expect_tools=("edit",),
        check_fs=lambda d: (d / "config.txt").exists()
        and "NEW" in (d / "config.txt").read_text()
        and "OLD" not in (d / "config.txt").read_text(),
        fs_desc="config.txt OLD->NEW",
    ),
    Scenario(
        id="bash",
        prompt="Use the bash tool to run exactly: echo PARITY_OK . Then reply "
               "with the command's output.",
        expect_tools=("bash",),
        fs_desc="no fs change",
    ),
]


# ---------------------------------------------------------------------------
# Run + parse
# ---------------------------------------------------------------------------

TOOL_BLOCK_TYPES = {"toolcall", "tool_call", "tooluse", "tool_use", "toolresult", "tool_result"}


def run_engine(engine: str, scenario: Scenario, key: str) -> dict:
    """Run one scenario through one engine in a fresh sandbox; return parsed result."""
    sandbox = Path(tempfile.mkdtemp(prefix=f"parity_{scenario.id}_{engine}_"))
    try:
        scenario.setup(sandbox)
        # Run BOTH engines with cwd=sandbox so file tools operate on identical
        # ground truth. Python uses the installed venv console script directly
        # (NOT `uv run --directory`, which would change cwd to the repo root).
        if engine == "python":
            cmd = [str(PY_ROOT / ".venv/bin/pi")]
        else:
            cmd = ["node", str(NODE_CLI)]
        cmd += ["-p", "--mode", "json", "--no-session", "--offline",
                "--provider", PROVIDER, "--model", MODEL, "--api-key", key,
                scenario.prompt]
        env = dict(os.environ, MINIMAX_API_KEY=key)
        proc = subprocess.run(cmd, cwd=sandbox, env=env, capture_output=True,
                              text=True, timeout=180)
        events = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return parse_run(events, proc, sandbox, scenario)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _walk_content(obj: Any, out_tools: set[str], texts: list[str], thinks: list[str]) -> None:
    """Recursively collect tool names, text blocks, and thinking blocks."""
    if isinstance(obj, dict):
        t = str(obj.get("type", "")).lower()
        if t in TOOL_BLOCK_TYPES:
            name = obj.get("name") or obj.get("toolName") or obj.get("tool")
            if name:
                out_tools.add(str(name))
        if t == "text" and isinstance(obj.get("text"), str):
            texts.append(obj["text"])
        if t == "thinking" and isinstance(obj.get("thinking"), str):
            thinks.append(obj["thinking"])
        for v in obj.values():
            _walk_content(v, out_tools, texts, thinks)
    elif isinstance(obj, list):
        for v in obj:
            _walk_content(v, out_tools, texts, thinks)


def parse_run(events: list[dict], proc, sandbox: Path, scenario: Scenario) -> dict:
    types = [e.get("type") for e in events]
    tools: set[str] = set()
    texts: list[str] = []
    thinks: list[str] = []
    _walk_content(events, tools, texts, thinks)
    # tool_execution_start events name the tool directly (both engines)
    for e in events:
        if e.get("type", "").startswith("tool_execution") and e.get("toolName"):
            tools.add(str(e["toolName"]))

    # streaming richness: do message_update events carry a payload beyond {type}?
    updates = [e for e in events if e.get("type") == "message_update"]
    rich_updates = sum(1 for e in updates if len(e) > 1)

    final_text = ""
    # final assistant text = last non-empty text block from a message_end/agent_end
    for e in events:
        if e.get("type") in ("message_end", "turn_end"):
            t2: list[str] = []
            _walk_content(e, set(), t2, [])
            if t2:
                final_text = t2[-1]
    if not final_text and texts:
        final_text = texts[-1]

    think_leak = bool(re.search(r"<think>|</think>", final_text, re.I))
    thinking_separated = len(thinks) > 0

    return {
        "exit": proc.returncode,
        "event_types": types,
        "type_histogram": {t: types.count(t) for t in dict.fromkeys(types)},
        "tools_invoked": sorted(tools),
        "n_updates": len(updates),
        "rich_updates": rich_updates,
        "thinking_separated": thinking_separated,
        "think_leak_in_final": think_leak,
        "final_text": final_text[:200],
        "fs_ok": bool(scenario.check_fs(sandbox)),
        "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr.strip() else [],
    }


# ---------------------------------------------------------------------------
# Compare & report
# ---------------------------------------------------------------------------

def compare(scn: Scenario, py: dict, nd: dict) -> dict:
    exp = set(scn.expect_tools)
    return {
        "scenario": scn.id,
        "both_completed": py["exit"] == 0 and nd["exit"] == 0,
        "py_tools_ok": exp.issubset(set(py["tools_invoked"])) if exp else None,
        "node_tools_ok": exp.issubset(set(nd["tools_invoked"])) if exp else None,
        "py_fs_ok": py["fs_ok"],
        "node_fs_ok": nd["fs_ok"],
        "event_types_match": set(py["type_histogram"]) == set(nd["type_histogram"]),
        "py_streaming_rich": py["rich_updates"] > 0,
        "node_streaming_rich": nd["rich_updates"] > 0,
        "py_think_leak": py["think_leak_in_final"],
        "node_think_leak": nd["think_leak_in_final"],
        "py_thinking_separated": py["thinking_separated"],
        "node_thinking_separated": nd["thinking_separated"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write machine-readable results here")
    ap.add_argument("--only", help="run a single scenario id")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    if not NODE_CLI.exists():
        raise SystemExit(f"Node engine not built: {NODE_CLI}\nRun: cd ../pi-mono-node-reference && npm install && npm run build")

    key = minimax_key()
    scns = [s for s in SCENARIOS if not args.only or s.id == args.only]
    results: list[dict] = []
    raw: dict[str, Any] = {}

    for scn in scns:
        for r in range(args.runs):
            tag = scn.id if args.runs == 1 else f"{scn.id}#{r+1}"
            print(f"… running {tag}  (python + node)", flush=True)
            py = run_engine("python", scn, key)
            nd = run_engine("node", scn, key)
            cmp = compare(scn, py, nd)
            cmp["tag"] = tag
            results.append(cmp)
            raw[tag] = {"python": py, "node": nd, "comparison": cmp}

    print("\n" + "=" * 96)
    print("  FUNCTIONAL PARITY EVAL  —  Python pi -p   vs   Node pi -p   "
          f"(provider={PROVIDER} model={MODEL})")
    print("=" * 96)
    hdr = f"{'scenario':<10}{'done':<6}{'tools py/nd':<14}{'fs py/nd':<10}{'evt=':<6}{'stream py/nd':<14}{'think-leak py/nd':<18}"
    print(hdr)
    print("-" * 96)

    def yn(v):
        return "-" if v is None else ("ok" if v else "XX")

    for c in results:
        tools = f"{yn(c['py_tools_ok'])}/{yn(c['node_tools_ok'])}"
        fs = f"{yn(c['py_fs_ok'])}/{yn(c['node_fs_ok'])}"
        stream = f"{yn(c['py_streaming_rich'])}/{yn(c['node_streaming_rich'])}"
        # for think-leak, "ok" means NO leak (good)
        leak = f"{yn(not c['py_think_leak'])}/{yn(not c['node_think_leak'])}"
        print(f"{c['tag']:<10}{yn(c['both_completed']):<6}{tools:<14}{fs:<10}"
              f"{yn(c['event_types_match']):<6}{stream:<14}{leak:<18}")

    print("-" * 96)
    print("legend: tools/fs/stream/think-leak shown as python/node. "
          "ok=pass XX=fail -=n/a. think-leak ok = NO <think> in final text.")

    # headline deltas
    print("\nKEY FINDINGS")
    any_py_no_stream = any(not c["py_streaming_rich"] and c["node_streaming_rich"] for c in results)
    any_py_leak = any(c["py_think_leak"] and not c["node_think_leak"] for c in results)
    any_evt_mismatch = any(not c["event_types_match"] for c in results)
    any_tool_gap = any(c["py_tools_ok"] is False and c["node_tools_ok"] for c in results)
    any_fs_gap = any((not c["py_fs_ok"]) and c["node_fs_ok"] for c in results)
    print(f"  - Python emits empty message_update (no streaming payload): "
          f"{'YES — headless consumers get no progress' if any_py_no_stream else 'no'}")
    print(f"  - <think> tags leak into Python final text (Node strips): "
          f"{'YES — polluted headless output' if any_py_leak else 'no'}")
    print(f"  - top-level event schema differs: {'YES' if any_evt_mismatch else 'no'}")
    print(f"  - Python missed a tool call Node made: {'YES' if any_tool_gap else 'no'}")
    print(f"  - Python file side-effect failed where Node succeeded: {'YES' if any_fs_gap else 'no'}")

    if args.json:
        Path(args.json).write_text(json.dumps(raw, indent=2))
        print(f"\nraw results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
