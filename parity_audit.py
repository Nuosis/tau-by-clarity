#!/usr/bin/env python3
"""Deterministic parity audit: pi-mono-python  vs  pi-mono-node-reference.

Compares the two monorepos along *named surfaces* whose identifiers are
string-identical across languages (slash commands, CLI flags, built-in tools,
providers, RPC methods, env keys, modules, model IDs). These are exactly the
places where a "missing feature" lives, so the diff is meaningful and
reproducible.

It does NOT judge behavioral fidelity (whether a tool/stream/TUI behaves the
same) — that needs runtime evals, not static comparison.

Usage:
    python3 parity_audit.py            # human-readable report
    python3 parity_audit.py --json     # machine-readable
    python3 parity_audit.py --strict   # exit 1 if any node-only surface found
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PY_ROOT = Path("/Users/marcusswift/cli/pi-mono-python")
TS_ROOT = Path("/Users/marcusswift/cli/pi-mono-node-reference")

PY_PKG = PY_ROOT / "packages"
TS_PKG = TS_ROOT / "packages"


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def find_all(pattern: str, text: str, group: int = 1) -> set[str]:
    return {m.group(group) for m in re.finditer(pattern, text)}


# ---------------------------------------------------------------------------
# Surface extractors. Each returns a set of identifier strings.
# ---------------------------------------------------------------------------

def slash_py() -> set[str]:
    t = read(PY_PKG / "coding-agent/src/pi_coding_agent/core/slash_commands.py")
    return find_all(r'BuiltinSlashCommand\(\s*"([a-z][a-z0-9-]*)"', t)


def slash_ts() -> set[str]:
    t = read(TS_PKG / "coding-agent/src/core/slash-commands.ts")
    return find_all(r'name:\s*"([a-z][a-z0-9-]*)"', t)


def flags_py() -> set[str]:
    # main.py / cli help text carries the canonical flag list
    t = read(PY_PKG / "coding-agent/src/pi_coding_agent/main.py")
    t += read(PY_PKG / "coding-agent/src/pi_coding_agent/cli_sub/args.py")
    t += read(PY_PKG / "coding-agent/src/pi_coding_agent/cli.py")
    return find_all(r'(--[a-z][a-z0-9-]+)', t)


def flags_ts() -> set[str]:
    t = read(TS_PKG / "coding-agent/src/cli/args.ts")
    t += read(TS_PKG / "coding-agent/src/cli.ts")
    t += read(TS_PKG / "coding-agent/src/main.ts")
    return find_all(r'(--[a-z][a-z0-9-]+)', t)


def tool_names_py() -> set[str]:
    names: set[str] = set()
    for f in (PY_PKG / "coding-agent/src/pi_coding_agent/core/tools").glob("*.py"):
        names |= find_all(r'(?:name|label)\s*=\s*"([a-z_][a-z0-9_]*)"', read(f))
    return names


def tool_names_ts() -> set[str]:
    names: set[str] = set()
    for f in (TS_PKG / "coding-agent/src/core/tools").glob("*.ts"):
        names |= find_all(r'(?:name|label):\s*"([a-z_][a-z0-9_]*)"', read(f))
    return names


def provider_files_py() -> set[str]:
    return {norm(f.stem) for f in (PY_PKG / "ai/src/pi_ai/providers").glob("*.py")
            if f.stem not in {"__init__"}}


def provider_files_ts() -> set[str]:
    return {norm(f.stem) for f in (TS_PKG / "ai/src/providers").glob("*.ts")}


def env_keys_py() -> set[str]:
    t = read(PY_PKG / "ai/src/pi_ai/env_api_keys.py")
    return find_all(r'"([A-Z][A-Z0-9_]{3,})"', t)


def env_keys_ts() -> set[str]:
    t = read(TS_PKG / "ai/src/env-api-keys.ts")
    return find_all(r'"([A-Z][A-Z0-9_]{3,})"', t)


def rpc_py() -> set[str]:
    t = read(PY_PKG / "coding-agent/src/pi_coding_agent/modes/rpc/types.py")
    t += read(PY_PKG / "coding-agent/src/pi_coding_agent/modes/rpc/mode.py")
    return find_all(r'"(method|[a-z]+/[a-z]+)"', t)


def rpc_ts() -> set[str]:
    t = read(TS_PKG / "coding-agent/src/modes/rpc/rpc-types.ts")
    t += read(TS_PKG / "coding-agent/src/modes/rpc/rpc-mode.ts")
    return find_all(r'"([a-z]+/[a-z]+)"', t)


def model_count_py() -> int:
    t = read(PY_PKG / "ai/src/pi_ai/models_generated.py")
    return len(find_all(r"""id=['"]([^'"]+)['"]""", t))


def model_count_ts() -> int:
    t = read(TS_PKG / "ai/src/models.generated.ts")
    return len(find_all(r'id:\s*"([^"]+)"', t))


# ---------------------------------------------------------------------------
# Module-map (normalized name) comparison per package.
# ---------------------------------------------------------------------------

def norm(stem: str) -> str:
    """Normalize a file stem to a language-neutral key."""
    return stem.replace("-", "").replace("_", "").lower()


# TS subtree -> PY subtree per package (src roots)
PKG_MAP = {
    "ai": ("ai/src", "ai/src/pi_ai"),
    "agent": ("agent/src", "agent/src/pi_agent"),
    "tui": ("tui/src", "tui/src/pi_tui"),
    "coding-agent": ("coding-agent/src", "coding-agent/src/pi_coding_agent"),
}

# Files that are language-runtime plumbing, not features — excluded from
# "missing module" noise.
TS_IGNORE = {
    "index", "node", "nodejs", "bun", "cli",  # barrels / runtime entry
    "typeboxhelpers", "abortsignals", "headers",  # TS-only typing/runtime helpers
}
PY_IGNORE = {"init", "main"}


def module_keys(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for f in root.rglob("*"):
        if f.suffix in {".ts", ".py"} and not f.name.endswith((".d.ts", ".test.ts")):
            key = norm(f.stem)
            if key in {"index", "init"}:
                # keep barrel awareness but key by parent dir
                key = norm(f.parent.name) + "/index"
            out[key] = str(f.relative_to(root))
    return out


def global_py_keys() -> set[str]:
    """All python module keys across the whole monorepo.

    Used so a node module that was *relocated* to a different python package
    (e.g. agent/harness/* -> coding-agent/core/*) is not falsely flagged.
    """
    keys: set[str] = set()
    for f in PY_PKG.rglob("*.py"):
        if "/.venv/" in str(f) or "/tests/" in str(f):
            continue
        keys.add(norm(f.stem))
    return keys


def module_delta(ts_sub: str, py_sub: str, py_global: set[str]) -> dict:
    ts_keys = module_keys(TS_PKG / ts_sub)
    py_keys = module_keys(PY_PKG / py_sub)
    # node-only: name absent from the ENTIRE python repo (not just this package)
    ts_only = sorted(k for k in ts_keys
                     if k.split("/")[0] not in py_global
                     and k.split("/")[0] not in TS_IGNORE
                     and not k.endswith("/index"))
    py_only = sorted(k for k in py_keys if k not in ts_keys and k.split("/")[0] not in PY_IGNORE
                     and not k.endswith("/index"))
    return {
        "ts_total": len(ts_keys),
        "py_total": len(py_keys),
        "ts_only": [ts_keys[k] for k in ts_only],
        "py_only": [py_keys[k] for k in py_only],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def diff_set(label: str, py: set[str], ts: set[str]) -> dict:
    return {
        "surface": label,
        "missing_in_python": sorted(ts - py),  # node has, python lacks
        "extra_in_python": sorted(py - ts),    # python has, node lacks
        "shared": len(py & ts),
        "py_count": len(py),
        "ts_count": len(ts),
    }


def stub_markers() -> list[str]:
    hits: list[str] = []
    pat = re.compile(r"NotImplementedError|TODO|FIXME|XXX|# *stub", re.I)
    for f in (PY_PKG).rglob("*.py"):
        if "/tests/" in str(f) or "/.venv/" in str(f):
            continue
        for i, line in enumerate(read(f).splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f.relative_to(PY_ROOT)}:{i}: {line.strip()[:90]}")
    return hits


def build_report() -> dict:
    surfaces = [
        diff_set("slash_commands", slash_py(), slash_ts()),
        diff_set("cli_flags", flags_py(), flags_ts()),
        diff_set("builtin_tools", tool_names_py(), tool_names_ts()),
        diff_set("ai_providers", provider_files_py(), provider_files_ts()),
        diff_set("env_api_keys", env_keys_py(), env_keys_ts()),
    ]
    py_global = global_py_keys()
    modules = {pkg: module_delta(ts, py, py_global) for pkg, (ts, py) in PKG_MAP.items()}
    return {
        "surfaces": surfaces,
        "models": {"python": model_count_py(), "node": model_count_ts()},
        "modules": modules,
        "python_stub_markers": stub_markers(),
    }


def print_report(rep: dict) -> None:
    print("=" * 78)
    print("  PARITY AUDIT  —  pi-mono-python  vs  pi-mono-node-reference")
    print("  (surface-area / structural deltas only; not behavioral fidelity)")
    print("=" * 78)

    print("\n## NAMED SURFACES (identifiers that map 1:1 across languages)\n")
    for s in rep["surfaces"]:
        miss, extra = s["missing_in_python"], s["extra_in_python"]
        flag = "OK " if not miss else "GAP"
        print(f"[{flag}] {s['surface']:<16} shared={s['shared']:<4} "
              f"py={s['py_count']:<4} node={s['ts_count']:<4}")
        if miss:
            print(f"        MISSING in python : {', '.join(miss)}")
        if extra:
            print(f"        extra in python  : {', '.join(extra)}")

    m = rep["models"]
    print(f"\n## MODEL REGISTRY   python={m['python']}  node={m['node']}  "
          f"(delta {m['node'] - m['python']:+d})")

    print("\n## MODULE COVERAGE (normalized name; barrels/runtime plumbing ignored)\n")
    for pkg, d in rep["modules"].items():
        print(f"### {pkg}  (node files={d['ts_total']}  python files={d['py_total']})")
        if d["ts_only"]:
            print(f"    node-only modules ({len(d['ts_only'])}) — candidate missing features:")
            for x in d["ts_only"]:
                print(f"        - {x}")
        if d["py_only"]:
            print(f"    python-only modules ({len(d['py_only'])}):")
            for x in d["py_only"]:
                print(f"        + {x}")
        if not d["ts_only"] and not d["py_only"]:
            print("    full module-name parity")
        print()

    stubs = rep["python_stub_markers"]
    print(f"## PYTHON STUB / TODO MARKERS  ({len(stubs)})")
    for h in stubs:
        print(f"    {h}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    rep = build_report()
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print_report(rep)
    if args.strict:
        gaps = any(s["missing_in_python"] for s in rep["surfaces"])
        gaps = gaps or any(d["ts_only"] for d in rep["modules"].values())
        return 1 if gaps else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
