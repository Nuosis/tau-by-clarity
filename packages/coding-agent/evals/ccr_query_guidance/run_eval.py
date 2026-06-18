#!/usr/bin/env python3
"""Live CCR query-guidance evals.

Purpose: measure whether a model can choose useful `ccr_retrieve.query` values
from ordinary compressed-search tasks. This is deliberately broader than any one
benchmark: code lookup, logs, JSON records, docs, config, notes, traces, and mixed
noise.

The eval captures both retrieval behavior and final-answer coverage:
- first query
- all queries
- whether expected task-specific terms appear in any query
- whether expected query targets appear in order
- whether expected answer terms appear in the final answer
- number of retrieval calls
- final status/output
- raw session path for evidence
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[4]
SCENARIOS = pathlib.Path(__file__).with_name("scenarios.json")
TAU = ROOT / ".venv" / "bin" / "tau"
PY = ROOT / ".venv" / "bin" / "python"
CCR_EXTENSION = ROOT / "packages" / "coding-agent" / "src" / "pi_coding_agent" / "active_compression" / "extension.py"


@dataclass
class RunResult:
    scenario_id: str
    expected_query_terms: list[str]
    expected_query_sequence: list[str]
    expected_answer_terms: list[str]
    queries: list[str]
    retrievals: list[dict[str, Any]]
    final_answer: str
    first_query: str | None
    matched_expected_term: bool
    matched_expected_sequence: bool
    matched_answer_terms: list[str]
    answer_term_coverage: float
    query_passed: bool
    answer_passed: bool
    passed: bool
    retrieval_calls: int
    returncode: int
    stdout_tail: str
    stderr_tail: str
    session_file: str | None


def _make_compressible_payload(payload: str) -> str:
    """Bury the scenario in ordinary noise so the model must retrieve it.

    Active compression keeps line heads/tails. Putting the target payload in the
    middle prevents the answer from being visible in the marker while preserving a
    realistic large-tool-output shape.
    """
    prefix = "\n".join(f"noise_prefix_{i:03d}: unrelated background row" for i in range(80))
    suffix = "\n".join(f"noise_suffix_{i:03d}: unrelated trailing row" for i in range(30))
    return f"{prefix}\n--- TARGET DATA START ---\n{payload}\n--- TARGET DATA END ---\n{suffix}\n"


def _compress_payload(agent_dir: pathlib.Path, payload: str) -> str:
    code = """
import os, sys
from pi_coding_agent.active_compression import compress
text = sys.stdin.read()
print(compress(text))
"""
    env = os.environ.copy()
    env["PI_AGENT_DIR"] = str(agent_dir)
    env["PYTHONPATH"] = ":".join([
        str(ROOT / "packages" / "coding-agent" / "src"),
        str(ROOT / "packages" / "agent" / "src"),
        str(ROOT / "packages" / "ai" / "src"),
        env.get("PYTHONPATH", ""),
    ])
    proc = subprocess.run(
        [str(PY), "-c", code],
        input=_make_compressible_payload(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()


def _text_blocks(msg: dict[str, Any]) -> str:
    return "\n".join(
        block.get("text", "")
        for block in (msg.get("content", []) or [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _extract_retrievals(session_file: pathlib.Path) -> list[dict[str, Any]]:
    """Extract ccr_retrieve calls and their exact tool-result text from a session."""
    retrievals: list[dict[str, Any]] = []
    pending_by_id: dict[str, dict[str, Any]] = {}
    if not session_file.exists():
        return retrievals
    for line_no, line in enumerate(session_file.read_text(errors="replace").splitlines(), 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msg = obj.get("message", {})
        role = msg.get("role")
        if role == "assistant":
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall" and block.get("name") == "ccr_retrieve":
                    args = block.get("arguments") or {}
                    q = args.get("query")
                    rec = {
                        "line": line_no,
                        "tool_call_id": block.get("id"),
                        "handle": args.get("handle"),
                        "query": q if isinstance(q, str) else None,
                        "result_text": None,
                        "result_preview": None,
                        "details": None,
                    }
                    retrievals.append(rec)
                    if rec["tool_call_id"]:
                        pending_by_id[str(rec["tool_call_id"])] = rec
        elif role == "toolResult" and msg.get("tool_name") == "ccr_retrieve":
            rec = pending_by_id.get(str(msg.get("tool_call_id")))
            if rec is not None:
                text = _text_blocks(msg)
                rec["result_text"] = text
                rec["result_preview"] = text[:1200]
                rec["details"] = msg.get("details")
    return retrievals


def _extract_final_answer(session_file: pathlib.Path) -> str:
    """Return the last assistant text response that was not solely a tool call."""
    final = ""
    if not session_file.exists():
        return final
    for line in session_file.read_text(errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msg = obj.get("message", {})
        if msg.get("role") != "assistant":
            continue
        text = _text_blocks(msg).strip()
        if text:
            final = text
    return final


def _matched_expected(queries: list[str], terms: list[str]) -> bool:
    q_blob = "\n".join(queries).lower()
    for term in terms:
        if term.lower() in q_blob:
            return True
    return False


def _matched_sequence(queries: list[str], sequence: list[str]) -> bool:
    """Whether expected retrieval targets appear in order across the query stream."""
    if not sequence:
        return True
    idx = 0
    for q in queries:
        ql = q.lower()
        while idx < len(sequence) and sequence[idx].lower() in ql:
            idx += 1
            if idx >= len(sequence):
                return True
    return False


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    blob = text.lower()
    return [term for term in terms if term.lower() in blob]


def run_one(scenario: dict[str, Any], args: argparse.Namespace, workdir: pathlib.Path) -> RunResult:
    sid = scenario["id"]
    agent_dir = workdir / sid / "agent"
    session_dir = workdir / sid / "sessions"
    agent_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    marker = _compress_payload(agent_dir, scenario["payload"])
    session_id = str(uuid.uuid4())
    prompt = f"""You are answering a search task from a compressed payload.

Task: {scenario['task']}

Compressed payload:
{marker}

Use ccr_retrieve when you need evidence from the compressed payload. The payload is not available through files or shell commands in this eval; use ccr_retrieve for payload evidence. Final answer should be concise.
"""

    env = os.environ.copy()
    env["PI_AGENT_DIR"] = str(agent_dir)
    env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
    env["OPENAI_COMPATIBLE_BASE_URL"] = args.base_url
    env["PYTHONPATH"] = ":".join([
        str(ROOT / "packages" / "coding-agent" / "src"),
        str(ROOT / "packages" / "agent" / "src"),
        str(ROOT / "packages" / "ai" / "src"),
        str(ROOT / "packages" / "tui" / "src"),
        env.get("PYTHONPATH", ""),
    ])

    cmd = [
        str(TAU),
        "--provider", "openai-compatible",
        "--model", args.model,
        "--api-key", args.api_key,
        "--mode", "json",
        "--extension", str(CCR_EXTENSION),
        "--exclude-tools", "bash,edit,find,grep,ls,read,write",
        "--session-dir", str(session_dir),
        "--session-id", session_id,
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--print",
        prompt,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=args.timeout)
    session_file = session_dir / f"{session_id}.jsonl"
    retrievals = _extract_retrievals(session_file)
    final_answer = _extract_final_answer(session_file)
    queries = [r["query"] for r in retrievals if isinstance(r.get("query"), str)]
    expected_terms = list(scenario.get("expected_query_terms", []))
    expected_sequence = list(scenario.get("expected_query_sequence", []))
    expected_answer_terms = list(scenario.get("expected_answer_terms", expected_terms))
    matched_term = _matched_expected(queries, expected_terms)
    matched_sequence = _matched_sequence(queries, expected_sequence)
    matched_answer_terms = _matched_terms(final_answer, expected_answer_terms)
    answer_term_coverage = (
        len(matched_answer_terms) / len(expected_answer_terms) if expected_answer_terms else 1.0
    )
    query_passed = matched_sequence if expected_sequence else matched_term
    answer_threshold = float(scenario.get("answer_term_threshold", 1.0))
    answer_passed = answer_term_coverage >= answer_threshold
    passed = answer_passed if expected_answer_terms else query_passed
    return RunResult(
        scenario_id=sid,
        expected_query_terms=expected_terms,
        expected_query_sequence=expected_sequence,
        expected_answer_terms=expected_answer_terms,
        queries=queries,
        retrievals=retrievals,
        final_answer=final_answer,
        first_query=queries[0] if queries else None,
        matched_expected_term=matched_term,
        matched_expected_sequence=matched_sequence,
        matched_answer_terms=matched_answer_terms,
        answer_term_coverage=answer_term_coverage,
        query_passed=query_passed,
        answer_passed=answer_passed,
        passed=passed,
        retrieval_calls=len(queries),
        returncode=proc.returncode,
        stdout_tail=proc.stdout[-1200:],
        stderr_tail=proc.stderr[-1200:],
        session_file=str(session_file) if session_file.exists() else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:2222/v1"))
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="limit scenarios for quick iteration")
    ap.add_argument("--ids", default="", help="comma-separated scenario ids to run")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()

    scenarios = json.loads(SCENARIOS.read_text())
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s.get("id") in wanted]
    if args.limit:
        scenarios = scenarios[: args.limit]

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ccr-query-eval-"))
    results: list[RunResult] = []
    try:
        for sc in scenarios:
            print(f"running {sc['id']}...", flush=True)
            res = run_one(sc, args, workdir)
            results.append(res)
            status = "PASS" if res.passed else "FAIL"
            seq = f" sequence={res.expected_query_sequence}" if res.expected_query_sequence else ""
            print(f"  {status} calls={res.retrieval_calls} first={res.first_query!r}{seq} queries={res.queries}", flush=True)

        report = {
            "workdir": str(workdir),
            "model": args.model,
            "passed": sum(1 for r in results if r.passed),
            "total": len(results),
            "results": [r.__dict__ for r in results],
        }
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({"passed": report["passed"], "total": report["total"], "workdir": str(workdir)}, indent=2))
        return 0 if report["passed"] == report["total"] else 1
    finally:
        if not args.keep_workdir:
            # Keep automatically when report path is absent and failures occurred? No: the
            # report carries copied session paths only if --keep-workdir is used.
            if not args.out:
                print(f"workdir removed; rerun with --keep-workdir to inspect sessions: {workdir}", file=sys.stderr)
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
