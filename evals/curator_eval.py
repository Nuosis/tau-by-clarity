#!/usr/bin/env python3
"""
Live curator eval: real-model extraction quality for Tau memory.

Claim tested:
  Given grounded coding evidence, the curator should auto-commit durable
  memories for user/tool-supported facts and avoid auto-committing assistant-only
  claims.

This is intentionally not a pytest unit test. It crosses the live provider
boundary and writes an artifact under evals/results/.

Usage:
  PYTHONPATH=packages/ai/src:packages/coding-agent/src \
    uv run python evals/curator_eval.py --provider minimax --model MiniMax-M3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "ai" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "coding-agent" / "src"))

from pi_ai import Context, SimpleStreamOptions, UserMessage, complete_simple, get_model  # noqa: E402
from pi_coding_agent.config import get_auth_path  # noqa: E402
from pi_coding_agent.core.auth_storage import AuthStorage  # noqa: E402
from pi_coding_agent.core.memory import (  # noqa: E402
    Curator,
    DeterministicEmbeddingProvider,
    Evidence,
    MemoryStore,
)


SCENARIO_ID = "memory-curator-grounded-coding-atoms"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(p for p in parts if p)


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            "e1",
            "user_turn",
            "Decision: use SQLite for Tau's project-local memory store, not Postgres, "
            "because Tau is a local CLI and should work without a server.",
        ),
        Evidence(
            "e2",
            "tool_result",
            "packages/coding-agent/src/pi_coding_agent/config.py contains "
            "MAX_RECONNECT_ATTEMPTS = 7741.",
        ),
        Evidence(
            "e3",
            "assistant_output",
            "I think we should probably add a Redis cache later.",
        ),
    ]


def _score(decisions: list[Any], written_count: int) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    auto = [d for d in decisions if d.verdict == "auto_commit"]
    active_auto = [d for d in auto if set(d.source_ids) <= {"e1", "e2"}]
    assistant_sourced_auto = [d for d in auto if "e3" in set(d.source_ids)]

    text = "\n".join(f"{d.title}\n{d.content}\n{d.key}" for d in decisions).lower()
    has_sqlite_decision = "sqlite" in text and ("postgres" in text or "local cli" in text)
    has_reconnect_fact = "7741" in text or "max_reconnect_attempts" in text

    if not has_sqlite_decision:
        failures.append("missing SQLite-vs-Postgres decision memory")
    if not has_reconnect_fact:
        failures.append("missing MAX_RECONNECT_ATTEMPTS file fact")
    if assistant_sourced_auto:
        failures.append("assistant-only evidence was auto-committed")
    if written_count < 1:
        failures.append("no active memories were committed")
    if not active_auto:
        failures.append("no grounded user/tool memory was auto-committed")

    metrics = {
        "decision_count": len(decisions),
        "auto_commit_count": len(auto),
        "grounded_auto_commit_count": len(active_auto),
        "assistant_sourced_auto_commit_count": len(assistant_sourced_auto),
        "written_count": written_count,
        "has_sqlite_decision": has_sqlite_decision,
        "has_reconnect_fact": has_reconnect_fact,
    }
    return not failures, failures, metrics


async def _run(args: argparse.Namespace) -> int:
    _load_dotenv(ROOT / ".env")

    model = get_model(args.provider, args.model)
    if model is None:
        print(f"Unknown model: {args.provider}/{args.model}", file=sys.stderr)
        return 2
    api_key = AuthStorage.create(get_auth_path()).resolve_api_key(args.provider)
    if not api_key:
        print(
            f"Missing API key for provider {args.provider!r}. "
            "Run tau /login, set the provider env var, or add it to .env.",
            file=sys.stderr,
        )
        return 2

    async def llm(system: str, user: str) -> str:
        result = await complete_simple(
            model,
            Context(system_prompt=system, messages=[UserMessage(content=user, timestamp=0)]),
            SimpleStreamOptions(temperature=0, max_tokens=args.max_tokens, api_key=api_key),
        )
        return _message_text(result)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="tau-curator-eval-") as tempdir:
        store = MemoryStore(tempdir, embedder=DeterministicEmbeddingProvider())
        curator = Curator(store=store, allm_fn=llm, provenance=SCENARIO_ID)
        decisions = await curator.acurate(_evidence())
        written = curator.commit(decisions)

        ok, failures, metrics = _score(decisions, len(written))
        artifact = {
            "scenario_id": SCENARIO_ID,
            "status": "pass" if ok else "fail",
            "provider": args.provider,
            "model": args.model,
            "duration_seconds": round(time.time() - started, 3),
            "metrics": metrics,
            "failures": failures,
            "decisions": [
                {
                    "title": d.title,
                    "content": d.content,
                    "memory_type": d.memory_type,
                    "key": d.key,
                    "source_ids": d.source_ids,
                    "verdict": d.verdict,
                    "confidence": d.confidence,
                    "rationale": d.rationale,
                }
                for d in decisions
            ],
            "written_ids": written,
        }

    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = results_dir / f"{SCENARIO_ID}-{int(started)}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "scenario_id": SCENARIO_ID,
        "status": artifact["status"],
        "artifact": str(artifact_path),
        "metrics": metrics,
        "failures": failures,
    }, indent=2))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=os.environ.get("TAU_CURATOR_EVAL_PROVIDER", "minimax"))
    parser.add_argument("--model", default=os.environ.get("TAU_CURATOR_EVAL_MODEL", "MiniMax-M3"))
    parser.add_argument("--max-tokens", type=int, default=1200)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
