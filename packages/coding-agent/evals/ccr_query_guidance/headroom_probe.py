#!/usr/bin/env python3
"""Probe real Headroom behavior against CCR hard-query scenarios.

This script intentionally bypasses Tau's legacy local active-compression package.
It answers two questions separately:

1. Public compression: does ``headroom.compress`` compress these payloads, and if
   so does the compressed message still contain expected evidence?
2. CCR search: if the original is stored in Headroom's CompressionStore, do
   Headroom's own BM25 search results contain the evidence the scenario requires?

It is a behavior probe, not a Tau integration test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCENARIOS = Path(__file__).with_name("scenarios.json")


def _load_headroom() -> tuple[Any, Any, Any]:
    try:
        import headroom
        from headroom.cache.compression_store import CompressionStore
        from headroom.compress import CompressConfig
    except Exception as exc:  # pragma: no cover - diagnostic CLI path
        raise SystemExit(
            "headroom-ai is required for this probe. Install it in the active "
            f"environment first. Import error: {exc}"
        ) from exc
    return headroom, CompressConfig, CompressionStore


def _make_compressible_payload(scenario: dict[str, Any]) -> str:
    payload = str(scenario["payload"])
    prefix = "\n".join(f"noise_prefix_{i:03d}: unrelated background row" for i in range(80))
    distractors: list[str] = []
    for i in range(int(scenario.get("distractor_repeat", 1))):
        for line in scenario.get("distractor_lines", []):
            distractors.append(f"distractor_{i:03d}: {line}")
    suffix = "\n".join(f"noise_suffix_{i:03d}: unrelated trailing row" for i in range(30))
    return (
        f"{prefix}\n"
        f"{chr(10).join(distractors)}\n"
        f"--- TARGET DATA START ---\n{payload}\n--- TARGET DATA END ---\n"
        f"{suffix}\n"
    )


def _terms_for(scenario: dict[str, Any]) -> list[str]:
    return list(
        scenario.get("expected_retrieval_terms")
        or scenario.get("expected_query_terms")
        or scenario.get("expected_answer_terms")
        or []
    )


def _queries_for(scenario: dict[str, Any]) -> list[str]:
    sequence = list(scenario.get("expected_query_sequence") or [])
    if sequence:
        return sequence
    terms = list(scenario.get("expected_query_terms") or [])
    return [" ".join(terms)] if terms else [str(scenario["task"])]


def _contains_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def _probe_public_compress(
    headroom: Any,
    CompressConfig: Any,
    *,
    scenario: dict[str, Any],
    payload: str,
    tool_name: str,
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": str(scenario["task"])},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "probe-1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "probe-1", "content": payload},
        {"role": "user", "content": "Use the tool result above to answer the task."},
    ]
    result = headroom.compress(
        messages,
        config=CompressConfig(protect_recent=0, kompress_model="disabled"),
    )
    compressed_content = ""
    for message in result.messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == "probe-1":
            compressed_content = str(message.get("content", ""))
            break
    terms = _terms_for(scenario)
    matched = _contains_terms(compressed_content, terms)
    return {
        "tool_name": tool_name,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "compression_ratio": result.compression_ratio,
        "transforms_applied": list(result.transforms_applied),
        "matched_terms": matched,
        "term_coverage": (len(matched) / len(terms)) if terms else 1.0,
    }


def _probe_ccr_search(
    CompressionStore: Any,
    *,
    scenario: dict[str, Any],
    payload: str,
    score_threshold: float,
) -> dict[str, Any]:
    store = CompressionStore(default_ttl=3600)
    key = store.store(
        original=payload,
        compressed="[headroom-probe compressed placeholder]",
        original_tokens=max(1, len(payload) // 4),
        compressed_tokens=4,
        original_item_count=len(payload.splitlines()),
        compressed_item_count=1,
        tool_name="headroom_probe",
        tool_call_id="probe-1",
        query_context=str(scenario["task"]),
        compression_strategy="probe",
    )

    terms = _terms_for(scenario)
    query_reports: list[dict[str, Any]] = []
    combined = ""
    for query in _queries_for(scenario):
        results = store.search(key, query, max_results=20, score_threshold=score_threshold)
        text = json.dumps(results, ensure_ascii=False)
        combined += "\n" + text
        query_reports.append(
            {
                "query": query,
                "result_count": len(results),
                "matched_terms": _contains_terms(text, terms),
                "preview": text[:600],
            }
        )

    matched = _contains_terms(combined, terms)
    return {
        "queries": query_reports,
        "matched_terms": matched,
        "term_coverage": (len(matched) / len(terms)) if terms else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", help="Scenario id to run. Default: all.")
    parser.add_argument("--tool-name", action="append", default=["read", "bash"])
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    headroom, CompressConfig, CompressionStore = _load_headroom()
    scenarios = json.loads(SCENARIOS.read_text())
    wanted = set(args.scenario or [])
    if wanted:
        scenarios = [scenario for scenario in scenarios if scenario["id"] in wanted]

    report: list[dict[str, Any]] = []
    for scenario in scenarios:
        payload = _make_compressible_payload(scenario)
        public = [
            _probe_public_compress(
                headroom,
                CompressConfig,
                scenario=scenario,
                payload=payload,
                tool_name=tool_name,
            )
            for tool_name in args.tool_name
        ]
        ccr = _probe_ccr_search(
            CompressionStore,
            scenario=scenario,
            payload=payload,
            score_threshold=args.score_threshold,
        )
        report.append(
            {
                "scenario_id": scenario["id"],
                "task": scenario["task"],
                "expected_terms": _terms_for(scenario),
                "public_compress": public,
                "ccr_search": ccr,
            }
        )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    for item in report:
        print(f"\n{item['scenario_id']}: {item['task']}")
        for row in item["public_compress"]:
            print(
                "  public "
                f"tool={row['tool_name']:<5} saved={row['tokens_saved']:<5} "
                f"after={row['tokens_after']:<6} transforms={row['transforms_applied']} "
                f"terms={len(row['matched_terms'])}/{len(item['expected_terms'])}"
            )
        ccr = item["ccr_search"]
        print(
            "  ccr_search "
            f"terms={len(ccr['matched_terms'])}/{len(item['expected_terms'])} "
            f"coverage={ccr['term_coverage']:.2f}"
        )
        for query in ccr["queries"]:
            print(
                f"    query={query['query']!r} results={query['result_count']} "
                f"terms={len(query['matched_terms'])}/{len(item['expected_terms'])}"
            )

    failed = [
        item["scenario_id"]
        for item in report
        if item["ccr_search"]["term_coverage"] < 1.0
    ]
    if failed:
        print(f"\nCCR search missing expected terms for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
