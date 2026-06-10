#!/usr/bin/env python3
"""
Fetch long agent-run traces from the clarity Langfuse (ClickHouse) and write them
as Tier-0 session fixtures.

Each qualifying trace is one accumulating-context agent run. We take the
observation with the largest `input` (the most-accumulated message array — i.e.
the full transcript at the agent's last LLM call), normalise its messages, and
write a v3-style session JSONL that evals/tier0_context_replay.py can replay with
--per-message-turns.

Access is over SSH to the clarity host, querying the langfuse ClickHouse container.
Credentials are read from the container env at runtime (not hard-coded here).

    python evals/langfuse_fetch.py --limit 21
    python evals/tier0_context_replay.py "evals/fixtures/langfuse/*.jsonl" \
        --per-message-turns --floor 0 --frontier 3

Read-only against Langfuse. Writes only local fixtures.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess

SSH_HOST = "clarity"
CH_CONTAINER = "clarity_backend_langfuse_clickhouse"
OUT_DIR = "evals/fixtures/langfuse"


def _ch(query: str) -> str:
    """Run a ClickHouse query in the container over SSH; return stdout."""
    # creds live in the container env; fetch the password indirectly so it never
    # lands in our argv/logs.
    remote = (
        f'PW=$(docker exec {CH_CONTAINER} printenv CLICKHOUSE_PASSWORD); '
        f'docker exec {CH_CONTAINER} clickhouse-client '
        f'--user clickhouse --password "$PW" -q {json.dumps(query)}'
    )
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", SSH_HOST, remote],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"clickhouse query failed: {out.stderr.strip()[:500]}")
    return out.stdout


def long_trace_ids(limit: int, min_obs: int, min_in_calls: int) -> list[str]:
    q = (
        "SELECT trace_id FROM observations GROUP BY trace_id "
        f"HAVING count() >= {min_obs} AND countIf(length(input) > 2) >= {min_in_calls} "
        f"ORDER BY count() DESC LIMIT {limit}"
    )
    return [ln for ln in _ch(q).splitlines() if ln.strip()]


def largest_input(trace_id: str) -> dict | None:
    """Return the parsed {model, messages:[...]} of the most-accumulated call."""
    q = (
        "SELECT base64Encode(input) FROM observations "
        f"WHERE trace_id = '{trace_id}' AND length(input) > 2 "
        "ORDER BY length(input) DESC LIMIT 1"
    )
    b64 = _ch(q).strip()
    if not b64:
        return None
    try:
        return json.loads(base64.b64decode(b64).decode("utf-8", "replace"))
    except (ValueError, json.JSONDecodeError):
        return None


def flatten_message(m: dict) -> tuple[str, str]:
    """Normalise an OpenAI/MiniMax-shaped message to (role, text)."""
    role = m.get("role", "?")
    parts: list[str] = []
    content = m.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or json.dumps(b)[:400])
            elif isinstance(b, str):
                parts.append(b)
    for tc in m.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        parts.append(f"[tool_call {fn.get('name', '?')}({str(fn.get('arguments', ''))[:300]})]")
    if role == "tool" and m.get("name"):
        parts.insert(0, f"[tool_result {m.get('name')}]")
    return role, "\n".join(p for p in parts if p)


def write_session(trace_id: str, payload: dict) -> tuple[str, int] | None:
    messages = payload.get("messages") or []
    if len(messages) < 4:
        return None
    path = os.path.join(OUT_DIR, f"{trace_id}.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({
            "type": "session", "id": trace_id, "version": 3,
            "timestamp": "1970-01-01T00:00:00+00:00",
            "cwd": "langfuse", "model": payload.get("model", ""),
        }) + "\n")
        for i, m in enumerate(messages):
            role, text = flatten_message(m)
            if not text:
                continue
            fh.write(json.dumps({
                "id": f"m{i}", "type": "message", "timestamp": i,
                "parentId": f"m{i-1}" if i else None,
                "message": {"role": role,
                            "content": [{"type": "text", "text": text}]},
            }) + "\n")
    return path, len(messages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=21)
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-in-calls", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ids = long_trace_ids(args.limit, args.min_obs, args.min_in_calls)
    print(f"found {len(ids)} qualifying traces")
    written = 0
    for tid in ids:
        payload = largest_input(tid)
        if not payload:
            print(f"  {tid[:16]}… no parseable input — skip")
            continue
        res = write_session(tid, payload)
        if not res:
            print(f"  {tid[:16]}… <4 messages — skip")
            continue
        path, n = res
        print(f"  {tid[:16]}… {n:>3} messages -> {path}")
        written += 1
    print(f"wrote {written} fixtures to {OUT_DIR}/")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
