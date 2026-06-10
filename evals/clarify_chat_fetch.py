#!/usr/bin/env python3
"""
Fetch Claire's chat conversations from the clarify Postgres (agent_chat_messages)
and write them as Tier-0 session fixtures — the low-recency / human-dialogue regime.

These are genuine multi-turn human<->Claire conversations with topic pivots and
cross-conversation recall (the regime the fidelity gradient is meant to help).
Replay WITHOUT --per-message-turns (from_type already gives real user/agent turns).

    python evals/clarify_chat_fetch.py --min-msgs 12
    python evals/tier0_context_replay.py "evals/fixtures/clarify_chat/*.jsonl" \
        --floor 0 --frontier 3

Read-only against clarify. Writes only local fixtures (kept out of git — real
customer/business content).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess

SSH_HOST = "clarity"
OUT_DIR = "evals/fixtures/clarify_chat"


def _psql(sql: str) -> str:
    remote = f'docker exec supabase-db psql -U postgres -At -c {json.dumps(sql)}'
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", SSH_HOST, remote],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"psql failed: {out.stderr.strip()[:500]}")
    return out.stdout


def session_ids(min_msgs: int) -> list[str]:
    sql = (
        "SELECT session_id FROM agent_chat_messages GROUP BY session_id "
        f"HAVING count(*) >= {min_msgs} ORDER BY count(*) DESC"
    )
    return [ln for ln in _psql(sql).splitlines() if ln.strip()]


def fetch_messages(session_id: str) -> list[tuple[str, str]]:
    """Return ordered (role, text). base64 the message so newlines survive psql."""
    sid = session_id.replace("'", "''")
    # base64 (newlines stripped — PG wraps at 76 chars) keeps message text on one
    # line; '|' is safe as a separator (absent from base64 alphabet and from_type).
    sql = (
        "SELECT from_type || '|' || "
        "translate(encode(convert_to(message,'UTF8'),'base64'), E'\\n', '') "
        f"FROM agent_chat_messages WHERE session_id = '{sid}' ORDER BY created_at"
    )
    rows: list[tuple[str, str]] = []
    for ln in _psql(sql).splitlines():
        if "|" not in ln:
            continue
        ft, b64 = ln.split("|", 1)
        try:
            text = base64.b64decode(b64).decode("utf-8", "replace")
        except ValueError:
            continue
        role = "user" if ft.lower() in ("user", "human", "customer") else "assistant"
        rows.append((role, text))
    return rows


def write_session(session_id: str, rows: list[tuple[str, str]]) -> tuple[str, int] | None:
    if len(rows) < 4:
        return None
    path = os.path.join(OUT_DIR, f"{session_id}.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "session", "id": session_id, "version": 3,
                             "timestamp": "1970-01-01T00:00:00+00:00",
                             "cwd": "clarify_chat"}) + "\n")
        for i, (role, text) in enumerate(rows):
            if not text.strip():
                continue
            fh.write(json.dumps({
                "id": f"m{i}", "type": "message", "timestamp": i,
                "parentId": f"m{i-1}" if i else None,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
            }) + "\n")
    return path, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-msgs", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ids = session_ids(args.min_msgs)
    print(f"found {len(ids)} chat sessions with >= {args.min_msgs} messages")
    written = 0
    for sid in ids:
        rows = fetch_messages(sid)
        res = write_session(sid, rows)
        if not res:
            continue
        path, n = res
        print(f"  {sid[:16]}… {n:>3} turns -> {path}")
        written += 1
    print(f"wrote {written} fixtures to {OUT_DIR}/")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
