"""Content-aware compressor (Smart-Crusher-lite, §12 / Headroom architecture).

Compresses large tool-output payloads by content type, ALWAYS preserving error
items, and caches the original in the CCR store so it stays retrievable. Lossy in
the prompt, reversible via CCR. Short payloads pass through untouched.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .ccr import CCRStore

MIN_TOKENS = 200  # below this, compression overhead exceeds the savings (pass through)
DEFAULT_ITEM_BUDGET = 15
_ERROR_LINE_RE = re.compile(r"error|fail|exception|traceback|fatal|critical", re.IGNORECASE)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _is_error_item(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        lk = str(k).lower()
        if lk in ("error", "errors", "exception", "traceback") and v:
            return True
        if lk in ("status", "status_code", "code"):
            try:
                if int(v) >= 400:
                    return True
            except (TypeError, ValueError):
                pass
        if lk in ("ok", "success", "passed") and v is False:
            return True
        if lk in ("level", "severity") and str(v).lower() in ("error", "fatal", "critical"):
            return True
    return False


def _crush_list_of_dicts(items: list, budget: int = DEFAULT_ITEM_BUDGET) -> tuple[list, int, int]:
    """Keep head (schema), tail (recency), an evenly-spaced importance sample, and
    ALWAYS every error item. Returns (kept, total, error_count)."""
    n = len(items)
    keep_idx: set[int] = set()
    head = max(1, int(budget * 0.30))
    tail = max(1, int(budget * 0.15))
    for i in range(min(head, n)):
        keep_idx.add(i)
    for i in range(max(0, n - tail), n):
        keep_idx.add(i)
    remaining = budget - len(keep_idx)
    if remaining > 0 and n > budget:
        step = max(1, n // remaining)
        for i in range(0, n, step):
            if len(keep_idx) >= budget:
                break
            keep_idx.add(i)
    kept = [items[i] for i in sorted(keep_idx)]
    errors = [x for x in items if _is_error_item(x)]
    for e in errors:
        if e not in kept:
            kept.append(e)
    return kept, n, len(errors)


_CCR_MARKER_RE = re.compile(r"\[CCR:([0-9a-f]{12})\]")


def compress(text: str, ccr: CCRStore) -> str:
    """Compress a tool-output string; cache the original in CCR. No-op if small.

    Phase-4 (Context Tracker) idempotence guards, in order:
    - Pass through our own already-compressed output (text carrying a [CCR:..] marker),
      so we never double-compress or churn handles.
    - Pass through content whose handle is already marked expanded — once the model
      has retrieved an original, re-compressing it would make retrieval futile and
      drive the read→retrieve→re-elide thrash loop.
    """
    if not text or _approx_tokens(text) < MIN_TOKENS:
        return text

    # Guard 1: don't re-compress our own compressed output.
    if _CCR_MARKER_RE.search(text):
        return text

    # Guard 2: don't re-compress an original the model has already expanded.
    if ccr.is_expanded(CCRStore.handle_for(text)):
        return text

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None

    # JSON array of dicts → statistical sample + anomaly (error) preservation.
    if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
        kept, n, n_err = _crush_list_of_dicts(parsed)
        if len(kept) >= n:
            return text
        handle = ccr.put(text)
        body = json.dumps(kept, ensure_ascii=False)
        return (
            f"[CCR:{handle}] compressed JSON array: {n} items → {len(kept)} kept "
            f"({n_err} error item(s) preserved). Retrieve a relevant subset with "
            f"ccr_retrieve(handle={handle}, query=<what you need>) — query required, "
            f"returns only matching items.\n{body}"
        )

    lines = stripped.splitlines()

    # Logs → keep head/tail + every error line.
    if len(lines) > 30:
        handle = ccr.put(text)
        errs = [ln for ln in lines if _ERROR_LINE_RE.search(ln)]
        keep = lines[:10] + [f"… ({max(0, len(lines) - 15)} lines elided) …"] + lines[-5:]
        if errs:
            keep += ["— preserved error lines —", *errs[:20]]
        body = "\n".join(keep)
        return (
            f"[CCR:{handle}] compressed log: {len(lines)} lines → head+tail+"
            f"{len(errs)} error line(s). Retrieve a relevant subset with "
            f"ccr_retrieve(handle={handle}, query=<what you need>) — query required, "
            f"returns only matching items.\n{body}"
        )

    # Generic large text → head + tail.
    handle = ccr.put(text)
    head, tail = stripped[:1200], stripped[-600:]
    return (
        f"[CCR:{handle}] compressed text ({len(stripped)} chars). "
        f"Retrieve relevant items with ccr_retrieve(handle={handle}, query=<what you need>) "
        f"— query required, returns only matching items.\n{head}\n… elided …\n{tail}"
    )
