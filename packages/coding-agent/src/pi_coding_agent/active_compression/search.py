"""Query-scoped retrieval over a cached CCR original (Headroom Phase-4 BM25).

When the model retrieves a compressed payload *with a query*, it should get back
only the relevant subset of the original — not the full thing. This is the piece
that lets CCR keep the peak token count low on tasks where the model needs a small
slice of a large tool output (e.g. "the edges touching node X" out of a 2000-line
graph), instead of expanding everything.

We split the original into its natural *items* (the same unit the compressor
samples): JSON-array elements when the payload is a JSON array of objects,
otherwise lines. We score items against the query with Okapi BM25 (dependency-free)
and return the top matches, preserving original order so structure is readable.
"""

from __future__ import annotations

import json
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# BM25 free parameters (Okapi defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _split_items(original: str) -> tuple[list[str], bool]:
    """Split a cached original into searchable items.

    Returns (items, is_json) — items are the rendered text of each unit. JSON
    arrays of objects split per element; everything else splits per line.
    """
    stripped = original.strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
        return [json.dumps(x, ensure_ascii=False) for x in parsed], True
    return stripped.splitlines(), False


def _bm25_scores(items: list[str], query: str) -> list[float]:
    """Okapi BM25 score of each item against the query (idf * tf saturation)."""
    q_terms = _tokenize(query)
    if not q_terms:
        return [0.0] * len(items)

    doc_tokens = [_tokenize(it) for it in items]
    doc_len = [len(d) for d in doc_tokens]
    n = len(items)
    avgdl = (sum(doc_len) / n) if n else 0.0

    # Document frequency per query term.
    q_set = set(q_terms)
    df: dict[str, int] = {t: 0 for t in q_set}
    for toks in doc_tokens:
        seen = set(toks)
        for t in q_set:
            if t in seen:
                df[t] += 1

    idf: dict[str, float] = {}
    for t in q_set:
        # Okapi idf with +1 smoothing to stay non-negative.
        idf[t] = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))

    scores: list[float] = []
    for toks, dl in zip(doc_tokens, doc_len):
        if not toks:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for w in toks:
            if w in q_set:
                tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for t, f in tf.items():
            denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * (dl / avgdl if avgdl else 0.0))
            s += idf[t] * (f * (_BM25_K1 + 1)) / (denom or 1.0)
        scores.append(s)
    return scores


def search_original(original: str, query: str, *, max_items: int = 40) -> dict:
    """Return the BM25-relevant subset of a cached original for `query`.

    Result dict:
      text        rendered subset (items in original order), ready for the model
      total_items number of items in the original
      kept_items  number of items returned
      is_json     whether items were JSON elements (vs lines)
    Items with a zero score are dropped; if nothing matches, returns an empty
    subset so the caller can fall back to a full retrieve.
    """
    items, is_json = _split_items(original)
    total = len(items)
    if total == 0:
        return {"text": "", "total_items": 0, "kept_items": 0, "is_json": is_json}

    scores = _bm25_scores(items, query)
    ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)
    chosen = [i for i in ranked[:max_items] if scores[i] > 0.0]
    chosen.sort()  # restore original order for readability

    if is_json:
        body = "[" + ", ".join(items[i] for i in chosen) + "]"
    else:
        body = "\n".join(items[i] for i in chosen)

    return {
        "text": body,
        "total_items": total,
        "kept_items": len(chosen),
        "is_json": is_json,
    }
