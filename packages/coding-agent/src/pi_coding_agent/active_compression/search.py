"""Query-scoped retrieval over a cached CCR original (Headroom Phase-4 BM25).

When the model retrieves a compressed payload *with a query*, it should get back
only the relevant subset of the original — not the full thing. This is the piece
that lets CCR keep the peak token count low on tasks where the model needs a small
slice of a large tool output.

The retriever is intentionally deterministic. It does two conservative things to
make search useful without turning the model into a search engineer:

1. BM25 ranks natural items (JSON-array elements or text lines).
2. For line-based payloads, matched lines are returned with a small surrounding
   context window, because a single matching line is often too thin to reveal the
   next query term (code bodies, config aliases, records, logs, etc.).
3. Retry once with high-signal tokens extracted from the query (IDs, constants,
   underscored/dashed terms, long rare terms) and merge any matches. This handles
   noisy model queries without any domain-specific rules.
"""

from __future__ import annotations

import json
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*")

# BM25 free parameters (Okapi defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75

# Conservative line context. Enough to reveal nearby relationship/body lines while
# keeping returned slices small.
_LINE_WINDOW_BEFORE = 1
_LINE_WINDOW_AFTER = 5
_MAX_RETURN_LINES = 80
_MAX_FALLBACK_TERMS = 4

_STOPWORDS = {
    "a", "an", "and", "api", "are", "as", "at", "body", "by", "call", "called",
    "current", "data", "error", "external", "file", "find", "for", "from", "get",
    "how", "in", "is", "it", "line", "lookup", "need", "of", "on", "or", "path",
    "payload", "provider", "request", "result", "return", "search", "status", "task",
    "the", "this", "to", "value", "what", "when", "where", "who", "why", "with",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        tokens.append(lowered)
        # Keep identifiers searchable both as exact symbols and as natural words.
        # Example: "escalation owner" should match "escalation_owner", while
        # "FEATURE_RECALL_ENABLED" and "stripe.Invoice.create" still retain their
        # exact-token evidence value.
        if any(sep in lowered for sep in ("_", "-", ".")):
            tokens.extend(part for part in re.split(r"[_\-.]+", lowered) if part)
    return tokens


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


def _ranked_matches(items: list[str], query: str, max_items: int) -> tuple[list[int], list[float]]:
    scores = _bm25_scores(items, query)
    ranked = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    chosen = [i for i in ranked[:max_items] if scores[i] > 0.0]
    return chosen, scores


def _signal_score(token: str) -> tuple[int, int, str]:
    """Rank query tokens by deterministic evidence value.

    Prefer exact-looking identifiers over prose: constants, IDs, underscored/dashed
    terms, dotted symbols, mixed alnum, and longer tokens.
    """
    t = token.strip("'\"`.,:;()[]{}")
    tl = t.lower()
    if not t or tl in _STOPWORDS or len(t) < 3:
        return (0, 0, t)
    score = 0
    if any(c.isdigit() for c in t):
        score += 5
    if any(c in t for c in "_-.:"):
        score += 4
    if t.upper() == t and any(c.isalpha() for c in t):
        score += 4
    if re.search(r"[a-z][A-Z]", t):
        score += 3
    if len(t) >= 8:
        score += 2
    if _IDENTIFIER_RE.fullmatch(t):
        score += 1
    return (score, len(t), t)


def _fallback_query(query: str) -> str:
    tokens = _TOKEN_RE.findall(query)
    ranked = sorted((_signal_score(t) for t in tokens), reverse=True)
    kept: list[str] = []
    seen: set[str] = set()
    for score, _length, token in ranked:
        if score <= 0:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
        if len(kept) >= _MAX_FALLBACK_TERMS:
            break
    return " ".join(reversed(kept))


def _expand_line_windows(indices: list[int], total: int) -> list[int]:
    expanded: set[int] = set()
    for i in indices:
        start = max(0, i - _LINE_WINDOW_BEFORE)
        end = min(total, i + _LINE_WINDOW_AFTER + 1)
        for j in range(start, end):
            expanded.add(j)
            if len(expanded) >= _MAX_RETURN_LINES:
                break
        if len(expanded) >= _MAX_RETURN_LINES:
            break
    return sorted(expanded)


def search_original(original: str, query: str, *, max_items: int = 40) -> dict:
    """Return the BM25-relevant subset of a cached original for `query`.

    Result dict:
      text             rendered subset, ready for the model
      total_items      number of items in the original
      kept_items       number of returned items/lines/elements
      matched_items    number of direct BM25 matches before window expansion
      is_json          whether items were JSON elements (vs lines)
      effective_query  original query, annotated with deterministic fallback when used
      fallback_used    whether high-signal fallback matches were merged
    """
    items, is_json = _split_items(original)
    total = len(items)
    if total == 0:
        return {
            "text": "",
            "total_items": 0,
            "kept_items": 0,
            "matched_items": 0,
            "is_json": is_json,
            "effective_query": query,
            "fallback_used": False,
        }

    chosen, _scores = _ranked_matches(items, query, max_items)
    effective_query = query
    fallback_used = False

    fallback = _fallback_query(query)
    if fallback and fallback.strip().lower() != query.strip().lower():
        fallback_chosen, _fallback_scores = _ranked_matches(items, fallback, max_items)
        if fallback_chosen:
            # Deterministically de-noise model queries. BM25 is OR-like: a noisy
            # query can match generic words and miss the best identifier-centered
            # slice. Add the high-signal retry matches to the candidate set so the
            # result includes evidence around the distinctive task values.
            chosen = sorted(set(chosen) | set(fallback_chosen))
            effective_query = fallback if not chosen else f"{query} | fallback:{fallback}"
            fallback_used = True

    matched_items = len(chosen)
    chosen.sort()  # stable basis for JSON and for line window expansion

    if is_json:
        kept = chosen
        body = "[" + ", ".join(items[i] for i in kept) + "]"
    else:
        kept = _expand_line_windows(chosen, total)
        body = "\n".join(items[i] for i in kept)

    return {
        "text": body,
        "total_items": total,
        "kept_items": len(kept),
        "matched_items": matched_items,
        "is_json": is_json,
        "effective_query": effective_query,
        "fallback_used": fallback_used,
    }
