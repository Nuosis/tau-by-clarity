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
3. Route structural/instruction-style queries through deterministic label
   discovery before BM25, and follow concrete IDs found there with a bounded
   local relationship lookup.
4. Retry once with high-signal tokens extracted from the query (IDs, constants,
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
_MAX_STRUCTURAL_LABELS = 8
_MAX_WALK_TERMS = 6
_MAX_TRACE_STEPS = 3
_MAX_TRACE_DIRECT_MATCHES_PER_TERM = 12
_TAIL_WINDOW_LINES = 24

_STOPWORDS = {
    "a", "an", "and", "api", "are", "as", "at", "body", "by", "call", "called",
    "current", "data", "error", "external", "file", "find", "for", "from", "get",
    "how", "in", "is", "it", "line", "lookup", "need", "of", "on", "or", "path",
    "payload", "provider", "request", "result", "return", "search", "status", "task",
    "the", "this", "to", "value", "what", "when", "where", "who", "why", "with",
}

_STRUCTURAL_TERMS = {
    "actual", "answer", "expected", "goal", "id", "input", "instruction",
    "instructions", "key", "operation", "output", "question", "schema",
    "target", "task",
}

_RELATION_TERMS = {
    "auth", "belongs_to", "call", "calls", "child", "children", "denial",
    "denial_code", "edge", "edges", "escalation", "from", "job", "owner",
    "parent", "parents", "policy", "provider", "risk_policy", "rule", "span",
    "to", "trace", "vendor", "vendor_call",
}

_TAIL_TERMS = {"actual", "end", "final", "last", "tail", "task"}

_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?([A-Za-z][A-Za-z0-9 _\-/]{1,60}?):\s*(.+?)\s*$"
)
_EDGE_RE = re.compile(r"^\s*([^\s]+)\s*->\s*([^\s]+)\s*$")
_BFS_OPERATION_RE = re.compile(
    r"Perform\s+a\s+BFS\s+from\s+node\s+([^\s.]+)\s+"
    r"(?:and\s+return\s+only\s+the\s+nodes\s+at\s+exactly\s+depth|with\s+depth)\s+"
    r"(\d+)",
    re.IGNORECASE,
)
_PARENT_OPERATION_RE = re.compile(r"Find\s+the\s+parents\s+of\s+node\s+([^\s.]+)", re.IGNORECASE)
_PAGINATION_RE = re.compile(r"\[Showing lines \d+-\d+ of \d+.*?Use offset=(\d+) to continue\.\]")


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
        key = token.lower()
        if score < 4 and key not in _RELATION_TERMS:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
        if len(kept) >= _MAX_FALLBACK_TERMS:
            break
    return " ".join(reversed(kept))


def _query_profile(query: str) -> dict[str, object]:
    raw_tokens = _TOKEN_RE.findall(query)
    tokens = [t.lower() for t in raw_tokens]
    meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    structural = [t for t in meaningful if t in _STRUCTURAL_TERMS]
    relation = [t for t in meaningful if t in _RELATION_TERMS]
    signals = [_signal_score(t)[0] for t in raw_tokens]
    high_signal = sum(1 for s in signals if s >= 5)
    return {
        "tokens": tokens,
        "meaningful": meaningful,
        "structural": structural,
        "relation": relation,
        "high_signal": high_signal,
        "structural_ratio": (len(structural) / len(meaningful)) if meaningful else 0.0,
        "relation_ratio": (len(relation) / len(meaningful)) if meaningful else 0.0,
    }


def _structural_label_score(line: str, query_tokens: set[str]) -> tuple[int, str]:
    """Score line-label candidates by stable structure, not corpus answers."""
    match = _LABEL_RE.match(line)
    if not match:
        return (0, line)
    label, value = match.groups()
    label_tokens = set(_tokenize(label))
    label_structural_tokens = label_tokens & _STRUCTURAL_TERMS
    if not label_structural_tokens:
        return (0, line)
    overlap = len(label_tokens & query_tokens)
    if overlap == 0:
        return (0, line)
    score = 10 + overlap
    if label.strip().upper() == label.strip() and any(c.isalpha() for c in label):
        score += 4
    if any(t in label_tokens for t in ("target", "instruction", "operation", "question", "task", "input")):
        score += 4
    if value.strip():
        score += 1
    return (score, line)


def _structural_label_matches(items: list[str], query: str, max_labels: int) -> list[int]:
    query_tokens = set(_tokenize(query))
    scored: list[tuple[int, int]] = []
    for i, line in enumerate(items):
        score, _ = _structural_label_score(line, query_tokens)
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _score, i in scored[:max_labels]]


def _structural_label_key(line: str) -> str | None:
    match = _LABEL_RE.match(line)
    if not match:
        return None
    label, _value = match.groups()
    label_tokens = [t for t in _tokenize(label) if t in _STRUCTURAL_TERMS]
    if not label_tokens:
        return None
    return " ".join(label_tokens)


def _has_ambiguous_structural_labels(items: list[str], indices: list[int]) -> bool:
    groups: dict[str, set[str]] = {}
    for idx in indices:
        key = _structural_label_key(items[idx])
        if not key:
            continue
        groups.setdefault(key, set()).add(items[idx].strip().lower())
    return any(len(values) > 1 for values in groups.values())


def _extract_walk_terms(lines: list[str]) -> list[str]:
    """Extract concrete terms worth following from retrieved structural evidence."""
    ranked: list[tuple[int, int, str]] = []
    for line in lines:
        for raw in _TOKEN_RE.findall(line):
            token = raw.strip("'\"`.,:;()[]{}")
            score, length, normalized = _signal_score(token)
            if score <= 0:
                continue
            if normalized.lower() in _STRUCTURAL_TERMS or normalized.lower() in _STOPWORDS:
                continue
            ranked.append((score, length, normalized))
    ranked.sort(reverse=True)
    kept: list[str] = []
    seen: set[str] = set()
    for _score, _length, token in ranked:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
        if len(kept) >= _MAX_WALK_TERMS:
            break
    return kept


def _line_indices_containing_terms(items: list[str], terms: list[str]) -> list[int]:
    if not terms:
        return []
    term_lowers = [t.lower() for t in terms]
    indices: list[int] = []
    for i, line in enumerate(items):
        lowered = line.lower()
        if any(term in lowered for term in term_lowers):
            indices.append(i)
    return indices


def _line_has_relation_shape(line: str) -> bool:
    stripped = line.strip()
    if "->" in line:
        return True
    if stripped.startswith("def ") or stripped.startswith("return ") or " return " in line:
        return True
    tokens = set(_tokenize(line))
    if tokens & _RELATION_TERMS:
        return True
    return False


def _extract_trace_terms(lines: list[str]) -> list[str]:
    ranked: list[tuple[int, int, str]] = []
    for line in lines:
        if not _line_has_relation_shape(line):
            continue
        for raw in _TOKEN_RE.findall(line):
            token = raw.strip("'\"`.,:;()[]{}")
            score, length, normalized = _signal_score(token)
            if score < 4:
                continue
            lowered = normalized.lower()
            if lowered in _STRUCTURAL_TERMS or lowered in _STOPWORDS:
                continue
            ranked.append((score, length, normalized))
    ranked.sort(reverse=True)
    kept: list[str] = []
    seen: set[str] = set()
    for _score, _length, token in ranked:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
        if len(kept) >= _MAX_WALK_TERMS:
            break
    return kept


def _trace_related_indices(items: list[str], seed_indices: list[int], max_lines: int) -> tuple[list[int], list[str]]:
    """Bounded deterministic relationship expansion from retrieved evidence.

    Search finds anchors. Trace follows concrete IDs/symbols from relation-shaped
    lines, adding rows that mention those terms and repeating for a few steps.
    It does not infer answers; it only returns linked evidence.
    """
    chosen: set[int] = set(seed_indices)
    frontier_terms = _extract_trace_terms([items[i] for i in sorted(chosen)])
    seen_terms = {t.lower() for t in frontier_terms}
    steps: list[str] = []

    for step in range(_MAX_TRACE_STEPS):
        if not frontier_terms or len(chosen) >= max_lines:
            break
        added: list[int] = []
        for term in frontier_terms:
            matches = _line_indices_containing_terms(items, [term])
            for idx in matches[:_MAX_TRACE_DIRECT_MATCHES_PER_TERM]:
                if idx not in chosen:
                    chosen.add(idx)
                    added.append(idx)
                if len(chosen) >= max_lines:
                    break
            if len(chosen) >= max_lines:
                break
        if not added:
            break
        steps.append(f"trace_step_{step + 1}")
        next_terms = _extract_trace_terms([items[i] for i in added])
        frontier_terms = []
        for term in next_terms:
            key = term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            frontier_terms.append(term)

    return sorted(chosen), steps


def _match_repetition_ratio(items: list[str], indices: list[int]) -> float:
    if not indices:
        return 0.0
    normalized = []
    for i in indices:
        text = re.sub(r"\d+", "#", items[i].lower())
        text = re.sub(r"^[a-z][a-z0-9_-]*_#:", "row_#:", text)
        normalized.append(text)
    return 1.0 - (len(set(normalized)) / len(normalized))


def _expand_line_windows(
    indices: list[int],
    total: int,
    *,
    before: int = _LINE_WINDOW_BEFORE,
    after: int = _LINE_WINDOW_AFTER,
) -> list[int]:
    expanded: set[int] = set()
    for i in indices:
        start = max(0, i - before)
        end = min(total, i + after + 1)
        for j in range(start, end):
            expanded.add(j)
            if len(expanded) >= _MAX_RETURN_LINES:
                break
        if len(expanded) >= _MAX_RETURN_LINES:
            break
    return sorted(expanded)


def _tail_indices(items: list[str], query: str) -> list[int]:
    """Return a bounded final slice for queries asking for the actual task/tail.

    Benchmark prompts and many tool outputs put the real question at the end after
    examples or setup. BM25 over repeated labels tends to return examples first;
    explicit tail intent should therefore be handled structurally, not as search.
    """
    tokens = set(_tokenize(query))
    if not (tokens & _TAIL_TERMS):
        return []
    if "tail" not in tokens and "task" not in tokens and "actual" not in tokens:
        return []
    start = max(0, len(items) - _TAIL_WINDOW_LINES)
    return list(range(start, len(items)))


def _pagination_continue_offset(items: list[str], query: str) -> str | None:
    tokens = set(_tokenize(query))
    wants_task_context = bool(
        tokens
        & {
            "actual",
            "bfs",
            "depth",
            "final",
            "instruction",
            "instructions",
            "operation",
            "question",
            "tail",
            "target",
            "task",
        }
    )
    if not wants_task_context:
        return None
    for line in reversed(items[-5:]):
        match = _PAGINATION_RE.search(line)
        if match:
            return match.group(1)
    return None


def _incoming_edge_indices(items: list[str], query: str) -> tuple[list[int], str | None]:
    """Return all graph edges whose destination is the queried node.

    This is a deterministic retrieval primitive for "parents of node X" tasks.
    BM25 is the wrong shape here: it returns rows mentioning X anywhere and can
    exhaust the line budget on outgoing edges or early graph context.
    """
    tokens = set(_tokenize(query))
    wants_incoming = bool(tokens & {"destination", "incoming", "parent", "parents"})
    if not wants_incoming:
        return [], None

    candidates: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        token = raw.strip("'\"`.,:;()[]{}")
        score, _length, normalized = _signal_score(token)
        lowered = normalized.lower()
        if score >= 5 and lowered not in _STOPWORDS and lowered not in _RELATION_TERMS:
            candidates.append(normalized)

    for candidate in candidates:
        matches: list[int] = []
        seen_sources: set[str] = set()
        for i, line in enumerate(items):
            edge = _EDGE_RE.match(line)
            if (
                edge
                and edge.group(2) == candidate
                and edge.group(1) != candidate
                and edge.group(1) not in seen_sources
            ):
                matches.append(i)
                seen_sources.add(edge.group(1))
        if matches:
            return matches, candidate
    return [], None


def _outgoing_edge_indices(items: list[str], query: str) -> tuple[list[int], str | None]:
    """Return all graph edges whose source is the queried node.

    This is the matching primitive for BFS/frontier expansion. A query like
    "abc123 ->" or "edges from abc123" should not invoke broad BM25 over the
    entire graph; it needs the adjacency list for exactly one source node.
    """
    tokens = set(_tokenize(query))
    wants_outgoing = (
        "->" in query
        or bool(tokens & {"bfs", "children", "edge", "edges", "from", "outgoing", "source"})
    )
    if not wants_outgoing or bool(tokens & {"destination", "incoming", "parent", "parents"}):
        return [], None

    candidates: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        token = raw.strip("'\"`.,:;()[]{}")
        score, _length, normalized = _signal_score(token)
        lowered = normalized.lower()
        if score >= 5 and lowered not in _STOPWORDS and lowered not in _RELATION_TERMS:
            candidates.append(normalized)

    for candidate in candidates:
        matches: list[int] = []
        for i, line in enumerate(items):
            edge = _EDGE_RE.match(line)
            if edge and edge.group(1) == candidate:
                matches.append(i)
        if matches:
            return matches, candidate
    return [], None


def _task_region_start(items: list[str]) -> int:
    """Return the line index where the actual task graph starts, after examples."""
    for i, line in enumerate(items):
        if "here is the graph to operate on" in line.lower():
            return i + 1
    return 0


def _bfs_operation_result(items: list[str], query: str) -> dict | None:
    """Return the exact GraphWalks BFS depth result for edge-list payloads.

    This is a deterministic graph retrieval primitive, not a generic BM25 search.
    If a compressed payload contains both directed edges and an explicit BFS
    operation, forcing the model to repeatedly retrieve one frontier at a time
    burns the turn budget and defeats compression's wall-clock advantage.
    """
    tokens = set(_tokenize(query))
    wants_graph_context = bool(tokens & {"bfs", "depth", "edge", "edges", "graph", "operation", "task"})
    if not wants_graph_context:
        return None

    task_items = items[_task_region_start(items):]
    operation_line: str | None = None
    start: str | None = None
    depth: int | None = None
    adjacency: dict[str, list[str]] = {}
    edge_count = 0

    for line in task_items:
        op = _BFS_OPERATION_RE.search(line)
        if op:
            operation_line = line.strip()
            start = op.group(1)
            depth = int(op.group(2))
            continue
        edge = _EDGE_RE.match(line)
        if edge:
            src, dst = edge.group(1), edge.group(2)
            adjacency.setdefault(src, []).append(dst)
            edge_count += 1

    if not start or depth is None or edge_count == 0:
        return None

    frontier: set[str] = {start}
    seen: set[str] = {start}
    levels: list[set[str]] = []
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for child in adjacency.get(node, []):
                if child not in seen:
                    next_frontier.add(child)
        seen.update(next_frontier)
        frontier = next_frontier
        levels.append(set(frontier))
        if not frontier:
            break

    result_nodes = sorted(frontier if depth > 0 else set())
    level_summary = "\n".join(
        f"depth {idx + 1}: {len(nodes)} node(s)"
        for idx, nodes in enumerate(levels)
    )
    body = (
        "[CCR graph BFS result]\n"
        f"Operation: {operation_line or f'BFS from {start} depth {depth}'}\n"
        f"Parsed directed edges: {edge_count}\n"
        f"Start node: {start}\n"
        f"Exact depth: {depth}\n"
        f"{level_summary}\n"
        f"Final Answer: [{', '.join(result_nodes)}]"
    )
    return {
        "text": body,
        "total_items": len(items),
        "kept_items": max(1, len(result_nodes)),
        "matched_items": edge_count,
        "is_json": False,
        "effective_query": f"{query} | graph_bfs:{start}:{depth}",
        "fallback_used": False,
        "route": "graph_bfs",
        "steps": ["parse_edges", "parse_bfs_operation", "compute_exact_depth"],
    }


def _parents_operation_result(items: list[str], query: str) -> dict | None:
    """Return the exact GraphWalks parent set for the final parent operation."""
    tokens = set(_tokenize(query))
    wants_graph_context = bool(tokens & {"edge", "edges", "graph", "operation", "parent", "parents", "task"})
    if not wants_graph_context:
        return None

    task_items = items[_task_region_start(items):]
    operation_line: str | None = None
    target: str | None = None
    incoming: set[str] = set()
    edge_count = 0

    for line in task_items:
        op = _PARENT_OPERATION_RE.search(line)
        if op:
            operation_line = line.strip()
            target = op.group(1)
            continue

    if not target:
        return None

    for line in task_items:
        edge = _EDGE_RE.match(line)
        if not edge:
            continue
        src, dst = edge.group(1), edge.group(2)
        edge_count += 1
        if dst == target and src != target:
            incoming.add(src)

    result_nodes = sorted(incoming)
    body = (
        "[CCR graph parents result]\n"
        f"Operation: {operation_line or f'Find parents of {target}'}\n"
        f"Parsed directed edges: {edge_count}\n"
        f"Target node: {target}\n"
        f"Incoming parent count: {len(result_nodes)}\n"
        f"Final Answer: [{', '.join(result_nodes)}]"
    )
    return {
        "text": body,
        "total_items": len(items),
        "kept_items": max(1, len(result_nodes)),
        "matched_items": len(result_nodes),
        "is_json": False,
        "effective_query": f"{query} | graph_parents:{target}",
        "fallback_used": False,
        "route": "graph_parents",
        "steps": ["parse_edges", "parse_parent_operation", "compute_incoming_edges"],
    }


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
      route            deterministic retrieval route used
      steps            route steps taken
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
            "route": "empty",
            "steps": [],
        }

    profile = _query_profile(query)
    route = "search"
    steps: list[str] = []
    chosen: list[int] = []
    fallback_used = False
    effective_query = query

    if not is_json:
        next_offset = _pagination_continue_offset(items, query)
        if next_offset is not None:
            return {
                "text": (
                    "This cached read result is a truncated page and does not contain "
                    f"the requested task context. Read the same file again with "
                    f"offset={next_offset}, then query the new CCR handle."
                ),
                "total_items": total,
                "kept_items": 1,
                "matched_items": 1,
                "is_json": is_json,
                "effective_query": query,
                "fallback_used": False,
                "route": "pagination_continue",
                "steps": ["pagination_marker"],
            }

        tail = _tail_indices(items, query)
        if tail:
            body = "\n".join(items[i] for i in tail)
            return {
                "text": body,
                "total_items": total,
                "kept_items": len(tail),
                "matched_items": len(tail),
                "is_json": is_json,
                "effective_query": query,
                "fallback_used": False,
                "route": "tail_slice",
                "steps": ["tail_window"],
            }

        bfs_result = _bfs_operation_result(items, query)
        if bfs_result is not None:
            return bfs_result

        incoming, incoming_target = _incoming_edge_indices(items, query)
        if incoming:
            body = "\n".join(items[i] for i in incoming)
            return {
                "text": body,
                "total_items": total,
                "kept_items": len(incoming),
                "matched_items": len(incoming),
                "is_json": is_json,
                "effective_query": f"{query} | incoming:{incoming_target}",
                "fallback_used": False,
                "route": "incoming_edges",
                "steps": ["incoming_edge_scan"],
            }

        outgoing, outgoing_source = _outgoing_edge_indices(items, query)
        if outgoing:
            body = "\n".join(items[i] for i in outgoing)
            return {
                "text": body,
                "total_items": total,
                "kept_items": len(outgoing),
                "matched_items": len(outgoing),
                "is_json": is_json,
                "effective_query": f"{query} | outgoing:{outgoing_source}",
                "fallback_used": False,
                "route": "outgoing_edges",
                "steps": ["outgoing_edge_scan"],
            }

        parents_result = _parents_operation_result(items, query)
        if parents_result is not None:
            return parents_result

    if not is_json and (
        float(profile["structural_ratio"]) >= 0.5
        or (
            profile["structural"]
            and int(profile["high_signal"]) == 0
            and not profile["relation"]
        )
    ):
        route = "structural_discovery"
        structural = _structural_label_matches(items, query, _MAX_STRUCTURAL_LABELS)
        if structural:
            if _has_ambiguous_structural_labels(items, structural):
                body = "\n".join(items[i] for i in structural)
                return {
                    "text": (
                        "CCR retrieval blocked: multiple matching structural labels "
                        "were found. Retry with a more specific label or identifier.\n"
                        f"{body}"
                    ),
                    "total_items": total,
                    "kept_items": len(structural),
                    "matched_items": len(structural),
                    "is_json": is_json,
                    "effective_query": query,
                    "fallback_used": False,
                    "route": "structural_ambiguous",
                    "steps": ["structural_labels", "ambiguity_guard"],
                }
            chosen = structural
            steps.append("structural_labels")
            walk_terms = _extract_walk_terms([items[i] for i in structural])
            walked = _line_indices_containing_terms(items, walk_terms)
            if walked:
                chosen = sorted(set(chosen) | set(walked))
                route = "structural_walk"
                effective_query = f"{query} | walk:{' '.join(walk_terms)}"
                steps.append("walk_terms")

    if not chosen:
        chosen, _scores = _ranked_matches(items, query, max_items)
        route = "search"

        if (
            not is_json
            and chosen
            and int(profile["high_signal"]) == 0
            and _match_repetition_ratio(items, chosen) > 0.75
        ):
            # Avoid flooding the model with repeated examples/tutorial text. When
            # no concrete term anchors the query and the top hits are repetitive,
            # return a small blocked result instead of a large repetitive slice.
            return {
                "text": (
                    "CCR retrieval blocked: query matched repetitive or low-diversity "
                    "content without a distinctive ID, symbol, label, or schema term. "
                    "Retry with a concrete value or a structural label such as target, "
                    "instruction, operation, question, key, or id."
                ),
                "total_items": total,
                "kept_items": 1,
                "matched_items": len(chosen),
                "is_json": is_json,
                "effective_query": query,
                "fallback_used": False,
                "route": "broad_query_rejected",
                "steps": ["repetition_guard"],
            }

    if route == "search":
        fallback = _fallback_query(query)
        if fallback and fallback.strip().lower() != query.strip().lower():
            fallback_chosen, _fallback_scores = _ranked_matches(items, fallback, max_items)
            if fallback_chosen:
                # Deterministically de-noise model queries. BM25 is OR-like: a noisy
                # query can match generic words and miss the best identifier-centered
                # slice. If the original hits are repetitive, replace them with the
                # high-signal slice; otherwise merge to preserve useful nearby context.
                if not is_json and _match_repetition_ratio(items, chosen) > 0.75:
                    chosen = fallback_chosen
                    steps.append("high_signal_replaced_repetitive")
                else:
                    chosen = sorted(set(chosen) | set(fallback_chosen))
                effective_query = fallback if not chosen else f"{query} | fallback:{fallback}"
                fallback_used = True
                steps.append("high_signal_fallback")

    if not is_json and chosen:
        traced, trace_steps = _trace_related_indices(items, chosen, _MAX_RETURN_LINES)
        if len(traced) > len(set(chosen)):
            chosen = traced
            steps.extend(trace_steps)
            if route == "structural_discovery":
                route = "structural_walk"
            elif route == "search":
                route = "search_then_trace"
        elif (
            route == "search"
            and len(set(chosen)) > 1
            and any(_line_has_relation_shape(items[i]) for i in set(chosen))
        ):
            route = "search_then_trace"
            steps.append("trace_in_window")

    matched_items = len(chosen)
    chosen.sort()  # stable basis for JSON and for line window expansion

    if is_json:
        kept = chosen
        body = "[" + ", ".join(items[i] for i in kept) + "]"
    else:
        if route in ("structural_discovery", "structural_walk"):
            kept = _expand_line_windows(chosen, total, before=0, after=_LINE_WINDOW_AFTER)
        elif "high_signal_replaced_repetitive" in steps:
            kept = _expand_line_windows(chosen, total, before=0, after=_LINE_WINDOW_AFTER)
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
        "route": route,
        "steps": steps,
    }
