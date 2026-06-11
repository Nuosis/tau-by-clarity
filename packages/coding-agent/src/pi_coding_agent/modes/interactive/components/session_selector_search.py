"""
Session selector search helpers.

Mirrors components/session-selector-search.ts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .selectors import fuzzy_score

SortMode = Literal["threaded", "recent", "relevance"]
NameFilter = Literal["all", "named"]


@dataclass(frozen=True)
class SearchToken:
    kind: Literal["fuzzy", "phrase"]
    value: str


@dataclass
class ParsedSearchQuery:
    mode: Literal["tokens", "regex"]
    tokens: list[SearchToken]
    regex: re.Pattern[str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class MatchResult:
    matches: bool
    score: float


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_whitespace_lower(text: str) -> str:
    return " ".join(text.lower().split())


def get_session_search_text(session: Any) -> str:
    return " ".join(
        str(part or "")
        for part in (
            _get_attr_or_key(session, "id", ""),
            _get_attr_or_key(session, "name", ""),
            _get_attr_or_key(session, "allMessagesText", _get_attr_or_key(session, "all_messages_text", "")),
            _get_attr_or_key(session, "cwd", ""),
        )
    )


def has_session_name(session: Any) -> bool:
    return bool(str(_get_attr_or_key(session, "name", "") or "").strip())


def parse_search_query(query: str) -> ParsedSearchQuery:
    trimmed = query.strip()
    if not trimmed:
        return ParsedSearchQuery(mode="tokens", tokens=[])
    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return ParsedSearchQuery(mode="regex", tokens=[], error="Empty regex")
        try:
            return ParsedSearchQuery(mode="regex", tokens=[], regex=re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            return ParsedSearchQuery(mode="regex", tokens=[], error=str(exc))

    tokens: list[SearchToken] = []
    buf = ""
    in_quote = False
    had_unclosed_quote = False

    def flush(kind: Literal["fuzzy", "phrase"]) -> None:
        nonlocal buf
        value = buf.strip()
        buf = ""
        if value:
            tokens.append(SearchToken(kind=kind, value=value))

    for char in trimmed:
        if char == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue
        if not in_quote and char.isspace():
            flush("fuzzy")
            continue
        buf += char

    if in_quote:
        had_unclosed_quote = True

    if had_unclosed_quote:
        return ParsedSearchQuery(
            mode="tokens",
            tokens=[SearchToken("fuzzy", item) for item in trimmed.split() if item.strip()],
        )

    flush("phrase" if in_quote else "fuzzy")
    return ParsedSearchQuery(mode="tokens", tokens=tokens)


def match_session(session: Any, parsed: ParsedSearchQuery) -> MatchResult:
    text = get_session_search_text(session)
    if parsed.mode == "regex":
        if parsed.regex is None:
            return MatchResult(False, 0)
        match = parsed.regex.search(text)
        if not match:
            return MatchResult(False, 0)
        return MatchResult(True, match.start() * 0.1)

    if not parsed.tokens:
        return MatchResult(True, 0)

    total_score = 0.0
    normalized: str | None = None
    for token in parsed.tokens:
        if token.kind == "phrase":
            normalized = normalized or _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(token.value)
            index = normalized.find(phrase)
            if index < 0:
                return MatchResult(False, 0)
            total_score += index * 0.1
            continue
        score = fuzzy_score(token.value, text)
        if score is None:
            return MatchResult(False, 0)
        total_score += score
    return MatchResult(True, total_score)


def _modified_timestamp(session: Any) -> float:
    modified = _get_attr_or_key(session, "modified", None)
    if isinstance(modified, datetime):
        return modified.timestamp()
    if isinstance(modified, (int, float)):
        return float(modified)
    if isinstance(modified, str):
        try:
            return datetime.fromisoformat(modified.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0
    return 0


def filter_and_sort_sessions(
    sessions: list[Any],
    query: str,
    sort_mode: SortMode,
    name_filter: NameFilter = "all",
) -> list[Any]:
    name_filtered = sessions if name_filter == "all" else [session for session in sessions if has_session_name(session)]
    if not query.strip():
        return list(name_filtered)
    parsed = parse_search_query(query)
    if parsed.error:
        return []
    if sort_mode == "recent":
        return [session for session in name_filtered if match_session(session, parsed).matches]

    scored: list[tuple[float, float, Any]] = []
    for session in name_filtered:
        result = match_session(session, parsed)
        if result.matches:
            scored.append((result.score, -_modified_timestamp(session), session))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [session for _, _, session in scored]


filterAndSortSessions = filter_and_sort_sessions
getSessionSearchText = get_session_search_text
hasSessionName = has_session_name
matchSession = match_session
parseSearchQuery = parse_search_query


__all__ = [
    "MatchResult",
    "NameFilter",
    "ParsedSearchQuery",
    "SearchToken",
    "SortMode",
    "filterAndSortSessions",
    "filter_and_sort_sessions",
    "getSessionSearchText",
    "get_session_search_text",
    "hasSessionName",
    "has_session_name",
    "matchSession",
    "match_session",
    "parseSearchQuery",
    "parse_search_query",
]
