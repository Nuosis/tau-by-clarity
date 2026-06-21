"""Legacy local content-aware compressor.

This is Tau's hand-rolled Smart-Crusher-like implementation, not Headroom.
Keep it isolated while real Headroom behavior is probed and integrated.

It compresses large tool-output payloads by content type, ALWAYS preserving
error items, and caches the original in the local CCR store so it stays
retrievable. Lossy in the prompt, reversible via local CCR. Short payloads pass
through untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from csv import writer as csv_writer
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from io import StringIO
from typing import Any, Protocol

from pi_ai.tokenization import count_text_tokens

from .ccr import CCRStore


class _CCRLike(Protocol):
    def put(self, content: str) -> str: ...

    def put_with_handle(self, handle: str, content: str) -> str: ...

    def get(self, handle: str) -> str | None: ...

    def is_expanded(self, handle: str) -> bool: ...

MIN_TOKENS = 200  # below this, compression overhead exceeds the savings (pass through)
DEFAULT_ITEM_BUDGET = 15
SEARCH_MAX_FILES = 15
SEARCH_MAX_MATCHES_TOTAL = 30
SEARCH_MAX_MATCHES_PER_FILE = 5
SEARCH_MIN_MATCHES = 10
DIFF_MIN_LINES = 50
LOG_MIN_LINES = 50
LOSSLESS_TABLE_MIN_SAVINGS_RATIO = 0.15
_ERROR_KEYWORDS = (
    "error",
    "exception",
    "failed",
    "failure",
    "critical",
    "fatal",
    "crash",
    "panic",
    "abort",
    "timeout",
    "denied",
    "rejected",
)
_ERROR_LINE_RE = re.compile(r"error|fail|exception|traceback|fatal|critical", re.IGNORECASE)
_WARN_LINE_RE = re.compile(r"warn|warning|deprecated", re.IGNORECASE)
_IMPORTANCE_LINE_RE = re.compile(r"important|note|todo|fixme|hack|xxx|bug|\bfix\b", re.IGNORECASE)
_DEDUPE_DIGIT_RE = re.compile(r"\d+")
_DEDUPE_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_DEDUPE_PATH_RE = re.compile(r"/[\w/]+/")
_SUMMARY_LINE_RE = re.compile(
    r"(^=+|^-+|\b\d+\s+(passed|failed|skipped|errors?|warnings?)\b|"
    r"\b(tests?|suites?):\s*\d+|\b(short test summary info|failures)\b)",
    re.IGNORECASE,
)
_SEARCH_COLON_LINE_RE = re.compile(r"^(?P<file>.+):(?P<line>\d+):(?P<body>.*)$")
_SEARCH_DASH_LINE_RE = re.compile(r"^(?P<file>.+)-(?P<line>\d+)-(?P<body>.*)$")
_DIFF_FILE_HEADER_RE = re.compile(r"^diff --(?:git |combined |cc )")
_DIFF_HUNK_RE = re.compile(r"^(?:@@ .* @@|@@@ .* @@@|@@@@ .* @@@@)")
_DIFF_BINARY_LINE_RE = re.compile(r"^Binary files .+ differ$", re.MULTILINE)
_DIFF_METADATA_LINE_RE = re.compile(
    r"^(similarity index|dissimilarity index|rename (from|to) |copy (from|to) |"
    r"new file mode|deleted file mode)",
    re.MULTILINE,
)
_CODE_SIGNATURE_RE = re.compile(
    r"^\s*(?:"
    r"(?:from\s+\S+\s+import\s+.+|import\s+.+)|"
    r"(?:async\s+)?def\s+\w+\s*\(.*|"
    r"class\s+\w+.*|"
    r"(?:export\s+)?(?:async\s+)?function\s+\w+\s*\(.*|"
    r"(?:const|let|var)\s+\w+\s*=.*=>.*|"
    r"(?:pub\s+)?(?:async\s+)?fn\s+\w+.*|"
    r"func\s+(?:\([^)]+\)\s*)?\w+\s*\(.*|"
    r"(?:public|private|protected)?\s*(?:class|interface|enum)\s+\w+.*"
    r")",
)
_PY_DOCSTRING_START_RE = re.compile(r"^(?P<indent>\s*)(?P<quote>\"\"\"|''')(?P<text>.*)$")
_FENCED_BLOCK_RE = re.compile(r"```(?P<label>[A-Za-z0-9_.+-]*)[ \t]*\n(?P<body>.*?)(?:\n```|```)", re.DOTALL)
_HTML_INDICATORS = (
    "<!doctype html",
    "<html",
    "<head",
    "<body",
    "<main",
    "<article",
    "<script",
    "<style",
    "<meta",
    "<link",
)

_TS_TYPE_RE = re.compile(r"^\s*(?:interface|type|enum|namespace)\s+\w+")
_JS_IMPORT_EXPORT_RE = re.compile(r"^\s*(?:import|export)\s+")
_GO_PACKAGE_RE = re.compile(r"^\s*package\s+\w+")
_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+[\w.]+;")
_CODE_TYPE_ANNOTATION_RE = re.compile(r":\s*(?:string|number|boolean|any|void)\b")
_HTML_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "br",
    "dd",
    "details",
    "dialog",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_HTML_NOISE_TAGS = {
    "aside",
    "canvas",
    "footer",
    "form",
    "header",
    "iframe",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}
_HTML_CONTENT_TAGS = {"article", "main"}
_HTML_TAG_NAMES = {
    "a",
    "abbr",
    "address",
    "area",
    "article",
    "aside",
    "audio",
    "b",
    "base",
    "bdi",
    "bdo",
    "blockquote",
    "body",
    "br",
    "button",
    "canvas",
    "caption",
    "cite",
    "code",
    "col",
    "colgroup",
    "data",
    "datalist",
    "dd",
    "del",
    "details",
    "dfn",
    "dialog",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "legend",
    "li",
    "link",
    "main",
    "map",
    "mark",
    "meta",
    "meter",
    "nav",
    "noscript",
    "object",
    "ol",
    "optgroup",
    "option",
    "output",
    "p",
    "picture",
    "pre",
    "progress",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "script",
    "section",
    "select",
    "small",
    "source",
    "span",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "u",
    "ul",
    "var",
    "video",
    "wbr",
}
_XML_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<name>[A-Za-z][A-Za-z0-9_.:-]*)(?P<body>[^<>]*?)>",
    re.DOTALL,
)


@dataclass(frozen=True)
class CompressionConfig:
    min_tokens: int = MIN_TOKENS
    max_items_after_crush: int = DEFAULT_ITEM_BUDGET
    lossless_min_savings_ratio: float = LOSSLESS_TABLE_MIN_SAVINGS_RATIO
    enable_ccr_marker: bool = True


def _approx_tokens(text: str) -> int:
    return count_text_tokens(text)


class _OriginalCCRStore:
    def __init__(self, ccr: CCRStore, original: str) -> None:
        self._ccr = ccr
        self._original = original

    def put(self, _content: str) -> str:
        return self._ccr.put(self._original)

    def put_with_handle(self, handle: str, content: str) -> str:
        return self._ccr.put_with_handle(handle, content)

    def get(self, handle: str) -> str | None:
        return self._ccr.get(handle)

    def is_expanded(self, handle: str) -> bool:
        return self._ccr.is_expanded(handle)


def _xml_tag_name(match: re.Match[str]) -> str:
    return match.group("name").split(":", 1)[-1].lower()


def _is_custom_xml_tag(match: re.Match[str]) -> bool:
    name = _xml_tag_name(match)
    return bool(name) and name not in _HTML_TAG_NAMES


def _is_self_closing_xml_tag(match: re.Match[str]) -> bool:
    return match.group("body").rstrip().endswith("/")


def _find_custom_tag_span(text: str, opener: re.Match[str]) -> int | None:
    """Return the exclusive end offset for an opening custom tag span."""
    tag_name = _xml_tag_name(opener)
    depth = 1
    cursor = opener.end()
    while True:
        match = _XML_TAG_RE.search(text, cursor)
        if match is None:
            return None
        cursor = match.end()
        if _xml_tag_name(match) != tag_name:
            continue
        if match.group("closing"):
            depth -= 1
            if depth == 0:
                return match.end()
        elif not _is_self_closing_xml_tag(match):
            depth += 1


def _tag_placeholder_prefix(text: str) -> str:
    base = "{{TAU_TAG_"
    if base not in text:
        return base
    salt = 1
    while f"{{{{TAU_TAG_{salt}_" in text:
        salt += 1
    return f"{{{{TAU_TAG_{salt}_"


def _protect_custom_tags(text: str) -> tuple[str, list[tuple[str, str]]]:
    if "<" not in text or ">" not in text:
        return text, []
    prefix = _tag_placeholder_prefix(text)
    protected: list[tuple[str, str]] = []
    parts: list[str] = []
    cursor = 0
    scan_at = 0
    while True:
        match = _XML_TAG_RE.search(text, scan_at)
        if match is None:
            break
        scan_at = match.end()
        if match.start() < cursor:
            continue
        if match.group("closing") or not _is_custom_xml_tag(match):
            continue
        end = match.end() if _is_self_closing_xml_tag(match) else _find_custom_tag_span(text, match)
        if end is None:
            continue

        parts.append(text[cursor : match.start()])
        placeholder = prefix + str(len(protected)) + "}}"
        original = text[match.start() : end]
        protected.append((placeholder, original))
        parts.append(placeholder)
        cursor = end
        scan_at = end
    if not protected:
        return text, []
    parts.append(text[cursor:])
    return "".join(parts), protected


def _restore_custom_tags(text: str, protected: list[tuple[str, str]]) -> str:
    restored = text
    for placeholder, original in protected:
        if placeholder in restored:
            restored = restored.replace(placeholder, original, 1)
    return restored


def _normalize_target_ratio(target_ratio: float | None) -> float | None:
    if target_ratio is None:
        return None
    try:
        ratio = float(target_ratio)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ratio) or ratio <= 0:
        return None
    return min(ratio, 1.0)


def _compress_with_custom_tag_protection(
    text: str,
    ccr: _CCRLike,
    *,
    target_ratio: float | None = None,
    config: CompressionConfig,
) -> str:
    cleaned, protected = _protect_custom_tags(text)
    if not protected:
        return _compress_inner(
            text,
            ccr,
            allow_mixed=True,
            protect_tags=False,
            target_ratio=target_ratio,
            config=config,
        )
    if not cleaned.strip():
        return text
    compressed = _compress_inner(
        cleaned,
        _OriginalCCRStore(ccr, text),
        allow_mixed=True,
        protect_tags=False,
        target_ratio=target_ratio,
        config=config,
    )
    if compressed == cleaned:
        return text
    restored = _restore_custom_tags(compressed, protected)
    return restored if len(restored) < len(text) else text


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.main_parts: list[str] = []
        self.body_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._content_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag in _HTML_NOISE_TAGS:
            self._skip_depth += 1
        if tag in _HTML_CONTENT_TAGS:
            self._content_depth += 1
        if tag in _HTML_BLOCK_TAGS:
            self._append_text("\n")
        if tag == "a":
            href = next((value for name, value in attrs if name.lower() == "href"), None)
            if href and self._content_depth > 0 and self._skip_depth == 0:
                self._append_text(f" ({href}) ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _HTML_BLOCK_TAGS:
            self._append_text("\n")
        if tag == "title":
            self._in_title = False
        if tag in _HTML_CONTENT_TAGS and self._content_depth > 0:
            self._content_depth -= 1
        if tag in _HTML_NOISE_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = unescape(data)
        if not text.strip():
            return
        if self._in_title:
            self.title_parts.append(text.strip())
            return
        if self._skip_depth > 0:
            return
        self._append_text(text)

    def _append_text(self, text: str) -> None:
        if self._skip_depth > 0:
            return
        target = self.main_parts if self._content_depth > 0 else self.body_parts
        target.append(text)


def _looks_like_html(text: str) -> bool:
    stripped = text.lstrip().lower()
    if stripped.startswith("<!doctype html") or stripped.startswith("<html"):
        return True
    sample = stripped[:4000]
    return sum(1 for indicator in _HTML_INDICATORS if indicator in sample) >= 2


def _detect_html_for_routing(text: str) -> bool:
    sample = text.lstrip()[:3000].lower()
    return sample.startswith("<!doctype html") or bool(re.search(r"<html[\s>]", sample))


def _normalize_extracted_html_text(parts: list[str]) -> str:
    text = "".join(parts)
    lines = []
    for raw in text.splitlines():
        collapsed = " ".join(raw.split())
        if collapsed:
            lines.append(collapsed)
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped)


def _extract_html_content(text: str) -> tuple[str | None, str | None]:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return None, None

    title = " ".join(" ".join(parser.title_parts).split()) or None
    main = _normalize_extracted_html_text(parser.main_parts)
    body = _normalize_extracted_html_text(parser.body_parts)
    extracted = main if len(main) >= 200 else body
    if not extracted or len(extracted) < 200:
        return None, title
    return extracted, title


def _compress_html(text: str, ccr: _CCRLike) -> str:
    extracted, title = _extract_html_content(text)
    if extracted is None:
        return text
    body_parts = []
    if title:
        body_parts.append(f"Title: {title}")
    body_parts.append(extracted)
    body = "\n\n".join(body_parts)
    return _maybe_wrap(
        original=text,
        compressed_body=body,
        ccr=ccr,
        label="extracted HTML content",
        original_count=len(text),
        compressed_count=len(body),
    )


def _is_error_item(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    serialized = _stable_value(d).lower()
    if any(keyword in serialized for keyword in _ERROR_KEYWORDS):
        return True
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


def _stable_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _category_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _detect_rare_status_value_indices(items: list[dict[str, Any]], common_fields: set[str]) -> set[int]:
    outliers: set[int] = set()
    for field_name in sorted(common_fields):
        values = [row[field_name] for row in items if field_name in row]
        unique_values = {_category_key(value) for value in values if value is not None}
        if not 2 <= len(unique_values) <= 50:
            continue

        counts: Counter[str] = Counter(
            "__none__" if value is None else (_category_key(value) or "")
            for value in values
        )
        if not counts:
            continue

        threshold = math.ceil(len(values) * 0.8)
        cumulative = 0
        top_values: set[str] = set()
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            cumulative += count
            top_values.add(value)
            if cumulative >= threshold:
                break

        if len(top_values) > 5:
            continue

        for index, row in enumerate(items):
            if field_name not in row:
                continue
            value = "__none__" if row[field_name] is None else (_category_key(row[field_name]) or "")
            if value not in top_values:
                outliers.add(index)
    return outliers


def _detect_structural_outlier_indices(items: list[dict[str, Any]]) -> set[int]:
    if len(items) < 5:
        return set()

    field_counts: Counter[str] = Counter()
    for row in items:
        field_counts.update(str(key) for key in row)

    n = len(items)
    common_fields = {key for key, count in field_counts.items() if count >= n * 0.8}
    rare_fields = {key for key, count in field_counts.items() if count < n * 0.2}

    outliers: set[int] = set()
    for index, row in enumerate(items):
        if any(str(key) in rare_fields for key in row):
            outliers.add(index)

    outliers.update(_detect_rare_status_value_indices(items, common_fields))
    return outliers


def _numeric_values_for_field(items: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in items:
        value = row.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _is_sequential_numeric(values: list[float]) -> bool:
    if len(values) < 5:
        return False
    diffs = [round(values[i + 1] - values[i], 12) for i in range(len(values) - 1)]
    first = diffs[0]
    if first == 0:
        return False
    return all(diff == first for diff in diffs)


def _detect_score_field_name(items: list[dict[str, Any]]) -> str | None:
    if len(items) < 5:
        return None

    field_counts: Counter[str] = Counter()
    for row in items:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            field_counts[str(key)] += 1

    best: tuple[float, str] | None = None
    for field_name, count in field_counts.items():
        if count < len(items) * 0.8:
            continue
        values = _numeric_values_for_field(items, field_name)
        if len(values) < 5 or _is_sequential_numeric(values[:50]):
            continue
        min_val = min(values)
        max_val = max(values)

        confidence = 0.0
        if 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0:
            confidence += 0.4
        elif 0.0 <= min_val <= 10.0 and 0.0 <= max_val <= 10.0:
            confidence += 0.3
        elif 0.0 <= min_val <= 100.0 and 0.0 <= max_val <= 100.0:
            confidence += 0.25
        elif min_val >= -1.0 and max_val <= 1.0:
            confidence += 0.35
        else:
            continue

        pairs = len(values) - 1
        if pairs >= 4:
            descending = sum(1 for left, right in zip(values, values[1:]) if left >= right)
            if descending / pairs > 0.7:
                confidence += 0.3

        first_20 = values[:20]
        if first_20:
            non_integer = sum(1 for value in first_20 if value != int(value))
            if non_integer / len(first_20) > 0.3:
                confidence += 0.1

        if any(word in field_name.lower() for word in ("score", "rank", "relevance", "priority")):
            confidence += 0.1

        if confidence >= 0.4 and (best is None or confidence > best[0]):
            best = (min(confidence, 0.95), field_name)

    return best[1] if best is not None else None


def _is_iso_temporal_string(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        datetime.fromisoformat(normalized)
        return True
    except ValueError:
        return False


def _has_temporal_field(items: list[dict[str, Any]]) -> bool:
    field_names = sorted({str(key) for row in items for key in row})
    for field_name in field_names:
        sample = [row.get(field_name) for row in items[:10] if isinstance(row.get(field_name), str)]
        if sample and sum(1 for value in sample if _is_iso_temporal_string(value)) / len(sample) > 0.5:
            return True

        values = _numeric_values_for_field(items, field_name)
        if not values:
            continue
        min_val = min(values)
        if 1_000_000_000.0 <= min_val <= 2_000_000_000.0:
            return True
        if 1_000_000_000_000.0 <= min_val <= 2_000_000_000_000.0:
            return True
    return False


def _sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _detect_change_points(values: list[float], *, window: int = 5, variance_threshold: float = 2.0) -> list[int]:
    if len(values) < window * 2:
        return []
    overall_std = _sample_stdev(values)
    if overall_std <= 0:
        return []
    threshold = variance_threshold * overall_std

    change_points: list[int] = []
    for index in range(window, len(values) - window):
        before = sum(values[index - window : index]) / window
        after = sum(values[index : index + window]) / window
        if abs(after - before) > threshold:
            change_points.append(index)

    if not change_points:
        return []

    deduped = [change_points[0]]
    for point in change_points[1:]:
        if point - deduped[-1] > window:
            deduped.append(point)
    return deduped


def _detect_time_series_change_point_indices(items: list[dict[str, Any]]) -> set[int]:
    if not _has_temporal_field(items):
        return set()

    keep: set[int] = set()
    field_names = sorted({str(key) for row in items for key in row})
    for field_name in field_names:
        values = _numeric_values_for_field(items, field_name)
        if len(values) != len(items) or _sample_stdev(values) <= 0:
            continue
        for point in _detect_change_points(values):
            for index in range(max(0, point - 2), min(len(items), point + 3)):
                keep.add(index)
    return keep


def _string_field_stats(items: list[dict[str, Any]]) -> dict[str, tuple[float, int, float]]:
    fields: dict[str, list[str]] = {}
    for row in items:
        for key, value in row.items():
            if isinstance(value, str):
                fields.setdefault(str(key), []).append(value)

    stats: dict[str, tuple[float, int, float]] = {}
    for field_name, values in fields.items():
        unique_count = len(set(values))
        unique_ratio = unique_count / max(1, len(values))
        avg_length = sum(len(value) for value in values) / max(1, len(values))
        stats[field_name] = (unique_ratio, unique_count, avg_length)
    return stats


def _detect_cluster_representative_indices(items: list[dict[str, Any]]) -> set[int]:
    stats = _string_field_stats(items)
    if not stats:
        return set()

    has_message_like = any(unique_ratio > 0.5 and avg_length > 20 for unique_ratio, _count, avg_length in stats.values())
    has_level_like = any(unique_ratio < 0.1 and 2 <= unique_count <= 10 for unique_ratio, unique_count, _avg in stats.values())
    if not (has_message_like and has_level_like):
        return set()

    message_stats = next(
        (field_stats for field_name, field_stats in sorted(stats.items()) if "message" in field_name.lower()),
        None,
    )
    if message_stats is None or message_stats[0] >= 0.5:
        return set()

    cluster_field: str | None = None
    max_uniqueness = 0.0
    for field_name, (unique_ratio, _count, _avg) in sorted(stats.items()):
        if unique_ratio > max_uniqueness and unique_ratio > 0.3:
            cluster_field = field_name
            max_uniqueness = unique_ratio
    if cluster_field is None:
        return set()

    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(items):
        value = row.get(cluster_field)
        message = value if isinstance(value, str) else ""
        digest = hashlib.md5(message[:50].encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        clusters.setdefault(digest, []).append(index)

    keep: set[int] = set()
    for indices in clusters.values():
        keep.update(indices[:2])
    return keep


def _compute_k_split_for_array(items: list[Any], budget: int = DEFAULT_ITEM_BUDGET) -> tuple[int, int, int, int]:
    k_total = min(budget, len(items))
    k_first_raw = max(1, round(k_total * 0.3))
    k_last_raw = max(1, round(k_total * 0.15))
    k_first = min(k_first_raw, k_total)
    k_last = min(k_last_raw, max(0, k_total - k_first))
    k_importance = max(0, k_total - k_first - k_last)
    return k_total, k_first, k_last, k_importance


def _sample_string_array(items: list[str], budget: int = DEFAULT_ITEM_BUDGET) -> list[str]:
    n = len(items)
    if n <= 8:
        return items

    k_total, k_first, k_last, _k_importance = _compute_k_split_for_array(items, budget)
    keep: set[int] = set(range(min(k_first, n)))
    keep.update(range(max(0, n - k_last), n))

    error_indices = {
        index
        for index, value in enumerate(items)
        if any(keyword in value.lower() for keyword in _ERROR_KEYWORDS)
    }
    keep.update(error_indices)

    lengths = [float(len(value)) for value in items]
    std_len = _sample_stdev(lengths)
    anomaly_indices: set[int] = set()
    if std_len > 0:
        mean_len = sum(lengths) / len(lengths)
        threshold = 2.0 * std_len
        anomaly_indices = {
            index
            for index, length in enumerate(lengths)
            if abs(length - mean_len) > threshold
        }
        keep.update(anomaly_indices)

    seen = {items[index] for index in keep}
    remaining_budget = max(0, k_total - len(keep))
    if remaining_budget > 0:
        stride = max(1, (n - 1) // (remaining_budget + 1))
        cap = k_total + len(error_indices) + len(anomaly_indices)
        index = 0
        while index < n and len(keep) < cap:
            if index not in keep and items[index] not in seen:
                keep.add(index)
                seen.add(items[index])
            index += stride

    return [items[index] for index in sorted(keep)]


def _sample_number_array(items: list[int | float], budget: int = DEFAULT_ITEM_BUDGET) -> list[int | float]:
    n = len(items)
    if n <= 8:
        return items

    finite = [float(value) for value in items if not isinstance(value, bool) and math.isfinite(float(value))]
    if not finite:
        return items

    k_total, k_first, k_last, _k_importance = _compute_k_split_for_array(items, budget)
    keep: set[int] = set(range(min(k_first, n)))
    keep.update(range(max(0, n - k_last), n))

    mean_value = sum(finite) / len(finite)
    std_value = _sample_stdev(finite)
    outlier_indices: set[int] = set()
    if std_value > 0:
        threshold = 2.0 * std_value
        for index, value in enumerate(items):
            numeric = float(value)
            if math.isfinite(numeric) and abs(numeric - mean_value) > threshold:
                outlier_indices.add(index)
        keep.update(outlier_indices)

    change_indices: set[int] = set()
    if n > 10 and std_value > 0:
        for point in _detect_change_points([float(value) for value in items]):
            change_indices.add(point)
        keep.update(change_indices)

    remaining_budget = max(0, k_total - len(keep))
    if remaining_budget > 0:
        stride = max(1, (n - 1) // (remaining_budget + 1))
        cap = k_total + len(outlier_indices)
        index = 0
        while index < n and len(keep) < cap:
            keep.add(index)
            index += stride

    return [items[index] for index in sorted(keep)]


def _mixed_group_key(value: Any) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "none"
    return "other"


def _stable_match_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _kept_value_keys(values: list[Any]) -> set[str]:
    return {_stable_match_key(value) for value in values}


def _sample_mixed_array(items: list[Any], budget: int = DEFAULT_ITEM_BUDGET) -> list[Any]:
    if len(items) <= 8:
        return items

    group_order: list[str] = []
    grouped_indices: dict[str, list[int]] = {}
    grouped_values: dict[str, list[Any]] = {}
    for index, value in enumerate(items):
        key = _mixed_group_key(value)
        if key not in grouped_indices:
            group_order.append(key)
            grouped_indices[key] = []
            grouped_values[key] = []
        grouped_indices[key].append(index)
        grouped_values[key].append(value)

    keep_indices: set[int] = set()
    for key in group_order:
        indices = grouped_indices[key]
        values = grouped_values[key]
        if len(values) < 5:
            keep_indices.update(indices)
            continue

        if key == "dict" and all(isinstance(value, dict) for value in values):
            sampled, _n, _n_err, _kept_indices = _crush_list_of_dicts(values, budget=budget)
            kept_keys = _kept_value_keys(sampled)
            for original_index, value in zip(indices, values, strict=True):
                if _stable_match_key(value) in kept_keys:
                    keep_indices.add(original_index)
        elif key == "str" and all(isinstance(value, str) for value in values):
            kept_values = set(_sample_string_array(values, budget=budget))
            for original_index, value in zip(indices, values, strict=True):
                if value in kept_values:
                    keep_indices.add(original_index)
        elif key == "number" and all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
            sampled_numbers = _sample_number_array(values, budget=budget)
            kept_keys = _kept_value_keys(sampled_numbers)
            for original_index, value in zip(indices, values, strict=True):
                if _stable_match_key(value) in kept_keys:
                    keep_indices.add(original_index)
        else:
            keep_indices.update(indices)

    return [items[index] for index in sorted(keep_indices)]


def _row_information_scores(items: list[dict[str, Any]]) -> dict[int, float]:
    field_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {}
    lengths: list[int] = []

    for row in items:
        lengths.append(len(_stable_value(row)))
        for key, value in row.items():
            field_counts[str(key)] += 1
            value_counts.setdefault(str(key), Counter())[_stable_value(value)] += 1

    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    total = max(1, len(items))
    common_fields = {key for key, count in field_counts.items() if count >= total * 0.8}
    rare_fields = {key for key, count in field_counts.items() if count <= max(1, total * 0.2)}

    scores: dict[int, float] = {}
    for index, row in enumerate(items):
        value_rarity = 0.0
        for key, value in row.items():
            counts = value_counts.get(str(key))
            if counts:
                value_rarity += 1.0 - (counts[_stable_value(value)] / total)
        value_rarity = value_rarity / max(1, len(row))

        fields = {str(key) for key in row}
        rare_field_score = len(fields & rare_fields) / max(1, len(rare_fields))
        missing_common_score = len(common_fields - fields) / max(1, len(common_fields))
        structural_score = min(1.0, rare_field_score + missing_common_score)

        if max_len == min_len:
            length_score = 0.0
        else:
            length_score = (lengths[index] - min_len) / (max_len - min_len)

        scores[index] = (0.45 * value_rarity) + (0.35 * structural_score) + (0.20 * length_score)
    return scores


def _adaptive_item_budget(items: list[dict[str, Any]], max_budget: int = DEFAULT_ITEM_BUDGET) -> int:
    if len(items) <= max_budget:
        return len(items)

    serialized = [_stable_value(row) for row in items]
    seen_bigrams: set[tuple[str, str]] = set()
    curve: list[int] = []
    for text in serialized:
        words = text.lower().split()
        if len(words) < 2:
            seen_bigrams.add((words[0] if words else "", ""))
        else:
            for index in range(len(words) - 1):
                seen_bigrams.add((words[index], words[index + 1]))
        curve.append(len(seen_bigrams))

    knee: int | None = None
    if len(curve) >= 3 and curve[-1] != curve[0]:
        x_range = len(curve) - 1
        y_range = curve[-1] - curve[0]
        best_diff = -1.0
        best_index = 0
        for index, value in enumerate(curve):
            x_norm = index / x_range
            y_norm = (value - curve[0]) / y_range
            diff = y_norm - x_norm
            if diff > best_diff:
                best_diff = diff
                best_index = index
        if best_diff >= 0.05:
            knee = best_index + 1

    unique_shapes = {_stable_value(sorted(row.keys())) for row in items}
    diversity_floor = min(max_budget, max(6, int(max_budget * min(1.0, len(unique_shapes) / 8))))
    if knee is None:
        return max(diversity_floor, max_budget)
    return max(6, min(max_budget, max(knee, diversity_floor)))


def _select_adaptive_indices(items: list[dict[str, Any]], budget: int) -> set[int]:
    n = len(items)
    if n <= budget:
        return set(range(n))

    keep_idx: set[int] = set()
    time_series_indices = _detect_time_series_change_point_indices(items)
    cluster_indices = _detect_cluster_representative_indices(items) if not time_series_indices else set()
    score_field = _detect_score_field_name(items)
    if score_field is not None and not time_series_indices and not cluster_indices:
        top_count = max(1, budget - 3)
        scored = [
            (index, float(row[score_field]))
            for index, row in enumerate(items)
            if isinstance(row.get(score_field), int | float) and not isinstance(row.get(score_field), bool)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        keep_idx.update(index for index, _score in scored[:top_count])
        for index, row in enumerate(items):
            if _is_error_item(row):
                keep_idx.add(index)
        keep_idx.update(_detect_structural_outlier_indices(items))
        return keep_idx

    head = max(1, int(budget * 0.25))
    tail = max(1, int(budget * 0.20))
    keep_idx.update(range(min(head, n)))
    keep_idx.update(range(max(0, n - tail), n))
    keep_idx.update(time_series_indices)
    keep_idx.update(cluster_indices)

    for index, row in enumerate(items):
        if _is_error_item(row):
            keep_idx.add(index)

    keep_idx.update(_detect_structural_outlier_indices(items))

    scores = _row_information_scores(items)
    for index, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if len(keep_idx) >= budget:
            break
        keep_idx.add(index)

    remaining = budget - len(keep_idx)
    if remaining > 0:
        stride = max(1, n // remaining)
        for index in range(0, n, stride):
            keep_idx.add(index)
            if len(keep_idx) >= budget:
                break

    return keep_idx


def _crush_list_of_dicts(
    items: list[dict[str, Any]],
    budget: int = DEFAULT_ITEM_BUDGET,
) -> tuple[list[dict[str, Any]], int, int, set[int]]:
    """Keep dynamic anchors, information-dense rows, and ALWAYS every error item."""
    n = len(items)
    budget = _adaptive_item_budget(items, budget)
    keep_idx = _select_adaptive_indices(items, budget)
    kept = [items[index] for index in sorted(keep_idx)]
    errors = [row for row in items if _is_error_item(row)]
    for row in errors:
        if row not in kept:
            kept.append(row)
            for index, candidate in enumerate(items):
                if candidate == row:
                    keep_idx.add(index)
                    break
    return kept, n, len(errors), keep_idx


_SUMMARY_CATEGORY_FIELDS = (
    "status",
    "state",
    "type",
    "kind",
    "level",
    "severity",
    "result",
    "outcome",
    "category",
)


def _summary_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > 80:
        return None
    lowered = stripped.lower()
    if "://" in lowered or lowered.startswith(("data:", "mailto:")):
        return None
    return stripped


def _summarize_dropped_dict_rows(items: list[dict[str, Any]], kept_indices: set[int]) -> str:
    dropped = [row for index, row in enumerate(items) if index not in kept_indices]
    if not dropped:
        return ""

    for field in _SUMMARY_CATEGORY_FIELDS:
        values = [_summary_value(row.get(field)) for row in dropped if field in row]
        values = [value for value in values if value is not None]
        if not values:
            continue
        counts = Counter(values)
        if not 1 <= len(counts) <= 12:
            continue
        parts = [f"{value}={count}" for value, count in counts.most_common(6)]
        if len(counts) > 6:
            parts.append(f"+{len(counts) - 6} more")
        return f"{len(dropped)} rows omitted; {field}: " + ", ".join(parts)

    field_counts: Counter[str] = Counter()
    for row in dropped:
        for key, value in row.items():
            if isinstance(value, str) and "://" in value:
                continue
            field_counts[str(key)] += 1
    if not field_counts:
        return f"{len(dropped)} rows omitted"
    parts = [f"{key}={count}" for key, count in field_counts.most_common(6)]
    if len(field_counts) > 6:
        parts.append(f"+{len(field_counts) - 6} more fields")
    return f"{len(dropped)} rows omitted; fields: " + ", ".join(parts)


def _canonical_json_array_for_ccr(items: list[dict[str, Any]]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _hash_canonical_for_ccr(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _is_id_like_field(key: str, values: list[Any], row_count: int) -> bool:
    key_lower = key.lower()
    if key_lower in {"id", "_id", "uuid"} or key_lower.endswith("_id"):
        return len(set(map(_stable_value, values))) >= row_count * 0.8
    return False


def _has_json_array_crush_signal(items: list[dict[str, Any]]) -> bool:
    if any(_is_error_item(row) for row in items):
        return True

    row_count = len(items)
    columns = _all_columns(items)
    if not columns:
        return False

    common_columns = _core_columns(items)
    if not common_columns:
        return False

    non_id_uniqueness: list[float] = []
    for key in common_columns:
        values = [row.get(key) for row in items if key in row]
        if not values:
            continue
        unique_ratio = len(set(map(_stable_value, values))) / max(1, len(values))
        if _is_id_like_field(key, values, row_count):
            continue
        non_id_uniqueness.append(unique_ratio)
        if any(word in key.lower() for word in ("score", "rank", "relevance", "priority", "error", "status")):
            return True

    if not non_id_uniqueness:
        return False

    return max(non_id_uniqueness) < 0.3


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _compact_stringified_json_cell(value: str) -> str | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None

    if isinstance(parsed, list) and len(parsed) >= 2 and all(isinstance(item, dict) for item in parsed):
        table = _compact_inline_table(parsed)
        if table is not None:
            return table
    if isinstance(parsed, dict | list):
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return None


def _inline_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact_inline_table(items: list[dict[str, Any]]) -> str | None:
    columns = _all_columns(items)
    if not columns:
        return None
    if not all(all(_is_scalar(row.get(col)) for col in columns) for row in items):
        return None

    buf = StringIO()
    writer = csv_writer(buf, lineterminator="\n")
    for row in items:
        writer.writerow([_inline_csv_cell(row.get(col)) for col in columns])
    csv_body = buf.getvalue().rstrip("\n")
    declarations = []
    for column in columns:
        nullable = any(column not in row or row.get(column) is None for row in items)
        values = [row.get(column) for row in items if column in row and row.get(column) is not None]
        if all(isinstance(value, bool) for value in values):
            type_tag = "bool"
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            type_tag = "int"
        elif all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
            type_tag = "float"
        elif all(isinstance(value, str) for value in values):
            type_tag = "string"
        else:
            type_tag = "json"
        declarations.append(f"{column}:{type_tag}{'?' if nullable else ''}")
    return f"[{len(items)}]{{{','.join(declarations)}}}\n{csv_body}"


def _flatten_uniform_nested_objects(items: list[dict[str, Any]], *, max_inner_keys: int = 6) -> list[dict[str, Any]]:
    if not items:
        return items
    columns = _all_columns(items)
    flattenable: dict[str, list[str]] = {}
    for column in columns:
        values = [row.get(column) for row in items]
        if not values or not all(isinstance(value, dict) for value in values):
            continue
        key_sets = [tuple(sorted(str(key) for key in value)) for value in values]
        first_keys = key_sets[0]
        if not first_keys or len(first_keys) > max_inner_keys:
            continue
        if all(keys == first_keys for keys in key_sets):
            flattenable[column] = list(first_keys)

    if not flattenable:
        return items

    flattened: list[dict[str, Any]] = []
    for row in items:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key not in flattenable:
                out[key] = value
                continue
            assert isinstance(value, dict)
            for inner_key in flattenable[key]:
                out[f"{key}.{inner_key}"] = value.get(inner_key)
        flattened.append(out)
    return flattened


def _csv_cell(value: Any, ccr: _CCRLike | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        if ccr is not None:
            kind = _classify_opaque_string(value)
            if kind is not None:
                handle = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
                ccr.put_with_handle(handle, value)
                size = _size_label(len(value.encode("utf-8")))
                return f"<<ccr:{handle},{kind},{size}>>"
        nested = _compact_stringified_json_cell(value)
        if nested is not None:
            return nested
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv_quote_if_needed(value: str) -> str:
    if value.startswith("<<ccr:") and value.endswith(">>"):
        return value
    if any(char in value for char in (",", '"', "\n", "\r")):
        return f'"{value.replace("\"", "\"\"")}"'
    return value


def _render_csv_row(values: list[str]) -> str:
    return ",".join(_csv_quote_if_needed(value) for value in values)


def _core_columns(items: list[dict[str, Any]]) -> list[str]:
    threshold = max(1, math.ceil(len(items) * 0.8))
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for row in items:
        for key in row:
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, len(first_seen))
    return sorted((key for key, count in counts.items() if count >= threshold), key=lambda k: first_seen[k])


def _all_columns(items: list[dict[str, Any]]) -> list[str]:
    first_seen: dict[str, int] = {}
    for row in items:
        for key in row:
            first_seen.setdefault(key, len(first_seen))
    return sorted(first_seen, key=lambda k: first_seen[k])


def _bucket_discriminator(items: list[dict[str, Any]]) -> str | None:
    if len(items) < 4:
        return None
    candidates: list[str] = []
    for key in _all_columns(items):
        values = [row.get(key) for row in items]
        if any(value is None or not isinstance(value, str | int | bool) for value in values):
            continue
        unique_values = set(values)
        if 2 <= len(unique_values) <= 8:
            min_bucket = min(values.count(value) for value in unique_values)
            if min_bucket >= 2:
                candidates.append(key)
    if not candidates:
        return None
    for preferred in ("type", "kind", "category", "event"):
        if preferred in candidates:
            return preferred
    return candidates[0]


def _compact_bucketed_table(items: list[dict[str, Any]], ccr: _CCRLike | None = None) -> str | None:
    discriminator = _bucket_discriminator(items)
    if discriminator is None:
        return None

    buckets: dict[Any, list[dict[str, Any]]] = {}
    for row in items:
        buckets.setdefault(row[discriminator], []).append(row)

    parts = [f"__buckets:{discriminator}"]
    for key in sorted(buckets, key=lambda value: str(value)):
        bucket_rows = buckets[key]
        table = _compact_homogeneous_table(bucket_rows, ccr)
        if table is None:
            return None
        parts.append(f"__key:{key}")
        parts.append(table)
    return "\n".join(parts)


def _column_type_tag(column: str, items: list[dict[str, Any]]) -> str:
    values = [row.get(column) for row in items if column in row and row.get(column) is not None]
    if not values:
        return "string"
    if all(isinstance(value, bool) for value in values):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "int"
    if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
        return "float"
    if all(isinstance(value, str) for value in values):
        return "string"
    return "json"


def _format_table_declaration(columns: list[str], items: list[dict[str, Any]]) -> str:
    declarations = []
    for column in columns:
        nullable = any(column not in row or row.get(column) is None for row in items)
        suffix = "?" if nullable else ""
        declarations.append(f"{column}:{_column_type_tag(column, items)}{suffix}")
    return f"[{len(items)}]{{{','.join(declarations)}}}"


def _ordered_table_columns(items: list[dict[str, Any]]) -> list[str]:
    core = _core_columns(items)
    all_columns = _all_columns(items)
    record_fields = {"record_id", "needle_id", "answer", "owner"}
    if record_fields & set(all_columns):
        preferred = ["record_id", "needle_id", "id", "answer", "status", "owner"]
        front = [column for column in preferred if column in all_columns]
        return front + [column for column in all_columns if column not in front]
    if any(any(column not in row or row.get(column) is None for row in items) for column in all_columns):
        return all_columns
    optional = [column for column in all_columns if column not in core]
    if optional:
        return core + optional
    return sorted(all_columns)


def _compact_homogeneous_table(items: list[dict[str, Any]], ccr: _CCRLike | None = None) -> str | None:
    items = _flatten_uniform_nested_objects(items)
    columns = _core_columns(items)
    if not columns:
        return None
    rows_with_core = sum(1 for row in items if all(col in row for col in columns))
    if rows_with_core / len(items) < 0.8:
        return None
    columns = _ordered_table_columns(items)
    if not any(all(_is_scalar(row.get(col)) for col in columns) for row in items[: min(10, len(items))]):
        return None

    buf = StringIO()
    for row in items:
        buf.write(_render_csv_row([_csv_cell(row.get(col), ccr) for col in columns]))
        buf.write("\n")
    csv_body = buf.getvalue().rstrip("\n")

    return f"{_format_table_declaration(columns, items)}\n{csv_body}"


def _compact_table(items: list[dict[str, Any]], ccr: _CCRLike | None = None) -> str | None:
    """Render a homogeneous JSON array as a compact CSV+schema table.

    This mirrors Headroom SmartCrusher's lossless-first table compaction in
    spirit: preserve all rows and navigable columns when the schema is stable.
    Values are JSON-encoded inside CSV cells, so strings, booleans, nulls, and
    nested values remain unambiguous for query-scoped CCR recovery.
    """
    if len(items) < 5:
        return None
    bucketed = _compact_bucketed_table(items, ccr)
    if bucketed is not None:
        return bucketed
    table = _compact_homogeneous_table(items, ccr)
    if table is None:
        return None

    return table


_CCR_MARKER_RE = re.compile(
    r"\[CCR:([0-9a-f]{12})\]|Retrieve (?:more|full [^:]+): hash=[0-9a-f]{12,24}|<<ccr:[^>]+>>"
)
_COMPACT_TABLE_DECL_RE = re.compile(r"(?m)^\[\d+\]\{[^\n{}]+\}$")
_QUERY_GUIDANCE = (
    "Use exact IDs, symbols, labels, or schema words in query. If the needed "
    "instruction/value is hidden inside the payload, query distinctive labels "
    "such as target, instruction, operation, question, key, or id; avoid broad "
    "generic terms that match many repeated lines."
)
_OPAQUE_STRING_MIN_BYTES = 256
_BASE64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def _ccr_prefix(handle: str, label: str, original_count: int, compressed_count: int) -> str:
    return (
        f"[CCR:{handle}] {label}: {original_count} -> {compressed_count}. "
        f"Retrieve a relevant subset with ccr_retrieve(handle={handle}, query=<what you need>) "
        f"-- query required, returns only matching items. {_QUERY_GUIDANCE}"
    )


def _maybe_wrap(
    *,
    original: str,
    compressed_body: str,
    ccr: _CCRLike,
    label: str,
    original_count: int,
    compressed_count: int,
    min_savings_chars: int = 200,
    max_compression_ratio_for_ccr: float | None = None,
) -> str:
    if not compressed_body or len(original) - len(compressed_body) < min_savings_chars:
        return original
    if (
        max_compression_ratio_for_ccr is not None
        and len(compressed_body) / max(1, len(original)) >= max_compression_ratio_for_ccr
    ):
        return original
    handle = ccr.put(original)
    return f"{_ccr_prefix(handle, label, original_count, compressed_count)}\n{compressed_body}"


def _lossless_savings_ratio(original: str, compressed_body: str) -> float:
    return 1.0 - (len(compressed_body) / max(1, len(original)))


def _looks_like_compact_table_text(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if (match := _COMPACT_TABLE_DECL_RE.search(stripped)) and ":" in match.group(0):
        return True
    return stripped.startswith("__buckets:") and any(":" in match.group(0) for match in _COMPACT_TABLE_DECL_RE.finditer(stripped))


def _contains_compact_table_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(value, str):
        return _looks_like_compact_table_text(value)
    if isinstance(value, dict):
        return any(_contains_compact_table_value(child, depth=depth + 1) for child in value.values())
    if isinstance(value, list):
        return any(_contains_compact_table_value(child, depth=depth + 1) for child in value)
    return False


def _looks_like_lossless_compact_table_output(text: str) -> bool:
    if _looks_like_compact_table_text(text):
        return True
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return False
    return _contains_compact_table_value(parsed)


def _parse_jsonish_lines(text: str) -> list[Any] | None:
    rows: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except Exception:
            return None
    return rows or None


def _minify_json_value(value: Any, original: str) -> str:
    try:
        minified = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return original
    return minified if len(minified) < len(original) else original


def _size_label(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    return f"{num_bytes / 1024:.1f}KB"


def _classify_opaque_string(value: str) -> str | None:
    encoded_len = len(value.encode("utf-8"))
    if encoded_len < _OPAQUE_STRING_MIN_BYTES:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    whitespace_count = sum(ch.isspace() for ch in stripped)
    if whitespace_count:
        return None

    base64_ratio = sum(ch in _BASE64_CHARS for ch in stripped) / len(stripped)
    if base64_ratio >= 0.95 and len(set(stripped)) >= 16:
        return "base64"

    if encoded_len >= 1024:
        return "string"

    return None


def _schema_sample(value: Any, *, depth: int = 0) -> Any:
    """Compact a JSON object while preserving navigable shape and identifiers."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in list(value.items())[:40]:
            out[str(key)] = _schema_sample(child, depth=depth + 1)
        if len(value) > 40:
            out["..."] = f"{len(value) - 40} keys omitted"
        return out
    if isinstance(value, list):
        sample = [_schema_sample(child, depth=depth + 1) for child in value[:3]]
        if len(value) > 3:
            sample.append(f"... {len(value) - 3} items omitted")
        return sample
    if isinstance(value, str):
        if len(value) <= 80:
            return value
        return value[:64] + "..."
    return value


def _compact_document_value(value: Any, ccr: _CCRLike, *, depth: int = 0) -> tuple[Any, bool]:
    """Lossless document walker for nested JSON tables.

    Headroom's SmartCrusher runs a document-level pass before lossy sampling:
    stable tabular sub-arrays are rendered as CSV+schema strings so all rows stay
    visible without a retrieval round-trip, and long opaque blobs are replaced
    with CCR markers while the exact value stays recoverable from the store.
    """
    if depth > 8:
        return value, False
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for key, child in value.items():
            compacted, child_changed = _compact_document_value(child, ccr, depth=depth + 1)
            out[key] = compacted
            changed = changed or child_changed
        return out, changed
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            table = _compact_table(value, ccr)
            if table is not None:
                return table, True
        changed = False
        out: list[Any] = []
        for item in value:
            compacted, child_changed = _compact_document_value(item, ccr, depth=depth + 1)
            out.append(compacted)
            changed = changed or child_changed
        return out, changed
    if isinstance(value, str):
        kind = _classify_opaque_string(value)
        if kind is None:
            return value, False
        handle = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        ccr.put_with_handle(handle, value)
        size = _size_label(len(value.encode("utf-8")))
        return f"<<ccr:{handle},{kind},{size}>>", True
    return value, False


def _looks_like_search(lines: list[str]) -> bool:
    matches = 0
    for line in lines[:200]:
        parsed = _parse_search_line(line)
        if parsed is None:
            continue
        path, _line_no, _body, _sep = parsed
        if "/" in path or "." in path or "\\" in path:
            matches += 1
        if matches >= 10:
            return True
    return False


def _detect_search_for_routing(lines: list[str]) -> bool:
    matches = 0
    for line in lines[:200]:
        parsed = _parse_search_line(line)
        if parsed is None:
            continue
        path, _line_no, _body, _sep = parsed
        if "/" in path or "." in path or "\\" in path:
            matches += 1
        if matches >= 2:
            return True
    return False


def _line_signal_score(line: str) -> tuple[int, int, int, int]:
    uppercase_runs = len(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", line))
    number_runs = len(re.findall(r"\b\d{3,}\b", line))
    path_like = 1 if "/" in line or "\\" in line else 0
    return (uppercase_runs, number_runs, path_like, len(line))


def _search_match_score(body: str) -> tuple[int, int, int, int, int]:
    if _ERROR_LINE_RE.search(body):
        priority = 3
    elif _WARN_LINE_RE.search(body):
        priority = 2
    elif _IMPORTANCE_LINE_RE.search(body):
        priority = 1
    else:
        priority = 0
    return (priority, *_line_signal_score(body))


def _parse_search_line(line: str) -> tuple[str, int, str, str] | None:
    m = _SEARCH_COLON_LINE_RE.match(line)
    sep = ":"
    if m is None:
        m = _SEARCH_DASH_LINE_RE.match(line)
        sep = "-"
    if m is None:
        return None

    file_path = m.group("file")
    if not file_path:
        return None
    if sep == "-" and file_path.endswith("-"):
        return None
    try:
        line_no = int(m.group("line"))
    except ValueError:
        return None
    return file_path, line_no, m.group("body"), sep


def _select_search_indices(rows: list[tuple[int, str, str]], *, max_matches: int = 5) -> list[int]:
    if len(rows) <= max_matches:
        return list(range(len(rows)))
    selected: set[int] = {0, len(rows) - 1}
    remaining = [idx for idx in range(1, len(rows) - 1)]
    remaining.sort(key=lambda idx: (_search_match_score(rows[idx][1]), -idx), reverse=True)
    for idx in remaining:
        if len(selected) >= max_matches:
            break
        selected.add(idx)
    return sorted(selected)


def _search_file_score(rows: list[tuple[int, str, str]]) -> tuple[int, int, int, int, int, int]:
    best_match = max((_search_match_score(body) for _line_no, body, _sep in rows), default=(0, 0, 0, 0, 0))
    return (*best_match, len(rows))


def _compress_search(lines: list[str], original: str, ccr: _CCRLike) -> str:
    by_file: dict[str, list[tuple[int, str, str]]] = {}
    passthrough: list[str] = []
    for line in lines:
        parsed = _parse_search_line(line)
        if parsed is None:
            if line.strip():
                passthrough.append(line)
            continue
        file_path, line_no, body, sep = parsed
        by_file.setdefault(file_path, []).append((line_no, body, sep))

    original_matches = sum(len(v) for v in by_file.values())
    if original_matches < SEARCH_MIN_MATCHES:
        return original

    out: list[str] = []
    kept_matches = 0
    selected_files = sorted(by_file)
    omitted_files = 0
    if len(selected_files) > SEARCH_MAX_FILES:
        selected_files = sorted(
            sorted(by_file),
            key=lambda file_path: (_search_file_score(by_file[file_path]), file_path),
            reverse=True,
        )[:SEARCH_MAX_FILES]
        selected_files.sort()
        omitted_files = len(by_file) - len(selected_files)

    selected_rows_by_file: dict[str, list[tuple[int, str, str]]] = {}
    remaining_match_budget = SEARCH_MAX_MATCHES_TOTAL
    ranked_selected_files = sorted(
        selected_files,
        key=lambda file_path: (_search_file_score(by_file[file_path]), file_path),
        reverse=True,
    )
    for file_path in ranked_selected_files:
        if remaining_match_budget <= 0:
            omitted_files += 1
            continue
        rows = by_file[file_path]
        max_matches = min(SEARCH_MAX_MATCHES_PER_FILE, remaining_match_budget)
        selected = [rows[i] for i in _select_search_indices(rows, max_matches=max_matches)]
        if not selected:
            omitted_files += 1
            continue
        selected_rows_by_file[file_path] = selected
        remaining_match_budget -= len(selected)

    for file_path in sorted(selected_rows_by_file):
        rows = by_file[file_path]
        selected = selected_rows_by_file[file_path]
        if len(rows) > len(selected):
            out.append(f"{file_path} ({len(rows)} matches, showing {len(selected)})")
            for line_no, body, _sep in selected:
                out.append(f"  {line_no}: {body}")
            out.append(f"[... and {len(rows) - len(selected)} more matches]")
        else:
            for line_no, body, sep in selected:
                out.append(f"{file_path}{sep}{line_no}{sep}{body}")
        kept_matches += len(selected)
    if omitted_files:
        out.append(f"... {omitted_files} files omitted ...")
    if passthrough:
        out.append("-- non-match lines --")
        out.extend(passthrough[:10])

    return _maybe_wrap(
        original=original,
        compressed_body="\n".join(out),
        ccr=ccr,
        label="compressed search results",
        original_count=original_matches,
        compressed_count=kept_matches,
        max_compression_ratio_for_ccr=0.8,
    )


def _looks_like_diff(lines: list[str]) -> bool:
    if len(lines) < DIFF_MIN_LINES:
        return False
    return _has_diff_signals(lines)


def _has_diff_signals(lines: list[str]) -> bool:
    sample = "\n".join(lines[:500])
    return (
        any(_DIFF_FILE_HEADER_RE.match(line) or _DIFF_HUNK_RE.match(line) for line in lines[:500])
        or ("--- " in sample and "+++ " in sample)
    )


def _diff_hunk_score(hunk: list[str]) -> tuple[int, int, int, int, int, int]:
    text = "\n".join(hunk)
    priority = 0
    if _ERROR_LINE_RE.search(text):
        priority = 3
    elif _IMPORTANCE_LINE_RE.search(text):
        priority = 2
    elif re.search(r"security|auth|password|secret|token", text, re.IGNORECASE):
        priority = 1
    changed = sum(
        1
        for line in hunk
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return (priority, changed, *_line_signal_score(text))


def _trim_diff_hunk_context(hunk: list[str], *, context: int = 2) -> list[str]:
    if not hunk:
        return hunk
    keep: set[int] = {0}
    for i, line in enumerate(hunk):
        if (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            for j in range(max(1, i - context), min(len(hunk), i + context + 1)):
                keep.add(j)
        elif line.startswith("\\ No newline"):
            keep.add(i)
    return [hunk[i] for i in sorted(keep)]


def _select_diff_hunks(hunks: list[list[str]], *, max_hunks: int = 10) -> tuple[list[list[str]], int]:
    if len(hunks) <= max_hunks:
        return hunks, 0
    selected: set[int] = {0, len(hunks) - 1}
    remaining = [idx for idx in range(1, len(hunks) - 1)]
    remaining.sort(key=lambda idx: (_diff_hunk_score(hunks[idx]), -idx), reverse=True)
    for idx in remaining:
        if len(selected) >= max_hunks:
            break
        selected.add(idx)
    return [hunks[i] for i in sorted(selected)], len(hunks) - len(selected)


def _diff_header_score(header: list[str]) -> tuple[int, int, int, int, int, int]:
    text = "\n".join(header)
    priority = 0
    if _ERROR_LINE_RE.search(text):
        priority = 3
    elif _DIFF_BINARY_LINE_RE.search(text) or _DIFF_METADATA_LINE_RE.search(text):
        priority = 2
    elif _IMPORTANCE_LINE_RE.search(text):
        priority = 2
    elif re.search(r"security|auth|password|secret|token", text, re.IGNORECASE):
        priority = 1
    return (priority, 0, *_line_signal_score(text))


def _diff_file_score(section: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    hunks = section.get("hunks", [])
    hunk_score = max((_diff_hunk_score(hunk) for hunk in hunks), default=(0, 0, 0, 0, 0, 0))
    return max(_diff_header_score(section.get("header", [])), hunk_score)


def _select_diff_file_sections(
    sections: list[dict[str, Any]], *, max_files: int = 20
) -> tuple[list[dict[str, Any]], int]:
    if len(sections) <= max_files:
        return sections, 0
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda item: (_diff_file_score(item[1]), -item[0]), reverse=True)
    selected_indices = {idx for idx, _section in indexed[:max_files]}
    selected = [section for idx, section in enumerate(sections) if idx in selected_indices]
    omitted_hunks = sum(len(section["hunks"]) for idx, section in enumerate(sections) if idx not in selected_indices)
    return selected, omitted_hunks


def _compress_diff(lines: list[str], original: str, ccr: _CCRLike) -> str:
    additions = deletions = files = 0
    structural_re = re.compile(
        r"^(diff --(?:git|combined|cc) |index |similarity index|rename (from|to) |"
        r"new file mode|deleted file mode|--- |\+\+\+ |@{2,4} |\\ No newline)"
    )
    prefix: list[str] = []
    file_sections: list[dict[str, Any]] = []
    current_file: dict[str, Any] | None = None
    current_hunk: list[str] | None = None

    def finish_hunk() -> None:
        nonlocal current_hunk
        if current_hunk is not None:
            if current_file is None:
                prefix.extend(current_hunk)
            else:
                current_file["hunks"].append(current_hunk)
        current_hunk = None

    def finish_file() -> None:
        nonlocal current_file
        finish_hunk()
        if current_file is not None:
            file_sections.append(current_file)
        current_file = None

    for i, line in enumerate(lines):
        if _DIFF_FILE_HEADER_RE.match(line):
            finish_file()
            current_file = {"header": [line], "hunks": []}
            files += 1
            continue
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        if line.startswith("-") and not line.startswith("---"):
            deletions += 1

        if _DIFF_HUNK_RE.match(line):
            finish_hunk()
            current_hunk = [line]
            continue
        if current_hunk is not None:
            current_hunk.append(line)
        elif current_file is not None:
            if structural_re.match(line) or line.strip():
                current_file["header"].append(line)
        else:
            prefix.append(line)

    finish_file()

    if not file_sections:
        return original

    out = list(prefix)
    selected_sections, omitted_hunks = _select_diff_file_sections(file_sections)
    for section in selected_sections:
        hunks = section["hunks"]
        if hunks:
            selected_hunks, omitted = _select_diff_hunks(hunks)
            omitted_hunks += omitted
            out.extend(section["header"])
            for hunk in selected_hunks:
                out.extend(_trim_diff_hunk_context(hunk))
        else:
            out.extend(section["header"])
    summary = f"[{files} files changed, +{additions} -{deletions} lines"
    if omitted_hunks:
        summary += f", {omitted_hunks} hunks omitted"
    summary += "]"
    out.append(summary)
    compressed_body = "\n".join(out)
    return _maybe_wrap(
        original=original,
        compressed_body=compressed_body,
        ccr=ccr,
        label="compressed diff",
        original_count=len(lines),
        compressed_count=len(compressed_body.splitlines()),
        max_compression_ratio_for_ccr=0.8,
    )


def _looks_like_code(lines: list[str]) -> bool:
    return _code_detector_hit_count(lines) >= 4


def _code_detector_hit_count(lines: list[str]) -> int:
    hits = 0
    for line in lines[:200]:
        stripped = line.strip()
        if (
            _CODE_SIGNATURE_RE.match(line)
            or _TS_TYPE_RE.match(line)
            or _JS_IMPORT_EXPORT_RE.match(line)
            or _GO_PACKAGE_RE.match(line)
            or _JAVA_PACKAGE_RE.match(line)
            or _CODE_TYPE_ANNOTATION_RE.search(line)
            or stripped.startswith(("@", "#[", "module.exports"))
        ):
            hits += 1
    return hits


def _detect_code_for_routing(lines: list[str]) -> bool:
    return _code_detector_hit_count(lines) >= 3


def _compact_python_docstring_line(line: str) -> str | None:
    match = _PY_DOCSTRING_START_RE.match(line)
    if match is None:
        return None
    quote = match.group("quote")
    text = match.group("text").strip()
    if text.endswith(quote):
        text = text[: -len(quote)].strip()
    if not text:
        return None
    return f"{match.group('indent')}{quote}{text}{quote}"


def _extract_code_signature_name(signature: str) -> str:
    stripped = signature.strip()
    match = re.search(r"(?:def|func|fn|function)\s+(?:\([^)]*\)\s*)?(\w+)", stripped)
    if match:
        return f"{match.group(1)}()"
    match = re.search(r"(?:public|private|protected|static|async|export)\s+.*?(\w+)\s*\(", stripped)
    if match:
        return f"{match.group(1)}()"
    match = re.search(r"class\s+(\w+)", stripped)
    if match:
        return match.group(1)
    match = re.search(r"(\w+)\s*\(", stripped)
    if match:
        return f"{match.group(1)}()"
    return ""


def _is_code_body_signature(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("import ", "from ")):
        return False
    return bool(_extract_code_signature_name(stripped))


def _summarize_compressed_code_signatures(signatures: list[str], compressed_count: int) -> str:
    if not signatures or compressed_count <= 0:
        return ""
    names: list[str] = []
    seen: set[str] = set()
    for signature in signatures:
        name = _extract_code_signature_name(signature)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        return f"{compressed_count} function bodies compressed"
    shown = names[:6]
    suffix = f" (+{len(names) - 6} more)" if len(names) > 6 else ""
    return f"{compressed_count} bodies compressed: {', '.join(shown)}{suffix}"


def _compress_code(lines: list[str], original: str, ccr: _CCRLike) -> str:
    auto_keep: set[int] = set(range(min(8, len(lines)))) | set(range(max(0, len(lines) - 8), len(lines)))
    keep: set[int] = set(auto_keep)
    replacements: dict[int, str] = {}
    docstring_body: set[int] = set()
    signatures: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if _CODE_SIGNATURE_RE.match(line):
            if _is_code_body_signature(line):
                signatures.append((i, line))
            keep.add(i)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith(("{", ":", "where")):
                keep.add(i + 1)
            if i + 1 < len(lines):
                docstring = _compact_python_docstring_line(lines[i + 1])
                if docstring is not None:
                    keep.add(i + 1)
                    replacements[i + 1] = docstring
                    quote_match = _PY_DOCSTRING_START_RE.match(lines[i + 1])
                    quote = quote_match.group("quote") if quote_match is not None else ""
                    first_line_body = quote_match.group("text") if quote_match is not None else ""
                    if quote and quote not in first_line_body:
                        for j in range(i + 2, len(lines)):
                            docstring_body.add(j)
                            if quote in lines[j]:
                                break
        elif line.strip().startswith(("@", "# type:", "// @")):
            keep.add(i)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith(("{", ":", "where")):
                keep.add(i + 1)
    keep.difference_update(docstring_body)
    for i in list(keep & auto_keep):
        stripped = lines[i].strip()
        if i in replacements or not stripped:
            continue
        if lines[i][:1].isspace() and not _CODE_SIGNATURE_RE.match(lines[i]):
            keep.discard(i)
    if len(keep) >= len(lines) * 0.75:
        return original
    out: list[str] = []
    compressed_signature_names = [line for i, line in signatures if i in keep]
    summary = _summarize_compressed_code_signatures(compressed_signature_names, len(compressed_signature_names))
    if summary:
        out.append(f"[{summary}]")
    last = -2
    for i in sorted(keep):
        if i != last + 1 and out:
            out.append("...")
        out.append(replacements.get(i, lines[i]))
        last = i
    return _maybe_wrap(
        original=original,
        compressed_body="\n".join(out),
        ccr=ccr,
        label="compressed code",
        original_count=len(lines),
        compressed_count=len(out),
    )


def _traceback_indices(lines: list[str], *, max_stacks: int = 3, max_lines: int = 20) -> set[int]:
    keep: set[int] = set()
    stacks_seen = 0
    i = 0
    while i < len(lines) and stacks_seen < max_stacks:
        if not lines[i].lstrip().startswith("Traceback (most recent call last)"):
            i += 1
            continue

        stacks_seen += 1
        count = 0
        j = i
        while j < len(lines) and count < max_lines:
            line = lines[j]
            stripped = line.strip()
            keep.add(j)
            count += 1
            j += 1

            if not stripped:
                break
            if j == i + 1:
                continue
            if line[:1].isspace():
                continue
            if stripped.startswith((
                "Traceback ",
                "During handling of the above exception",
                "The above exception was the direct cause",
            )):
                continue
            break
        i = max(j, i + 1)
    return keep


def _warning_dedupe_key(line: str) -> str:
    split_at = min((idx for idx in (line.find(":"), line.find("=")) if idx >= 0), default=len(line))
    prefix = line[:split_at]
    suffix = line[split_at:]
    suffix = _DEDUPE_DIGIT_RE.sub("N", suffix)
    suffix = _DEDUPE_HEX_RE.sub("ADDR", suffix)
    suffix = _DEDUPE_PATH_RE.sub("/PATH/", suffix)
    return prefix + suffix


def _warning_indices(lines: list[str], *, max_warnings: int = 5) -> set[int]:
    keep: set[int] = set()
    seen: set[str] = set()
    for i, line in enumerate(lines):
        if not _WARN_LINE_RE.search(line):
            continue
        key = _warning_dedupe_key(line)
        if key in seen:
            continue
        seen.add(key)
        keep.add(i)
        if len(keep) >= max_warnings:
            break
    return keep


def _select_first_last_high_signal(indices: list[int], lines: list[str], *, max_count: int = 10) -> set[int]:
    if len(indices) <= max_count:
        return set(indices)
    selected: set[int] = {indices[0], indices[-1]}
    remaining = [idx for idx in indices[1:-1] if idx not in selected]
    remaining.sort(key=lambda idx: (_line_signal_score(lines[idx]), -idx), reverse=True)
    for idx in remaining:
        if len(selected) >= max_count:
            break
        selected.add(idx)
    return selected


def _tokenize_log_template_line(line: str) -> list[str]:
    return line.split()


def _template_extends_run(template: list[str | None], tokens: list[str], *, threshold: float = 0.4) -> bool:
    if len(tokens) != len(template) or not tokens:
        return False
    matches = 0
    for slot, token in zip(template, tokens, strict=True):
        if slot is None or slot == token:
            matches += 1
    return (matches / len(tokens)) >= threshold


def _merge_log_template(template: list[str | None], tokens: list[str]) -> None:
    for index, token in enumerate(tokens):
        if template[index] is not None and template[index] != token:
            template[index] = None


def _flush_log_template_run(
    run_indices: list[int],
    template: list[str | None],
    tokenized: list[list[str]],
    lines: list[str],
    out: list[str],
    *,
    template_id: int,
    min_run: int = 3,
    min_constant_tokens: int = 2,
) -> bool:
    constant_count = sum(1 for slot in template if slot is not None)
    varying_count = len(template) - constant_count
    if len(run_indices) < min_run or constant_count < min_constant_tokens or varying_count == 0:
        out.extend(lines[index] for index in run_indices)
        return False

    template_text = " ".join(slot if slot is not None else "<*>" for slot in template)
    out.append(f"[Template T{template_id}: {template_text}] ({len(run_indices)} occurrences)")
    for line_index in run_indices:
        variants = [
            token
            for token, slot in zip(tokenized[line_index], template, strict=True)
            if slot is None
        ]
        out.append(" ".join(variants))
    return True


def _compact_log_templates(lines: list[str], original: str, ccr: _CCRLike) -> str:
    if len(lines) < 20 or any(_ERROR_LINE_RE.search(line) or _WARN_LINE_RE.search(line) for line in lines):
        return original

    tokenized = [_tokenize_log_template_line(line) for line in lines]
    out: list[str] = []
    run_indices: list[int] = []
    run_template: list[str | None] = []
    template_id = 1
    collapsed = False

    def flush() -> None:
        nonlocal collapsed, run_indices, run_template, template_id
        if not run_indices:
            return
        did_collapse = _flush_log_template_run(
            run_indices,
            run_template,
            tokenized,
            lines,
            out,
            template_id=template_id,
        )
        if did_collapse:
            collapsed = True
            template_id += 1
        run_indices = []
        run_template = []

    for index, tokens in enumerate(tokenized):
        if not tokens:
            flush()
            out.append(lines[index])
            continue
        if run_indices and _template_extends_run(run_template, tokens):
            run_indices.append(index)
            _merge_log_template(run_template, tokens)
            continue
        flush()
        run_indices = [index]
        run_template = [token for token in tokens]
    flush()

    if not collapsed:
        return original
    body = "\n".join(out)
    return _maybe_wrap(
        original=original,
        compressed_body=body,
        ccr=ccr,
        label="compacted log templates",
        original_count=len(lines),
        compressed_count=len(out),
        max_compression_ratio_for_ccr=0.5,
    )


_LOG_LEVEL_RE = re.compile(
    r"\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL|PASSED|FAILED|SKIPPED)\b",
    re.IGNORECASE,
)
_LOG_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b")


def _looks_like_log(lines: list[str]) -> bool:
    if len(lines) < LOG_MIN_LINES:
        return False
    return _has_log_signals(lines)


def _has_log_signals(lines: list[str]) -> bool:
    sample = lines[: min(len(lines), 200)]
    hits = 0
    for line in sample:
        if _LOG_LEVEL_RE.search(line) or _LOG_TIMESTAMP_RE.search(line) or _SUMMARY_LINE_RE.search(line):
            hits += 1
        if line.startswith(("Traceback ", "npm ERR!", "cargo ", "pytest ", "FAIL ", "PASS ")):
            hits += 2
    return hits >= 4


def _compress_log(lines: list[str], original: str, ccr: _CCRLike) -> str:
    templated = _compact_log_templates(lines, original, ccr)
    if templated != original:
        return templated

    keep: set[int] = set(range(min(12, len(lines)))) | set(range(max(0, len(lines) - 18), len(lines)))
    keep.update(_traceback_indices(lines))
    keep.update(_warning_indices(lines))
    error_indices = [
        i
        for i, line in enumerate(lines)
        if _ERROR_LINE_RE.search(line) or _SUMMARY_LINE_RE.search(line)
    ]
    selected_error_indices = _select_first_last_high_signal(error_indices, lines)
    for i, line in enumerate(lines):
        if i in selected_error_indices:
            for j in range(max(0, i - 2), min(len(lines), i + 4)):
                keep.add(j)
    if len(keep) >= len(lines):
        return original
    out: list[str] = []
    last = -2
    for i in sorted(keep):
        if i != last + 1 and out:
            out.append(f"... ({i - last - 1} lines elided) ...")
        out.append(lines[i])
        last = i
    return _maybe_wrap(
        original=original,
        compressed_body="\n".join(out),
        ccr=ccr,
        label="compressed log",
        original_count=len(lines),
        compressed_count=len(out),
        max_compression_ratio_for_ccr=0.5,
    )


_TEXT_HEADER_RE = re.compile(r"^\s*(#{1,6}\s+\S.*|[A-Z][A-Z0-9 _/-]{8,}|[A-Z][\w .:/-]{2,}:)\s*$")
_TEXT_ANCHOR_RE = re.compile(
    r"(?<!\w)([A-Z][A-Z0-9_-]{4,}|\d{4,}|[a-f0-9]{12,}|[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+|[A-Za-z_]\w+\(\))(?!\w)"
)


def _text_anchor_score(line: str) -> tuple[int, int, int]:
    header = 1 if _TEXT_HEADER_RE.match(line.strip()) else 0
    anchors = len(_TEXT_ANCHOR_RE.findall(line))
    return (header, anchors, len(line))


def _compress_plain_text(
    lines: list[str],
    original: str,
    ccr: _CCRLike,
    *,
    target_ratio: float | None = None,
) -> str:
    keep: set[int] = set(range(min(12, len(lines)))) | set(range(max(0, len(lines) - 12), len(lines)))
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _TEXT_HEADER_RE.match(stripped):
            keep.update(range(max(0, i - 1), min(len(lines), i + 3)))
        elif _TEXT_ANCHOR_RE.search(stripped):
            keep.add(i)

    if len(keep) > 80:
        required = set(range(min(12, len(lines)))) | set(range(max(0, len(lines) - 12), len(lines)))
        optional = [idx for idx in keep if idx not in required]
        optional.sort(key=lambda idx: (_text_anchor_score(lines[idx]), -idx), reverse=True)
        keep = required | set(optional[: max(0, 80 - len(required))])
    if target_ratio is not None:
        required = set(range(min(12, len(lines)))) | set(range(max(0, len(lines) - 12), len(lines)))
        optional = [idx for idx in keep if idx not in required]
        optional.sort(key=lambda idx: (_text_anchor_score(lines[idx]), -idx), reverse=True)
        max_keep = max(len(required), int(len(lines) * target_ratio))
        keep = required | set(optional[: max(0, max_keep - len(required))])

    if len(keep) >= len(lines) * 0.75:
        return original

    out: list[str] = []
    last = -2
    for i in sorted(keep):
        if i != last + 1 and out:
            out.append(f"... ({i - last - 1} lines elided) ...")
        out.append(lines[i])
        last = i

    return _maybe_wrap(
        original=original,
        compressed_body="\n".join(out),
        ccr=ccr,
        label="compressed text",
        original_count=len(lines),
        compressed_count=len(out),
    )


def _compress_mixed_sections(
    text: str,
    ccr: _CCRLike,
    *,
    target_ratio: float | None = None,
    config: CompressionConfig,
) -> str:
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        return text

    parts: list[str] = []
    cursor = 0
    changed = False
    for match in matches:
        parts.append(text[cursor : match.start()])
        label = match.group("label")
        body = match.group("body")
        compressed_body = _compress_inner(
            body,
            ccr,
            allow_mixed=False,
            protect_tags=False,
            target_ratio=target_ratio,
            config=config,
        )
        changed = changed or compressed_body != body
        parts.append(f"```{label}\n{compressed_body}\n```")
        cursor = match.end()
    parts.append(text[cursor:])
    if not changed:
        return text

    mixed_body = "".join(parts)
    return _maybe_wrap(
        original=text,
        compressed_body=mixed_body,
        ccr=ccr,
        label="compressed mixed content",
        original_count=len(text.splitlines()),
        compressed_count=len(mixed_body.splitlines()),
    )


def _compress_inner(
    text: str,
    ccr: _CCRLike,
    *,
    allow_mixed: bool,
    protect_tags: bool = True,
    target_ratio: float | None = None,
    config: CompressionConfig,
) -> str:
    target_ratio = _normalize_target_ratio(target_ratio)
    if protect_tags:
        return _compress_with_custom_tag_protection(text, ccr, target_ratio=target_ratio, config=config)

    stripped = text.strip()
    if allow_mixed:
        mixed = _compress_mixed_sections(text, ccr, target_ratio=target_ratio, config=config)
        if mixed != text:
            return mixed

    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    parsed_from_json = parsed is not None

    # JSONL arrays/objects are common tool output and Headroom routes them
    # through the same structured-data path as JSON arrays.
    if parsed is None:
        parsed = _parse_jsonish_lines(stripped)

    # JSON array of dicts -> statistical sample + anomaly (error) preservation.
    if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
        table = _compact_table(parsed, ccr)
        if table is not None and _lossless_savings_ratio(text, table) >= config.lossless_min_savings_ratio:
            return table

        if not _has_json_array_crush_signal(parsed):
            minified = _minify_json_value(parsed, text)
            return minified if minified != text else text

        kept, n, _n_err, kept_indices = _crush_list_of_dicts(parsed, budget=config.max_items_after_crush)
        if len(kept) >= n:
            return text
        if not config.enable_ccr_marker:
            return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
        canonical = _canonical_json_array_for_ccr(parsed)
        handle = _hash_canonical_for_ccr(canonical)
        ccr.put_with_handle(handle, canonical)
        dropped_count = n - len(kept)
        sentinel = {"_ccr_dropped": f"<<ccr:{handle} {dropped_count}_rows_offloaded>>"}
        summary = _summarize_dropped_dict_rows(parsed, kept_indices)
        if summary:
            sentinel["_ccr_summary"] = summary
        return json.dumps([*kept, sentinel], ensure_ascii=False, separators=(",", ":"))

    if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
        kept_strings = _sample_string_array(parsed, budget=config.max_items_after_crush)
        if len(kept_strings) < len(parsed):
            return json.dumps(kept_strings, ensure_ascii=False, separators=(",", ":"))

    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in parsed)
    ):
        kept_numbers = _sample_number_array(parsed, budget=config.max_items_after_crush)
        if len(kept_numbers) < len(parsed):
            return json.dumps(kept_numbers, ensure_ascii=False, separators=(",", ":"))

    if isinstance(parsed, list) and parsed:
        kept_mixed = _sample_mixed_array(parsed, budget=config.max_items_after_crush)
        if len(kept_mixed) < len(parsed):
            return json.dumps(kept_mixed, ensure_ascii=False, separators=(",", ":"))

    if isinstance(parsed, dict):
        compacted_doc, doc_changed = _compact_document_value(parsed, ccr)
        if doc_changed:
            body = json.dumps(compacted_doc, ensure_ascii=False, indent=2)
            if len(body) < len(text):
                return body

        minified = _minify_json_value(parsed, text)
        if minified != text:
            return minified

        body = json.dumps(_schema_sample(parsed), ensure_ascii=False, indent=2)
        return _maybe_wrap(
            original=text,
            compressed_body=body,
            ccr=ccr,
            label="compressed JSON object",
            original_count=len(stripped),
            compressed_count=len(body),
        )

    if parsed_from_json and parsed is not None:
        minified = _minify_json_value(parsed, text)
        if minified != text:
            return minified

    if _looks_like_html(stripped):
        html = _compress_html(text, ccr)
        if html != text:
            return html

    lines = stripped.splitlines()

    if _looks_like_search(lines):
        return _compress_search(lines, text, ccr)

    if _looks_like_diff(lines):
        return _compress_diff(lines, text, ccr)

    if len(lines) < DIFF_MIN_LINES and _has_diff_signals(lines):
        return text

    if len(lines) > 80 and _looks_like_code(lines):
        return _compress_code(lines, text, ccr)

    # Logs -> keep head/tail + errors, stack-adjacent context, warnings, summaries.
    if _looks_like_log(lines):
        return _compress_log(lines, text, ccr)

    if len(lines) < LOG_MIN_LINES and _has_log_signals(lines):
        return text

    if len(lines) > 30:
        return _compress_plain_text(lines, text, ccr, target_ratio=target_ratio)

    # Generic large text -> head + tail.
    handle = ccr.put(text)
    if target_ratio is None:
        head_chars, tail_chars = 1200, 600
    else:
        keep_chars = max(200, int(len(stripped) * target_ratio))
        head_chars = max(100, int(keep_chars * 2 / 3))
        tail_chars = max(100, keep_chars - head_chars)
    head, tail = stripped[:head_chars], stripped[-tail_chars:]
    return (
        f"[CCR:{handle}] compressed text ({len(stripped)} chars). "
        f"Retrieve relevant items with ccr_retrieve(handle={handle}, query=<what you need>) "
        f"-- query required, returns only matching items. {_QUERY_GUIDANCE}\n{head}\n... elided ...\n{tail}"
    )


def _detect_content_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "text"

    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return "json_array"

    lines = stripped.splitlines()
    if _has_diff_signals(lines):
        return "diff"

    if _detect_html_for_routing(stripped):
        return "html"

    if _detect_search_for_routing(lines):
        return "search"

    if _has_log_signals(lines):
        return "build"

    if _detect_code_for_routing(lines):
        return "source_code"

    return "text"


def compress(
    text: str,
    ccr: CCRStore,
    *,
    target_ratio: float | None = None,
    force: bool = False,
    config: CompressionConfig | None = None,
) -> str:
    """Compress a tool-output string; cache the original in CCR. No-op if small.

    Phase-4 (Context Tracker) idempotence guards, in order:
    - Pass through our own already-compressed output (text carrying a [CCR:..] marker),
      so we never double-compress or churn handles.
    - Pass through content whose handle is already marked expanded — once the model
      has retrieved an original, re-compressing it would make retrieval futile and
      drive the read->retrieve->re-elide thrash loop.
    """
    config = config or CompressionConfig()
    if not text or (not force and _approx_tokens(text) < config.min_tokens):
        return text

    # Guard 1: don't re-compress our own compressed output.
    if _CCR_MARKER_RE.search(text):
        return text

    # Guard 2: don't wrap lossless JSON table compaction in a later CCR pass.
    if _looks_like_lossless_compact_table_output(text):
        return text

    # Guard 3: don't re-compress an original the model has already expanded.
    if ccr.is_expanded(CCRStore.handle_for(text)):
        return text

    return _compress_inner(text, ccr, allow_mixed=True, target_ratio=target_ratio, config=config)
