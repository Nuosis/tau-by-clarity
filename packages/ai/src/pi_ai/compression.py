"""Universal active-compression hook for pi_ai.

Same chokepoint shape as the PII filter (pii.py): a registered, default-no-op
compressor transforms the outbound `Context` before dispatch, so large payloads
are compressed for ALL calls regardless of source. tau-by-clarity registers the real
content-aware compressor (which also caches the original in a CCR store so it can
be retrieved out-of-band).

Unlike PII, compression is ONE-WAY at this layer: there is no response transform.
Recovery of an original is out-of-band via the CCR store's retrieve path — never
by un-transforming the stream here.

If nothing is registered this is an exact no-op.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any
import uuid as _uuid

from .compression_policy import CompressionPolicy, policy_default_payg, resolve_policy
from .tokenization import count_text_tokens

# compress(text) -> text. The compressor itself decides what to touch (it size-
# gates and content-type-routes internally) and owns its CCR cache.
CompressFn = Callable[[str], str]
CompressionObserver = Callable[["CompressionEvent"], None]


@dataclass(frozen=True, slots=True)
class CompressionEvent:
    strategy: str
    original_tokens: int
    compressed_tokens: int
    original_bytes: int
    compressed_bytes: int
    role: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class CompressionStats:
    total_compressions: int
    total_original_tokens: int
    total_compressed_tokens: int
    total_tokens_saved: int
    total_original_bytes: int
    total_compressed_bytes: int
    total_bytes_saved: int
    compressions_by_strategy: dict[str, int]
    tokens_saved_by_strategy: dict[str, int]
    bytes_saved_by_strategy: dict[str, int]


@dataclass(frozen=True, slots=True)
class CompressionLearningStats:
    total_events: int
    total_skipped_read_only: int
    total_tokens_saved: int
    total_bytes_saved: int
    events_by_strategy: dict[str, int]
    skipped_by_strategy: dict[str, int]


@dataclass(frozen=True, slots=True)
class VolatileFinding:
    label: str
    sample: str


@dataclass(frozen=True, slots=True)
class CacheAlignmentStats:
    total_scans: int
    total_findings: int
    skipped_by_policy: int
    findings_by_label: dict[str, int]


@dataclass(frozen=True, slots=True)
class UnitOutcomeStats:
    total_units: int
    outcomes_by_reason: dict[str, int]
    outcomes_by_category: dict[str, int]


@dataclass(frozen=True, slots=True)
class CompressionCacheStats:
    hits: int
    misses: int
    entries: int
    tokens_saved: int


EXCLUDED_TOOL_NAMES = frozenset({"ccr_retrieve"})
DEFAULT_EXCLUDED_TOOL_RESULT_NAMES = frozenset({
    "Read",
    "Glob",
    "Grep",
    "Write",
    "Edit",
    "read",
    "glob",
    "grep",
    "write",
    "edit",
})
READ_TOOL_NAMES = frozenset({"read", "Read"})
MUTATING_TOOL_NAMES = frozenset({"edit", "write", "Edit", "Write"})
LIVE_CACHE_ZONE = "live"
IMAGE_MAX_DIMENSION = 512
IMAGE_MIN_BYTES = 64_000
IMAGE_JPEG_QUALITY = 85
OCR_MIN_CONFIDENCE = 0.7
READ_LIFECYCLE_MIN_BYTES = 512
LIVE_USER_TEXT_MIN_BYTES = 512
MIN_TOKENS_TO_COMPRESS = 200
ERROR_PROTECTION_MAX_CHARS = 8000
RECENT_CODE_PROTECTION_MESSAGES = 4
ERROR_INDICATOR_KEYWORDS = (
    "traceback",
    "error",
    "exception",
    "failed",
    "fail",
    "fatal",
    "critical",
    "crash",
    "timeout",
    "abort",
    "denied",
    "rejected",
)
ANALYSIS_INTENT_KEYWORDS = frozenset({
    "analyze",
    "analyse",
    "review",
    "audit",
    "inspect",
    "security",
    "vulnerability",
    "bug",
    "issue",
    "problem",
    "explain",
    "understand",
    "how does",
    "what does",
    "debug",
    "fix",
    "error",
    "wrong",
    "broken",
    "refactor",
    "improve",
    "optimize",
    "clean up",
})
IMAGE_OCR_INTENT_RE = re.compile(
    r"\b("
    r"read|ocr|extract|transcribe|text|document|invoice|receipt|error|traceback|stack\s+trace|"
    r"log|screenshot|terminal|console|message|what\s+does\s+.*say"
    r")\b",
    re.IGNORECASE,
)
CODE_SIGNATURE_RE = re.compile(
    r"^\s*(?:"
    r"(?:from\s+\S+\s+import\s+.+|import\s+.+)|"
    r"(?:async\s+)?def\s+\w+\s*\(.*|"
    r"class\s+\w+.*|"
    r"(?:export\s+)?(?:async\s+)?function\s+\w+\s*\(.*|"
    r"(?:const|let|var)\s+\w+\s*=.*=>.*|"
    r"(?:pub\s+)?(?:async\s+)?fn\s+\w+.*|"
    r"func\s+(?:\([^)]+\)\s*)?\w+\s*\(.*|"
    r"(?:public|private|protected)?\s*(?:class|interface|enum)\s+\w+.*"
    r")"
)
TOOL_SCHEMA_DROP_KEYS = frozenset({
    "$comment",
    "$id",
    "$schema",
    "deprecated",
    "example",
    "examples",
    "markdownDescription",
    "readOnly",
    "title",
    "writeOnly",
})
_CCR_MARKER_RE = re.compile(
    r"\[CCR:[0-9a-f]{12}\]|Retrieve (?:more|original|full [^:]+): hash=[0-9a-f]{12,24}|<<ccr:[^>]+>>"
)
_CCR_HANDLE_RE = re.compile(r"\[CCR:([0-9a-f]{12})\]")
_OWN_CCR_MARKER_RE = re.compile(r"\[CCR:[0-9a-f]{12}\]|<<ccr:[^>]+>>")
_READ_LIFECYCLE_MARKER_RE = re.compile(r"^\[Read content (?:stale|superseded): .*Retrieve original: hash=[0-9a-f]{12,24}\.")

_compressor: CompressFn | None = None
_observer: CompressionObserver | None = None
_stats_lock = Lock()
_breaker_lock = Lock()
_total_compressions = 0
_total_original_tokens = 0
_total_compressed_tokens = 0
_total_original_bytes = 0
_total_compressed_bytes = 0
_compressions_by_strategy: dict[str, int] = {}
_tokens_saved_by_strategy: dict[str, int] = {}
_bytes_saved_by_strategy: dict[str, int] = {}
_learning_total_events = 0
_learning_total_skipped_read_only = 0
_learning_total_tokens_saved = 0
_learning_total_bytes_saved = 0
_learning_events_by_strategy: dict[str, int] = {}
_learning_skipped_by_strategy: dict[str, int] = {}
_cache_alignment_scans = 0
_cache_alignment_findings = 0
_cache_alignment_skipped_by_policy = 0
_cache_alignment_findings_by_label: dict[str, int] = {}
_unit_outcomes_by_reason: dict[str, int] = {}
_unit_outcomes_by_category: dict[str, int] = {}
_compression_cache: OrderedDict[str, tuple[str, int]] = OrderedDict()
_compression_cache_hits = 0
_compression_cache_misses = 0
_compression_cache_tokens_saved = 0
_current_target_ratio: ContextVar[float | None] = ContextVar(
    "pi_ai_compression_target_ratio",
    default=None,
)
_current_force_compression: ContextVar[bool] = ContextVar(
    "pi_ai_compression_force_compression",
    default=False,
)
_current_min_tokens: ContextVar[int | None] = ContextVar(
    "pi_ai_compression_min_tokens",
    default=None,
)
_current_max_items_after_crush: ContextVar[int | None] = ContextVar(
    "pi_ai_compression_max_items_after_crush",
    default=None,
)
_current_lossless_min_savings_ratio: ContextVar[float | None] = ContextVar(
    "pi_ai_compression_lossless_min_savings_ratio",
    default=None,
)
_current_enable_ccr_marker: ContextVar[bool | None] = ContextVar(
    "pi_ai_compression_enable_ccr_marker",
    default=None,
)
_current_image_optimize: ContextVar[bool] = ContextVar(
    "pi_ai_compression_image_optimize",
    default=True,
)
_current_policy: ContextVar[CompressionPolicy] = ContextVar(
    "pi_ai_compression_policy",
    default=policy_default_payg(),
)
_current_text_compression_cache: ContextVar[dict[str, str] | None] = ContextVar(
    "pi_ai_text_compression_cache",
    default=None,
)
_breaker_failures = 0
_breaker_open_until = 0.0


def _breaker_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _breaker_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _breaker_threshold() -> int:
    return _breaker_env_int("HEADROOM_PIPELINE_BREAKER_THRESHOLD", 3)


def _breaker_cooldown_s() -> float:
    return _breaker_env_float("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", 60.0)


def register_compressor(fn: CompressFn) -> None:
    """Install the universal outbound compressor. Called by tau-by-clarity on import."""
    global _compressor
    _compressor = fn
    reset_compression_circuit_breaker()


def unregister_compressor() -> None:
    global _compressor
    _compressor = None
    reset_compression_circuit_breaker()


def has_compressor() -> bool:
    return _compressor is not None


def reset_compression_circuit_breaker() -> None:
    global _breaker_failures, _breaker_open_until
    with _breaker_lock:
        _breaker_failures = 0
        _breaker_open_until = 0.0


def get_compression_circuit_breaker_state() -> dict[str, Any]:
    threshold = _breaker_threshold()
    cooldown = _breaker_cooldown_s()
    now = time.monotonic()
    with _breaker_lock:
        open_until = _breaker_open_until
        failures = _breaker_failures
    return {
        "threshold": threshold,
        "cooldown_s": cooldown,
        "consecutive_failures": failures,
        "open": threshold > 0 and now < open_until,
        "open_remaining_s": max(0.0, open_until - now) if threshold > 0 else 0.0,
    }


def _compression_breaker_is_open() -> bool:
    threshold = _breaker_threshold()
    if threshold <= 0:
        return False
    with _breaker_lock:
        return time.monotonic() < _breaker_open_until


def _compression_breaker_record_failure() -> None:
    global _breaker_failures, _breaker_open_until
    threshold = _breaker_threshold()
    if threshold <= 0:
        return
    cooldown = _breaker_cooldown_s()
    with _breaker_lock:
        _breaker_failures += 1
        if _breaker_failures >= threshold:
            _breaker_open_until = time.monotonic() + cooldown
            _breaker_failures = 0


def _compression_breaker_record_success() -> None:
    global _breaker_failures
    if _breaker_threshold() <= 0:
        return
    with _breaker_lock:
        _breaker_failures = 0


def get_current_compression_target_ratio() -> float | None:
    return _current_target_ratio.get()


def get_current_compression_force_compression() -> bool:
    return _current_force_compression.get()


def get_current_compression_min_tokens() -> int | None:
    return _current_min_tokens.get()


def get_current_compression_max_items_after_crush() -> int | None:
    return _current_max_items_after_crush.get()


def get_current_compression_policy() -> CompressionPolicy:
    return _current_policy.get()


def get_current_compression_lossless_min_savings_ratio() -> float | None:
    return _current_lossless_min_savings_ratio.get()


def get_current_compression_enable_ccr_marker() -> bool | None:
    return _current_enable_ccr_marker.get()


def get_current_compression_image_optimize() -> bool:
    return _current_image_optimize.get()


def _current_min_tokens_to_compress() -> int:
    return _current_min_tokens.get() or MIN_TOKENS_TO_COMPRESS


def register_compression_observer(fn: CompressionObserver) -> None:
    global _observer
    _observer = fn


def unregister_compression_observer() -> None:
    global _observer
    _observer = None


def reset_compression_stats() -> None:
    global _total_compressions, _total_original_tokens, _total_compressed_tokens
    global _total_original_bytes, _total_compressed_bytes
    global _compression_cache_hits, _compression_cache_misses, _compression_cache_tokens_saved
    with _stats_lock:
        _total_compressions = 0
        _total_original_tokens = 0
        _total_compressed_tokens = 0
        _total_original_bytes = 0
        _total_compressed_bytes = 0
        _compressions_by_strategy.clear()
        _tokens_saved_by_strategy.clear()
        _bytes_saved_by_strategy.clear()
        _unit_outcomes_by_reason.clear()
        _unit_outcomes_by_category.clear()
        _compression_cache.clear()
        _compression_cache_hits = 0
        _compression_cache_misses = 0
        _compression_cache_tokens_saved = 0


def get_compression_stats() -> CompressionStats:
    with _stats_lock:
        return CompressionStats(
            total_compressions=_total_compressions,
            total_original_tokens=_total_original_tokens,
            total_compressed_tokens=_total_compressed_tokens,
            total_tokens_saved=max(0, _total_original_tokens - _total_compressed_tokens),
            total_original_bytes=_total_original_bytes,
            total_compressed_bytes=_total_compressed_bytes,
            total_bytes_saved=max(0, _total_original_bytes - _total_compressed_bytes),
            compressions_by_strategy=dict(_compressions_by_strategy),
            tokens_saved_by_strategy=dict(_tokens_saved_by_strategy),
            bytes_saved_by_strategy=dict(_bytes_saved_by_strategy),
        )


def reset_compression_learning_stats() -> None:
    global _learning_total_events, _learning_total_skipped_read_only
    global _learning_total_tokens_saved, _learning_total_bytes_saved
    with _stats_lock:
        _learning_total_events = 0
        _learning_total_skipped_read_only = 0
        _learning_total_tokens_saved = 0
        _learning_total_bytes_saved = 0
        _learning_events_by_strategy.clear()
        _learning_skipped_by_strategy.clear()


def get_compression_learning_stats() -> CompressionLearningStats:
    with _stats_lock:
        return CompressionLearningStats(
            total_events=_learning_total_events,
            total_skipped_read_only=_learning_total_skipped_read_only,
            total_tokens_saved=_learning_total_tokens_saved,
            total_bytes_saved=_learning_total_bytes_saved,
            events_by_strategy=dict(_learning_events_by_strategy),
            skipped_by_strategy=dict(_learning_skipped_by_strategy),
        )


_UNIT_REASON_CATEGORIES = {
    "applied": "applied",
    "protected_user_message": "protected_role",
    "protected_system_message": "protected_role",
    "protected_assistant_message": "protected_role",
    "immutable": "immutable",
    "below_unit_floor": "size_floor",
    "compressor_no_change": "compressor_noop",
    "compressor_unavailable": "compressor_noop",
    "already_compressed": "already_compressed",
    "read_lifecycle_marker": "already_compressed",
    "rejected_not_smaller": "rejected_not_smaller",
    "protected_error_output": "protected_content",
    "protected_code_context": "protected_content",
    "excluded_tool_result": "protected_content",
    "image_optimize_disabled": "protected_content",
}


def _unit_reason_category(reason: str) -> str:
    if reason.startswith("cache_zone_"):
        return "cache_zone"
    return _UNIT_REASON_CATEGORIES.get(reason, "other")


def _record_unit_outcome(reason: str) -> None:
    category = _unit_reason_category(reason)
    with _stats_lock:
        _unit_outcomes_by_reason[reason] = _unit_outcomes_by_reason.get(reason, 0) + 1
        _unit_outcomes_by_category[category] = _unit_outcomes_by_category.get(category, 0) + 1


def get_unit_outcome_stats() -> UnitOutcomeStats:
    with _stats_lock:
        return UnitOutcomeStats(
            total_units=sum(_unit_outcomes_by_reason.values()),
            outcomes_by_reason=dict(_unit_outcomes_by_reason),
            outcomes_by_category=dict(_unit_outcomes_by_category),
        )


def reset_compression_cache() -> None:
    global _compression_cache_hits, _compression_cache_misses, _compression_cache_tokens_saved
    with _stats_lock:
        _compression_cache.clear()
        _compression_cache_hits = 0
        _compression_cache_misses = 0
        _compression_cache_tokens_saved = 0


def get_compression_cache_stats() -> CompressionCacheStats:
    with _stats_lock:
        return CompressionCacheStats(
            hits=_compression_cache_hits,
            misses=_compression_cache_misses,
            entries=len(_compression_cache),
            tokens_saved=_compression_cache_tokens_saved,
        )


def reset_cache_alignment_stats() -> None:
    global _cache_alignment_scans, _cache_alignment_findings, _cache_alignment_skipped_by_policy
    with _stats_lock:
        _cache_alignment_scans = 0
        _cache_alignment_findings = 0
        _cache_alignment_skipped_by_policy = 0
        _cache_alignment_findings_by_label.clear()


def get_cache_alignment_stats() -> CacheAlignmentStats:
    with _stats_lock:
        return CacheAlignmentStats(
            total_scans=_cache_alignment_scans,
            total_findings=_cache_alignment_findings,
            skipped_by_policy=_cache_alignment_skipped_by_policy,
            findings_by_label=dict(_cache_alignment_findings_by_label),
        )


def _record_compression_learning_event(
    event: CompressionEvent,
    *,
    token_savings: int,
    byte_savings: int,
) -> None:
    global _learning_total_events, _learning_total_skipped_read_only
    global _learning_total_tokens_saved, _learning_total_bytes_saved
    if _current_policy.get().toin_read_only:
        _learning_total_skipped_read_only += 1
        _learning_skipped_by_strategy[event.strategy] = _learning_skipped_by_strategy.get(event.strategy, 0) + 1
        return
    _learning_total_events += 1
    _learning_total_tokens_saved += token_savings
    _learning_total_bytes_saved += byte_savings
    _learning_events_by_strategy[event.strategy] = _learning_events_by_strategy.get(event.strategy, 0) + 1


def _restore_compression_stats(snapshot: CompressionStats) -> None:
    global _total_compressions, _total_original_tokens, _total_compressed_tokens
    global _total_original_bytes, _total_compressed_bytes
    with _stats_lock:
        _total_compressions = snapshot.total_compressions
        _total_original_tokens = snapshot.total_original_tokens
        _total_compressed_tokens = snapshot.total_compressed_tokens
        _total_original_bytes = snapshot.total_original_bytes
        _total_compressed_bytes = snapshot.total_compressed_bytes
        _compressions_by_strategy.clear()
        _compressions_by_strategy.update(snapshot.compressions_by_strategy)
        _tokens_saved_by_strategy.clear()
        _tokens_saved_by_strategy.update(snapshot.tokens_saved_by_strategy)
        _bytes_saved_by_strategy.clear()
        _bytes_saved_by_strategy.update(snapshot.bytes_saved_by_strategy)


def _restore_unit_outcome_stats(snapshot: UnitOutcomeStats) -> None:
    with _stats_lock:
        _unit_outcomes_by_reason.clear()
        _unit_outcomes_by_reason.update(snapshot.outcomes_by_reason)
        _unit_outcomes_by_category.clear()
        _unit_outcomes_by_category.update(snapshot.outcomes_by_category)


def _restore_compression_learning_stats(snapshot: CompressionLearningStats) -> None:
    global _learning_total_events, _learning_total_skipped_read_only
    global _learning_total_tokens_saved, _learning_total_bytes_saved
    with _stats_lock:
        _learning_total_events = snapshot.total_events
        _learning_total_skipped_read_only = snapshot.total_skipped_read_only
        _learning_total_tokens_saved = snapshot.total_tokens_saved
        _learning_total_bytes_saved = snapshot.total_bytes_saved
        _learning_events_by_strategy.clear()
        _learning_events_by_strategy.update(snapshot.events_by_strategy)
        _learning_skipped_by_strategy.clear()
        _learning_skipped_by_strategy.update(snapshot.skipped_by_strategy)


def _approx_tokens_from_text(text: str) -> int:
    return count_text_tokens(text)


_HEX_HASH_LENGTHS = frozenset({32, 40, 64})


def _is_uuid_token(token: str) -> bool:
    if len(token) != 36 or token.count("-") != 4:
        return False
    try:
        _uuid.UUID(token)
    except (AttributeError, ValueError):
        return False
    return True


def _is_iso8601_token(token: str) -> bool:
    if len(token) < 8 or ("T" not in token and "-" not in token):
        return False
    candidate = token[:-1] + "+00:00" if token.endswith("Z") else token
    try:
        datetime.fromisoformat(candidate)
    except (TypeError, ValueError):
        return False
    return True


def _is_jwt_shape_token(token: str) -> bool:
    if token.count(".") != 2:
        return False
    segments = token.split(".")
    if len(segments) != 3:
        return False
    for segment in segments:
        if len(segment) < 4:
            return False
        padded = segment + "=" * (-len(segment) % 4)
        try:
            base64.urlsafe_b64decode(padded.encode("ascii"))
        except (binascii.Error, UnicodeEncodeError, ValueError):
            return False
    return True


def _is_hex_hash_token(token: str) -> bool:
    if len(token) not in _HEX_HASH_LENGTHS:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True


def _volatile_label(token: str) -> str | None:
    if _is_uuid_token(token):
        return "uuid"
    if "." in token and _is_jwt_shape_token(token):
        return "jwt"
    if _is_iso8601_token(token):
        return "iso8601"
    if _is_hex_hash_token(token):
        return "hex_hash"
    return None


def _volatile_tokens(content: str) -> list[str]:
    tokens: list[str] = []
    for raw in content.split():
        cleaned = raw.strip(".,;:!?\"'()[]{}<>")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def detect_volatile_content(content: str) -> list[VolatileFinding]:
    if not content:
        return []
    findings: list[VolatileFinding] = []
    for token in _volatile_tokens(content):
        label = _volatile_label(token)
        if label is None:
            continue
        sample = token if len(token) <= 16 else f"{token[:8]}...{token[-4:]}"
        findings.append(VolatileFinding(label=label, sample=sample))
    return findings


def _cache_alignment_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def _cache_alignment_message_content(msg: Any) -> str | None:
    if isinstance(msg, dict):
        return _cache_alignment_text(msg.get("content"))
    return _cache_alignment_text(getattr(msg, "content", None))


def cache_alignment_score(messages: list[Any]) -> float:
    score = 100.0
    for msg in messages:
        if _message_role(msg) != "system":
            continue
        content = _cache_alignment_message_content(msg)
        if content:
            score -= len(detect_volatile_content(content)) * 10
    return max(0.0, min(100.0, score))


def _record_cache_alignment_scan(texts: list[str], *, skipped_by_policy: bool) -> None:
    global _cache_alignment_scans, _cache_alignment_findings, _cache_alignment_skipped_by_policy
    if not texts:
        return
    with _stats_lock:
        if skipped_by_policy:
            _cache_alignment_skipped_by_policy += len(texts)
            return
        _cache_alignment_scans += len(texts)
        for text in texts:
            for finding in detect_volatile_content(text):
                _cache_alignment_findings += 1
                _cache_alignment_findings_by_label[finding.label] = (
                    _cache_alignment_findings_by_label.get(finding.label, 0) + 1
                )


def _cache_alignment_system_texts(context: Any, *, frozen_count: int) -> list[str]:
    texts: list[str] = []
    system_prompt = getattr(context, "system_prompt", None)
    if isinstance(system_prompt, str) and system_prompt:
        texts.append(system_prompt)
    for index, msg in enumerate(list(getattr(context, "messages", []) or [])):
        if index < frozen_count or _message_role(msg) != "system":
            continue
        content = _cache_alignment_message_content(msg)
        if content:
            texts.append(content)
    return texts


def _approx_tokens_from_bytes(data: int) -> int:
    return max(1, data // 4)


def estimate_openai_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate OpenAI-style image input tokens from rendered dimensions.

    This mirrors Headroom's tile-boundary accounting closely enough for active
    compression telemetry: low detail is a flat 85 tokens; high detail is the
    base 85-token image cost plus 170 tokens per 512x512 tile.
    """
    if width <= 0 or height <= 0:
        return 0
    if detail == "low":
        return 85
    return 85 + 170 * math.ceil(width / 512) * math.ceil(height / 512)


def estimate_anthropic_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    pixels = width * height
    if pixels > 1_150_000:
        scale = (1_150_000 / pixels) ** 0.5
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    if max(width, height) > 1568:
        scale = 1568 / max(width, height)
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    return max(1, (width * height) // 750)


def _emit_compression_event(event: CompressionEvent) -> None:
    global _total_compressions, _total_original_tokens, _total_compressed_tokens
    global _total_original_bytes, _total_compressed_bytes
    token_savings = max(0, event.original_tokens - event.compressed_tokens)
    byte_savings = max(0, event.original_bytes - event.compressed_bytes)
    with _stats_lock:
        _total_compressions += 1
        _total_original_tokens += event.original_tokens
        _total_compressed_tokens += event.compressed_tokens
        _total_original_bytes += event.original_bytes
        _total_compressed_bytes += event.compressed_bytes
        _compressions_by_strategy[event.strategy] = _compressions_by_strategy.get(event.strategy, 0) + 1
        if token_savings:
            _tokens_saved_by_strategy[event.strategy] = (
                _tokens_saved_by_strategy.get(event.strategy, 0) + token_savings
            )
        if byte_savings:
            _bytes_saved_by_strategy[event.strategy] = (
                _bytes_saved_by_strategy.get(event.strategy, 0) + byte_savings
            )
        _record_compression_learning_event(event, token_savings=token_savings, byte_savings=byte_savings)

    observer = _observer
    if observer is None:
        return
    try:
        observer(event)
    except Exception:
        return


def _details_value(msg: Any, key: str, default: Any = None) -> Any:
    details = getattr(msg, "details", None)
    if isinstance(details, dict):
        return details.get(key, default)
    return default


def _message_role(msg: Any) -> str | None:
    value = getattr(msg, "role", None)
    if isinstance(value, str):
        return value
    if isinstance(msg, dict):
        value = msg.get("role")
        if isinstance(value, str):
            return value
    return None


def _cache_zone(value: Any, default: str = LIVE_CACHE_ZONE) -> str:
    if isinstance(value, dict):
        if value.get("cache_control") is not None:
            return "prefix"
        zone = value.get("cache_zone", value.get("cacheZone", default))
        return str(zone or default)
    if getattr(value, "cache_control", None) is not None:
        return "prefix"
    zone = getattr(value, "cache_zone", None)
    if zone is None:
        zone = getattr(value, "cacheZone", None)
    if zone is None:
        zone = default
    return str(zone or default)


def _is_mutable(value: Any, default: bool = True) -> bool:
    if isinstance(value, dict):
        mutable = value.get("mutable", default)
        return bool(mutable)
    mutable = getattr(value, "mutable", None)
    if mutable is None:
        mutable = default
    return bool(mutable)


def _is_live_user_text_candidate(text: str) -> bool:
    return len(text.encode("utf-8")) >= LIVE_USER_TEXT_MIN_BYTES


def _has_strong_error_indicators(text: str) -> bool:
    lowered = text.lower()
    hits = 0
    for keyword in ERROR_INDICATOR_KEYWORDS:
        if keyword in lowered:
            hits += 1
            if hits >= 2:
                return True
    return False


def _is_protected_error_output(text: str, *, explicit_error: bool = False) -> bool:
    if len(text) > ERROR_PROTECTION_MAX_CHARS:
        return False
    return explicit_error or _has_strong_error_indicators(text)


def _looks_like_code_text(text: str) -> bool:
    hits = 0
    for line in text.splitlines()[:200]:
        if CODE_SIGNATURE_RE.match(line):
            hits += 1
            if hits >= 4:
                return True
    return False


def _message_text(msg: Any) -> str | None:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(getattr(block, "text", None), str):
                texts.append(block.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
        if texts:
            return "\n".join(texts)
    return None


def _has_analysis_intent(messages: list[Any], frozen_count: int) -> bool:
    for index in range(len(messages) - 1, frozen_count - 1, -1):
        msg = messages[index]
        if _message_role(msg) != "user":
            continue
        text = _message_text(msg)
        if text is None:
            return False
        lowered = text.lower()
        return any(keyword in lowered for keyword in ANALYSIS_INTENT_KEYWORDS)
    return False


def _is_protected_code_output(text: str, *, messages_from_end: int, analysis_intent: bool) -> bool:
    if not _looks_like_code_text(text):
        return False
    if messages_from_end <= RECENT_CODE_PROTECTION_MESSAGES:
        return True
    return analysis_intent


@dataclass(frozen=True, slots=True)
class _FileOperation:
    msg_index: int
    tool_call_id: str
    tool_name: str
    file_path: str
    operation: str
    read_offset: int | None = None
    read_limit: int | None = None


@dataclass(frozen=True, slots=True)
class _ReadReplacement:
    state: str
    file_path: str


def _tool_call_arguments(call: Any) -> dict[str, Any]:
    args = getattr(call, "arguments", None)
    if isinstance(args, dict):
        return args
    if isinstance(call, dict):
        value = call.get("input")
        if isinstance(value, dict):
            return value
        value = call.get("arguments")
        if isinstance(value, dict):
            return value
        function = call.get("function")
        if isinstance(function, dict):
            raw = function.get("arguments")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_call_id(call: Any) -> str | None:
    value = getattr(call, "id", None)
    if isinstance(value, str) and value:
        return value
    if isinstance(call, dict):
        value = call.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def _tool_call_name(call: Any) -> str | None:
    value = getattr(call, "name", None)
    if isinstance(value, str) and value:
        return value
    if isinstance(call, dict):
        function = call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        value = call.get("name")
        if isinstance(value, str) and value:
            return value
    return None


def _assistant_tool_calls(msg: Any) -> list[Any]:
    calls: list[Any] = []
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, list):
        calls.extend(block for block in content if getattr(block, "type", None) == "toolCall")
        calls.extend(
            block
            for block in content
            if isinstance(block, dict) and block.get("type") in {"toolCall", "tool_use"}
        )
    tool_calls = getattr(msg, "tool_calls", None)
    if isinstance(tool_calls, list):
        calls.extend(tool_calls)
    if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
        calls.extend(msg["tool_calls"])
    return calls


def _tool_name_by_call_id(messages: list[Any], frozen_count: int) -> dict[str, str]:
    names: dict[str, str] = {}
    for index, msg in enumerate(messages):
        if index < frozen_count or _message_role(msg) != "assistant":
            continue
        for call in _assistant_tool_calls(msg):
            call_id = _tool_call_id(call)
            name = _tool_call_name(call)
            if call_id and name:
                names[call_id] = name
    return names


def _is_excluded_tool_result_name(name: Any) -> bool:
    return isinstance(name, str) and name in DEFAULT_EXCLUDED_TOOL_RESULT_NAMES


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    parsed = _int_or_none(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _bounded_ratio_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return min(parsed, 1.0)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def _read_covers(later: _FileOperation, earlier: _FileOperation) -> bool:
    if later.read_offset is None and later.read_limit is None:
        return True
    if earlier.read_offset is None and earlier.read_limit is None:
        return False
    later_start = later.read_offset or 0
    earlier_start = earlier.read_offset or 0
    later_end = later_start + (later.read_limit or 2000)
    earlier_end = earlier_start + (earlier.read_limit or 2000)
    return later_start <= earlier_start and later_end >= earlier_end


def _read_lifecycle_replacements(
    messages: list[Any],
    frozen_count: int,
    *,
    compress_stale: bool = True,
    compress_superseded: bool = False,
) -> dict[str, _ReadReplacement]:
    operations: list[_FileOperation] = []
    for index, msg in enumerate(messages):
        if index < frozen_count or _message_role(msg) != "assistant":
            continue
        for call in _assistant_tool_calls(msg):
            call_id = _tool_call_id(call)
            name = _tool_call_name(call)
            if not call_id or not name:
                continue
            args = _tool_call_arguments(call)
            file_path = args.get("path") or args.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            if name in READ_TOOL_NAMES:
                operations.append(
                    _FileOperation(
                        msg_index=index,
                        tool_call_id=call_id,
                        tool_name=name,
                        file_path=file_path,
                        operation="read",
                        read_offset=_int_or_none(args.get("offset")),
                        read_limit=_int_or_none(args.get("limit")),
                    )
                )
            elif name in MUTATING_TOOL_NAMES:
                operations.append(
                    _FileOperation(
                        msg_index=index,
                        tool_call_id=call_id,
                        tool_name=name,
                        file_path=file_path,
                        operation="edit",
                    )
                )

    by_file: dict[str, list[_FileOperation]] = {}
    for op in operations:
        by_file.setdefault(op.file_path, []).append(op)

    replacements: dict[str, _ReadReplacement] = {}
    for file_path, ops in by_file.items():
        reads = [op for op in ops if op.operation == "read"]
        edits = [op for op in ops if op.operation == "edit"]
        for read_op in reads:
            state: str | None = None
            if compress_stale and any(edit.msg_index > read_op.msg_index for edit in edits):
                state = "stale"
            elif compress_superseded and any(
                later.msg_index > read_op.msg_index and _read_covers(later, read_op)
                for later in reads
            ):
                state = "superseded"
            if state is not None:
                replacements[read_op.tool_call_id] = _ReadReplacement(state=state, file_path=file_path)
    return replacements


def _text_blocks_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [getattr(block, "text", None) for block in content]
        if texts and all(isinstance(text, str) for text in texts):
            return "\n".join(texts)
    return None


def _safe_compress_text(text: str, fn: CompressFn) -> str | None:
    cache = _current_text_compression_cache.get()
    if cache is not None and text in cache:
        return cache[text]
    if _compression_breaker_is_open():
        return None
    try:
        compressed = fn(text)
    except Exception:
        _compression_breaker_record_failure()
        return None
    if not isinstance(compressed, str):
        _compression_breaker_record_failure()
        return None
    _compression_breaker_record_success()
    if cache is not None:
        cache[text] = compressed
    return compressed


def _read_lifecycle_marker(original: str, replacement: _ReadReplacement, fn: CompressFn) -> str | None:
    force_token = _current_force_compression.set(True)
    try:
        compressed = _safe_compress_text(original, fn)
    finally:
        _current_force_compression.reset(force_token)
    if compressed is None:
        return None
    match = _CCR_HANDLE_RE.search(compressed)
    if match is None:
        return None
    handle = match.group(1)
    if replacement.state == "stale":
        reason = f"{replacement.file_path} was modified after this read"
    else:
        reason = f"{replacement.file_path} was re-read later"
    return (
        f"[Read content {replacement.state}: {reason}. "
        f"Retrieve original: hash={handle}. Re-read the file for current content if needed.]"
    )


def _replace_text_content(content: Any, marker: str) -> Any:
    if isinstance(content, str):
        return marker
    if isinstance(content, list):
        replaced = False
        blocks: list[Any] = []
        for block in content:
            if not replaced and isinstance(getattr(block, "text", None), str):
                blocks.append(block.model_copy(update={"text": marker}))
                replaced = True
            elif isinstance(getattr(block, "text", None), str):
                blocks.append(block.model_copy(update={"text": ""}))
            else:
                blocks.append(block)
        return blocks
    return content


def _apply_read_lifecycle_to_text(
    original: str,
    fn: CompressFn,
    replacement: _ReadReplacement,
    *,
    role: str | None,
    tool_name: str | None,
    min_size_bytes: int = READ_LIFECYCLE_MIN_BYTES,
) -> str | None:
    if len(original.encode("utf-8")) < min_size_bytes:
        return None
    marker = _read_lifecycle_marker(original, replacement, fn)
    if marker is None:
        return None
    if len(marker.encode("utf-8")) >= len(original.encode("utf-8")):
        return None
    _emit_compression_event(
        CompressionEvent(
            strategy=f"read_lifecycle:{replacement.state}",
            original_tokens=_approx_tokens_from_text(original),
            compressed_tokens=_approx_tokens_from_text(marker),
            original_bytes=len(original.encode("utf-8")),
            compressed_bytes=len(marker.encode("utf-8")),
            role=role,
            tool_name=tool_name,
        )
    )
    return marker


def _apply_anthropic_read_lifecycle(
    msg: dict[str, Any],
    fn: CompressFn,
    replacements: dict[str, _ReadReplacement],
    *,
    min_size_bytes: int = READ_LIFECYCLE_MIN_BYTES,
) -> Any:
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    changed = False
    updated_blocks: list[Any] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            updated_blocks.append(block)
            continue
        replacement = replacements.get(str(block.get("tool_use_id") or ""))
        original = block.get("content")
        if replacement is None or not isinstance(original, str):
            updated_blocks.append(block)
            continue
        if _is_protected_error_output(original, explicit_error=bool(block.get("is_error"))):
            updated_blocks.append(block)
            continue
        marker = _apply_read_lifecycle_to_text(
            original,
            fn,
            replacement,
            role="user",
            tool_name="Read",
            min_size_bytes=min_size_bytes,
        )
        if marker is None:
            updated_blocks.append(block)
            continue
        updated = dict(block)
        updated["content"] = marker
        updated_blocks.append(updated)
        changed = True

    if not changed:
        return msg
    updated_msg = dict(msg)
    updated_msg["content"] = updated_blocks
    return updated_msg


def _apply_read_lifecycle(
    msg: Any,
    fn: CompressFn,
    replacements: dict[str, _ReadReplacement],
    *,
    min_size_bytes: int = READ_LIFECYCLE_MIN_BYTES,
) -> Any:
    if _message_role(msg) == "user" and isinstance(msg, dict):
        return _apply_anthropic_read_lifecycle(
            msg,
            fn,
            replacements,
            min_size_bytes=min_size_bytes,
        )
    replacement = replacements.get(str(getattr(msg, "tool_call_id", "")))
    if replacement is None:
        return msg
    if _message_role(msg) != "toolResult":
        return msg
    if getattr(msg, "tool_name", None) not in READ_TOOL_NAMES:
        return msg
    msg_cache_zone = str(_details_value(msg, "cache_zone", _details_value(msg, "cacheZone", LIVE_CACHE_ZONE)))
    if msg_cache_zone != LIVE_CACHE_ZONE or not bool(_details_value(msg, "mutable", True)):
        return msg
    original = _text_blocks_content(getattr(msg, "content", None))
    if original is None:
        return msg
    marker = _apply_read_lifecycle_to_text(
        original,
        fn,
        replacement,
        role="toolResult",
        tool_name=getattr(msg, "tool_name", None),
        min_size_bytes=min_size_bytes,
    )
    if marker is None:
        return msg
    return msg.model_copy(update={"content": _replace_text_content(getattr(msg, "content", None), marker)})


def _anthropic_tool_result_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
        return None
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(content)
    return None


def _replace_anthropic_tool_result_content(content: Any, compressed: str) -> Any:
    if isinstance(content, list):
        replaced = False
        updated_items: list[Any] = []
        for item in content:
            if (
                not replaced
                and isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                updated = dict(item)
                updated["text"] = compressed
                updated_items.append(updated)
                replaced = True
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                updated = dict(item)
                updated["text"] = ""
                updated_items.append(updated)
            else:
                updated_items.append(item)
        return updated_items
    return compressed


def _compress_anthropic_tool_result_block(
    block: dict[str, Any],
    fn: CompressFn,
    *,
    protect_code: bool = False,
) -> tuple[dict[str, Any], bool]:
    block_cache_zone = _cache_zone(block)
    if block_cache_zone != LIVE_CACHE_ZONE:
        if _has_text_unit_candidate(block):
            _record_unit_outcome(f"cache_zone_{block_cache_zone}")
        return block, False
    if not _is_mutable(block):
        if _has_text_unit_candidate(block):
            _record_unit_outcome("immutable")
        return block, False
    if block.get("type") != "tool_result":
        return block, False
    content = block.get("content")
    original = _anthropic_tool_result_text(content)
    if original is None:
        return block, False
    if _is_protected_error_output(original, explicit_error=bool(block.get("is_error"))):
        _record_unit_outcome("protected_error_output")
        return block, False
    if protect_code and _looks_like_code_text(original):
        _record_unit_outcome("protected_code_context")
        return block, False
    if _OWN_CCR_MARKER_RE.search(original):
        _record_unit_outcome("already_compressed")
        return block, False
    if _READ_LIFECYCLE_MARKER_RE.search(original):
        _record_unit_outcome("read_lifecycle_marker")
        return block, False
    original_tokens = _approx_tokens_from_text(original)
    if original_tokens < _current_min_tokens_to_compress():
        _record_unit_outcome("below_unit_floor")
        return block, False
    compressed = _compress_text_preserving_markers(original, fn)
    if compressed == original:
        _record_unit_outcome("compressor_no_change")
        return block, False
    compressed_tokens = _approx_tokens_from_text(compressed)
    if compressed_tokens >= original_tokens:
        _record_unit_outcome("rejected_not_smaller")
        return block, False
    _emit_compression_event(
        CompressionEvent(
            strategy="text",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_bytes=len(original.encode("utf-8")),
            compressed_bytes=len(compressed.encode("utf-8")),
            role="toolResult",
        )
    )
    _record_unit_outcome("applied")
    updated = dict(block)
    updated["content"] = _replace_anthropic_tool_result_content(content, compressed)
    return updated, True


def _compress_anthropic_tool_results(
    msg: dict[str, Any],
    fn: CompressFn,
    *,
    protect_code: bool = False,
    tool_name_by_id: dict[str, str] | None = None,
) -> Any:
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    changed = False
    updated_blocks: list[Any] = []
    for block in content:
        if isinstance(block, dict):
            tool_use_id = str(block.get("tool_use_id") or "")
            if _is_excluded_tool_result_name((tool_name_by_id or {}).get(tool_use_id)):
                if _has_text_unit_candidate(block):
                    _record_unit_outcome("excluded_tool_result")
                updated_blocks.append(block)
                continue
            updated, did_change = _compress_anthropic_tool_result_block(block, fn, protect_code=protect_code)
            updated_blocks.append(updated)
            changed = changed or did_change
        else:
            updated_blocks.append(block)
    if not changed:
        return msg
    updated_msg = dict(msg)
    updated_msg["content"] = updated_blocks
    return updated_msg


def _compress_text_value(
    original: str,
    fn: CompressFn,
    *,
    role: str | None,
    tool_name: str | None = None,
    explicit_error: bool = False,
    protect_code: bool = False,
) -> str | None:
    if role == "toolResult" and _is_protected_error_output(original, explicit_error=explicit_error):
        _record_unit_outcome("protected_error_output")
        return None
    if role == "toolResult" and protect_code and _looks_like_code_text(original):
        _record_unit_outcome("protected_code_context")
        return None
    if role == "toolResult" and _OWN_CCR_MARKER_RE.search(original):
        _record_unit_outcome("already_compressed")
        return None
    if _READ_LIFECYCLE_MARKER_RE.search(original):
        _record_unit_outcome("read_lifecycle_marker")
        return None
    original_tokens = _approx_tokens_from_text(original)
    if original_tokens < _current_min_tokens_to_compress():
        _record_unit_outcome("below_unit_floor")
        return None
    compressed = _compress_text_preserving_markers(original, fn)
    if compressed == original:
        _record_unit_outcome("compressor_no_change")
        return None
    compressed_tokens = _approx_tokens_from_text(compressed)
    if compressed_tokens >= original_tokens:
        _record_unit_outcome("rejected_not_smaller")
        return None
    _emit_compression_event(
        CompressionEvent(
            strategy="text",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_bytes=len(original.encode("utf-8")),
            compressed_bytes=len(compressed.encode("utf-8")),
            role=role,
            tool_name=tool_name,
        )
    )
    _record_unit_outcome("applied")
    return compressed


def _compress_dict_text_block(
    block: dict[str, Any],
    fn: CompressFn,
    *,
    role: str | None,
    explicit_error: bool = False,
    protect_code: bool = False,
) -> tuple[dict[str, Any], bool]:
    block_cache_zone = _cache_zone(block)
    if block_cache_zone != LIVE_CACHE_ZONE:
        if _has_text_unit_candidate(block):
            _record_unit_outcome(f"cache_zone_{block_cache_zone}")
        return block, False
    if not _is_mutable(block):
        if _has_text_unit_candidate(block):
            _record_unit_outcome("immutable")
        return block, False
    if block.get("type") != "text" or not isinstance(block.get("text"), str):
        return block, False
    compressed = _compress_text_value(
        block["text"],
        fn,
        role=role,
        explicit_error=explicit_error,
        protect_code=protect_code,
    )
    if compressed is None:
        return block, False
    updated = dict(block)
    updated["text"] = compressed
    return updated, True


def _data_url_payload(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url.split(",", 1)
    media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
    if not data:
        return None
    return media_type, data


def _image_ocr_intent(text: str | None) -> bool:
    if not text:
        return False
    return bool(IMAGE_OCR_INTENT_RE.search(text))


def _message_text_for_image_intent(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif block.get("type") == "text" and isinstance(block.get("content"), str):
                    texts.append(block["content"])
            elif isinstance(getattr(block, "text", None), str):
                texts.append(block.text)
    return "\n".join(texts)


def _resolve_rapidocr() -> tuple[Any | None, str | None]:
    try:
        from rapidocr_onnxruntime import RapidOCR as rapid_ocr_v1

        return rapid_ocr_v1, "v1"
    except ImportError:
        pass
    try:
        from rapidocr import RapidOCR as rapid_ocr_v3  # type: ignore[import-not-found]

        return rapid_ocr_v3, "v3"
    except ImportError:
        return None, None


def _ocr_extract_image_text(image_bytes: bytes, min_confidence: float = OCR_MIN_CONFIDENCE) -> str | None:
    ocr_cls, api_version = _resolve_rapidocr()
    if ocr_cls is None:
        return None
    try:
        raw = ocr_cls()(image_bytes)
    except Exception:
        return None

    if api_version == "v1":
        try:
            rows, _elapsed = raw
        except (TypeError, ValueError):
            return None
        if not rows:
            return None
        try:
            texts = [str(row[1]) for row in rows]
            scores = [float(row[2]) for row in rows]
        except (IndexError, TypeError, ValueError):
            return None
    elif api_version == "v3":
        texts_attr = getattr(raw, "txts", None)
        scores_attr = getattr(raw, "scores", None)
        texts = [str(text) for text in (texts_attr or [])]
        try:
            scores = [float(score) for score in (scores_attr or [])]
        except (TypeError, ValueError):
            return None
        if len(texts) != len(scores):
            return None
    else:
        return None

    if not texts or not scores:
        return None
    if (sum(scores) / len(scores)) < min_confidence:
        return None
    text = "\n".join(line for line in texts if line.strip()).strip()
    return text or None


def _emit_image_ocr_event(
    original_raw: bytes,
    extracted_text: str,
    *,
    role: str | None,
    tool_name: str | None = None,
) -> None:
    dimensions = _image_dimensions_from_bytes(original_raw)
    if dimensions is not None:
        original_tokens = estimate_openai_image_tokens(*dimensions)
    else:
        original_tokens = _approx_tokens_from_bytes(len(original_raw))
    compressed_tokens = _approx_tokens_from_text(extracted_text)
    _emit_compression_event(
        CompressionEvent(
            strategy="image_ocr",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_bytes=len(original_raw),
            compressed_bytes=len(extracted_text.encode("utf-8")),
            role=role,
            tool_name=tool_name,
        )
    )


def _ocr_text(extracted_text: str) -> str:
    return f"[OCR from image]\n{extracted_text}"


def _image_bytes_from_dict_block(block: dict[str, Any]) -> bytes | None:
    if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
        url = block["image_url"].get("url")
        if not isinstance(url, str):
            return None
        payload = _data_url_payload(url)
        if payload is None:
            return None
        _media_type, raw_data = payload
    elif block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") != "base64" or not isinstance(source.get("data"), str):
            return None
        raw_data = source["data"]
    else:
        inline_data = block.get("inlineData")
        if not isinstance(inline_data, dict):
            inline_data = block.get("inline_data")
        if not isinstance(inline_data, dict) or not isinstance(inline_data.get("data"), str):
            return None
        raw_data = inline_data["data"]
    try:
        return base64.b64decode(raw_data, validate=False)
    except Exception:
        return None


def _ocr_text_dict_block(block: dict[str, Any], extracted_text: str) -> dict[str, Any]:
    text = _ocr_text(extracted_text)
    if "inlineData" in block or "inline_data" in block:
        return {"text": text}
    return {"type": "text", "text": text}


def _compress_image_bytes(image_bytes: bytes) -> bytes | None:
    if len(image_bytes) < IMAGE_MIN_BYTES:
        return None

    try:
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        if width <= IMAGE_MAX_DIMENSION and height <= IMAGE_MAX_DIMENSION:
            return None

        img.thumbnail((IMAGE_MAX_DIMENSION, IMAGE_MAX_DIMENSION), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        compressed = out.getvalue()
    except Exception:
        return None

    if len(compressed) >= len(image_bytes):
        return None
    return compressed


def _emit_image_resize_event(
    original_raw: bytes,
    compressed_raw: bytes,
    *,
    role: str | None,
    tool_name: str | None = None,
) -> None:
    original_dimensions = _image_dimensions_from_bytes(original_raw)
    compressed_dimensions = _image_dimensions_from_bytes(compressed_raw)
    if original_dimensions is not None and compressed_dimensions is not None:
        original_tokens = estimate_openai_image_tokens(*original_dimensions)
        compressed_tokens = estimate_openai_image_tokens(*compressed_dimensions)
    else:
        original_tokens = _approx_tokens_from_bytes(len(original_raw))
        compressed_tokens = _approx_tokens_from_bytes(len(compressed_raw))
    _emit_compression_event(
        CompressionEvent(
            strategy="image_resize",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_bytes=len(original_raw),
            compressed_bytes=len(compressed_raw),
            role=role,
            tool_name=tool_name,
        )
    )


def _compress_dict_image_block(
    block: dict[str, Any],
    *,
    role: str | None,
    tool_name: str | None = None,
    ocr_query: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if _cache_zone(block) != LIVE_CACHE_ZONE or not _is_mutable(block):
        return block, False
    if not _current_image_optimize.get():
        if _image_bytes_from_dict_block(block) is not None:
            _record_unit_outcome("image_optimize_disabled")
        return block, False

    if _image_ocr_intent(ocr_query):
        original_raw = _image_bytes_from_dict_block(block)
        if original_raw is not None:
            extracted = _ocr_extract_image_text(original_raw)
            if extracted:
                _emit_image_ocr_event(original_raw, extracted, role=role, tool_name=tool_name)
                return _ocr_text_dict_block(block, extracted), True

    if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
        image_url = block["image_url"]
        url = image_url.get("url")
        if not isinstance(url, str):
            return block, False
        payload = _data_url_payload(url)
        if payload is None:
            return block, False
        _media_type, raw_data = payload
        try:
            original_raw = base64.b64decode(raw_data, validate=False)
        except Exception:
            return block, False
        compressed_raw = _compress_image_bytes(original_raw)
        if compressed_raw is None:
            return block, False
        updated_url = dict(image_url)
        updated_url["url"] = f"data:image/jpeg;base64,{base64.b64encode(compressed_raw).decode('ascii')}"
        updated_block = dict(block)
        updated_block["image_url"] = updated_url
        _emit_image_resize_event(original_raw, compressed_raw, role=role, tool_name=tool_name)
        return updated_block, True

    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") != "base64" or not isinstance(source.get("data"), str):
            return block, False
        try:
            original_raw = base64.b64decode(source["data"], validate=False)
        except Exception:
            return block, False
        compressed_raw = _compress_image_bytes(original_raw)
        if compressed_raw is None:
            return block, False
        updated_source = dict(source)
        updated_source["media_type"] = "image/jpeg"
        updated_source["data"] = base64.b64encode(compressed_raw).decode("ascii")
        updated_block = dict(block)
        updated_block["source"] = updated_source
        _emit_image_resize_event(original_raw, compressed_raw, role=role, tool_name=tool_name)
        return updated_block, True

    inline_data = block.get("inlineData")
    if not isinstance(inline_data, dict):
        inline_data = block.get("inline_data")
    if isinstance(inline_data, dict) and isinstance(inline_data.get("data"), str):
        try:
            original_raw = base64.b64decode(inline_data["data"], validate=False)
        except Exception:
            return block, False
        compressed_raw = _compress_image_bytes(original_raw)
        if compressed_raw is None:
            return block, False
        updated_inline = dict(inline_data)
        if "mimeType" in updated_inline:
            updated_inline["mimeType"] = "image/jpeg"
            key = "inlineData"
        else:
            updated_inline["mime_type"] = "image/jpeg"
            key = "inline_data"
        updated_inline["data"] = base64.b64encode(compressed_raw).decode("ascii")
        updated_block = dict(block)
        updated_block[key] = updated_inline
        _emit_image_resize_event(original_raw, compressed_raw, role=role, tool_name=tool_name)
        return updated_block, True

    return block, False


def _compress_dict_message_image_blocks(
    msg: dict[str, Any],
    *,
    role: str | None,
    tool_name: str | None = None,
    ocr_query: str | None = None,
) -> Any:
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    changed = False
    updated_blocks: list[Any] = []
    for block in content:
        if isinstance(block, dict):
            updated, did_change = _compress_dict_image_block(
                block,
                role=role,
                tool_name=tool_name,
                ocr_query=ocr_query,
            )
            updated_blocks.append(updated)
            changed = changed or did_change
        else:
            updated_blocks.append(block)
    if not changed:
        return msg
    updated_msg = dict(msg)
    updated_msg["content"] = updated_blocks
    return updated_msg


def _compress_dict_message_text_blocks(
    msg: dict[str, Any],
    fn: CompressFn,
    *,
    role: str | None,
    protect_code: bool = False,
    ocr_query: str | None = None,
) -> Any:
    content = msg.get("content")
    if isinstance(content, str):
        compressed = _compress_text_value(
            content,
            fn,
            role=role,
            protect_code=protect_code,
        )
        if compressed is None:
            return msg
        updated = dict(msg)
        updated["content"] = compressed
        return updated
    if not isinstance(content, list):
        return msg

    changed = False
    updated_blocks: list[Any] = []
    for block in content:
        if isinstance(block, dict):
            updated, did_change = _compress_dict_image_block(block, role=role, ocr_query=ocr_query)
            if not did_change:
                updated, did_change = _compress_dict_text_block(
                    block,
                    fn,
                    role=role,
                    protect_code=protect_code,
                )
            updated_blocks.append(updated)
            changed = changed or did_change
        else:
            updated_blocks.append(block)
    if not changed:
        return msg
    updated_msg = dict(msg)
    updated_msg["content"] = updated_blocks
    return updated_msg


def _compress_tool_role_dict_message(
    msg: dict[str, Any],
    fn: CompressFn,
    *,
    protect_code: bool = False,
    tool_name_by_id: dict[str, str] | None = None,
) -> Any:
    role = _message_role(msg)
    if role not in {"tool", "function"}:
        return msg
    msg_cache_zone = _cache_zone(msg)
    if msg_cache_zone != LIVE_CACHE_ZONE:
        if _has_text_unit_candidate(msg):
            _record_unit_outcome(f"cache_zone_{msg_cache_zone}")
        return msg
    if not _is_mutable(msg):
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("immutable")
        return msg
    tool_call_id = str(msg.get("tool_call_id") or msg.get("id") or "")
    tool_name = str(msg.get("name") or msg.get("tool_name") or (tool_name_by_id or {}).get(tool_call_id) or "")
    if _is_excluded_tool_result_name(tool_name):
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("excluded_tool_result")
        return msg
    content = msg.get("content")
    if isinstance(content, str):
        compressed = _compress_text_value(
            content,
            fn,
            role="toolResult",
            tool_name=tool_name or None,
            explicit_error=bool(msg.get("is_error")),
            protect_code=protect_code,
        )
        if compressed is None:
            return msg
        updated_msg = dict(msg)
        updated_msg["content"] = compressed
        return updated_msg
    if not isinstance(content, list):
        return msg

    changed = False
    updated_blocks: list[Any] = []
    explicit_error = bool(msg.get("is_error"))
    for block in content:
        if isinstance(block, dict):
            updated, did_change = _compress_dict_image_block(
                block,
                role="toolResult",
                tool_name=tool_name or None,
            )
            if not did_change:
                updated, did_change = _compress_dict_text_block(
                    block,
                    fn,
                    role="toolResult",
                    explicit_error=explicit_error,
                    protect_code=protect_code,
                )
            updated_blocks.append(updated)
            changed = changed or did_change
        else:
            updated_blocks.append(block)
    if not changed:
        return msg
    updated_msg = dict(msg)
    updated_msg["content"] = updated_blocks
    return updated_msg


def _compress_text_preserving_markers(text: str, fn: CompressFn) -> str:
    """Compress marker-free spans while preserving existing CCR marker bytes."""
    if not _CCR_MARKER_RE.search(text):
        return _safe_compress_text(text, fn) or text

    parts: list[str] = []
    cursor = 0
    changed = False
    for match in _CCR_MARKER_RE.finditer(text):
        prefix = text[cursor:match.start()]
        if prefix:
            compressed_prefix = _safe_compress_text(prefix, fn)
            if compressed_prefix is None:
                return text
            changed = changed or compressed_prefix != prefix
            parts.append(compressed_prefix)
        parts.append(match.group(0))
        cursor = match.end()
    suffix = text[cursor:]
    if suffix:
        compressed_suffix = _safe_compress_text(suffix, fn)
        if compressed_suffix is None:
            return text
        changed = changed or compressed_suffix != suffix
        parts.append(compressed_suffix)
    return "".join(parts) if changed else text


def _compress_image_block(block: Any, *, ocr_query: str | None = None) -> Any:
    if getattr(block, "type", None) != "image":
        return block
    if not _current_image_optimize.get():
        if getattr(block, "data", None):
            _record_unit_outcome("image_optimize_disabled")
        return block
    raw_data = getattr(block, "data", None)
    if not isinstance(raw_data, str) or not raw_data:
        return block
    try:
        image_bytes = base64.b64decode(raw_data, validate=False)
    except Exception:
        return block
    if _image_ocr_intent(ocr_query):
        extracted = _ocr_extract_image_text(image_bytes)
        if extracted:
            from .types import TextContent

            return TextContent(type="text", text=_ocr_text(extracted))
    compressed = _compress_image_bytes(image_bytes)
    if compressed is None:
        return block
    return block.model_copy(
        update={
            "data": base64.b64encode(compressed).decode("ascii"),
            "mime_type": "image/jpeg",
        }
    )


def _has_text_unit_candidate(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, list | tuple):
        for item in value:
            if _has_text_unit_candidate(item):
                return True
        return False
    if isinstance(value, dict):
        for key in ("content", "text"):
            if _has_text_unit_candidate(value.get(key)):
                return True
        return False
    content = getattr(value, "content", None)
    if _has_text_unit_candidate(content):
        return True
    text = getattr(value, "text", None)
    return isinstance(text, str) and bool(text)


def _image_dimensions_from_bytes(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return None


def _xform_block(
    block: Any,
    fn: CompressFn,
    *,
    parent_cache_zone: str,
    parent_mutable: bool,
    role: str | None = None,
    tool_name: str | None = None,
    protect_code: bool = False,
    ocr_query: str | None = None,
) -> Any:
    block_cache_zone = _cache_zone(block, parent_cache_zone)
    if block_cache_zone != LIVE_CACHE_ZONE:
        if _has_text_unit_candidate(block):
            _record_unit_outcome(f"cache_zone_{block_cache_zone}")
        return block
    if not _is_mutable(block, parent_mutable):
        if _has_text_unit_candidate(block):
            _record_unit_outcome("immutable")
        return block
    if isinstance(getattr(block, "text", None), str):
        original = block.text
        compressed = _compress_text_value(
            original,
            fn,
            role=role,
            tool_name=tool_name,
            protect_code=protect_code,
        )
        if compressed is None:
            return block
        return block.model_copy(update={"text": compressed})
    if getattr(block, "type", None) == "image":
        original_data = getattr(block, "data", "")
        out = _compress_image_block(block, ocr_query=ocr_query)
        compressed_data = getattr(out, "data", "")
        if (
            out is not block
            and getattr(out, "type", None) == "image"
            and isinstance(original_data, str)
            and isinstance(compressed_data, str)
        ):
            try:
                original_raw = base64.b64decode(original_data, validate=False)
                compressed_raw = base64.b64decode(compressed_data, validate=False)
            except Exception:
                pass
            else:
                _emit_image_resize_event(original_raw, compressed_raw, role=role, tool_name=tool_name)
        elif out is not block and getattr(out, "type", None) == "text" and isinstance(original_data, str):
            try:
                original_raw = base64.b64decode(original_data, validate=False)
            except Exception:
                pass
            else:
                _emit_image_ocr_event(
                    original_raw,
                    getattr(out, "text", "").removeprefix("[OCR from image]\n"),
                    role=role,
                    tool_name=tool_name,
                )
        return out
    return block


def _xform_message(
    msg: Any,
    fn: CompressFn,
    *,
    frozen: bool = False,
    live_user: bool = False,
    protect_code: bool = False,
    compress_user_messages: bool = False,
    compress_system_messages: bool = True,
    tool_name_by_id: dict[str, str] | None = None,
) -> Any:
    if frozen:
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("cache_zone_frozen")
        return msg
    role = _message_role(msg)
    if role == "user" and isinstance(msg, dict):
        ocr_query = _message_text_for_image_intent(msg) if live_user else None
        if not compress_user_messages:
            content = msg.get("content")
            if isinstance(content, str) and content:
                _record_unit_outcome("protected_user_message")
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                        and block.get("text")
                    ):
                        _record_unit_outcome("protected_user_message")
            updated = _compress_anthropic_tool_results(
                msg,
                fn,
                protect_code=protect_code,
                tool_name_by_id=tool_name_by_id,
            )
            return _compress_dict_message_image_blocks(updated, role="user", ocr_query=ocr_query)
        updated = _compress_anthropic_tool_results(
            msg,
            fn,
            protect_code=protect_code,
            tool_name_by_id=tool_name_by_id,
        )
        return _compress_dict_message_text_blocks(
            updated,
            fn,
            role="user",
            protect_code=False,
            ocr_query=ocr_query,
        )
    if role in {"system", "developer"} and isinstance(msg, dict):
        if not compress_system_messages:
            if _has_text_unit_candidate(msg):
                _record_unit_outcome("protected_system_message")
            return msg
        return _compress_dict_message_text_blocks(
            msg,
            fn,
            role=role,
            protect_code=protect_code,
        )
    if role in {"tool", "function"} and isinstance(msg, dict):
        return _compress_tool_role_dict_message(
            msg,
            fn,
            protect_code=protect_code,
            tool_name_by_id=tool_name_by_id,
        )
    if getattr(msg, "role", None) == "user":
        content = getattr(msg, "content", None)
        ocr_query = _message_text_for_image_intent(msg) if live_user else None
        if not compress_user_messages and isinstance(content, str):
            if content:
                _record_unit_outcome("protected_user_message")
            return msg
        if isinstance(content, str) and live_user and _is_live_user_text_candidate(content):
            if _READ_LIFECYCLE_MARKER_RE.search(content):
                return msg
            compressed = _compress_text_value(content, fn, role="user")
            if compressed is None:
                return msg
            return msg.model_copy(update={"content": compressed})
        if isinstance(content, list):
            if not compress_user_messages:
                for block in content:
                    if isinstance(getattr(block, "text", None), str) and block.text:
                        _record_unit_outcome("protected_user_message")
            return msg.model_copy(
                update={
                    "content": [
                        _xform_block(
                            b,
                            fn,
                            parent_cache_zone=LIVE_CACHE_ZONE,
                            parent_mutable=True,
                            role="user",
                            protect_code=False,
                            ocr_query=ocr_query,
                        )
                        if getattr(b, "type", None) == "image"
                        or (
                            compress_user_messages
                            and live_user
                            and isinstance(getattr(b, "text", None), str)
                            and _is_live_user_text_candidate(b.text)
                        )
                        else b
                        for b in content
                    ]
                }
            )
        return msg
    # Only compress tool-result payloads — never the live user prompt or the
    # assistant's own messages. Tool outputs are the read-side bloat Headroom-style
    # compression targets; the current instruction must reach the model verbatim.
    if getattr(msg, "role", None) != "toolResult":
        if getattr(msg, "role", None) == "assistant" and _has_text_unit_candidate(msg):
            _record_unit_outcome("protected_assistant_message")
        return msg
    if _is_excluded_tool_result_name(getattr(msg, "tool_name", None)):
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("excluded_tool_result")
        return msg
    # CCR retrieval is already a scoped decompression result. Re-compressing it
    # hides the evidence the model explicitly asked for and causes retrieve loops.
    if getattr(msg, "tool_name", None) in EXCLUDED_TOOL_NAMES:
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("excluded_tool_result")
        return msg
    msg_cache_zone = str(_details_value(msg, "cache_zone", _details_value(msg, "cacheZone", LIVE_CACHE_ZONE)))
    if msg_cache_zone != LIVE_CACHE_ZONE:
        if _has_text_unit_candidate(msg):
            _record_unit_outcome(f"cache_zone_{msg_cache_zone}")
        return msg
    msg_mutable = bool(_details_value(msg, "mutable", True))
    if not msg_mutable:
        if _has_text_unit_candidate(msg):
            _record_unit_outcome("immutable")
        return msg
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        compressed = _compress_text_value(
            content,
            fn,
            role="toolResult",
            tool_name=getattr(msg, "tool_name", None),
            explicit_error=bool(getattr(msg, "is_error", False)),
            protect_code=protect_code,
        )
        if compressed is None:
            return msg
        return msg.model_copy(update={"content": compressed})
    if isinstance(content, list):
        if bool(getattr(msg, "is_error", False)):
            original = _text_blocks_content(content)
            if original is not None and _is_protected_error_output(original, explicit_error=True):
                return msg
        return msg.model_copy(
            update={
                "content": [
                    _xform_block(
                        b,
                        fn,
                        parent_cache_zone=msg_cache_zone,
                        parent_mutable=msg_mutable,
                        role="toolResult",
                        tool_name=getattr(msg, "tool_name", None),
                        protect_code=protect_code,
                    )
                    for b in content
                ]
            }
        )
    return msg


def _json_byte_len(value: Any) -> int:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return len(text.encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(child) for child in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:
            pass
    return str(value)


def _context_token_count(context: Any) -> int:
    payload = {
        "system_prompt": _jsonable(getattr(context, "system_prompt", None)),
        "messages": _jsonable(getattr(context, "messages", None)),
        "tools": _jsonable(getattr(context, "tools", None)),
    }
    return count_text_tokens(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _compression_cache_max_entries() -> int:
    try:
        value = int(os.environ.get("TAU_COMPRESSION_CACHE_MAX_ENTRIES", "1024"))
    except ValueError:
        return 1024
    return max(0, value)


def _compression_content_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _compression_cache_get(text: str) -> str | None:
    global _compression_cache_hits, _compression_cache_misses
    if _compression_cache_max_entries() <= 0:
        return None
    key = _compression_content_hash(text)
    with _stats_lock:
        item = _compression_cache.get(key)
        if item is None:
            _compression_cache_misses += 1
            return None
        _compression_cache.move_to_end(key)
        _compression_cache_hits += 1
        return item[0]


def _compression_cache_store(original: str, compressed: str) -> None:
    global _compression_cache_tokens_saved
    if original == compressed:
        return
    max_entries = _compression_cache_max_entries()
    if max_entries <= 0:
        return
    tokens_saved = max(0, _approx_tokens_from_text(original) - _approx_tokens_from_text(compressed))
    key = _compression_content_hash(original)
    with _stats_lock:
        previous = _compression_cache.get(key)
        if previous is not None:
            _compression_cache_tokens_saved = max(0, _compression_cache_tokens_saved - previous[1])
        _compression_cache[key] = (compressed, tokens_saved)
        _compression_cache.move_to_end(key)
        _compression_cache_tokens_saved += tokens_saved
        while len(_compression_cache) > max_entries:
            _old_key, (_old_text, old_saved) = _compression_cache.popitem(last=False)
            _compression_cache_tokens_saved = max(0, _compression_cache_tokens_saved - old_saved)


def _dict_tool_texts(msg: dict[str, Any]) -> list[str]:
    role = _message_role(msg)
    if role in {"tool", "function"}:
        content = msg.get("content")
        if isinstance(content, str) and content:
            return [content]
        if isinstance(content, list):
            return [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block.get("text")
            ]
    if role == "user" and isinstance(msg.get("content"), list):
        texts: list[str] = []
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    item["text"]
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item.get("text")
                )
        return texts
    return []


def _object_tool_texts(msg: Any) -> list[str]:
    if getattr(msg, "role", None) != "toolResult":
        return []
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list):
        return [
            block.text
            for block in content
            if isinstance(getattr(block, "text", None), str) and block.text
        ]
    return []


def _tool_texts(msg: Any) -> list[str]:
    if isinstance(msg, dict):
        return _dict_tool_texts(msg)
    return _object_tool_texts(msg)


def _apply_cached_to_dict_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = _message_role(msg)
    if role in {"tool", "function"}:
        content = msg.get("content")
        if isinstance(content, str):
            cached = _compression_cache_get(content)
            if cached is None:
                return msg
            updated = dict(msg)
            updated["content"] = cached
            return updated
        if isinstance(content, list):
            changed = False
            blocks: list[Any] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    cached = _compression_cache_get(block["text"])
                    if cached is not None:
                        new_block = dict(block)
                        new_block["text"] = cached
                        blocks.append(new_block)
                        changed = True
                        continue
                blocks.append(block)
            if changed:
                updated = dict(msg)
                updated["content"] = blocks
                return updated
        return msg

    if role == "user" and isinstance(msg.get("content"), list):
        changed = False
        blocks: list[Any] = []
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                blocks.append(block)
                continue
            content = block.get("content")
            if isinstance(content, str):
                cached = _compression_cache_get(content)
                if cached is not None:
                    new_block = dict(block)
                    new_block["content"] = cached
                    blocks.append(new_block)
                    changed = True
                    continue
            elif isinstance(content, list):
                content_changed = False
                items: list[Any] = []
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                    ):
                        cached = _compression_cache_get(item["text"])
                        if cached is not None:
                            new_item = dict(item)
                            new_item["text"] = cached
                            items.append(new_item)
                            content_changed = True
                            continue
                    items.append(item)
                if content_changed:
                    new_block = dict(block)
                    new_block["content"] = items
                    blocks.append(new_block)
                    changed = True
                    continue
            blocks.append(block)
        if changed:
            updated = dict(msg)
            updated["content"] = blocks
            return updated
    return msg


def _apply_cached_to_object_message(msg: Any) -> Any:
    if getattr(msg, "role", None) != "toolResult":
        return msg
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        cached = _compression_cache_get(content)
        if cached is None:
            return msg
        return msg.model_copy(update={"content": cached})
    if isinstance(content, list):
        changed = False
        blocks: list[Any] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                cached = _compression_cache_get(text)
                if cached is not None:
                    blocks.append(block.model_copy(update={"text": cached}))
                    changed = True
                    continue
            blocks.append(block)
        if changed:
            return msg.model_copy(update={"content": blocks})
    return msg


def _apply_compression_cache(context: Any, *, frozen_count: int) -> Any:
    messages = list(getattr(context, "messages", []) or [])
    if not messages:
        return context
    changed = False
    updated: list[Any] = []
    for index, msg in enumerate(messages):
        if index < frozen_count:
            updated.append(msg)
            continue
        next_msg = _apply_cached_to_dict_message(msg) if isinstance(msg, dict) else _apply_cached_to_object_message(msg)
        changed = changed or next_msg is not msg
        updated.append(next_msg)
    if not changed:
        return context
    return context.model_copy(update={"messages": updated})


def _update_compression_cache_from_messages(original_messages: list[Any], compressed_messages: list[Any], *, frozen_count: int) -> None:
    for index, (original, compressed) in enumerate(zip(original_messages, compressed_messages, strict=False)):
        if index < frozen_count:
            continue
        original_texts = _tool_texts(original)
        compressed_texts = _tool_texts(compressed)
        for original_text, compressed_text in zip(original_texts, compressed_texts, strict=False):
            _compression_cache_store(original_text, compressed_text)


def _iter_token_payloads(value: Any) -> list[str]:
    payloads: list[str] = []
    if value is None:
        return payloads
    if isinstance(value, str):
        return [value]
    for attr in ("content", "text", "data", "mime_type", "name", "description", "parameters", "system_prompt"):
        child = getattr(value, attr, None)
        if child is not None:
            payloads.extend(_iter_token_payloads(child))
    return payloads

def _compact_tool_schema_value(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, list):
        return [_compact_tool_schema_value(item, parent_key) for item in value]
    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        # A schema annotation named "title" can be dropped, but a property named
        # "title" under JSON Schema `properties` is an actual tool argument.
        if parent_key != "properties" and key in TOOL_SCHEMA_DROP_KEYS:
            continue
        if key == "description" and isinstance(child, str):
            compacted[key] = " ".join(child.split())
            continue
        compacted[key] = _compact_tool_schema_value(child, key)
    return compacted


def _tool_to_debug_dict(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return dict(tool)
    dump = getattr(tool, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "parameters": getattr(tool, "parameters", None),
    }


def _xform_tool_schema(tool: Any) -> tuple[Any, int, int, bool]:
    before_dict = _tool_to_debug_dict(tool)
    before_bytes = _json_byte_len(before_dict)

    if isinstance(tool, dict):
        updated = dict(tool)
        parameters = updated.get("parameters")
        if isinstance(parameters, dict):
            updated["parameters"] = _compact_tool_schema_value(parameters)
        if isinstance(updated.get("description"), str):
            updated["description"] = " ".join(str(updated["description"]).split())
        after_bytes = _json_byte_len(updated)
        if after_bytes < before_bytes:
            return updated, before_bytes, after_bytes, True
        return tool, before_bytes, after_bytes, False

    parameters = getattr(tool, "parameters", None)
    description = getattr(tool, "description", None)
    update: dict[str, Any] = {}
    if isinstance(parameters, dict):
        update["parameters"] = _compact_tool_schema_value(parameters)
    if isinstance(description, str):
        update["description"] = " ".join(description.split())
    if not update:
        return tool, before_bytes, before_bytes, False

    model_copy = getattr(tool, "model_copy", None)
    if callable(model_copy):
        updated_tool = model_copy(update=update)
    else:
        updated_tool = tool
    after_bytes = _json_byte_len(_tool_to_debug_dict(updated_tool))
    if after_bytes < before_bytes:
        return updated_tool, before_bytes, after_bytes, True
    return tool, before_bytes, after_bytes, False


def _xform_tools(tools: Any) -> tuple[Any, int, int, bool]:
    if not isinstance(tools, list) or not tools:
        return tools, 0, 0, False

    updated_tools: list[Any] = []
    total_before = 0
    total_after = 0
    changed = False
    for tool in tools:
        updated, before_bytes, after_bytes, did_change = _xform_tool_schema(tool)
        updated_tools.append(updated)
        total_before += before_bytes
        total_after += after_bytes
        changed = changed or did_change
    if not changed or total_after >= total_before:
        return tools, total_before, total_after, False
    return updated_tools, total_before, total_after, True


def compress_context(context: Any) -> Any:
    """Compress large content in the outbound context. No-op if no compressor is
    registered. Never mutates the caller's context."""
    fn = _compressor
    if fn is None:
        return context
    target_ratio = _bounded_ratio_or_none(getattr(context, "compression_target_ratio", None))
    if target_ratio == 0:
        target_ratio = None
    min_tokens = _positive_int_or_none(getattr(context, "compression_min_tokens", None))
    policy = resolve_policy(getattr(context, "compression_auth_mode", None))
    max_items_after_crush = _positive_int_or_none(getattr(context, "compression_max_items_after_crush", None))
    lossless_min_savings_ratio = _bounded_ratio_or_none(
        getattr(context, "compression_lossless_min_savings_ratio", None)
    )
    enable_ccr_marker = _bool_or_none(getattr(context, "compression_enable_ccr_marker", None))
    image_optimize = bool(getattr(context, "compression_image_optimize", True))
    target_token = _current_target_ratio.set(target_ratio)
    min_tokens_token = _current_min_tokens.set(min_tokens)
    max_items_token = _current_max_items_after_crush.set(max_items_after_crush)
    savings_ratio_token = _current_lossless_min_savings_ratio.set(lossless_min_savings_ratio)
    ccr_marker_token = _current_enable_ccr_marker.set(enable_ccr_marker)
    image_optimize_token = _current_image_optimize.set(image_optimize)
    policy_token = _current_policy.set(policy)
    cache_token = _current_text_compression_cache.set({})
    try:
        tokens_before = _context_token_count(context)
        stats_before = get_compression_stats()
        learning_stats_before = get_compression_learning_stats()
        unit_stats_before = get_unit_outcome_stats()
        frozen_count = max(0, int(getattr(context, "compression_frozen_message_count", 0) or 0))
        _record_cache_alignment_scan(
            _cache_alignment_system_texts(context, frozen_count=frozen_count),
            skipped_by_policy=not policy.cache_aligner_enabled,
        )
        cache_context = _apply_compression_cache(context, frozen_count=frozen_count)
        transformed = _compress_context_inner(cache_context, fn, policy)
        tokens_after = _context_token_count(transformed)
        if tokens_after > tokens_before:
            _restore_compression_stats(stats_before)
            _restore_compression_learning_stats(learning_stats_before)
            _restore_unit_outcome_stats(unit_stats_before)
            return context
        _update_compression_cache_from_messages(
            list(getattr(context, "messages", []) or []),
            list(getattr(transformed, "messages", []) or []),
            frozen_count=frozen_count,
        )
        return transformed
    finally:
        _current_policy.reset(policy_token)
        _current_image_optimize.reset(image_optimize_token)
        _current_enable_ccr_marker.reset(ccr_marker_token)
        _current_lossless_min_savings_ratio.reset(savings_ratio_token)
        _current_max_items_after_crush.reset(max_items_token)
        _current_min_tokens.reset(min_tokens_token)
        _current_target_ratio.reset(target_token)
        _current_text_compression_cache.reset(cache_token)


def _compress_context_inner(context: Any, fn: CompressFn, policy: CompressionPolicy) -> Any:
    frozen_count = max(0, int(getattr(context, "compression_frozen_message_count", 0) or 0))
    messages = list(context.messages)
    compress_user_messages = bool(getattr(context, "compression_compress_user_messages", False))
    compress_system_messages = bool(getattr(context, "compression_compress_system_messages", False))
    if policy.live_zone_only:
        compress_user_messages = False
        compress_system_messages = False
    compress_stale_reads = bool(getattr(context, "compression_compress_stale_reads", True))
    compress_superseded_reads = bool(getattr(context, "compression_compress_superseded_reads", False))
    min_read_lifecycle_value = _int_or_none(
        getattr(context, "compression_read_lifecycle_min_bytes", READ_LIFECYCLE_MIN_BYTES)
    )
    min_read_lifecycle_bytes = max(
        0,
        min_read_lifecycle_value if min_read_lifecycle_value is not None else READ_LIFECYCLE_MIN_BYTES,
    )
    protect_recent_value = _int_or_none(getattr(context, "compression_protect_recent", RECENT_CODE_PROTECTION_MESSAGES))
    protect_recent = max(0, protect_recent_value if protect_recent_value is not None else RECENT_CODE_PROTECTION_MESSAGES)
    replacements = _read_lifecycle_replacements(
        messages,
        frozen_count,
        compress_stale=compress_stale_reads,
        compress_superseded=compress_superseded_reads,
    )
    tool_name_by_id = _tool_name_by_call_id(messages, frozen_count)
    analysis_intent = _has_analysis_intent(messages, frozen_count)
    tools, tools_before_bytes, tools_after_bytes, tools_changed = (
        _xform_tools(getattr(context, "tools", None))
        if not policy.live_zone_only
        else (getattr(context, "tools", None), 0, 0, False)
    )
    if tools_changed:
        _emit_compression_event(
            CompressionEvent(
                strategy="tool_schema",
                original_tokens=_approx_tokens_from_bytes(tools_before_bytes),
                compressed_tokens=_approx_tokens_from_bytes(tools_after_bytes),
                original_bytes=tools_before_bytes,
                compressed_bytes=tools_after_bytes,
                role="tool_schema",
            )
        )
    latest_user_index = -1
    for i, msg in enumerate(messages):
        if i >= frozen_count and _message_role(msg) == "user":
            latest_user_index = i
    updates: dict[str, Any] = {
        "messages": [
            _xform_message(
                m
                if i < frozen_count
                else _apply_read_lifecycle(
                    m,
                    fn,
                    replacements,
                    min_size_bytes=min_read_lifecycle_bytes,
                ),
                fn,
                frozen=i < frozen_count,
                live_user=i == latest_user_index,
                protect_code=(
                    (protect_recent > 0 and (len(messages) - i) <= protect_recent)
                    or analysis_intent
                ),
                compress_user_messages=compress_user_messages,
                compress_system_messages=compress_system_messages,
                tool_name_by_id=tool_name_by_id,
            )
            for i, m in enumerate(messages)
        ]
    }
    if tools_changed:
        updates["tools"] = tools
    system_prompt = getattr(context, "system_prompt", None)
    if compress_system_messages and isinstance(system_prompt, str):
        compressed_system = _compress_text_value(
            system_prompt,
            fn,
            role="system",
            protect_code=analysis_intent,
        )
        if compressed_system is not None:
            updates["system_prompt"] = compressed_system
    elif isinstance(system_prompt, str) and system_prompt:
        _record_unit_outcome("protected_system_message")
    return context.model_copy(
        update=updates
    )
