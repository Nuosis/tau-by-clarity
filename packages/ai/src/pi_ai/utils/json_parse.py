"""
Partial JSON parsing for streaming tool arguments — mirrors packages/ai/src/utils/json-parse.ts
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass(frozen=True)
class StreamingJsonParseResult:
    raw: str
    value: dict[str, Any] | None
    repaired_text: str | None = None
    repair_applied: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return isinstance(self.value, dict)


def parse_partial_json(text: str) -> dict[str, Any] | None:
    # Alias for compatibility
    return _parse_partial_json_impl(text)


def parse_streaming_json(text: str) -> dict[str, Any]:
    """
    Parse potentially incomplete JSON from a streaming response.
    Returns empty dict {} on failure (mirrors TS behavior).
    """
    result = _parse_partial_json_impl(text)
    return result if result is not None else {}  # Return {} instead of None


def parse_streaming_json_result(text: str) -> StreamingJsonParseResult:
    """
    Parse streamed tool arguments with evidence.

    Unlike parse_streaming_json(), this never collapses malformed non-empty JSON
    to {}. Callers can execute repaired values or preserve the raw malformed
    payload for diagnostics.
    """
    raw = text or ""
    if not raw.strip():
        return StreamingJsonParseResult(raw=raw, value=None, error="empty")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return StreamingJsonParseResult(raw=raw, value=parsed)
        return StreamingJsonParseResult(raw=raw, value=None, error=f"expected object, got {type(parsed).__name__}")
    except json.JSONDecodeError as exc:
        original_error = f"{exc.msg} at line {exc.lineno} column {exc.colno} (char {exc.pos})"

    repaired = _try_repair_json_text(raw)
    if repaired is not None:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return StreamingJsonParseResult(
                    raw=raw,
                    value=parsed,
                    repaired_text=repaired,
                    repair_applied=True,
                    error=original_error,
                )
            return StreamingJsonParseResult(
                raw=raw,
                value=None,
                repaired_text=repaired,
                repair_applied=True,
                error=f"expected object after repair, got {type(parsed).__name__}",
            )
        except json.JSONDecodeError as exc:
            return StreamingJsonParseResult(
                raw=raw,
                value=None,
                repaired_text=repaired,
                repair_applied=True,
                error=f"{original_error}; repair failed: {exc.msg} at line {exc.lineno} column {exc.colno} (char {exc.pos})",
            )

    return StreamingJsonParseResult(raw=raw, value=None, error=original_error)


def _parse_partial_json_impl(text: str) -> dict[str, Any] | None:
    """
    Parse potentially incomplete JSON from a streaming response.

    Tries exact parse first, then attempts to fix common truncation issues.
    Returns None if the text cannot be parsed even partially.
    """
    if not text or not text.strip():
        return None

    # Try exact parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to complete truncated JSON by closing open structures
    fixed = _try_fix_json(text)
    if fixed is not None:
        return fixed

    return None


def _try_fix_json(text: str) -> dict[str, Any] | None:
    """Attempt to fix truncated JSON by closing unclosed delimiters."""
    candidate = _try_repair_json_text(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _try_repair_json_text(text: str) -> str | None:
    """Attempt mechanical JSON completion without inventing semantic content."""
    stripped = text.strip()

    # Must start with {
    if not stripped.startswith("{"):
        return None

    # Count open/close braces and brackets, handle strings
    stack: list[str] = []
    in_string = False
    escape_next = False

    for char in stripped:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}",  "]"):
            if stack and stack[-1] == char:
                stack.pop()

    repaired = stripped
    if in_string:
        repaired += '"'

    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r":\s*$", ": null", repaired)

    # Close unclosed structures.
    closing = "".join(reversed(stack))
    candidate = repaired + closing

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None
