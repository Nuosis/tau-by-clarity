"""
Text truncation utilities — mirrors packages/coding-agent/src/core/tools/truncate.ts
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KB (matches TS)
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None  # "lines" or "bytes"
    output_lines: int
    total_lines: int
    output_bytes: int
    first_line_exceeds_limit: bool = False
    last_line_partial: bool = False


def format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_}B"
    kb = bytes_ / 1024
    if kb < 1024:
        return f"{kb:.1f}KB"
    mb = kb / 1024
    return f"{mb:.1f}MB"


def truncate_head(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """
    Truncate text from the head (start), keeping the beginning.
    Mirrors truncateHead() in TypeScript.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    selected_lines: list[str] = []
    current_bytes = 0
    truncated_by: str | None = None

    # Check first line for byte-exceeds
    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            output_lines=0,
            total_lines=total_lines,
            output_bytes=0,
            first_line_exceeds_limit=True,
        )

    for i, line in enumerate(lines):
        if len(selected_lines) >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
        if current_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        selected_lines.append(line)
        current_bytes += line_bytes

    truncated = truncated_by is not None
    content = "\n".join(selected_lines)

    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        output_lines=len(selected_lines),
        total_lines=total_lines,
        output_bytes=current_bytes,
    )


def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    """
    Truncate a string to fit within a byte limit, keeping the end.
    Mirrors truncateStringToBytesFromEnd() in TypeScript.
    """
    buf = s.encode("utf-8")
    if len(buf) <= max_bytes:
        return s

    # Start from the end, skip max_bytes back
    start = len(buf) - max_bytes

    # Find a valid UTF-8 boundary (start of a character)
    # UTF-8 continuation bytes have the pattern 10xxxxxx (0x80)
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1

    return buf[start:].decode("utf-8", errors="ignore")


def truncate_tail(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """
    Truncate text from the tail (end), keeping the end.
    Mirrors truncateTail() in TypeScript.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))

    # Check if no truncation needed
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            truncated_by=None,
            output_lines=total_lines,
            total_lines=total_lines,
            output_bytes=total_bytes,
        )

    # Work from the end
    selected_lines: list[str] = []
    current_bytes = 0
    truncated_by: str | None = None
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(selected_lines) >= max_lines:
            truncated_by = "lines"
            break
        
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if selected_lines else 0)  # +1 for newline

        if current_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds maxBytes,
            # take the end of the line (partial)
            if not selected_lines:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                selected_lines.insert(0, truncated_line)
                current_bytes = len(truncated_line.encode("utf-8"))
                last_line_partial = True
            break

        selected_lines.insert(0, line)
        current_bytes += line_bytes

    # If we exited due to line limit
    if len(selected_lines) >= max_lines and current_bytes <= max_bytes:
        truncated_by = "lines"

    content = "\n".join(selected_lines)
    final_output_bytes = len(content.encode("utf-8"))

    return TruncationResult(
        content=content,
        truncated=True,
        truncated_by=truncated_by,
        output_lines=len(selected_lines),
        total_lines=total_lines,
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
    )


def truncate_line(line: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max_length with [truncated] suffix. Mirrors truncateLine() in TypeScript."""
    if len(line) <= max_length:
        return line, False
    return line[:max_length] + "... [truncated]", True
