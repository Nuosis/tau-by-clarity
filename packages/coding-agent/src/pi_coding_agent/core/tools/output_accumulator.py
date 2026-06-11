"""Streaming output accumulator.

Mirrors the Node helper used by bash/tool execution: keep bounded display
output in memory while preserving full output to a temp file once limits are
exceeded.
"""
from __future__ import annotations

import codecs
import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO
from uuid import uuid4

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


@dataclass
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    full_output_path: str | None = None


class OutputAccumulator:
    def __init__(
        self,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        temp_file_prefix: str = "pi-output",
    ) -> None:
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.max_rolling_bytes = max(max_bytes * 2, 1)
        self.temp_file_prefix = temp_file_prefix
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

        self._raw_chunks: list[bytes] = []
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False
        self._temp_file_path: str | None = None
        self._temp_file: BinaryIO | None = None

    @property
    def full_output_path(self) -> str | None:
        return self._temp_file_path

    @property
    def total_lines(self) -> int:
        return self._total_lines

    @property
    def total_bytes(self) -> int:
        return self._total_decoded_bytes

    def append(self, data: bytes | bytearray | memoryview | str) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self._total_raw_bytes += len(raw)
        self._append_decoded_text(self.decoder.decode(raw, final=False))

        if self._temp_file is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            assert self._temp_file is not None
            self._temp_file.write(raw)
        elif raw:
            self._raw_chunks.append(raw)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._append_decoded_text(self.decoder.decode(b"", final=True))
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, *, persist_if_truncated: bool = False) -> OutputSnapshot:
        tail = truncate_tail(
            self._snapshot_text(),
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )
        truncated = self._total_lines > self.max_lines or self._total_decoded_bytes > self.max_bytes
        truncation = TruncationResult(
            content=tail.content,
            truncated=truncated,
            truncated_by=tail.truncated_by if truncated else None,
            output_lines=tail.output_lines,
            total_lines=self._total_lines,
            output_bytes=tail.output_bytes,
            first_line_exceeds_limit=tail.first_line_exceeds_limit,
            last_line_partial=tail.last_line_partial,
        )
        if persist_if_truncated and truncation.truncated:
            self._ensure_temp_file()
        return OutputSnapshot(
            content=truncation.content,
            truncation=truncation,
            full_output_path=self._temp_file_path,
        )

    def close_temp_file(self) -> None:
        if self._temp_file is None:
            return
        self._temp_file.flush()
        self._temp_file.close()
        self._temp_file = None

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return
        encoded_len = len(text.encode("utf-8"))
        self._total_decoded_bytes += encoded_len
        self._tail_text += text
        self._tail_bytes += encoded_len
        if self._tail_bytes > self.max_rolling_bytes * 2:
            self._trim_tail()

        newlines = text.count("\n")
        if newlines == 0:
            self._current_line_bytes += encoded_len
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            tail = text.rsplit("\n", 1)[1]
            self._current_line_bytes = len(tail.encode("utf-8"))
            self._has_open_line = bool(tail)
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        raw = self._tail_text.encode("utf-8")
        if len(raw) <= self.max_rolling_bytes:
            self._tail_bytes = len(raw)
            return
        start = len(raw) - self.max_rolling_bytes
        while start < len(raw) and (raw[start] & 0xC0) == 0x80:
            start += 1
        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else raw[start - 1] == 0x0A
        )
        self._tail_text = raw[start:].decode("utf-8", errors="replace")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self.max_bytes
            or self._total_decoded_bytes > self.max_bytes
            or self._total_lines > self.max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        self._temp_file_path = os.path.join(
            tempfile.gettempdir(),
            f"{self.temp_file_prefix}-{uuid4().hex[:16]}.log",
        )
        self._temp_file = open(self._temp_file_path, "wb")
        for chunk in self._raw_chunks:
            self._temp_file.write(chunk)
        self._raw_chunks = []


__all__ = ["OutputAccumulator", "OutputSnapshot"]
