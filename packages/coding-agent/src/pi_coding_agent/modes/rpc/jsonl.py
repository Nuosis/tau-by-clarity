"""Strict LF-framed JSONL helpers."""
from __future__ import annotations

import codecs
import json
from collections.abc import Callable, Iterable
from typing import Any


def serialize_json_line(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n"


class JsonlLineReader:
    """Incremental UTF-8 JSONL line reader that splits only on LF."""

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self.on_line = on_line
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.buffer = ""

    def feed(self, chunk: bytes | str) -> None:
        self.buffer += chunk if isinstance(chunk, str) else self.decoder.decode(chunk, final=False)
        while True:
            newline_index = self.buffer.find("\n")
            if newline_index == -1:
                return
            line = self.buffer[:newline_index]
            self.buffer = self.buffer[newline_index + 1 :]
            self._emit(line)

    def end(self) -> None:
        self.buffer += self.decoder.decode(b"", final=True)
        if self.buffer:
            self._emit(self.buffer)
            self.buffer = ""

    def _emit(self, line: str) -> None:
        self.on_line(line[:-1] if line.endswith("\r") else line)


def read_jsonl_lines(chunks: Iterable[bytes | str]) -> list[str]:
    lines: list[str] = []
    reader = JsonlLineReader(lines.append)
    for chunk in chunks:
        reader.feed(chunk)
    reader.end()
    return lines


__all__ = ["JsonlLineReader", "read_jsonl_lines", "serialize_json_line"]
