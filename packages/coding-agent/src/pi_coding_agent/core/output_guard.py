"""Raw stdout guard used by harness/runtime modes."""
from __future__ import annotations

import asyncio
import sys
from typing import TextIO


class _StdoutProxy:
    def __init__(self, stderr: TextIO) -> None:
        self._stderr = stderr

    def write(self, text: str) -> int:
        return self._stderr.write(text)

    def flush(self) -> None:
        self._stderr.flush()

    def __getattr__(self, name: str):
        return getattr(self._stderr, name)


_original_stdout: TextIO | None = None
_raw_stdout: TextIO | None = None
_write_lock = asyncio.Lock()


def take_over_stdout() -> None:
    global _original_stdout, _raw_stdout
    if _original_stdout is not None:
        return
    _original_stdout = sys.stdout
    _raw_stdout = sys.stdout
    sys.stdout = _StdoutProxy(sys.stderr)  # type: ignore[assignment]


def restore_stdout() -> None:
    global _original_stdout, _raw_stdout
    if _original_stdout is None:
        return
    sys.stdout = _original_stdout
    _original_stdout = None
    _raw_stdout = None


def is_stdout_taken_over() -> bool:
    return _original_stdout is not None


async def write_raw_stdout(text: str) -> None:
    if not text:
        return
    stream = _raw_stdout or sys.stdout
    async with _write_lock:
        stream.write(text)
        stream.flush()


async def wait_for_raw_stdout_backpressure() -> None:
    async with _write_lock:
        return


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()
    stream = _raw_stdout or sys.stdout
    stream.flush()


__all__ = [
    "flush_raw_stdout",
    "is_stdout_taken_over",
    "restore_stdout",
    "take_over_stdout",
    "wait_for_raw_stdout_backpressure",
    "write_raw_stdout",
]
