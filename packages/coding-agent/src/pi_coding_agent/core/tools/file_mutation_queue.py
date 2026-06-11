"""Serialize file mutations targeting the same canonical file."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_locks: dict[str, asyncio.Lock] = {}
_registration_lock = asyncio.Lock()


async def get_mutation_queue_key(file_path: str) -> str:
    resolved = os.path.abspath(file_path)
    try:
        return os.path.realpath(resolved)
    except OSError:
        return resolved


async def with_file_mutation_queue(file_path: str, fn: Callable[[], Awaitable[T]]) -> T:
    key = await get_mutation_queue_key(file_path)
    async with _registration_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
    try:
        async with lock:
            return await fn()
    finally:
        if not lock.locked():
            async with _registration_lock:
                if _locks.get(key) is lock and not lock.locked():
                    _locks.pop(key, None)


def active_file_mutation_queue_count() -> int:
    return len(_locks)


__all__ = [
    "active_file_mutation_queue_count",
    "get_mutation_queue_key",
    "with_file_mutation_queue",
]
