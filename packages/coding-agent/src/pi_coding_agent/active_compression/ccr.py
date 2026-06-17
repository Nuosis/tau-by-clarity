"""CCR — Compress · Cache · Retrieve store.

A local, durable, hash-indexed store of original payloads. When the compressor
crushes a tool output it caches the original here under a short handle, so the
full content is recoverable out-of-band (the §12 reversibility guarantee). Local
SQLite only — nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time


class CCRStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Phase-4 (Context Tracker): handles whose original has been retrieved/
        # expanded for the model *in the current conversation*. Once expanded, that
        # content must NOT be re-compressed on subsequent turns, or retrieval is
        # futile (the model re-elides forever). Per the Headroom spec this is
        # per-conversation state ("Across multiple turns... remembers what was
        # compressed in earlier turns"), so it is in-memory and process-scoped — it
        # MUST NOT persist across runs/sessions, or AC silently degrades to a no-op
        # for any payload ever retrieved once. The durable `ccr` cache below is
        # separate: originals stay retrievable on disk; only expansion state resets.
        self._expanded: set[str] = set()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ccr "
                "(handle TEXT PRIMARY KEY, content TEXT NOT NULL, created_at REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def put(self, content: str) -> str:
        """Store original content, returns its stable hash handle (idempotent)."""
        handle = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        with self._lock, self._connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO ccr (handle, content, created_at) VALUES (?, ?, ?)",
                (handle, content, time.time()),
            )
        return handle

    def get(self, handle: str) -> str | None:
        with self._connect() as c:
            row = c.execute("SELECT content FROM ccr WHERE handle = ?", (handle,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def handle_for(content: str) -> str:
        """The handle a given content would hash to (matches put())."""
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]

    def mark_expanded(self, handle: str) -> None:
        """Mark a handle as expanded for the model in the current conversation.

        In-memory only: this is per-process/per-conversation state and must not
        persist across runs (a durable mark would permanently disable compression
        for that content everywhere)."""
        with self._lock:
            self._expanded.add(handle)

    def is_expanded(self, handle: str) -> bool:
        with self._lock:
            return handle in self._expanded
