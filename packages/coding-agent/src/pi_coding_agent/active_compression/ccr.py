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
