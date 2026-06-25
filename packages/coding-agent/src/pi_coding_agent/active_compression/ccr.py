"""CCR — Compress · Cache · Retrieve store.

A local, durable, hash-indexed store of original payloads. When the compressor
crushes a tool output it caches the original here under a short handle, so the
full content is recoverable out-of-band (the §12 reversibility guarantee). Local
SQLite only — nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_CCR_TTL_SECONDS = 1800
CCR_TTL_SECONDS_ENV = "HEADROOM_CCR_TTL_SECONDS"

logger = logging.getLogger(__name__)

_RETRIEVAL_LOG_PREVIEW_CHARS = 4096
_SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_-]*)"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]+)"
)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")
_API_KEY_VALUE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _env_default_ttl() -> int:
    raw = os.environ.get(CCR_TTL_SECONDS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_CCR_TTL_SECONDS
    try:
        ttl = int(raw)
    except ValueError:
        return DEFAULT_CCR_TTL_SECONDS
    return ttl if ttl > 0 else DEFAULT_CCR_TTL_SECONDS


def _redact_retrieval_log_payload(payload: str) -> str:
    redacted = _SECRET_KEY_VALUE_RE.sub(r"\1\2\3[REDACTED]", payload)
    redacted = _AUTH_VALUE_RE.sub(r"\1 [REDACTED]", redacted)
    return _API_KEY_VALUE_RE.sub("sk-[REDACTED]", redacted)


def _payload_for_retrieval_log(payload: str) -> dict[str, Any]:
    redacted = _redact_retrieval_log_payload(payload)
    preview = redacted[:_RETRIEVAL_LOG_PREVIEW_CHARS]
    return {
        "payload_chars": len(payload),
        "payload_preview_chars": len(preview),
        "payload_truncated": len(redacted) > len(preview),
        "payload_preview": preview,
    }


@dataclass
class CompressionEntry:
    hash: str
    original_content: str
    compressed_content: str = ""
    original_tokens: int = 0
    compressed_tokens: int = 0
    original_item_count: int = 0
    compressed_item_count: int = 0
    tool_name: str | None = None
    tool_call_id: str | None = None
    query_context: str | None = None
    created_at: float = 0.0
    ttl: int = DEFAULT_CCR_TTL_SECONDS
    compression_strategy: str | None = None
    retrieval_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    def record_access(self, query: str | None = None) -> None:
        self.retrieval_count += 1
        self.last_accessed = time.time()
        if query and query not in self.search_queries:
            self.search_queries.append(query)
            if len(self.search_queries) > 10:
                self.search_queries = self.search_queries[-10:]


@dataclass(frozen=True)
class RetrievalEvent:
    hash: str
    query: str | None
    items_retrieved: int
    total_items: int
    tool_name: str | None
    timestamp: float
    retrieval_type: str


class CCRStore:
    def __init__(
        self,
        path: str,
        *,
        max_entries: int = 1000,
        default_ttl: int | None = None,
        enable_feedback: bool = True,
    ) -> None:
        self.path = path
        self.max_entries = max_entries
        self.default_ttl = default_ttl if default_ttl is not None else _env_default_ttl()
        self.enable_feedback = enable_feedback
        self._lock = threading.Lock()
        self._retrieval_events: list[RetrievalEvent] = []
        self._max_events = 1000
        self._eviction_heap: list[tuple[float, str]] = []
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
            self._migrate(c)
            for handle, created_at in c.execute("SELECT handle, created_at FROM ccr").fetchall():
                heapq.heappush(self._eviction_heap, (float(created_at or 0.0), str(handle)))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _migrate(self, c: sqlite3.Connection) -> None:
        existing = {row[1] for row in c.execute("PRAGMA table_info(ccr)").fetchall()}
        columns = {
            "compressed_content": "TEXT NOT NULL DEFAULT ''",
            "original_tokens": "INTEGER NOT NULL DEFAULT 0",
            "compressed_tokens": "INTEGER NOT NULL DEFAULT 0",
            "original_item_count": "INTEGER NOT NULL DEFAULT 0",
            "compressed_item_count": "INTEGER NOT NULL DEFAULT 0",
            "tool_name": "TEXT",
            "tool_call_id": "TEXT",
            "query_context": "TEXT",
            "ttl": f"INTEGER NOT NULL DEFAULT {self.default_ttl}",
            "compression_strategy": "TEXT",
            "retrieval_count": "INTEGER NOT NULL DEFAULT 0",
            "search_queries": "TEXT NOT NULL DEFAULT '[]'",
            "last_accessed": "REAL",
        }
        for name, ddl in columns.items():
            if name not in existing:
                c.execute(f"ALTER TABLE ccr ADD COLUMN {name} {ddl}")

    def put(
        self,
        content: str,
        *,
        compressed_content: str = "",
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        ttl: int | None = None,
        compression_strategy: str | None = None,
    ) -> str:
        """Store original content, returns its stable hash handle (idempotent)."""
        handle = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        return self.put_with_handle(
            handle,
            content,
            compressed_content=compressed_content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_item_count,
            compressed_item_count=compressed_item_count,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=query_context,
            ttl=ttl,
            compression_strategy=compression_strategy,
        )

    def put_with_handle(
        self,
        handle: str,
        content: str,
        *,
        compressed_content: str = "",
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        ttl: int | None = None,
        compression_strategy: str | None = None,
    ) -> str:
        """Store original content under an explicit compatible handle."""
        if not handle or not all(c in "0123456789abcdefABCDEF" for c in handle):
            raise ValueError(f"handle must be a non-empty hex string, got {handle!r}")
        created_at = time.time()
        ttl_value = ttl if ttl is not None else self.default_ttl
        with self._lock, self._connect() as c:
            self._clean_expired_locked(c)
            self._evict_if_needed_locked(c)
            c.execute(
                """
                INSERT INTO ccr (
                    handle, content, created_at, compressed_content, original_tokens,
                    compressed_tokens, original_item_count, compressed_item_count,
                    tool_name, tool_call_id, query_context, ttl, compression_strategy,
                    retrieval_count, search_queries, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '[]', NULL)
                ON CONFLICT(handle) DO UPDATE SET
                    content=excluded.content,
                    created_at=excluded.created_at,
                    compressed_content=excluded.compressed_content,
                    original_tokens=excluded.original_tokens,
                    compressed_tokens=excluded.compressed_tokens,
                    original_item_count=excluded.original_item_count,
                    compressed_item_count=excluded.compressed_item_count,
                    tool_name=excluded.tool_name,
                    tool_call_id=excluded.tool_call_id,
                    query_context=excluded.query_context,
                    ttl=excluded.ttl,
                    compression_strategy=excluded.compression_strategy,
                    retrieval_count=0,
                    search_queries='[]',
                    last_accessed=NULL
                """,
                (
                    handle.lower(),
                    content,
                    created_at,
                    compressed_content,
                    original_tokens,
                    compressed_tokens,
                    original_item_count,
                    compressed_item_count,
                    tool_name,
                    tool_call_id,
                    query_context,
                    ttl_value,
                    compression_strategy,
                ),
            )
            heapq.heappush(self._eviction_heap, (created_at, handle.lower()))
        return handle.lower()

    def get(self, handle: str) -> str | None:
        entry = self.retrieve(handle, log_event=False)
        return entry.original_content if entry is not None else None

    def retrieve(
        self,
        handle: str,
        query: str | None = None,
        *,
        log_event: bool = True,
    ) -> CompressionEntry | None:
        with self._connect() as c:
            with self._lock:
                entry = self._entry_locked(c, handle)
                if entry is None:
                    return None
                if entry.is_expired():
                    c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
                    return None
                entry.record_access(query)
                self._write_access_locked(c, entry)
                if log_event:
                    self._record_event_locked(
                        handle=entry.hash,
                        query=query,
                        items_retrieved=entry.original_item_count,
                        total_items=entry.original_item_count,
                        tool_name=entry.tool_name,
                        retrieval_type="full",
                    )
                    self._log_retrieval_payload(entry, query, "full", entry.original_content)
                return replace(entry, search_queries=list(entry.search_queries))

    def exists(self, handle: str, *, clean_expired: bool = False) -> bool:
        with self._lock, self._connect() as c:
            entry = self._entry_locked(c, handle)
            if entry is None:
                return False
            if not entry.is_expired():
                return True
            if clean_expired:
                c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
            return False

    def refresh(self, handle: str, *, ttl: int | None = None) -> bool:
        """Refresh a live CCR entry so session-held refs do not expire mid-loop.

        Returns False for missing or already-expired handles. Expired rows are
        deleted instead of revived because callers that need durable recovery
        must rebuild from the session-owned original.
        """
        now = time.time()
        with self._lock, self._connect() as c:
            entry = self._entry_locked(c, handle)
            if entry is None:
                return False
            if entry.is_expired():
                c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
                return False
            ttl_value = ttl if ttl is not None else entry.ttl
            c.execute(
                "UPDATE ccr SET created_at = ?, ttl = ? WHERE handle = ?",
                (now, ttl_value, handle.lower()),
            )
            heapq.heappush(self._eviction_heap, (now, handle.lower()))
            return True

    def get_metadata(self, handle: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as c:
            entry = self._entry_locked(c, handle)
            if entry is None:
                return None
            if entry.is_expired():
                c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
                return None
            return {
                "hash": entry.hash,
                "tool_name": entry.tool_name,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "query_context": entry.query_context,
                "compressed_content": entry.compressed_content,
                "created_at": entry.created_at,
                "ttl": entry.ttl,
            }

    def get_entry_status(self, handle: str, *, clean_expired: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as c:
            entry = self._entry_locked(c, handle)
            if entry is None:
                return {"hash": handle, "status": "missing", "default_ttl_seconds": self.default_ttl}
            age = now - entry.created_at
            expired = age > entry.ttl
            if expired and clean_expired:
                c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
            return {
                "hash": handle,
                "status": "expired" if expired else "available",
                "ttl_seconds": entry.ttl,
                "default_ttl_seconds": self.default_ttl,
                "created_at": entry.created_at,
                "expires_at": entry.created_at + entry.ttl,
                "age_seconds": age,
            }

    def search(self, handle: str, query: str, *, max_results: int = 20) -> list[Any]:
        entry = self.retrieve(handle, query=query, log_event=False)
        if entry is None:
            return []
        from .search import search_original

        result = search_original(entry.original_content, query, max_items=max_results)
        text = result.get("text") or ""
        if not text or result.get("kept_items", 0) <= 0:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [{"type": "text", "text": text}]
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [parsed]
        with self._lock:
            self._record_event_locked(
                handle=entry.hash,
                query=query,
                items_retrieved=len(items),
                total_items=int(result.get("total_items") or len(items)),
                tool_name=entry.tool_name,
                retrieval_type="search",
            )
        self._log_retrieval_payload(entry, query, "search", json.dumps(items, ensure_ascii=False))
        return items[:max_results]

    def get_retrieval_events(
        self,
        *,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[RetrievalEvent]:
        with self._lock:
            events = list(self._retrieval_events)
        if tool_name:
            events = [event for event in events if event.tool_name == tool_name]
        return list(reversed(events[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as c:
            self._clean_expired_locked(c)
            rows = c.execute(
                "SELECT original_tokens, compressed_tokens, retrieval_count FROM ccr"
            ).fetchall()
            return {
                "entry_count": len(rows),
                "max_entries": self.max_entries,
                "default_ttl_seconds": self.default_ttl,
                "total_original_tokens": sum(int(row[0] or 0) for row in rows),
                "total_compressed_tokens": sum(int(row[1] or 0) for row in rows),
                "total_retrievals": sum(int(row[2] or 0) for row in rows),
                "event_count": len(self._retrieval_events),
                "backend": {"backend_type": "sqlite", "path": self.path},
            }

    def clear(self) -> None:
        with self._lock, self._connect() as c:
            c.execute("DELETE FROM ccr")
            self._retrieval_events.clear()
            self._eviction_heap.clear()

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

    def _entry_locked(self, c: sqlite3.Connection, handle: str) -> CompressionEntry | None:
        row = c.execute(
            """
            SELECT handle, content, created_at, compressed_content, original_tokens,
                   compressed_tokens, original_item_count, compressed_item_count,
                   tool_name, tool_call_id, query_context, ttl, compression_strategy,
                   retrieval_count, search_queries, last_accessed
            FROM ccr WHERE handle = ?
            """,
            (handle.lower(),),
        ).fetchone()
        if row is None:
            return None
        try:
            search_queries = json.loads(row[14] or "[]")
        except Exception:
            search_queries = []
        return CompressionEntry(
            hash=row[0],
            original_content=row[1],
            created_at=float(row[2] or 0.0),
            compressed_content=row[3] or "",
            original_tokens=int(row[4] or 0),
            compressed_tokens=int(row[5] or 0),
            original_item_count=int(row[6] or 0),
            compressed_item_count=int(row[7] or 0),
            tool_name=row[8],
            tool_call_id=row[9],
            query_context=row[10],
            ttl=int(row[11] or self.default_ttl),
            compression_strategy=row[12],
            retrieval_count=int(row[13] or 0),
            search_queries=list(search_queries) if isinstance(search_queries, list) else [],
            last_accessed=row[15],
        )

    def _write_access_locked(self, c: sqlite3.Connection, entry: CompressionEntry) -> None:
        c.execute(
            """
            UPDATE ccr
            SET retrieval_count = ?, search_queries = ?, last_accessed = ?
            WHERE handle = ?
            """,
            (
                entry.retrieval_count,
                json.dumps(entry.search_queries, ensure_ascii=False),
                entry.last_accessed,
                entry.hash,
            ),
        )

    def _clean_expired_locked(self, c: sqlite3.Connection) -> None:
        now = time.time()
        c.execute("DELETE FROM ccr WHERE (? - created_at) > ttl", (now,))

    def _evict_if_needed_locked(self, c: sqlite3.Connection) -> None:
        while True:
            count = c.execute("SELECT COUNT(*) FROM ccr").fetchone()[0]
            if count < self.max_entries:
                return
            if not self._eviction_heap:
                row = c.execute("SELECT handle FROM ccr ORDER BY created_at ASC LIMIT 1").fetchone()
                if row is None:
                    return
                c.execute("DELETE FROM ccr WHERE handle = ?", (row[0],))
                continue
            created_at, handle = heapq.heappop(self._eviction_heap)
            row = c.execute("SELECT created_at FROM ccr WHERE handle = ?", (handle,)).fetchone()
            if row is not None and float(row[0]) == created_at:
                c.execute("DELETE FROM ccr WHERE handle = ?", (handle,))
                return

    def _record_event_locked(
        self,
        *,
        handle: str,
        query: str | None,
        items_retrieved: int,
        total_items: int,
        tool_name: str | None,
        retrieval_type: str,
    ) -> None:
        if not self.enable_feedback:
            return
        self._retrieval_events.append(
            RetrievalEvent(
                hash=handle,
                query=query,
                items_retrieved=items_retrieved,
                total_items=total_items,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type=retrieval_type,
            )
        )
        if len(self._retrieval_events) > self._max_events:
            self._retrieval_events = self._retrieval_events[-self._max_events :]

    def _log_retrieval_payload(
        self,
        entry: CompressionEntry,
        query: str | None,
        retrieval_type: str,
        payload: str,
    ) -> None:
        event = {
            "event": "ccr_retrieve",
            "hash": entry.hash,
            "retrieval_type": retrieval_type,
            "query": query,
            "items_retrieved": entry.original_item_count,
            "total_items": entry.original_item_count,
            "tool_name": entry.tool_name,
            "tool_call_id": entry.tool_call_id,
            "compression_strategy": entry.compression_strategy,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
            **_payload_for_retrieval_log(payload),
        }
        logger.info(
            "event=ccr_retrieve %s",
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        )
