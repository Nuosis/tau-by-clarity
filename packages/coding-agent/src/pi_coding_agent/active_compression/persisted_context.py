"""Persisted CCR-backed context support.

The outbound compression hook keeps the in-memory agent transcript raw. This
module is the session-persistence companion: it stores the model-facing
compressed message in JSONL while preserving every CCR original in non-context
metadata so refs can be refreshed or rebuilt on resume.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from .ccr import CCRStore

_BRACKET_HANDLE_RE = re.compile(r"\[CCR:([0-9a-fA-F]{12})\]")
_INLINE_HANDLE_RE = re.compile(r"<<ccr:([0-9a-fA-F]{12})(?:[,\s][^>]*)?>>")

METADATA_KEY = "activeCompression"


def extract_ccr_handles(text: str) -> list[str]:
    """Return unique CCR handles in first-seen order from Tau marker formats."""
    seen: set[str] = set()
    out: list[str] = []
    for pattern in (_BRACKET_HANDLE_RE, _INLINE_HANDLE_RE):
        for match in pattern.finditer(text or ""):
            handle = match.group(1).lower()
            if handle not in seen:
                seen.add(handle)
                out.append(handle)
    return out


def _text_items(message: dict[str, Any]) -> list[tuple[list[Any], str]]:
    role = message.get("role")
    content = message.get("content")
    if role == "toolResult" and isinstance(content, list):
        out: list[tuple[list[Any], str]] = []
        for index, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                out.append((["content", index, "text"], item["text"]))
        return out
    return []


def _set_path(target: dict[str, Any], path: list[Any], value: str) -> None:
    node: Any = target
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _ccr_store() -> CCRStore:
    from pi_coding_agent import active_compression

    return active_compression._ccr()


def _tool_call_id_from_message(message: Any) -> str | None:
    if isinstance(message, dict):
        value = message.get("tool_call_id") or message.get("toolCallId")
    else:
        value = getattr(message, "tool_call_id", None) or getattr(message, "toolCallId", None)
    return value if isinstance(value, str) and value else None


def _entry_message(entry: dict[str, Any]) -> dict[str, Any] | None:
    msg = entry.get("message")
    return msg if isinstance(msg, dict) else None


def _compress_text_with_tool_context(text: str, *, tool_name: str | None, tool_call_id: str | None) -> str:
    from pi_coding_agent import active_compression

    try:
        from pi_ai import (
            get_current_compression_tool_call_id,
            get_current_compression_tool_name,
            set_current_compression_tool_context,
        )
    except Exception:
        return active_compression.compress(text)

    previous_name = get_current_compression_tool_name()
    previous_call_id = get_current_compression_tool_call_id()
    set_current_compression_tool_context(tool_name, tool_call_id)
    try:
        return active_compression.compress(text)
    finally:
        set_current_compression_tool_context(previous_name, previous_call_id)


def compress_message_for_persistence(message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a compressed model-facing message plus durable CCR metadata.

    Only tool-result text is compressed here. User intent and assistant output
    stay exact, matching the outbound active-compression policy.
    """
    text_items = _text_items(message)
    if not text_items:
        return message, None

    store = _ccr_store()
    compressed = copy.deepcopy(message)
    refs: list[dict[str, Any]] = []
    changed = False
    tool_name = message.get("tool_name") or message.get("toolName")
    tool_call_id = message.get("tool_call_id") or message.get("toolCallId")
    for path, original_text in text_items:
        compressed_text = _compress_text_with_tool_context(
            original_text,
            tool_name=tool_name if isinstance(tool_name, str) else None,
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
        )
        if compressed_text == original_text:
            continue
        handles = extract_ccr_handles(compressed_text)
        if not handles:
            continue
        _set_path(compressed, path, compressed_text)
        changed = True
        for handle in handles:
            original = store.get(handle)
            if original is None and handle == CCRStore.handle_for(original_text):
                original = original_text
            if original is None:
                continue
            refs.append({
                "handle": handle,
                "path": path,
                "original": original,
                "compressed": compressed_text,
                "toolName": tool_name,
                "toolCallId": tool_call_id,
            })

    if not changed or not refs:
        return message, None
    return compressed, {"version": 1, "refs": refs}


def refresh_or_rebuild_session_refs(entries: list[dict[str, Any]], *, store: CCRStore | None = None) -> dict[str, int]:
    """Refresh live CCR refs and rebuild missing/expired refs from session metadata."""
    ccr = store or _ccr_store()
    refreshed = 0
    rebuilt = 0
    missing = 0
    for entry in entries:
        meta = entry.get(METADATA_KEY)
        if not isinstance(meta, dict):
            continue
        refs = meta.get("refs")
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            handle = ref.get("handle")
            original = ref.get("original")
            if not isinstance(handle, str) or not isinstance(original, str):
                missing += 1
                continue
            if ccr.refresh(handle):
                refreshed += 1
                continue
            ccr.put_with_handle(
                handle,
                original,
                compressed_content=str(ref.get("compressed") or ""),
                tool_name=ref.get("toolName") if isinstance(ref.get("toolName"), str) else None,
                tool_call_id=ref.get("toolCallId") if isinstance(ref.get("toolCallId"), str) else None,
            )
            rebuilt += 1
    return {"refreshed": refreshed, "rebuilt": rebuilt, "missing": missing}


def apply_persisted_compressed_messages(messages: list[Any], entries: list[dict[str, Any]]) -> list[Any]:
    """Replace live raw tool-result messages with their persisted compressed form.

    Agent state remains the runtime source of truth, but the model-facing context
    should use the same CCR-backed refs persisted in the session JSONL. Matching
    by tool_call_id keeps this scoped to concrete tool results.
    """
    by_call_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry.get(METADATA_KEY), dict):
            continue
        msg = _entry_message(entry)
        if not msg or msg.get("role") != "toolResult":
            continue
        tool_call_id = _tool_call_id_from_message(msg)
        if tool_call_id:
            by_call_id[tool_call_id] = msg
    if not by_call_id:
        return messages

    changed = False
    out: list[Any] = []
    for msg in messages:
        if (tool_call_id := _tool_call_id_from_message(msg)) and tool_call_id in by_call_id:
            out.append(copy.deepcopy(by_call_id[tool_call_id]))
            changed = True
        else:
            out.append(msg)
    return out if changed else messages
