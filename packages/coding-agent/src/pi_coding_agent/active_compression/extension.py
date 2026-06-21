"""CCR retrieve trigger — the §12 reversibility path made usable.

A single, consistent retrieval path: the **`ccr_retrieve` tool**, which is always
query-scoped. The model passes a required `query` and gets back only the matching
items (a BM25 search within the cached original), so retrieval can never reinflate
the context with the full payload. There is deliberately no harness-driven
full-expansion fallback and no full-dump escape hatch — those were back-doors that
defeated CCR's purpose of keeping the peak context small.

Loaded standalone by the extension loader → absolute imports.
"""

from __future__ import annotations

import re
from typing import Any

from pi_coding_agent.active_compression import is_enabled, retrieve
from pi_coding_agent.active_compression.search import search_original

_VALID_CCR_HANDLE_RE = re.compile(r"^[0-9a-fA-F]{12}$")


def _format_count_map(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(values.items()))


def _stats_payload() -> dict[str, Any]:
    try:
        from pi_ai import (
            get_cache_alignment_stats,
            get_compression_cache_stats,
            get_compression_learning_stats,
            get_compression_stats,
            get_unit_outcome_stats,
            has_compressor,
        )
    except Exception as exc:
        return {"error": f"pi_ai unavailable: {exc}"}

    stats = get_compression_stats()
    learning = get_compression_learning_stats()
    cache_alignment = get_cache_alignment_stats()
    compression_cache = get_compression_cache_stats()
    unit_outcomes = get_unit_outcome_stats()
    return {
        "enabled": is_enabled(),
        "compressor_registered": has_compressor(),
        "total_compressions": stats.total_compressions,
        "total_original_tokens": stats.total_original_tokens,
        "total_compressed_tokens": stats.total_compressed_tokens,
        "total_tokens_saved": stats.total_tokens_saved,
        "total_original_bytes": stats.total_original_bytes,
        "total_compressed_bytes": stats.total_compressed_bytes,
        "total_bytes_saved": stats.total_bytes_saved,
        "compressions_by_strategy": stats.compressions_by_strategy,
        "tokens_saved_by_strategy": stats.tokens_saved_by_strategy,
        "bytes_saved_by_strategy": stats.bytes_saved_by_strategy,
        "learning_events": learning.total_events,
        "learning_skipped_read_only": learning.total_skipped_read_only,
        "learning_tokens_saved": learning.total_tokens_saved,
        "learning_bytes_saved": learning.total_bytes_saved,
        "learning_events_by_strategy": learning.events_by_strategy,
        "learning_skipped_by_strategy": learning.skipped_by_strategy,
        "cache_alignment_scans": cache_alignment.total_scans,
        "cache_alignment_findings": cache_alignment.total_findings,
        "cache_alignment_skipped_by_policy": cache_alignment.skipped_by_policy,
        "cache_alignment_findings_by_label": cache_alignment.findings_by_label,
        "compression_cache_hits": compression_cache.hits,
        "compression_cache_misses": compression_cache.misses,
        "compression_cache_entries": compression_cache.entries,
        "compression_cache_tokens_saved": compression_cache.tokens_saved,
        "unit_outcomes": unit_outcomes.total_units,
        "unit_outcomes_by_reason": unit_outcomes.outcomes_by_reason,
        "unit_outcomes_by_category": unit_outcomes.outcomes_by_category,
    }


def _stats_text() -> str:
    payload = _stats_payload()
    if "error" in payload:
        return f"Active compression stats are unavailable because {payload['error']}."

    return "\n".join([
        "Active compression stats:",
        f"  enabled: {'yes' if payload['enabled'] else 'no'}",
        f"  compressor registered: {'yes' if payload['compressor_registered'] else 'no'}",
        f"  total compressions: {payload['total_compressions']}",
        f"  tokens: {payload['total_original_tokens']} -> {payload['total_compressed_tokens']} "
        f"(saved {payload['total_tokens_saved']})",
        f"  bytes: {payload['total_original_bytes']} -> {payload['total_compressed_bytes']} "
        f"(saved {payload['total_bytes_saved']})",
        f"  by strategy: {_format_count_map(payload['compressions_by_strategy'])}",
        f"  tokens saved by strategy: {_format_count_map(payload['tokens_saved_by_strategy'])}",
        f"  bytes saved by strategy: {_format_count_map(payload['bytes_saved_by_strategy'])}",
        f"  learning events: {payload['learning_events']} "
        f"(read-only skipped {payload['learning_skipped_read_only']})",
        f"  learning by strategy: {_format_count_map(payload['learning_events_by_strategy'])}",
        f"  learning skipped by strategy: {_format_count_map(payload['learning_skipped_by_strategy'])}",
        f"  cache alignment scans: {payload['cache_alignment_scans']} "
        f"(findings {payload['cache_alignment_findings']}, "
        f"policy skipped {payload['cache_alignment_skipped_by_policy']})",
        f"  volatile findings by label: {_format_count_map(payload['cache_alignment_findings_by_label'])}",
        f"  compression cache: hits {payload['compression_cache_hits']}, "
        f"misses {payload['compression_cache_misses']}, "
        f"entries {payload['compression_cache_entries']}, "
        f"tokens saved {payload['compression_cache_tokens_saved']}",
        f"  unit outcomes: {payload['unit_outcomes']}",
        f"  unit outcomes by category: {_format_count_map(payload['unit_outcomes_by_category'])}",
        f"  unit outcomes by reason: {_format_count_map(payload['unit_outcomes_by_reason'])}",
        "Subcommands: stats | status | reset",
    ])


def _retrieve_tool_response(handle: str, query: str, *, tool_name: str) -> dict[str, Any]:
    if not _VALID_CCR_HANDLE_RE.fullmatch(handle or ""):
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Invalid CCR handle {handle!r}. CCR handles are 12 hex characters "
                    "from a [CCR:<handle>] marker; do not use task IDs or needle IDs as handles."
                ),
            }],
            "isError": True,
        }

    original = retrieve(handle)
    if original is None:
        return {
            "content": [{"type": "text", "text": f"No CCR entry for handle {handle!r}."}],
            "isError": True,
        }

    # Retrieval is ALWAYS query-scoped (Headroom BM25): there is no full-payload
    # escape hatch. Returning only the relevant subset is the entire point —
    # it keeps the peak context small and the original stays compressed for
    # future turns (we never mark_expanded here). If the model needs more, it
    # issues another scoped query; it cannot dump the whole payload back in.
    res = search_original(original, query)
    if res["kept_items"] > 0:
        note = (
            f"[CCR query '{query}': route={res.get('route', 'unknown')} "
            f"steps={','.join(res.get('steps') or []) or 'none'}; "
            f"{res['kept_items']} of {res['total_items']} items. If insufficient, "
            f"issue another {tool_name} with a more specific ID, symbol, label, "
            f"schema word, or relationship term.]"
        )
        return {
            "content": [{"type": "text", "text": f"{note}\n{res['text']}"}],
            "details": {
                "handle": handle,
                "query": query,
                "kept_items": res["kept_items"],
                "matched_items": res.get("matched_items"),
                "total_items": res["total_items"],
                "effective_query": res.get("effective_query"),
                "fallback_used": res.get("fallback_used"),
                "route": res.get("route"),
                "steps": res.get("steps"),
                "chars": len(res["text"]),
            },
        }
    # No matches → tell the model so it can refine. Never dump the full payload.
    return {
        "content": [{"type": "text", "text": (
            f"[CCR query '{query}': no matching items in {handle} "
            f"({res['total_items']} items total). Try a different/broader query.]"
        )}],
        "details": {
            "handle": handle,
            "query": query,
            "kept_items": 0,
            "matched_items": res.get("matched_items"),
            "effective_query": res.get("effective_query"),
            "fallback_used": res.get("fallback_used"),
            "route": res.get("route"),
            "steps": res.get("steps"),
        },
    }


def extension_factory(pi: Any) -> None:
    # ---- model-driven: the explicit tool ---------------------------------- #
    async def execute(tool_call_id, params, signal, on_update, ctx):
        params = params or {}
        handle = params.get("handle", "")
        query = (params.get("query") or "").strip()
        return _retrieve_tool_response(handle, query, tool_name="ccr_retrieve")

    pi.register_tool(
        name="ccr_retrieve",
        label="Retrieve from compressed payload",
        description=(
            "Retrieve the specific parts of a compressed payload needed for the "
            "current task. Use semantically rich task terms for `query`: exact IDs, "
            "node names, file paths, symbols, entity names, timestamps, or other "
            "distinctive values from the task. If the needed instruction or value is "
            "hidden inside the compressed payload, search for distinctive labels or "
            "schema words such as target, instruction, operation, question, key, or "
            "id instead of broad generic terms. Call again with another "
            "specific query when you need more."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "12-char CCR handle"},
                "query": {
                    "type": "string",
                    "description": (
                        "Required. Semantically rich search term for the next needed "
                        "piece of information: prefer exact IDs, node names, file "
                        "paths, symbols, entity names, timestamps, or other distinctive "
                        "task values. For hidden payload instructions, use labels or "
                        "schema words such as target, instruction, operation, question, "
                        "key, or id."
                    ),
                },
            },
            "required": ["handle", "query"],
        },
        execute=execute,
    )

    async def compression_command(args: str, ctx: Any = None) -> str:
        parts = (args or "").strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "stats"

        try:
            from pi_ai import (
                reset_cache_alignment_stats,
                reset_compression_cache,
                reset_compression_learning_stats,
                reset_compression_stats,
            )
        except Exception:
            return "Active compression stats are unavailable because pi_ai could not be loaded."

        if sub == "reset":
            reset_compression_stats()
            reset_compression_learning_stats()
            reset_cache_alignment_stats()
            reset_compression_cache()
            return "Active compression stats reset."

        if sub not in ("stats", "status"):
            return "Usage: /compression [stats|status|reset]"

        return _stats_text()

    pi.register_command(
        "compression",
        {
            "description": "Inspect/control active compression (stats | status | reset)",
            "handler": compression_command,
        },
    )

    # NOTE: there is intentionally NO harness-driven full-expansion fallback.
    # An earlier version rehydrated the entire original in place whenever the model
    # merely *mentioned* a [CCR:<handle>] in its text. That was a back-door to the
    # full-payload dump — it silently reinflated the context (the exact thing CCR
    # exists to prevent) without the model ever issuing a scoped query. Retrieval is
    # now a single, consistent path: the query-scoped `ccr_retrieve` tool, whose
    # `query` parameter is structurally required (the harness bounces query-less
    # calls back to the model). The model must say what it's looking for; it can
    # never expand everything.


activate = extension_factory
default = extension_factory
