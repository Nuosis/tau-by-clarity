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

from typing import Any

from pi_coding_agent.active_compression import retrieve
from pi_coding_agent.active_compression.search import search_original


def extension_factory(pi: Any) -> None:
    # ---- model-driven: the explicit tool ---------------------------------- #
    async def execute(tool_call_id, params, signal, on_update, ctx):
        params = params or {}
        handle = params.get("handle", "")
        query = (params.get("query") or "").strip()
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
                f"[CCR query '{query}': {res['kept_items']} of {res['total_items']} "
                f"items. Issue another ccr_retrieve with a different query to fetch "
                f"other items.]"
            )
            return {
                "content": [{"type": "text", "text": f"{note}\n{res['text']}"}],
                "details": {
                    "handle": handle,
                    "query": query,
                    "kept_items": res["kept_items"],
                    "total_items": res["total_items"],
                    "chars": len(res["text"]),
                },
            }
        # No matches → tell the model so it can refine. Never dump the full payload.
        return {
            "content": [{"type": "text", "text": (
                f"[CCR query '{query}': no matching items in {handle} "
                f"({res['total_items']} items total). Try a different/broader query.]"
            )}],
            "details": {"handle": handle, "query": query, "kept_items": 0},
        }

    pi.register_tool(
        name="ccr_retrieve",
        label="Retrieve from compressed payload",
        description=(
            "Search a CCR-compressed payload by its [CCR:<handle>] handle. `query` "
            "is required: pass the keywords/identifier you need (e.g. the node, "
            "file, or term you're looking for) to get back only the relevant items "
            "— a BM25 search within the cached data. Issue multiple scoped queries "
            "to gather what you need; there is no full-payload dump."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "12-char CCR handle"},
                "query": {
                    "type": "string",
                    "description": (
                        "Required. Keywords/identifier to search within the cached "
                        "payload; returns only matching items."
                    ),
                },
            },
            "required": ["handle", "query"],
        },
        execute=execute,
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
