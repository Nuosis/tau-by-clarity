"""CCR retrieve trigger — the §12 reversibility path made usable.

Two mechanisms, both bundled on-by-default with active compression:

1. **`ccr_retrieve` tool** (model-driven) — the model can explicitly fetch a
   compressed original by its `[CCR:<handle>]` handle.
2. **Harness-driven rehydration** (the §12 de-risk for the "model won't call the
   tool" finding) — a `context` hook that watches for a handle the model
   *referenced in its own output* and **expands the compressed block in place**,
   without depending on a formal tool call. The model only has to mention the
   handle (cheap, reliable); the harness does the retrieval (deterministic).

Loaded standalone by the extension loader → absolute imports.
"""

from __future__ import annotations

import re
from typing import Any

from pi_coding_agent.active_compression import retrieve
from pi_coding_agent.clarity_pii.walk import apply_to_message

_HANDLE_RE = re.compile(r"ccr_retrieve\s+([0-9a-f]{12})|\[CCR:([0-9a-f]{12})\]")


def _role(m: Any) -> Any:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _text_of(m: Any) -> str:
    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            t = b.get("text") if isinstance(b, dict) else getattr(b, "text", None)
            if isinstance(t, str):
                parts.append(t)
        return " ".join(parts)
    return ""


def _handles_referenced_by_assistant(messages: list) -> list[str]:
    out: list[str] = []
    for m in messages:
        if _role(m) != "assistant":
            continue
        for mt in _HANDLE_RE.finditer(_text_of(m)):
            h = mt.group(1) or mt.group(2)
            if h and h not in out:
                out.append(h)
    return out


def extension_factory(pi: Any) -> None:
    state: dict[str, set] = {"expanded": set()}

    # ---- model-driven: the explicit tool ---------------------------------- #
    async def execute(tool_call_id, params, signal, on_update, ctx):
        handle = (params or {}).get("handle", "")
        original = retrieve(handle)
        if original is None:
            return {
                "content": [{"type": "text", "text": f"No CCR entry for handle {handle!r}."}],
                "isError": True,
            }
        state["expanded"].add(handle)
        return {
            "content": [{"type": "text", "text": original}],
            "details": {"handle": handle, "chars": len(original)},
        }

    pi.register_tool(
        name="ccr_retrieve",
        label="Retrieve compressed original",
        description="Fetch the full original of a CCR-compressed payload by its [CCR:<handle>] handle.",
        parameters={
            "type": "object",
            "properties": {"handle": {"type": "string", "description": "12-char CCR handle"}},
            "required": ["handle"],
        },
        execute=execute,
    )

    # ---- harness-driven: rehydrate on cue-hit ----------------------------- #
    async def on_context(event: dict[str, Any], ctx: Any) -> dict[str, Any] | None:
        messages = event.get("messages") or []
        wanted = [h for h in _handles_referenced_by_assistant(messages) if h not in state["expanded"]]
        if not wanted:
            return None
        changed = False
        for h in wanted:
            state["expanded"].add(h)
            original = retrieve(h)
            if original is None:
                continue
            # Replace the compressed block (text starting with [CCR:<h>]) with the
            # full original, in place, across the transcript.
            def _expand(text: str, h=h, original=original) -> str:
                return original if text.startswith(f"[CCR:{h}]") else text

            for m in messages:
                apply_to_message(m, _expand)
            changed = True
        return {"messages": messages} if changed else None

    pi.on("context", on_context)


activate = extension_factory
default = extension_factory
