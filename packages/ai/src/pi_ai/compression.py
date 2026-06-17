"""Universal active-compression hook for pi_ai.

Same chokepoint shape as the PII filter (pii.py): a registered, default-no-op
compressor transforms the outbound `Context` before dispatch, so large payloads
are compressed for ALL calls regardless of source. tau-by-clarity registers the real
content-aware compressor (which also caches the original in a CCR store so it can
be retrieved out-of-band).

Unlike PII, compression is ONE-WAY at this layer: there is no response transform.
Recovery of an original is out-of-band via the CCR store's retrieve path — never
by un-transforming the stream here.

pi_ai stays dependency-free: if nothing is registered this is an exact no-op.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# compress(text) -> text. The compressor itself decides what to touch (it size-
# gates and content-type-routes internally) and owns its CCR cache.
CompressFn = Callable[[str], str]

_compressor: Optional[CompressFn] = None


def register_compressor(fn: CompressFn) -> None:
    """Install the universal outbound compressor. Called by tau-by-clarity on import."""
    global _compressor
    _compressor = fn


def unregister_compressor() -> None:
    global _compressor
    _compressor = None


def has_compressor() -> bool:
    return _compressor is not None


def _xform_block(block: Any, fn: CompressFn) -> Any:
    if isinstance(getattr(block, "text", None), str):
        return block.model_copy(update={"text": fn(block.text)})
    return block


def _xform_message(msg: Any, fn: CompressFn) -> Any:
    # Only compress tool-result payloads — never the live user prompt or the
    # assistant's own messages. Tool outputs are the read-side bloat Headroom-style
    # compression targets; the current instruction must reach the model verbatim.
    if getattr(msg, "role", None) != "toolResult":
        return msg
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return msg.model_copy(update={"content": fn(content)})
    if isinstance(content, list):
        return msg.model_copy(update={"content": [_xform_block(b, fn) for b in content]})
    return msg


def compress_context(context: Any) -> Any:
    """Compress large content in the outbound context. No-op if no compressor is
    registered. Never mutates the caller's context."""
    fn = _compressor
    if fn is None:
        import os, sys
        if os.environ.get("LB_AC_DEBUG"):
            print("[AC_DEBUG] compress_context: NO compressor registered", file=sys.stderr)
        return context
    import os as _os, sys as _sys
    if _os.environ.get("LB_AC_DEBUG"):
        roles = [getattr(m, "role", "?") for m in context.messages]
        sizes = []
        for m in context.messages:
            c = getattr(m, "content", None)
            if isinstance(c, list):
                sizes.append(sum(len(getattr(b, "text", "") or "") for b in c))
            elif isinstance(c, str):
                sizes.append(len(c))
            else:
                sizes.append(0)
        print(f"[AC_DEBUG] compress_context in: roles={roles} sizes={sizes}", file=_sys.stderr)
    new = context.model_copy(
        update={"messages": [_xform_message(m, fn) for m in context.messages]}
    )
    if _os.environ.get("LB_AC_DEBUG"):
        sizes = []
        for m in new.messages:
            c = getattr(m, "content", None)
            if isinstance(c, list):
                sizes.append(sum(len(getattr(b, "text", "") or "") for b in c))
            elif isinstance(c, str):
                sizes.append(len(c))
            else:
                sizes.append(0)
        print(f"[AC_DEBUG] compress_context out: sizes={sizes}", file=_sys.stderr)
    return new
