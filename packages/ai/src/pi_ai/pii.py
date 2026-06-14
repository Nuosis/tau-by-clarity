"""Universal PII filter hook for pi_ai.

This is the single chokepoint every LLM call funnels through (`stream_simple` /
`stream`, and therefore `complete_simple` / `complete`). A registered filter
tokenizes the outbound `Context` before it reaches the provider and detokenizes
the response — so PII is captured for ALL calls regardless of source (agent
sessions, the outer loop, evals, any direct pi_ai use).

pi_ai stays dependency-free: the filter is *registered* (clarity_pii does this).
If nothing is registered, every function here is an exact no-op.

The registered factory returns a fresh `(tokenize, detokenize)` pair PER CALL, so
the per-call token namespace never collides with a session-level vault that may
have already tokenized the same context upstream (that case is a clean no-op).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

TokenizeFn = Callable[[str], str]
# Returns (tokenize, detokenize) for ONE call.
PiiFilterFactory = Callable[[], Tuple[TokenizeFn, TokenizeFn]]

_factory: Optional[PiiFilterFactory] = None


def register_pii_filter(factory: PiiFilterFactory) -> None:
    """Install the universal PII filter factory. Called by clarity_pii on import."""
    global _factory
    _factory = factory


def unregister_pii_filter() -> None:
    global _factory
    _factory = None


def has_pii_filter() -> bool:
    return _factory is not None


# --- text walkers over pi_ai's own types (no external deps) ----------------- #


def _xform_dict(d: dict, fn: TokenizeFn) -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = _xform_value(v, fn)
    return out


def _xform_list(lst: list, fn: TokenizeFn) -> list:
    return [_xform_value(v, fn) for v in lst]


def _xform_value(v: Any, fn: TokenizeFn) -> Any:
    if isinstance(v, str):
        return fn(v)
    if isinstance(v, dict):
        return _xform_dict(v, fn)
    if isinstance(v, list):
        return _xform_list(v, fn)
    return v


def _xform_block(block: Any, fn: TokenizeFn) -> Any:
    """A content block (TextContent / ThinkingContent / ToolCall / ImageContent)."""
    if isinstance(getattr(block, "text", None), str):
        return block.model_copy(update={"text": fn(block.text)})
    if isinstance(getattr(block, "thinking", None), str):
        return block.model_copy(update={"thinking": fn(block.thinking)})
    if isinstance(getattr(block, "arguments", None), dict):
        return block.model_copy(update={"arguments": _xform_dict(block.arguments, fn)})
    return block


def _xform_message(msg: Any, fn: TokenizeFn) -> Any:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return msg.model_copy(update={"content": fn(content)})
    if isinstance(content, list):
        return msg.model_copy(update={"content": [_xform_block(b, fn) for b in content]})
    return msg


def protect_context(context: Any) -> Tuple[Any, Optional[TokenizeFn]]:
    """Tokenize the outbound context. Returns (possibly-new context, detokenizer).

    No-op (returns the same context and None) when no filter is registered. Never
    mutates the caller's context — the canonical transcript stays cleartext.
    """
    factory = _factory
    if factory is None:
        return context, None
    tokenize, detokenize = factory()
    updates: dict[str, Any] = {
        "messages": [_xform_message(m, tokenize) for m in context.messages]
    }
    if isinstance(getattr(context, "system_prompt", None), str) and context.system_prompt:
        updates["system_prompt"] = tokenize(context.system_prompt)
    return context.model_copy(update=updates), detokenize


def detok_event(event: Any, detokenize: TokenizeFn) -> Any:
    """Restore PII in a streamed response event (text/thinking deltas, the final
    message). Tool-call arg streaming is left to the final message to avoid
    corrupting partial JSON."""
    etype = getattr(event, "type", None)
    if etype in ("text_delta", "thinking_delta") and isinstance(getattr(event, "delta", None), str):
        return event.model_copy(update={"delta": detokenize(event.delta)})
    if etype in ("text_end", "thinking_end") and isinstance(getattr(event, "content", None), str):
        return event.model_copy(update={"content": detokenize(event.content)})
    if etype == "done" and getattr(event, "message", None) is not None:
        return event.model_copy(update={"message": _xform_message(event.message, detokenize)})
    return event
