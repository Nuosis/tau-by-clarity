"""Message / payload walkers — yield (get, set) accessors for every editable text
string inside messages, tool-call arguments, and provider payloads, so the
tokenizer/detokenizer can be applied in place. Ported unchanged from the
prototype.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable


def content_text_slots(content: Any) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, dict):
            t = item.get("type")
            if t == "text" and isinstance(item.get("text"), str):
                yield (lambda i=item: i["text"], lambda v, i=item: i.__setitem__("text", v))
            elif t == "thinking" and isinstance(item.get("thinking"), str):
                yield (lambda i=item: i["thinking"], lambda v, i=item: i.__setitem__("thinking", v))
            elif t == "toolCall" and isinstance(item.get("arguments"), dict):
                yield from dict_string_slots(item["arguments"])
        else:
            t = getattr(item, "type", None)
            if t == "text" and isinstance(getattr(item, "text", None), str):
                yield (lambda i=item: i.text, lambda v, i=item: setattr(i, "text", v))
            elif t == "thinking" and isinstance(getattr(item, "thinking", None), str):
                yield (lambda i=item: i.thinking, lambda v, i=item: setattr(i, "thinking", v))
            elif t == "toolCall" and isinstance(getattr(item, "arguments", None), dict):
                yield from dict_string_slots(item.arguments)


def dict_string_slots(d: dict[str, Any]) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for key, val in list(d.items()):
        if isinstance(val, str):
            yield (lambda k=key: d[k], lambda v, k=key: d.__setitem__(k, v))
        elif isinstance(val, dict):
            yield from dict_string_slots(val)
        elif isinstance(val, list):
            yield from list_string_slots(val)


def list_string_slots(lst: list[Any]) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for i, val in enumerate(lst):
        if isinstance(val, str):
            yield (lambda i=i: lst[i], lambda v, i=i: lst.__setitem__(i, v))
        elif isinstance(val, dict):
            yield from dict_string_slots(val)
        elif isinstance(val, list):
            yield from list_string_slots(val)


def apply_to_message(msg: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every text string in ``msg`` in place. Returns msg."""
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = fn(content)
            return msg
        get_content = lambda: msg.get("content")  # noqa: E731
    else:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            try:
                setattr(msg, "content", fn(content))
            except Exception:
                pass
            return msg
        get_content = lambda: getattr(msg, "content", None)  # noqa: E731

    for get, set_ in content_text_slots(get_content()):
        try:
            set_(fn(get()))
        except Exception:
            continue
    return msg


def deep_copy_message(msg: Any) -> Any:
    model_copy = getattr(msg, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(deep=True)
        except Exception:
            pass
    try:
        return copy.deepcopy(msg)
    except Exception:
        return msg
