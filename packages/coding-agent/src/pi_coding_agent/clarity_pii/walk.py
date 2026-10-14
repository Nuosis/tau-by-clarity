"""Message / payload walkers — yield (get, set) accessors for every editable text
string inside messages, tool-call arguments, and provider payloads, so the
tokenizer/detokenizer can be applied in place. Ported unchanged from the
prototype.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Iterable


PROVIDER_PAYLOAD_PROTOCOL_KEYS = frozenset({
    "api",
    "call_id",
    "encrypted_content",
    "finish_reason",
    "id",
    "include",
    "model",
    "name",
    "object",
    "previous_response_id",
    "role",
    "status",
    "store",
    "stream",
    "tool_choice",
    "tool_call_id",
    "toolCallId",
    "tool_use_id",
    "toolUseId",
    "type",
})


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


def dict_string_slots(
    d: dict[str, Any],
    *,
    skip_keys: set[str] | frozenset[str] | None = None,
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for key, val in list(d.items()):
        if skip_keys and key in skip_keys:
            continue
        if isinstance(val, str):
            yield (lambda k=key: d[k], lambda v, k=key: d.__setitem__(k, v))
        elif isinstance(val, dict):
            yield from dict_string_slots(val, skip_keys=skip_keys)
        elif isinstance(val, list):
            yield from list_string_slots(val, skip_keys=skip_keys)


def list_string_slots(
    lst: list[Any],
    *,
    skip_keys: set[str] | frozenset[str] | None = None,
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for i, val in enumerate(lst):
        if isinstance(val, str):
            yield (lambda i=i: lst[i], lambda v, i=i: lst.__setitem__(i, v))
        elif isinstance(val, dict):
            yield from dict_string_slots(val, skip_keys=skip_keys)
        elif isinstance(val, list):
            yield from list_string_slots(val, skip_keys=skip_keys)


_BASE64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")


def _is_base64(value: Any) -> bool:
    return isinstance(value, str) and _BASE64_RE.fullmatch(value) is not None


def _is_base64_data_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("data:"):
        return False
    header, sep, data = value.partition(",")
    return bool(sep) and header.endswith(";base64") and _is_base64(data)


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _collect_media_blocks(blocks: Any, out: set[tuple[int, str]]) -> None:
    for block in _dicts(blocks):
        kind = block.get("type")
        if kind == "input_image" and _is_base64_data_url(block.get("image_url")):
            out.add((id(block), "image_url"))
        elif kind == "input_file" and _is_base64_data_url(block.get("file_data")):
            out.add((id(block), "file_data"))
        elif kind == "image_url" and isinstance(block.get("image_url"), dict):
            if _is_base64_data_url(block["image_url"].get("url")):
                out.add((id(block["image_url"]), "url"))
        elif kind == "input_audio" and isinstance(block.get("input_audio"), dict):
            if _is_base64(block["input_audio"].get("data")):
                out.add((id(block["input_audio"]), "data"))
        elif kind in ("image", "document") and isinstance(block.get("source"), dict):
            source = block["source"]
            if source.get("type") == "base64" and _is_base64(source.get("data")):
                out.add((id(source), "data"))
        elif kind == "tool_result":
            _collect_media_blocks(block.get("content"), out)


def _media_fields(payload: dict[str, Any]) -> set[tuple[int, str]]:
    """(container id, key) of encoded media in provider message content.

    Only message content is media-bearing: Responses ``input`` items, Chat
    Completions/Anthropic ``messages`` and Gemini ``contents[].parts`` (also
    under a ``request`` wrapper). Schemas, tool inputs and other application
    objects stay on the plain walker.
    """
    out: set[tuple[int, str]] = set()
    roots = [payload]
    if isinstance(payload.get("request"), dict):
        roots.append(payload["request"])
    for root in roots:
        for message in _dicts(root.get("input")) + _dicts(root.get("messages")):
            _collect_media_blocks(message.get("content"), out)
            _collect_media_blocks(message.get("output"), out)
        for content in _dicts(root.get("contents")):
            for part in _dicts(content.get("parts")):
                for key in ("inlineData", "inline_data"):
                    inline = part.get(key)
                    if isinstance(inline, dict) and _is_base64(inline.get("data")):
                        out.add((id(inline), "data"))
    return out


def provider_payload_string_slots(d: dict[str, Any]) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    """Yield editable provider payload text while leaving protocol fields intact.

    Encoded media in provider message content is binary, not text: a detector
    match inside base64 corrupts the image, so those fields are skipped.
    """
    yield from _payload_dict_slots(d, _media_fields(d))


def _payload_dict_slots(
    d: dict[str, Any],
    media: set[tuple[int, str]],
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for key, val in list(d.items()):
        if key in PROVIDER_PAYLOAD_PROTOCOL_KEYS or (id(d), key) in media:
            continue
        if isinstance(val, str):
            yield (lambda k=key: d[k], lambda v, k=key: d.__setitem__(k, v))
        elif isinstance(val, dict):
            yield from _payload_dict_slots(val, media)
        elif isinstance(val, list):
            yield from _payload_list_slots(val, media)


def _payload_list_slots(
    lst: list[Any],
    media: set[tuple[int, str]],
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for i, val in enumerate(lst):
        if isinstance(val, str):
            yield (lambda i=i: lst[i], lambda v, i=i: lst.__setitem__(i, v))
        elif isinstance(val, dict):
            yield from _payload_dict_slots(val, media)
        elif isinstance(val, list):
            yield from _payload_list_slots(val, media)


def provider_payload_protocol_slots(d: dict[str, Any]) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    """Yield protected provider protocol fields that must round-trip exactly."""
    for key, val in list(d.items()):
        if key in PROVIDER_PAYLOAD_PROTOCOL_KEYS and isinstance(val, str):
            yield (lambda k=key: d[k], lambda v, k=key: d.__setitem__(k, v))
            continue
        if isinstance(val, dict):
            yield from provider_payload_protocol_slots(val)
        elif isinstance(val, list):
            yield from _provider_payload_protocol_list_slots(val)


def _provider_payload_protocol_list_slots(
    lst: list[Any],
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for val in lst:
        if isinstance(val, dict):
            yield from provider_payload_protocol_slots(val)
        elif isinstance(val, list):
            yield from _provider_payload_protocol_list_slots(val)


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
