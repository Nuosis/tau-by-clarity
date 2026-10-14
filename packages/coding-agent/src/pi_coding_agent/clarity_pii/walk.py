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
# Tool definitions carry schemas, defaults and examples, never media; they keep
# the plain walker so a media-shaped example is still tokenized.
_TOOL_DEFINITION_KEYS = frozenset({"tools", "functions"})


def _is_base64(value: Any) -> bool:
    return isinstance(value, str) and _BASE64_RE.fullmatch(value) is not None


def _is_base64_data_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("data:"):
        return False
    header, sep, data = value.partition(",")
    return bool(sep) and header.endswith(";base64") and _is_base64(data)


def _base64_media_keys(d: dict[str, Any], parent_key: str | None) -> frozenset[str]:
    """Keys of ``d`` that hold encoded media in a known provider media field.

    OpenAI Responses ``input_image``/``input_file``, Chat Completions
    ``image_url.url`` and ``input_audio.data``, Anthropic base64 ``source``
    and Gemini ``inlineData``.
    """
    kind = d.get("type")
    if kind == "input_image" and _is_base64_data_url(d.get("image_url")):
        return frozenset({"image_url"})
    if kind == "input_file" and _is_base64_data_url(d.get("file_data")):
        return frozenset({"file_data"})
    if kind == "base64" and "media_type" in d and _is_base64(d.get("data")):
        return frozenset({"data"})
    if parent_key == "image_url" and _is_base64_data_url(d.get("url")):
        return frozenset({"url"})
    if parent_key in ("input_audio", "inlineData", "inline_data") and _is_base64(d.get("data")):
        return frozenset({"data"})
    return frozenset()


def provider_payload_string_slots(d: dict[str, Any]) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    """Yield editable provider payload text while leaving protocol fields intact.

    Encoded media in provider content is binary, not text: a detector match
    inside base64 corrupts the image, so those fields are skipped.
    """
    for key, val in list(d.items()):
        if key in _TOOL_DEFINITION_KEYS and isinstance(val, (dict, list)):
            container = {key: val}
            yield from dict_string_slots(container, skip_keys=PROVIDER_PAYLOAD_PROTOCOL_KEYS)
        else:
            yield from _content_string_slots(d, key, val, parent_key=None)


def _content_string_slots(
    owner: dict[str, Any],
    key: str,
    val: Any,
    *,
    parent_key: str | None,
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    if key in PROVIDER_PAYLOAD_PROTOCOL_KEYS or key in _base64_media_keys(owner, parent_key):
        return
    if isinstance(val, str):
        yield (lambda: owner[key], lambda v: owner.__setitem__(key, v))
    elif isinstance(val, dict):
        for child_key, child in list(val.items()):
            yield from _content_string_slots(val, child_key, child, parent_key=key)
    elif isinstance(val, list):
        yield from _content_list_string_slots(val)


def _content_list_string_slots(
    lst: list[Any],
) -> Iterable[tuple[Callable[[], str], Callable[[str], None]]]:
    for i, val in enumerate(lst):
        if isinstance(val, str):
            yield (lambda i=i: lst[i], lambda v, i=i: lst.__setitem__(i, v))
        elif isinstance(val, dict):
            for child_key, child in list(val.items()):
                yield from _content_string_slots(val, child_key, child, parent_key=None)
        elif isinstance(val, list):
            yield from _content_list_string_slots(val)


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
