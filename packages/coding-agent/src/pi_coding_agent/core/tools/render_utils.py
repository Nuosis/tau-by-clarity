"""Tool render helpers mirrored from the Node harness."""
from __future__ import annotations

import os


def shorten_path(path: object) -> str:
    if not isinstance(path, str):
        return ""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def str_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    return text.replace("\r", "")


def get_text_output(result: dict | None, show_images: bool = False) -> str:
    if not result:
        return ""
    content = result.get("content") or []
    text_blocks = [item for item in content if item.get("type") == "text"]
    image_blocks = [item for item in content if item.get("type") == "image"]
    output = "\n".join(normalize_display_text(str(item.get("text") or "")) for item in text_blocks)
    if image_blocks and not show_images:
        indicators = "\n".join(f"[image: {item.get('mimeType') or item.get('mime_type') or 'image/unknown'}]" for item in image_blocks)
        output = f"{output}\n{indicators}" if output else indicators
    return output


def invalid_arg_text() -> str:
    return "[invalid arg]"


def render_tool_path(raw_path: str | None, cwd: str, *, empty_fallback: str | None = None) -> str:
    if raw_path is None:
        return invalid_arg_text()
    value = raw_path or empty_fallback
    if not value:
        return "..."
    absolute = value if os.path.isabs(value) else os.path.abspath(os.path.join(cwd, value))
    return shorten_path(absolute)


__all__ = [
    "get_text_output",
    "invalid_arg_text",
    "normalize_display_text",
    "render_tool_path",
    "replace_tabs",
    "shorten_path",
    "str_or_none",
]
