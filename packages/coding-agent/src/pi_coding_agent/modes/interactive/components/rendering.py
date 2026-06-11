"""
Rendering helpers and lightweight components for interactive mode.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

from pi_tui.components.image import Image, ImageOptions, ImageTheme


def format_key_text(key: str, capitalize: bool = False, platform: str | None = None) -> str:
    platform_name = platform or os.sys.platform

    def format_part(part: str) -> str:
        display = "option" if platform_name == "darwin" and part.lower() == "alt" else part
        return display[:1].upper() + display[1:] if capitalize and display else display

    return "/".join("+".join(format_part(part) for part in combo.split("+")) for combo in key.split("/"))


DEFAULT_KEYBINDINGS = {
    "tui.select.cancel": ["escape"],
    "tui.select.confirm": ["enter"],
    "tui.select.up": ["up"],
    "tui.select.down": ["down"],
}


def key_text(keybinding: str, keybindings: dict[str, list[str]] | None = None) -> str:
    bindings = keybindings or DEFAULT_KEYBINDINGS
    return format_key_text("/".join(bindings.get(keybinding, [keybinding])))


def key_display_text(keybinding: str, keybindings: dict[str, list[str]] | None = None) -> str:
    bindings = keybindings or DEFAULT_KEYBINDINGS
    return format_key_text("/".join(bindings.get(keybinding, [keybinding])), capitalize=True)


def key_hint(keybinding: str, description: str, keybindings: dict[str, list[str]] | None = None) -> str:
    return f"{key_text(keybinding, keybindings)} {description}"


def raw_key_hint(key: str, description: str) -> str:
    return f"{format_key_text(key)} {description}"


@dataclass(frozen=True)
class VisualTruncateResult:
    visual_lines: list[str]
    skipped_count: int


def truncate_to_visual_lines(
    text: str,
    max_visual_lines: int,
    width: int,
    padding_x: int = 0,
) -> VisualTruncateResult:
    if not text:
        return VisualTruncateResult([], 0)
    render_width = max(1, width - padding_x * 2)
    visual: list[str] = []
    for logical in text.splitlines() or [""]:
        wrapped = textwrap.wrap(logical, render_width, replace_whitespace=False, drop_whitespace=False)
        visual.extend((" " * padding_x + line) for line in (wrapped or [""]))
    if len(visual) <= max_visual_lines:
        return VisualTruncateResult(visual, 0)
    return VisualTruncateResult(visual[-max_visual_lines:], len(visual) - max_visual_lines)


class DynamicBorder:
    def __init__(self, color: Callable[[str], str] | None = None) -> None:
        self.color = color or (lambda text: text)

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        return [self.color("─" * max(1, width))]


class BorderedLoader:
    def __init__(self, message: str, cancellable: bool = True) -> None:
        self.message = message
        self.cancellable = cancellable
        self.aborted = False
        self.on_abort: Callable[[], None] | None = None

    def handle_input(self, key_data: str) -> None:
        if self.cancellable and key_data in {"escape", "esc", "\x1b"}:
            self.aborted = True
            if self.on_abort:
                self.on_abort()

    def dispose(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        border = DynamicBorder().render(width)[0]
        lines = [border, self.message]
        if self.cancellable:
            lines.append(key_hint("tui.select.cancel", "cancel"))
        lines.append(border)
        return lines


_DIFF_LINE = re.compile(r"^([+\-\s])(\s*\d*)\s(.*)$")


def parse_diff_line(line: str) -> dict[str, str] | None:
    match = _DIFF_LINE.match(line)
    if not match:
        return None
    return {"prefix": match.group(1), "line_num": match.group(2), "content": match.group(3)}


def replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def render_intra_line_diff(old_content: str, new_content: str) -> tuple[str, str]:
    old = replace_tabs(old_content)
    new = replace_tabs(new_content)
    prefix_len = 0
    max_prefix = min(len(old), len(new))
    while prefix_len < max_prefix and old[prefix_len] == new[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    max_suffix = min(len(old) - prefix_len, len(new) - prefix_len)
    while suffix_len < max_suffix and old[len(old) - suffix_len - 1] == new[len(new) - suffix_len - 1]:
        suffix_len += 1

    old_mid_end = len(old) - suffix_len if suffix_len else len(old)
    new_mid_end = len(new) - suffix_len if suffix_len else len(new)
    old_mid = old[prefix_len:old_mid_end]
    new_mid = new[prefix_len:new_mid_end]
    suffix = old[old_mid_end:] if suffix_len else ""
    removed = old[:prefix_len] + (f"{{-{old_mid}-}}" if old_mid else "") + suffix
    added = new[:prefix_len] + (f"{{+{new_mid}+}}" if new_mid else "") + suffix
    return removed, added


def render_diff(diff_text: str, file_path: str | None = None) -> str:
    lines = diff_text.split("\n")
    result: list[str] = []
    index = 0
    while index < len(lines):
        parsed = parse_diff_line(lines[index])
        if not parsed:
            result.append(lines[index])
            index += 1
            continue
        if parsed["prefix"] == "-":
            removed: list[dict[str, str]] = []
            while index < len(lines):
                item = parse_diff_line(lines[index])
                if not item or item["prefix"] != "-":
                    break
                removed.append(item)
                index += 1
            added: list[dict[str, str]] = []
            while index < len(lines):
                item = parse_diff_line(lines[index])
                if not item or item["prefix"] != "+":
                    break
                added.append(item)
                index += 1
            if len(removed) == 1 and len(added) == 1:
                removed_line, added_line = render_intra_line_diff(removed[0]["content"], added[0]["content"])
                result.append(f"-{removed[0]['line_num']} {removed_line}")
                result.append(f"+{added[0]['line_num']} {added_line}")
            else:
                result.extend(f"-{item['line_num']} {replace_tabs(item['content'])}" for item in removed)
                result.extend(f"+{item['line_num']} {replace_tabs(item['content'])}" for item in added)
        elif parsed["prefix"] == "+":
            result.append(f"+{parsed['line_num']} {replace_tabs(parsed['content'])}")
            index += 1
        else:
            result.append(f" {parsed['line_num']} {replace_tabs(parsed['content'])}")
            index += 1
    return "\n".join(result)


def _image_fallback_color() -> Callable[[str], str]:
    """A muted theme color for image text-fallbacks; identity if unavailable."""
    try:
        from ..theme.theme import get_theme

        t = get_theme()
        avail = getattr(t, "_fg_ansi", {})
        name = "muted" if "muted" in avail else ("dim" if "dim" in avail else None)
        if name:
            return lambda s: t.fg(name, s)
    except Exception:
        pass
    return lambda s: s


def _image_blocks(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract image content blocks (with base64 data) from a tool result."""
    if not result:
        return []
    out: list[dict[str, Any]] = []
    for block in result.get("content", []) or []:
        if block.get("type") == "image" and (block.get("data") or block.get("base64")):
            out.append(block)
    return out


def get_text_output(result: dict[str, Any] | None, show_images: bool = True) -> str:
    if not result:
        return ""
    parts: list[str] = []
    for block in result.get("content", []) or []:
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block.get("type") == "image" and not show_images:
            parts.append(f"[image: {block.get('mimeType') or block.get('mime_type') or 'unknown'}]")
    return "\n".join(parts)


class ToolExecutionComponent:
    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        args: Any,
        show_images: bool = True,
        image_width_cells: int = 60,
    ) -> None:
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.args = args
        self.show_images = show_images
        self.image_width_cells = image_width_cells
        self.expanded = False
        self.execution_started = False
        self.args_complete = False
        self.is_partial = True
        self.result: dict[str, Any] | None = None
        # Inline-image components, lazily built from the result and cached so the
        # image is not re-transmitted to the terminal on every frame.
        self._image_components: list[Image] = []
        self._image_cache_key: tuple[int, bool, int] | None = None

    def update_args(self, args: Any) -> None:
        self.args = args

    def mark_execution_started(self) -> None:
        self.execution_started = True

    def set_args_complete(self) -> None:
        self.args_complete = True

    def update_result(self, result: dict[str, Any], is_partial: bool = False) -> None:
        self.result = result
        self.is_partial = is_partial

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)

    def set_show_images(self, show: bool) -> None:
        self.show_images = bool(show)

    def set_image_width_cells(self, width: int) -> None:
        self.image_width_cells = max(1, int(width))

    def get_text_output(self) -> str:
        return get_text_output(self.result, self.show_images)

    def format_tool_execution(self) -> str:
        text = self.tool_name
        try:
            args_text = json.dumps(self.args, indent=2, sort_keys=True)
        except TypeError:
            args_text = str(self.args)
        if args_text and args_text != "null":
            text += f"\n\n{args_text}"
        output = self.get_text_output()
        if output:
            text += f"\n{output}"
        return text

    def _sync_image_components(self) -> None:
        """(Re)build cached Image components when result / visibility changes."""
        key = (id(self.result), self.show_images, self.image_width_cells)
        if key == self._image_cache_key:
            return
        self._image_cache_key = key
        self._image_components = []
        if not self.show_images:
            return
        fallback_color = _image_fallback_color()
        for block in _image_blocks(self.result):
            data = block.get("data") or block.get("base64") or ""
            mime = block.get("mimeType") or block.get("mime_type") or "image/png"
            try:
                self._image_components.append(
                    Image(
                        data,
                        mime,
                        ImageTheme(fallback_color=fallback_color),
                        ImageOptions(max_width_cells=self.image_width_cells),
                    )
                )
            except Exception:
                # A malformed image must never break tool-result rendering.
                pass

    def render(self, width: int | None = None) -> list[str]:
        lines = self.format_tool_execution().splitlines()
        if self.result and self.result.get("isError"):
            lines.append("(error)")
        elif self.result and not self.is_partial:
            lines.append("(complete)")
        elif self.execution_started:
            lines.append("(running)")
        if width is not None:
            lines = [line[:width] for line in lines]
        # Append inline images AFTER truncation — their escape sequences must
        # stay intact, so they are never width-sliced like text lines.
        self._sync_image_components()
        if self._image_components:
            eff_width = width if width is not None else self.image_width_cells + 4
            for img in self._image_components:
                lines.append("")  # spacer between text/result and each image
                lines.extend(img.render(eff_width))
        return lines


__all__ = [
    "BorderedLoader",
    "DynamicBorder",
    "ToolExecutionComponent",
    "VisualTruncateResult",
    "format_key_text",
    "get_text_output",
    "key_display_text",
    "key_hint",
    "key_text",
    "parse_diff_line",
    "raw_key_hint",
    "render_diff",
    "render_intra_line_diff",
    "replace_tabs",
    "truncate_to_visual_lines",
]
