"""ANSI SGR to HTML conversion for exported sessions."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

ANSI_COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#c0c0c0",
    "#808080",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
]

_ANSI_RE = re.compile(r"\x1b\[([\d;]*)m")


@dataclass
class TextStyle:
    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False


def color_256_to_hex(index: int) -> str:
    if index < 16:
        return ANSI_COLORS[index]
    if index < 232:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def component(n: int) -> int:
            return 0 if n == 0 else 55 + n * 40

        return f"#{component(r):02x}{component(g):02x}{component(b):02x}"
    gray = 8 + (index - 232) * 10
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def _style_css(style: TextStyle) -> str:
    parts: list[str] = []
    if style.fg:
        parts.append(f"color:{style.fg}")
    if style.bg:
        parts.append(f"background-color:{style.bg}")
    if style.bold:
        parts.append("font-weight:bold")
    if style.dim:
        parts.append("opacity:0.6")
    if style.italic:
        parts.append("font-style:italic")
    if style.underline:
        parts.append("text-decoration:underline")
    return ";".join(parts)


def _has_style(style: TextStyle) -> bool:
    return bool(style.fg or style.bg or style.bold or style.dim or style.italic or style.underline)


def _apply_sgr(params: list[int], style: TextStyle) -> None:
    i = 0
    while i < len(params):
        code = params[i]
        if code == 0:
            style.fg = style.bg = None
            style.bold = style.dim = style.italic = style.underline = False
        elif code == 1:
            style.bold = True
        elif code == 2:
            style.dim = True
        elif code == 3:
            style.italic = True
        elif code == 4:
            style.underline = True
        elif code == 22:
            style.bold = style.dim = False
        elif code == 23:
            style.italic = False
        elif code == 24:
            style.underline = False
        elif 30 <= code <= 37:
            style.fg = ANSI_COLORS[code - 30]
        elif code == 38:
            if i + 2 < len(params) and params[i + 1] == 5:
                style.fg = color_256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                style.fg = f"rgb({params[i + 2]},{params[i + 3]},{params[i + 4]})"
                i += 4
        elif code == 39:
            style.fg = None
        elif 40 <= code <= 47:
            style.bg = ANSI_COLORS[code - 40]
        elif code == 48:
            if i + 2 < len(params) and params[i + 1] == 5:
                style.bg = color_256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                style.bg = f"rgb({params[i + 2]},{params[i + 3]},{params[i + 4]})"
                i += 4
        elif code == 49:
            style.bg = None
        elif 90 <= code <= 97:
            style.fg = ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:
            style.bg = ANSI_COLORS[code - 100 + 8]
        i += 1


def ansi_to_html(text: str) -> str:
    style = TextStyle()
    result = []
    last_index = 0
    in_span = False
    for match in _ANSI_RE.finditer(text):
        before = text[last_index : match.start()]
        if before:
            result.append(html.escape(before, quote=True).replace("&#x27;", "&#039;"))
        if in_span:
            result.append("</span>")
            in_span = False
        params = [int(part) if part else 0 for part in match.group(1).split(";")] if match.group(1) else [0]
        _apply_sgr(params, style)
        if _has_style(style):
            result.append(f'<span style="{_style_css(style)}">')
            in_span = True
        last_index = match.end()
    remaining = text[last_index:]
    if remaining:
        result.append(html.escape(remaining, quote=True).replace("&#x27;", "&#039;"))
    if in_span:
        result.append("</span>")
    return "".join(result)


def ansi_lines_to_html(lines: list[str]) -> str:
    return "".join(f'<div class="ansi-line">{ansi_to_html(line) or "&nbsp;"}</div>' for line in lines)


__all__ = ["ANSI_COLORS", "TextStyle", "ansi_lines_to_html", "ansi_to_html", "color_256_to_hex"]
