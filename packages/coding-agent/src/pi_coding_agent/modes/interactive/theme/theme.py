"""Interactive theme utilities mirroring the TypeScript theme module."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from pi_coding_agent.config import get_custom_themes_dir, get_themes_dir

ColorValue = str | int
ColorMode = str

BG_KEYS = {
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
}

EXT_TO_LANG = {
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "kt": "kotlin",
    "swift": "swift",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "php": "php",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "fish": "fish",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "md": "markdown",
    "markdown": "markdown",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmake": "cmake",
    "lua": "lua",
    "perl": "perl",
    "r": "r",
    "scala": "scala",
    "clj": "clojure",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "ml": "ocaml",
    "vim": "vim",
    "graphql": "graphql",
    "proto": "protobuf",
    "tf": "hcl",
    "hcl": "hcl",
}

BASE_COLORS: dict[str, ColorValue] = {
    "accent": "#00a6ff",
    "border": "#60606a",
    "borderAccent": "#00a6ff",
    "borderMuted": "#6f6f78",
    "success": "#45d483",
    "error": "#ff5c5c",
    "warning": "#e5b84b",
    "muted": "#8d8d96",
    "dim": "#6f6f78",
    "text": "#e5e5e7",
    "thinkingText": "#b8c4ff",
    "selectedBg": "#153247",
    "userMessageBg": "#142433",
    "userMessageText": "#e5e5e7",
    "customMessageBg": "#202027",
    "customMessageText": "#e5e5e7",
    "customMessageLabel": "#a7d8ff",
    "toolPendingBg": "#312b18",
    "toolSuccessBg": "#16351f",
    "toolErrorBg": "#3a1919",
    "toolTitle": "#a7d8ff",
    "toolOutput": "#d8d8dc",
    "mdHeading": "#ffffff",
    "mdLink": "#80c8ff",
    "mdLinkUrl": "#8d8d96",
    "mdCode": "#ffd37a",
    "mdCodeBlock": "#d8d8dc",
    "mdCodeBlockBorder": "#6f6f78",
    "mdQuote": "#b8b8bf",
    "mdQuoteBorder": "#6f6f78",
    "mdHr": "#6f6f78",
    "mdListBullet": "#00a6ff",
    "toolDiffAdded": "#45d483",
    "toolDiffRemoved": "#ff5c5c",
    "toolDiffContext": "#8d8d96",
    "syntaxComment": "#8d8d96",
    "syntaxKeyword": "#ff8bd1",
    "syntaxFunction": "#80c8ff",
    "syntaxVariable": "#ffd37a",
    "syntaxString": "#9ae68f",
    "syntaxNumber": "#c9a4ff",
    "syntaxType": "#7ad9ff",
    "syntaxOperator": "#e5e5e7",
    "syntaxPunctuation": "#b8b8bf",
    "thinkingOff": "#6f6f78",
    "thinkingMinimal": "#80c8ff",
    "thinkingLow": "#45d483",
    "thinkingMedium": "#e5b84b",
    "thinkingHigh": "#ff8bd1",
    "thinkingXhigh": "#ff5c5c",
    "bashMode": "#45d483",
}

BUILTIN_THEME_JSON: dict[str, dict[str, Any]] = {
    "dark": {"name": "dark", "colors": dict(BASE_COLORS)},
    "light": {
        "name": "light",
        "colors": {
            **BASE_COLORS,
            "text": "#111111",
            "mdHeading": "#111111",
            "mdCodeBlock": "#222222",
            "toolOutput": "#222222",
            "selectedBg": "#d9ecff",
            "userMessageBg": "#eef7ff",
            "customMessageBg": "#f2f2f5",
            "toolPendingBg": "#fff3cc",
            "toolSuccessBg": "#daf4df",
            "toolErrorBg": "#ffe0e0",
        },
    },
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _fg_ansi(color: ColorValue, mode: ColorMode) -> str:
    if color == "":
        return "\x1b[39m"
    if isinstance(color, int):
        return f"\x1b[38;5;{color}m"
    if color.startswith("#"):
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[38;2;{r};{g};{b}m" if mode == "truecolor" else f"\x1b[38;5;{_rgb_to_256(r, g, b)}m"
    raise ValueError(f"Invalid color value: {color}")


def _bg_ansi(color: ColorValue, mode: ColorMode) -> str:
    if color == "":
        return "\x1b[49m"
    if isinstance(color, int):
        return f"\x1b[48;5;{color}m"
    if color.startswith("#"):
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[48;2;{r};{g};{b}m" if mode == "truecolor" else f"\x1b[48;5;{_rgb_to_256(r, g, b)}m"
    raise ValueError(f"Invalid color value: {color}")


def _rgb_to_256(r: int, g: int, b: int) -> int:
    if r == g == b:
        if r < 8:
            return 16
        if r > 238:
            return 231
        return 232 + round((r - 8) / 10)
    return 16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)


def _style(code: str, reset: str, text: str) -> str:
    return f"{code}{text}{reset}"


def _resolve_var(value: ColorValue, vars_map: dict[str, ColorValue], seen: set[str] | None = None) -> ColorValue:
    if isinstance(value, int) or value == "" or str(value).startswith("#"):
        return value
    seen = seen or set()
    if value in seen:
        raise ValueError(f"Circular variable reference detected: {value}")
    if value not in vars_map:
        raise ValueError(f"Variable reference not found: {value}")
    seen.add(value)
    return _resolve_var(vars_map[value], vars_map, seen)


def _resolve_colors(colors: dict[str, ColorValue], vars_map: dict[str, ColorValue] | None = None) -> dict[str, ColorValue]:
    vars_map = vars_map or {}
    return {key: _resolve_var(value, vars_map) for key, value in colors.items()}


@dataclass
class Theme:
    fg_colors: dict[str, ColorValue]
    bg_colors: dict[str, ColorValue]
    mode: ColorMode = "truecolor"
    name: str | None = None
    source_path: str | None = None
    source_info: Any = None

    def __post_init__(self) -> None:
        self._fg_ansi = {key: _fg_ansi(value, self.mode) for key, value in self.fg_colors.items()}
        self._bg_ansi = {key: _bg_ansi(value, self.mode) for key, value in self.bg_colors.items()}

    def fg(self, color: str, text: str) -> str:
        if color not in self._fg_ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return f"{self._fg_ansi[color]}{text}\x1b[39m"

    def bg(self, color: str, text: str) -> str:
        if color not in self._bg_ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return f"{self._bg_ansi[color]}{text}\x1b[49m"

    def bold(self, text: str) -> str:
        return _style("\x1b[1m", "\x1b[22m", text)

    def italic(self, text: str) -> str:
        return _style("\x1b[3m", "\x1b[23m", text)

    def underline(self, text: str) -> str:
        return _style("\x1b[4m", "\x1b[24m", text)

    def inverse(self, text: str) -> str:
        return _style("\x1b[7m", "\x1b[27m", text)

    def strikethrough(self, text: str) -> str:
        return _style("\x1b[9m", "\x1b[29m", text)

    def get_fg_ansi(self, color: str) -> str:
        if color not in self._fg_ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return self._fg_ansi[color]

    def get_bg_ansi(self, color: str) -> str:
        if color not in self._bg_ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return self._bg_ansi[color]

    def get_color_mode(self) -> ColorMode:
        return self.mode

    def get_thinking_border_color(self, level: str) -> Callable[[str], str]:
        mapping = {
            "off": "thinkingOff",
            "minimal": "thinkingMinimal",
            "low": "thinkingLow",
            "medium": "thinkingMedium",
            "high": "thinkingHigh",
            "xhigh": "thinkingXhigh",
        }
        return lambda text: self.fg(mapping.get(level, "thinkingOff"), text)

    def get_bash_mode_border_color(self) -> Callable[[str], str]:
        return lambda text: self.fg("bashMode", text)


def _theme_from_json(theme_json: dict[str, Any], mode: ColorMode = "truecolor", source_path: str | None = None) -> Theme:
    colors = theme_json.get("colors") or {}
    if not isinstance(colors, dict) or "accent" not in colors:
        raise ValueError(f"Invalid theme \"{theme_json.get('name', source_path or '<theme>')}\"")
    resolved = _resolve_colors(colors, theme_json.get("vars") or {})
    fg_colors = {key: value for key, value in resolved.items() if key not in BG_KEYS}
    bg_colors = {key: value for key, value in resolved.items() if key in BG_KEYS}
    return Theme(fg_colors=fg_colors, bg_colors=bg_colors, mode=mode, name=theme_json.get("name"), source_path=source_path)


def load_theme_from_path(theme_path: str, mode: ColorMode = "truecolor") -> Theme:
    with open(theme_path, encoding="utf-8") as handle:
        return _theme_from_json(json.load(handle), mode=mode, source_path=theme_path)


def _load_theme_json(name: str) -> dict[str, Any]:
    if name in BUILTIN_THEME_JSON:
        return BUILTIN_THEME_JSON[name]
    custom_path = os.path.join(get_custom_themes_dir(), f"{name}.json")
    if os.path.exists(custom_path):
        with open(custom_path, encoding="utf-8") as handle:
            return json.load(handle)
    bundled_path = os.path.join(get_themes_dir(), f"{name}.json")
    if os.path.exists(bundled_path):
        with open(bundled_path, encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Theme not found: {name}")


_registered_themes: dict[str, Theme] = {}
_current_theme: Theme | None = None
_current_theme_name: str | None = None


def set_registered_themes(themes: list[Theme]) -> None:
    _registered_themes.clear()
    for item in themes:
        name = getattr(item, "name", None)
        if not name:
            continue
        if isinstance(item, Theme):
            _registered_themes[name] = item
            continue
        source_path = getattr(item, "source_path", None) or getattr(item, "path", None)
        if source_path:
            try:
                loaded = load_theme_from_path(source_path)
                if loaded.name:
                    _registered_themes[loaded.name] = loaded
                else:
                    _registered_themes[name] = loaded
                continue
            except Exception:
                pass
        colors = getattr(item, "colors", None)
        if isinstance(colors, dict):
            try:
                theme_json = dict(colors)
                theme_json.setdefault("name", name)
                _registered_themes[name] = _theme_from_json(theme_json, source_path=source_path)
            except Exception:
                continue


def get_available_themes_with_paths() -> list[dict[str, str | None]]:
    items: dict[str, str | None] = {
        "dark": os.path.join(get_themes_dir(), "dark.json"),
        "light": os.path.join(get_themes_dir(), "light.json"),
    }
    custom_dir = get_custom_themes_dir()
    if os.path.isdir(custom_dir):
        for file_name in os.listdir(custom_dir):
            if file_name.endswith(".json"):
                theme_path = os.path.join(custom_dir, file_name)
                try:
                    custom_theme = load_theme_from_path(theme_path)
                    if custom_theme.name and custom_theme.name not in items:
                        items[custom_theme.name] = theme_path
                except Exception:
                    pass
    for name, registered in _registered_themes.items():
        items.setdefault(name, registered.source_path)
    return [{"name": name, "path": path} for name, path in sorted(items.items())]


def get_available_themes() -> list[str]:
    return [item["name"] for item in get_available_themes_with_paths()]


def get_theme_by_name(name: str) -> Theme | None:
    try:
        if name in _registered_themes:
            return _registered_themes[name]
        return _theme_from_json(_load_theme_json(name))
    except Exception:
        return None


def get_default_theme() -> str:
    colorfgbg = os.environ.get("COLORFGBG", "")
    match = re.search(r"(?:^|;)(\d{1,3})$", colorfgbg)
    if match:
        return "light" if int(match.group(1)) >= 7 else "dark"
    return "dark"


def init_theme(theme_name: str | None = None, enable_watcher: bool = False) -> None:
    del enable_watcher
    set_theme(theme_name or get_default_theme())


def set_theme(name: str, enable_watcher: bool = False) -> dict[str, Any]:
    del enable_watcher
    global _current_theme, _current_theme_name
    loaded = get_theme_by_name(name)
    if loaded is None:
        _current_theme = _theme_from_json(BUILTIN_THEME_JSON["dark"])
        _current_theme_name = "dark"
        return {"success": False, "error": f"Theme not found: {name}"}
    _current_theme = loaded
    _current_theme_name = name
    return {"success": True}


def set_theme_instance(theme_instance: Theme) -> None:
    global _current_theme, _current_theme_name
    _current_theme = theme_instance
    _current_theme_name = "<in-memory>"


def get_theme() -> Theme:
    global _current_theme
    if _current_theme is None:
        init_theme()
    return _current_theme


def get_language_from_path(file_path: str) -> str | None:
    name = os.path.basename(file_path).lower()
    if name in EXT_TO_LANG:
        return EXT_TO_LANG[name]
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    return EXT_TO_LANG.get(ext)


# pygments token category -> theme syntax color name. Ordered most-specific
# first; matched via pygments' subtype `in` semantics (e.g. Keyword.Constant in
# Keyword). Mirrors the highlight.js class -> syntax* mapping in the Node theme.
def _syntax_color_map() -> list[tuple[Any, str]]:
    from pygments.token import (
        Comment, Keyword, Name, Number, Operator, Punctuation, String,
    )
    return [
        (Comment, "syntaxComment"),
        (Keyword.Type, "syntaxType"),
        (Keyword, "syntaxKeyword"),
        (Name.Function, "syntaxFunction"),
        (Name.Class, "syntaxType"),
        (Name.Builtin, "syntaxType"),
        (Name.Decorator, "syntaxFunction"),
        (String, "syntaxString"),
        (Number, "syntaxNumber"),
        (Operator, "syntaxOperator"),
        (Punctuation, "syntaxPunctuation"),
        (Name.Variable, "syntaxVariable"),
        (Name, "syntaxVariable"),
    ]


def highlight_code(code: str, lang: str | None = None) -> list[str]:
    """Token-aware syntax highlighting using the theme's syntax* palette.

    Falls back to flat code coloring if pygments is unavailable, the language
    is unknown/unguessable, or the active theme lacks the syntax colors.
    """
    t = get_theme()

    def flat() -> list[str]:
        return [t.fg("mdCodeBlock", line) for line in code.split("\n")]

    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.util import ClassNotFound
    except Exception:
        return flat()

    lexer = None
    if lang:
        try:
            lexer = get_lexer_by_name(lang)
        except ClassNotFound:
            lexer = None
    if lexer is None:
        try:
            lexer = guess_lexer(code)
        except Exception:
            return flat()

    color_map = _syntax_color_map()
    available = getattr(t, "_fg_ansi", {})

    def color_for(tok_type: Any) -> str:
        for ttype, cname in color_map:
            if tok_type in ttype and cname in available:
                return cname
        return "mdCodeBlock"

    try:
        lines: list[str] = [""]
        for tok_type, value in lex(code, lexer):
            cname = color_for(tok_type)
            segments = value.split("\n")
            for i, seg in enumerate(segments):
                if i > 0:
                    lines.append("")
                if seg:
                    lines[-1] += t.fg(cname, seg)
        # lex appends a trailing newline -> drop the empty last line if present
        if len(lines) > 1 and lines[-1] == "":
            lines.pop()
        return lines or flat()
    except Exception:
        return flat()


def get_markdown_theme() -> Any:
    from pi_tui.components.markdown import MarkdownTheme

    t = get_theme()
    return MarkdownTheme(
        heading=lambda text: t.fg("mdHeading", text),
        link=lambda text: t.fg("mdLink", text),
        link_url=lambda text: t.fg("mdLinkUrl", text),
        code=lambda text: t.fg("mdCode", text),
        code_block=lambda text: t.fg("mdCodeBlock", text),
        code_block_border=lambda text: t.fg("mdCodeBlockBorder", text),
        quote=lambda text: t.fg("mdQuote", text),
        quote_border=lambda text: t.fg("mdQuoteBorder", text),
        hr=lambda text: t.fg("mdHr", text),
        list_bullet=lambda text: t.fg("mdListBullet", text),
        bold=t.bold,
        italic=t.italic,
        underline=t.underline,
        strikethrough=t.strikethrough,
        highlight_code=highlight_code,
    )


def get_select_list_theme() -> Any:
    from pi_tui.components.select_list import SelectListTheme

    t = get_theme()
    return SelectListTheme(
        selected_prefix=lambda text: t.fg("accent", text),
        selected_text=lambda text: t.fg("accent", text),
        description=lambda text: t.fg("muted", text),
        scroll_info=lambda text: t.fg("muted", text),
        no_match=lambda text: t.fg("muted", text),
    )


def get_editor_theme() -> Any:
    from pi_tui.components.editor import EditorTheme

    t = get_theme()
    return EditorTheme(border_color=lambda text: t.fg("borderMuted", text), select_list=get_select_list_theme())


def get_settings_list_theme() -> Any:
    from pi_tui.components.settings_list import SettingsListTheme

    t = get_theme()
    return SettingsListTheme(
        label=lambda text, selected: t.fg("accent", text) if selected else text,
        value=lambda text, selected: t.fg("accent", text) if selected else t.fg("muted", text),
        description=lambda text: t.fg("dim", text),
        cursor=t.fg("accent", "-> "),
        hint=lambda text: t.fg("dim", text),
    )


getLanguageFromPath = get_language_from_path
getMarkdownTheme = get_markdown_theme
getSelectListTheme = get_select_list_theme
getEditorTheme = get_editor_theme
getSettingsListTheme = get_settings_list_theme
highlightCode = highlight_code
initTheme = init_theme
setTheme = set_theme
setThemeInstance = set_theme_instance
setRegisteredThemes = set_registered_themes
getAvailableThemes = get_available_themes
getAvailableThemesWithPaths = get_available_themes_with_paths
getThemeByName = get_theme_by_name
loadThemeFromPath = load_theme_from_path
getDefaultTheme = get_default_theme


__all__ = [
    "Theme",
    "get_available_themes",
    "get_available_themes_with_paths",
    "get_default_theme",
    "get_editor_theme",
    "get_language_from_path",
    "get_markdown_theme",
    "get_select_list_theme",
    "get_settings_list_theme",
    "get_theme",
    "get_theme_by_name",
    "highlight_code",
    "init_theme",
    "load_theme_from_path",
    "set_registered_themes",
    "set_theme",
    "set_theme_instance",
    "getAvailableThemes",
    "getAvailableThemesWithPaths",
    "getDefaultTheme",
    "getEditorTheme",
    "getLanguageFromPath",
    "getMarkdownTheme",
    "getSelectListTheme",
    "getSettingsListTheme",
    "getThemeByName",
    "highlightCode",
    "initTheme",
    "loadThemeFromPath",
    "setRegisteredThemes",
    "setTheme",
    "setThemeInstance",
]
