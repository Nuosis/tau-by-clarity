"""
Path resolution utilities — mirrors packages/coding-agent/src/core/tools/path-utils.ts
"""
from __future__ import annotations

import os
import re
import unicodedata


# Unicode spaces pattern (matches U+00A0, U+2000-U+200A, U+202F, U+205F, U+3000)
UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_unicode_spaces(s: str) -> str:
    """Replace all Unicode spaces with ASCII space."""
    return UNICODE_SPACES.sub(" ", s)


def _normalize_at_prefix(s: str) -> str:
    """Remove @ prefix if present."""
    return s[1:] if s.startswith("@") else s


def expand_path(file_path: str) -> str:
    """
    Expand ~ to home directory and normalize Unicode spaces.
    Mirrors expandPath() in TypeScript.
    """
    normalized = _normalize_unicode_spaces(_normalize_at_prefix(file_path))
    if normalized == "~":
        return os.path.expanduser("~")
    if normalized.startswith("~/"):
        return os.path.expanduser(normalized)
    return normalized


def _try_macos_screenshot_path(file_path: str) -> str:
    """Replace ' AM.' and ' PM.' with narrow no-break space + AM/PM."""
    return re.sub(r" (AM|PM)\.", rf"{NARROW_NO_BREAK_SPACE}\1.", file_path)


def _try_nfd_variant(file_path: str) -> str:
    """Convert to NFD (decomposed) form, as macOS stores filenames."""
    return unicodedata.normalize("NFD", file_path)


def _try_curly_quote_variant(file_path: str) -> str:
    """Replace straight apostrophe (') with right single quotation mark (\u2019)."""
    return file_path.replace("'", "\u2019")


def _file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(file_path)


def resolve_to_cwd(path: str, cwd: str) -> str:
    """
    Resolve a path relative to cwd.
    Handles ~ expansion and absolute paths.
    Mirrors resolveToCwd() in TypeScript.
    """
    expanded = expand_path(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(cwd, expanded))


def resolve_read_path(path: str, cwd: str) -> str:
    """
    Resolve a read path with macOS variant fallbacks.
    Mirrors resolveReadPath() in TypeScript.
    """
    resolved = resolve_to_cwd(path, cwd)

    if _file_exists(resolved):
        return resolved

    # Try macOS AM/PM variant (narrow no-break space before AM/PM)
    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and _file_exists(am_pm_variant):
        return am_pm_variant

    # Try NFD variant (macOS stores filenames in NFD form)
    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and _file_exists(nfd_variant):
        return nfd_variant

    # Try curly quote variant (macOS uses U+2019 in screenshot names)
    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and _file_exists(curly_variant):
        return curly_variant

    # Try combined NFD + curly quote (for French macOS screenshots like "Capture d'écran")
    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and _file_exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved
