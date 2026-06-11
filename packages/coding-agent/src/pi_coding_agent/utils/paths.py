"""Path normalization helpers."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse, unquote

UNICODE_SPACES = "".join(chr(code) for code in [0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000])


@dataclass(frozen=True)
class PathInputOptions:
    trim: bool = False
    expand_tilde: bool = True
    home_dir: str | None = None
    strip_at_prefix: bool = False
    normalize_unicode_spaces: bool = False


def canonicalize_path(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def is_local_path(value: str) -> bool:
    trimmed = value.strip()
    return not trimmed.startswith(("npm:", "git:", "github:", "http:", "https:", "ssh:"))


def normalize_path(input_path: str, options: PathInputOptions | None = None) -> str:
    opts = options or PathInputOptions()
    normalized = input_path.strip() if opts.trim else input_path
    if opts.normalize_unicode_spaces:
        normalized = normalized.translate({ord(char): " " for char in UNICODE_SPACES})
    if opts.strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    if opts.expand_tilde:
        home = opts.home_dir or os.path.expanduser("~")
        if normalized == "~":
            return home
        if normalized.startswith("~/") or normalized.startswith("~\\"):
            return os.path.join(home, normalized[2:])
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        return unquote(parsed.path)
    return normalized


def resolve_path(input_path: str, base_dir: str | None = None, options: PathInputOptions | None = None) -> str:
    normalized = normalize_path(input_path, options)
    normalized_base = normalize_path(base_dir or os.getcwd())
    return os.path.abspath(normalized if os.path.isabs(normalized) else os.path.join(normalized_base, normalized))


def get_cwd_relative_path(file_path: str, cwd: str) -> str | None:
    resolved_cwd = resolve_path(cwd)
    resolved_path = resolve_path(file_path, resolved_cwd)
    relative_path = os.path.relpath(resolved_path, resolved_cwd)
    if relative_path == ".":
        return "."
    if relative_path == ".." or relative_path.startswith(f"..{os.sep}") or os.path.isabs(relative_path):
        return None
    return relative_path


def format_path_relative_to_cwd_or_absolute(file_path: str, cwd: str) -> str:
    absolute_path = resolve_path(file_path, cwd)
    display = get_cwd_relative_path(absolute_path, cwd) or absolute_path
    return display.replace(os.sep, "/")


def mark_path_ignored_by_cloud_sync(path: str) -> None:
    if sys_platform := os.uname().sysname.lower() if hasattr(os, "uname") else "":
        commands = []
        if sys_platform == "darwin":
            commands = [
                ["xattr", "-w", "com.dropbox.ignored", "1", path],
                ["xattr", "-w", "com.apple.fileprovider.ignore#P", "1", path],
            ]
        elif sys_platform == "linux":
            commands = [["setfattr", "-n", "user.com.dropbox.ignored", "-v", "1", path]]
        for cmd in commands:
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except OSError:
                pass


__all__ = [
    "PathInputOptions",
    "canonicalize_path",
    "format_path_relative_to_cwd_or_absolute",
    "get_cwd_relative_path",
    "is_local_path",
    "mark_path_ignored_by_cloud_sync",
    "normalize_path",
    "resolve_path",
]
