"""Tau by Clarity startup banner for the tau harness.

Self-contained: the block lettering is embedded (no figlet dependency at
runtime). Renders in cyan on truecolor terminals, degrades to plain text
elsewhere.
"""
from __future__ import annotations

import os
import sys

# "TAU" — figlet `standard` font.
_TAU = r""" _____     _     _   _
|_   _|   / \   | | | |
  | |    / _ \  | | | |
  | |   / ___ \ | |_| |
  |_|  /_/   \_\ \___/ """

_LINE1 = "claritycustomsoftware.com"
_LINE2 = "tau · by clarity"

# 256-color cyan — renders in Terminal.app AND iTerm2/kitty (truecolor 38;2 is
# silently dropped by Terminal.app, which is why it showed gray).
_CYAN = "\x1b[1;38;5;45m"   # bold bright cyan
_DIM = "\x1b[38;5;37m"      # softer teal-cyan for the sub-lines
_RST = "\x1b[0m"


def _supports_color(stream: object) -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("PI_NO_BANNER"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    # Banner only shows interactively; treat a TTY on stdout OR stderr as color-ok.
    out_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    err_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return out_tty or err_tty


def render_banner(color: bool = True) -> str:
    """Return the banner as a string (ANSI-colored when color=True)."""
    art_lines = _TAU.split("\n")
    width = max(len(line) for line in art_lines)

    def center(text: str) -> str:
        return text.center(width)

    # Block art stays left-aligned EXACTLY as generated (centering each line
    # independently skews the diagonal strokes). Only the sub-lines are centered.
    if color:
        body = "\n".join(f"{_CYAN}{line}{_RST}" for line in art_lines)
        sub = f"{_DIM}{center(_LINE1)}{_RST}\n{_DIM}{center(_LINE2)}{_RST}"
    else:
        body = "\n".join(art_lines)
        sub = f"{center(_LINE1)}\n{center(_LINE2)}"

    return f"{body}\n\n{sub}\n"


def print_banner(stream=None) -> None:
    """Print the banner to a stream (stderr by default), colored if it's a TTY."""
    stream = stream or sys.stderr
    stream.write(render_banner(color=_supports_color(stream)))
    stream.flush()


if __name__ == "__main__":
    print_banner(sys.stdout)
