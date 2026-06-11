"""Decorative interactive components exported for Node API parity."""
from __future__ import annotations

from typing import Any


class _StaticDecorativeComponent:
    def __init__(self, ui: Any | None = None) -> None:
        self.ui = ui
        self._cached_width: int | None = None
        self._cached_lines: list[str] = []
        self.disposed = False

    def invalidate(self) -> None:
        self._cached_width = None

    def dispose(self) -> None:
        self.disposed = True

    def _center(self, text: str, width: int) -> str:
        visible = len(text)
        left = max(0, (width - visible) // 2)
        return " " * left + text

    def _lines(self) -> list[str]:
        raise NotImplementedError

    def render(self, width: int) -> list[str]:
        if self._cached_width == width:
            return self._cached_lines
        self._cached_lines = [self._center(line[:width], width) for line in self._lines()]
        self._cached_width = width
        return self._cached_lines


class ArminComponent(_StaticDecorativeComponent):
    """Small deterministic version of the Node Armin easter-egg component."""

    def _lines(self) -> list[str]:
        return [
            "   [pi]",
            "  /----\\",
            "  \\----/",
            "ARMIN SAYS HI",
        ]


class DaxnutsComponent(_StaticDecorativeComponent):
    """Small deterministic version of the Node Daxnuts easter-egg component."""

    def _lines(self) -> list[str]:
        return [
            "POWERED BY DAXNUTS",
            "Free Kimi K2.5 via OpenCode Zen",
            '"Powered by daxnuts"',
            "- @thdxr",
        ]


__all__ = ["ArminComponent", "DaxnutsComponent"]
