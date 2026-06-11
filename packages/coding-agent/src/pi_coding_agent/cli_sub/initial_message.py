"""Initial prompt assembly for CLI non-interactive mode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .args import Args


@dataclass
class InitialMessageResult:
    initial_message: str | None = None
    initial_images: list[dict[str, Any]] | None = None


def build_initial_message(
    *,
    parsed: Args,
    file_text: str | None = None,
    file_images: list[dict[str, Any]] | None = None,
    stdin_content: str | None = None,
) -> InitialMessageResult:
    """Combine stdin, @file text, and first CLI message into one prompt.

    Mirrors the Node implementation by consuming the first positional message
    from ``parsed.messages`` after it is folded into the initial prompt.
    """
    parts: list[str] = []
    if stdin_content is not None:
        parts.append(stdin_content)
    if file_text:
        parts.append(file_text)
    if parsed.messages:
        parts.append(parsed.messages.pop(0))
    return InitialMessageResult(
        initial_message="".join(parts) if parts else None,
        initial_images=file_images if file_images else None,
    )


__all__ = ["InitialMessageResult", "build_initial_message"]
