"""
Project trust selector component.

Mirrors the state and input behavior of TypeScript components/trust-selector.ts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pi_coding_agent.core.trust_manager import ProjectTrustDecision


@dataclass(frozen=True)
class TrustOption:
    label: str
    trusted: bool


TRUST_OPTIONS = [
    TrustOption("Trust", True),
    TrustOption("Do not trust", False),
]


def format_decision(decision: ProjectTrustDecision) -> str:
    if decision is True:
        return "trusted"
    if decision is False:
        return "untrusted"
    return "none"


class TrustSelectorComponent:
    def __init__(
        self,
        cwd: str,
        saved_decision: ProjectTrustDecision,
        project_trusted: bool,
        on_select: Callable[[bool], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.cwd = cwd
        self.saved_decision = saved_decision
        self.project_trusted = project_trusted
        self.on_select = on_select or (lambda trusted: None)
        self.on_cancel = on_cancel or (lambda: None)
        initial = next((idx for idx, option in enumerate(TRUST_OPTIONS) if option.trusted == saved_decision), 0)
        self.selected_index = max(0, initial)

    def render(self, width: int | None = None) -> list[str]:
        lines = [
            "Project trust",
            self.cwd,
            f"Saved decision: {format_decision(self.saved_decision)}",
            f"Current session: {'trusted' if self.project_trusted else 'untrusted'}",
            "",
        ]
        for index, option in enumerate(TRUST_OPTIONS):
            selected = "→ " if index == self.selected_index else "  "
            current = " ✓" if option.trusted == self.saved_decision else ""
            lines.append(f"{selected}{option.label}{current}")
        lines.append("")
        lines.append("↑↓ navigate  Enter save  Esc cancel")
        if width is None:
            return lines
        return [line[:width] for line in lines]

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"}:
            self.selected_index = max(0, self.selected_index - 1)
        elif key_data in {"down", "j", "\x1b[B"}:
            self.selected_index = min(len(TRUST_OPTIONS) - 1, self.selected_index + 1)
        elif key_data in {"\n", "enter", "return"}:
            self.on_select(TRUST_OPTIONS[self.selected_index].trusted)
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()


__all__ = [
    "TRUST_OPTIONS",
    "TrustOption",
    "TrustSelectorComponent",
    "format_decision",
]
