"""Small deterministic text input used by Python component parity shims."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TextInput:
    value: str = ""
    on_submit: Callable[[str], None] | None = None
    on_escape: Callable[[], None] | None = None

    def get_value(self) -> str:
        return self.value

    def set_value(self, value: str) -> None:
        self.value = value

    def handle_input(self, key_data: str) -> None:
        if key_data in {"\n", "enter", "return"}:
            if self.on_submit:
                self.on_submit(self.value)
            return
        if key_data in {"escape", "esc", "\x1b"}:
            if self.on_escape:
                self.on_escape()
            return
        if key_data in {"backspace", "\b", "\x7f"}:
            self.value = self.value[:-1]
            return
        if len(key_data) == 1 and key_data.isprintable():
            self.value += key_data

    def render(self, width: int | None = None) -> list[str]:
        text = self.value
        if width is not None and width >= 0:
            text = text[-width:] if len(text) > width else text
        return [text]
