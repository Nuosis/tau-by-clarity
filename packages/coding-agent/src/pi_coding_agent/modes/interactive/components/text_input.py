"""Small deterministic text input used by Python component parity shims."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TextInput:
    value: str = ""
    on_submit: Callable[[str], None] | None = None
    on_escape: Callable[[], None] | None = None
    _paste_buffer: str = ""
    _is_in_paste: bool = False

    def get_value(self) -> str:
        return self.value

    def set_value(self, value: str) -> None:
        self.value = value

    def handle_input(self, key_data: str) -> None:
        if "\x1b[200~" in key_data:
            self._is_in_paste = True
            self._paste_buffer = ""
            key_data = key_data.replace("\x1b[200~", "")

        if self._is_in_paste:
            self._paste_buffer += key_data
            end_idx = self._paste_buffer.find("\x1b[201~")
            if end_idx >= 0:
                pasted = self._paste_buffer[:end_idx]
                remaining = self._paste_buffer[end_idx + len("\x1b[201~"):]
                self._paste_buffer = ""
                self._is_in_paste = False
                self._append_text(pasted)
                if remaining:
                    self.handle_input(remaining)
            return

        if key_data in {"\n", "\r", "\x1bOM", "enter", "return"}:
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
        self._append_text(key_data)

    def _append_text(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        filtered = "".join(ch for ch in text if ch == " " or ch.isprintable())
        self.value += filtered

    def render(self, width: int | None = None) -> list[str]:
        text = self.value
        if width is not None and width >= 0:
            text = text[-width:] if len(text) > width else text
        return [text]
