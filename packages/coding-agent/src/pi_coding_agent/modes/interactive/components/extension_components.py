"""Extension prompt components."""
from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from typing import Any, Callable

from .countdown_timer import CountdownTimer
from .text_input import TextInput


class ExtensionSelectorComponent:
    def __init__(
        self,
        title: str,
        options: list[str],
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        opts: dict[str, Any] | None = None,
    ) -> None:
        self.base_title = title
        self.title = title
        self.options = list(options)
        self.selected_index = 0
        self.on_select = on_select or (lambda option: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.on_toggle_tools_expanded = (opts or {}).get("onToggleToolsExpanded") or (opts or {}).get("on_toggle_tools_expanded")
        timeout = (opts or {}).get("timeout")
        self.countdown = (
            CountdownTimer(
                int(timeout),
                on_tick=lambda seconds: setattr(self, "title", f"{self.base_title} ({seconds}s)"),
                on_expire=self.on_cancel,
            )
            if timeout and int(timeout) > 0
            else None
        )

    def handle_input(self, key_data: str) -> None:
        if key_data in {"toggle_tools", "ctrl+o"}:
            if self.on_toggle_tools_expanded:
                self.on_toggle_tools_expanded()
        elif key_data in {"up", "k", "\x1b[A"} and self.options:
            self.selected_index = max(0, self.selected_index - 1)
        elif key_data in {"down", "j", "\x1b[B"} and self.options:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
        elif key_data in {"\n", "enter", "return"} and self.options:
            self.on_select(self.options[self.selected_index])
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()

    def dispose(self) -> None:
        if self.countdown:
            self.countdown.dispose()

    def render(self, width: int | None = None) -> list[str]:
        return [
            self.title,
            *[
                f"{'→ ' if idx == self.selected_index else '  '}{option}"
                for idx, option in enumerate(self.options)
            ],
        ]


class ExtensionInputComponent:
    def __init__(
        self,
        title: str,
        placeholder: str | None = None,
        on_submit: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        opts: dict[str, Any] | None = None,
    ) -> None:
        self.base_title = title
        self.title = title
        self.placeholder = placeholder
        self.on_submit = on_submit or (lambda value: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.input = TextInput(on_submit=self.on_submit, on_escape=self.on_cancel)
        timeout = (opts or {}).get("timeout")
        self.countdown = (
            CountdownTimer(
                int(timeout),
                on_tick=lambda seconds: setattr(self, "title", f"{self.base_title} ({seconds}s)"),
                on_expire=self.on_cancel,
            )
            if timeout and int(timeout) > 0
            else None
        )

    def handle_input(self, key_data: str) -> None:
        self.input.handle_input(key_data)

    def dispose(self) -> None:
        if self.countdown:
            self.countdown.dispose()

    def render(self, width: int | None = None) -> list[str]:
        hint = self.placeholder or ""
        return [self.title, hint, *self.input.render(width)]


class ExtensionEditorComponent:
    def __init__(
        self,
        title: str,
        prefill: str | None = None,
        on_submit: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        tui: Any | None = None,
        keybindings: Any | None = None,
        options: Any | None = None,
    ) -> None:
        self.title = title
        self.text = prefill or ""
        self.on_submit = on_submit or (lambda value: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.tui = tui
        self.keybindings = keybindings
        self.options = options

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text

    def handle_input(self, key_data: str) -> None:
        if key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()
        elif self._matches_external_editor(key_data):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.open_external_editor())
            else:
                loop.create_task(self.open_external_editor())
        elif key_data in {"\n", "enter", "return"}:
            self.on_submit(self.text)
        elif key_data in {"backspace", "\b", "\x7f"}:
            self.text = self.text[:-1]
        elif len(key_data) == 1 and key_data.isprintable():
            self.text += key_data

    def _matches_external_editor(self, key_data: str) -> bool:
        if key_data in {"external_editor", "ctrl+g"}:
            return True
        if self.keybindings is not None and hasattr(self.keybindings, "matches"):
            return bool(self.keybindings.matches(key_data, "app.editor.external"))
        return False

    def _tui_call(self, method: str, *args: Any) -> None:
        target = getattr(self.tui, method, None)
        if callable(target):
            target(*args)

    async def open_external_editor(self) -> bool:
        editor_cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor_cmd:
            return False

        fd, tmp_path = tempfile.mkstemp(prefix="pi-extension-editor-", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.text)

            self._tui_call("stop")
            cmd_parts = shlex.split(editor_cmd)
            if not cmd_parts:
                return False

            proc = await asyncio.create_subprocess_exec(
                cmd_parts[0],
                *cmd_parts[1:],
                tmp_path,
                stdin=None,
                stdout=None,
                stderr=None,
            )
            status = await proc.wait()
            if status == 0:
                with open(tmp_path, encoding="utf-8") as handle:
                    self.text = handle.read().removesuffix("\n")
                return True
            return False
        except (OSError, ValueError):
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            self._tui_call("start")
            self._tui_call("request_render", True)
            self._tui_call("requestRender", True)

    def render(self, width: int | None = None) -> list[str]:
        return [self.title, self.text]


class CustomEditor:
    def __init__(self, keybindings: Any | None = None, text: str = "") -> None:
        self.keybindings = keybindings
        self.text = text
        self.action_handlers: dict[str, Callable[[], None]] = {}
        self.on_escape: Callable[[], None] | None = None
        self.on_ctrl_d: Callable[[], None] | None = None
        self.on_paste_image: Callable[[], None] | None = None
        self.on_extension_shortcut: Callable[[str], bool] | None = None

    def on_action(self, action: str, handler: Callable[[], None]) -> None:
        self.action_handlers[action] = handler

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text

    def _matches(self, key_data: str, action: str) -> bool:
        if self.keybindings is not None and hasattr(self.keybindings, "matches"):
            return bool(self.keybindings.matches(key_data, action))
        fallback = {
            "escape": "app.interrupt",
            "ctrl+d": "app.exit",
            "paste_image": "app.clipboard.pasteImage",
        }
        return fallback.get(key_data) == action

    def handle_input(self, key_data: str) -> None:
        if self.on_extension_shortcut and self.on_extension_shortcut(key_data):
            return
        if self._matches(key_data, "app.clipboard.pasteImage"):
            if self.on_paste_image:
                self.on_paste_image()
            return
        if self._matches(key_data, "app.interrupt"):
            handler = self.on_escape or self.action_handlers.get("app.interrupt")
            if handler:
                handler()
            return
        if self._matches(key_data, "app.exit") and not self.text:
            handler = self.on_ctrl_d or self.action_handlers.get("app.exit")
            if handler:
                handler()
            return
        for action, handler in self.action_handlers.items():
            if action not in {"app.interrupt", "app.exit"} and self._matches(key_data, action):
                handler()
                return
        if key_data in {"backspace", "\b", "\x7f"}:
            self.text = self.text[:-1]
        elif len(key_data) == 1 and key_data.isprintable():
            self.text += key_data
