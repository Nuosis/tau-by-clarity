"""
Keybindings management — mirrors packages/coding-agent/src/core/keybindings.ts

Provides KeybindingsManager and DEFAULT_KEYBINDINGS for configurable app/editor actions.
"""
from __future__ import annotations

import json
import os
from pi_coding_agent.config import CONFIG_DIR_NAME
from typing import Union


# Application-level actions (coding agent specific)
LEGACY_KEYBINDING_NAME_MIGRATIONS: dict[str, str] = {
    "interrupt": "app.interrupt",
    "clear": "app.clear",
    "exit": "app.exit",
    "suspend": "app.suspend",
    "cycleThinkingLevel": "app.thinking.cycle",
    "cycleModelForward": "app.model.cycleForward",
    "cycleModelBackward": "app.model.cycleBackward",
    "selectModel": "app.model.select",
    "expandTools": "app.tools.expand",
    "toggleThinking": "app.thinking.toggle",
    "toggleSessionNamedFilter": "app.session.toggleNamedFilter",
    "externalEditor": "app.editor.external",
    "followUp": "app.message.followUp",
    "dequeue": "app.message.dequeue",
    "pasteImage": "app.clipboard.pasteImage",
    "newSession": "app.session.new",
    "tree": "app.session.tree",
    "fork": "app.session.fork",
    "resume": "app.session.resume",
    "moveLeft": "tui.editor.cursorLeft",
    "moveRight": "tui.editor.cursorRight",
    "moveWordLeft": "tui.editor.cursorWordLeft",
    "moveWordRight": "tui.editor.cursorWordRight",
    "moveToLineStart": "tui.editor.cursorLineStart",
    "moveToLineEnd": "tui.editor.cursorLineEnd",
    "moveUp": "tui.editor.cursorUp",
    "moveDown": "tui.editor.cursorDown",
    "selectLeft": "tui.editor.selectLeft",
    "selectRight": "tui.editor.selectRight",
    "selectWordLeft": "tui.editor.selectWordLeft",
    "selectWordRight": "tui.editor.selectWordRight",
    "selectToLineStart": "tui.editor.selectToLineStart",
    "selectToLineEnd": "tui.editor.selectToLineEnd",
    "selectAll": "tui.input.copy",
    "deleteLeft": "tui.editor.deleteCharBackward",
    "deleteRight": "tui.editor.deleteCharForward",
    "deleteWordLeft": "tui.editor.deleteWordBackward",
    "deleteWordRight": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "newline": "tui.input.newLine",
    "submit": "tui.input.submit",
    "historyPrev": "tui.editor.historyPrev",
    "historyNext": "tui.editor.historyNext",
    "tab": "tui.input.tab",
}

APP_ACTIONS: list[str] = [
    "app.interrupt",
    "app.clear",
    "app.exit",
    "app.suspend",
    "app.thinking.cycle",
    "app.model.cycleForward",
    "app.model.cycleBackward",
    "app.model.select",
    "app.tools.expand",
    "app.thinking.toggle",
    "app.session.toggleNamedFilter",
    "app.editor.external",
    "app.message.followUp",
    "app.message.dequeue",
    "app.clipboard.pasteImage",
    "app.session.new",
    "app.session.tree",
    "app.session.fork",
    "app.session.resume",
]

# Default editor actions (subset, since we don't use pi-tui)
EDITOR_ACTIONS: list[str] = [
    "tui.editor.cursorLeft",
    "tui.editor.cursorRight",
    "tui.editor.cursorWordLeft",
    "tui.editor.cursorWordRight",
    "tui.editor.cursorLineStart",
    "tui.editor.cursorLineEnd",
    "tui.editor.cursorUp",
    "tui.editor.cursorDown",
    "tui.editor.selectLeft",
    "tui.editor.selectRight",
    "tui.editor.selectWordLeft",
    "tui.editor.selectWordRight",
    "tui.editor.selectToLineStart",
    "tui.editor.selectToLineEnd",
    "tui.input.copy",
    "tui.editor.deleteCharBackward",
    "tui.editor.deleteCharForward",
    "tui.editor.deleteWordBackward",
    "tui.editor.deleteWordForward",
    "tui.editor.deleteToLineStart",
    "tui.editor.deleteToLineEnd",
    "tui.input.newLine",
    "tui.input.submit",
    "tui.editor.historyPrev",
    "tui.editor.historyNext",
    "tui.input.tab",
]

# KeyId = str (e.g. "escape", "ctrl+c", "alt+enter")
KeyId = str

# Default application keybindings — mirrors DEFAULT_APP_KEYBINDINGS in TS
DEFAULT_APP_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    "app.interrupt": "escape",
    "app.clear": "ctrl+c",
    "app.exit": "ctrl+d",
    "app.suspend": "ctrl+z",
    "app.thinking.cycle": "shift+tab",
    "app.model.cycleForward": "ctrl+p",
    "app.model.cycleBackward": "shift+ctrl+p",
    "app.model.select": "ctrl+l",
    "app.tools.expand": "ctrl+o",
    "app.thinking.toggle": "ctrl+t",
    "app.session.toggleNamedFilter": "ctrl+n",
    "app.editor.external": "ctrl+g",
    "app.message.followUp": "alt+enter",
    "app.message.dequeue": "alt+up",
    "app.clipboard.pasteImage": "ctrl+v",
    "app.session.new": [],
    "app.session.tree": [],
    "app.session.fork": [],
    "app.session.resume": [],
}

# Default editor keybindings (readline-compatible)
DEFAULT_EDITOR_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    "tui.editor.cursorLeft": "left",
    "tui.editor.cursorRight": "right",
    "tui.editor.cursorWordLeft": "ctrl+left",
    "tui.editor.cursorWordRight": "ctrl+right",
    "tui.editor.cursorLineStart": "home",
    "tui.editor.cursorLineEnd": "end",
    "tui.editor.cursorUp": "up",
    "tui.editor.cursorDown": "down",
    "tui.editor.selectLeft": "shift+left",
    "tui.editor.selectRight": "shift+right",
    "tui.editor.selectWordLeft": "shift+ctrl+left",
    "tui.editor.selectWordRight": "shift+ctrl+right",
    "tui.editor.selectToLineStart": "shift+home",
    "tui.editor.selectToLineEnd": "shift+end",
    "tui.input.copy": "ctrl+a",
    "tui.editor.deleteCharBackward": "backspace",
    "tui.editor.deleteCharForward": "delete",
    "tui.editor.deleteWordBackward": "ctrl+backspace",
    "tui.editor.deleteWordForward": "ctrl+delete",
    "tui.editor.deleteToLineStart": "ctrl+u",
    "tui.editor.deleteToLineEnd": "ctrl+k",
    "tui.input.newLine": "shift+enter",
    "tui.input.submit": "enter",
    "tui.editor.historyPrev": "ctrl+up",
    "tui.editor.historyNext": "ctrl+down",
    "tui.input.tab": "tab",
}

# All default keybindings (app + editor) — mirrors DEFAULT_KEYBINDINGS in TS
DEFAULT_KEYBINDINGS: dict[str, Union[KeyId, list[KeyId]]] = {
    **DEFAULT_EDITOR_KEYBINDINGS,
    **DEFAULT_APP_KEYBINDINGS,
}


class KeybindingsManager:
    """
    Manages all keybindings (app + editor).
    Mirrors KeybindingsManager in TypeScript.
    """

    def __init__(self, config: dict[str, Union[KeyId, list[KeyId]]] | None = None):
        self._config: dict[str, Union[KeyId, list[KeyId]]] = self._normalize_config(config or {})
        self._app_action_to_keys: dict[str, list[KeyId]] = {}
        self._build_maps()

    @classmethod
    def create(cls, agent_dir: str | None = None) -> "KeybindingsManager":
        """
        Load keybindings from ~/.pi/agent/keybindings.json and merge with defaults.
        """
        if agent_dir is None:
            agent_dir = os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME, "agent")

        keybindings_path = os.path.join(agent_dir, "keybindings.json")
        user_config: dict = {}

        if os.path.exists(keybindings_path):
            try:
                with open(keybindings_path, encoding="utf-8") as f:
                    user_config = json.load(f)
            except Exception:
                pass

        merged = {**DEFAULT_KEYBINDINGS, **cls._normalize_config(user_config)}
        return cls(merged)

    @staticmethod
    def normalize_action(action: str) -> str:
        return LEGACY_KEYBINDING_NAME_MIGRATIONS.get(action, action)

    @classmethod
    def _normalize_config(
        cls,
        config: dict[str, Union[KeyId, list[KeyId]]],
    ) -> dict[str, Union[KeyId, list[KeyId]]]:
        normalized: dict[str, Union[KeyId, list[KeyId]]] = {}
        for action, keys in config.items():
            normalized[cls.normalize_action(action)] = keys
        return normalized

    def _build_maps(self) -> None:
        """Build internal lookup maps from config."""
        self._app_action_to_keys.clear()
        for action in APP_ACTIONS:
            keys = self._config.get(action) or DEFAULT_APP_KEYBINDINGS.get(action, [])
            if isinstance(keys, str):
                self._app_action_to_keys[action] = [keys]
            else:
                self._app_action_to_keys[action] = list(keys)

    def get_keys_for_action(self, action: str) -> list[KeyId]:
        """Get the list of key IDs bound to an action."""
        action = self.normalize_action(action)
        if action in APP_ACTIONS:
            return self._app_action_to_keys.get(action, [])
        # Editor actions
        keys = self._config.get(action) or DEFAULT_EDITOR_KEYBINDINGS.get(action, [])
        if isinstance(keys, str):
            return [keys]
        return list(keys)

    def matches(self, first: str, second: KeyId) -> bool:
        """Check if a key matches an action.

        Node uses matches(key, action). Older Python callers used matches(action, key).
        Accept both orders so editor code and existing callers share one contract.
        """
        if self.normalize_action(second) in DEFAULT_KEYBINDINGS:
            key = first
            action = second
        else:
            action = first
            key = second
        return key in self.get_keys_for_action(action)

    def get_config(self) -> dict[str, Union[KeyId, list[KeyId]]]:
        """Get the full keybindings config."""
        return dict(self._config)

    def get_effective_config(self) -> dict[str, Union[KeyId, list[KeyId]]]:
        """Return the resolved Node-style keybindings config."""
        return self.get_config()

    def set_keybinding(self, action: str, keys: Union[KeyId, list[KeyId]]) -> None:
        """Update a keybinding at runtime."""
        self._config[self.normalize_action(action)] = keys
        self._build_maps()

    getKeys = get_keys_for_action
    getEffectiveConfig = get_effective_config
