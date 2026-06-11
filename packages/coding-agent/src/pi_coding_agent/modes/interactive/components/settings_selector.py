"""Settings selector parity contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .selectors import SelectItem, SelectList

THINKING_DESCRIPTIONS = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Maximum reasoning (~32k tokens)",
}


@dataclass
class SettingItem:
    id: str
    label: str
    description: str
    current_value: str
    values: list[str]


class SettingsSelectorComponent:
    def __init__(self, config: Any, callbacks: Any) -> None:
        self.config = config
        self.callbacks = callbacks
        self.items = self._build_items(config)
        self.select_list = SelectList(
            [
                SelectItem(item.id, item.label, f"{item.current_value} - {item.description}")
                for item in self.items
            ],
            visible_count=10,
            on_select=lambda selected: self.cycle_setting(str(selected.value)),
            on_cancel=lambda: self._call("on_cancel"),
        )

    def _cfg(self, name: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    def _call(self, name: str, *args: Any) -> None:
        callback = self.callbacks.get(name) if isinstance(self.callbacks, dict) else getattr(self.callbacks, name, None)
        if callback:
            callback(*args)

    def _build_items(self, config: Any) -> list[SettingItem]:
        def bool_value(name: str, default: bool = False) -> str:
            return "true" if bool(self._cfg(name, default)) else "false"

        items = [
            SettingItem("autocompact", "Auto-compact", "Automatically compact context", bool_value("autoCompact", bool(self._cfg("auto_compact", True))), ["true", "false"]),
            SettingItem("auto-resize-images", "Auto-resize images", "Resize large images", bool_value("autoResizeImages", bool(self._cfg("auto_resize_images", True))), ["true", "false"]),
            SettingItem("block-images", "Block images", "Prevent images from being sent", bool_value("blockImages", bool(self._cfg("block_images", False))), ["true", "false"]),
            SettingItem("skill-commands", "Skill commands", "Register skills as slash commands", bool_value("enableSkillCommands", True), ["true", "false"]),
            SettingItem("steering-mode", "Steering mode", "Streaming steering behavior", str(self._cfg("steeringMode", self._cfg("steering_mode", "one-at-a-time"))), ["one-at-a-time", "all"]),
            SettingItem("follow-up-mode", "Follow-up mode", "Queued follow-up behavior", str(self._cfg("followUpMode", self._cfg("follow_up_mode", "one-at-a-time"))), ["one-at-a-time", "all"]),
            SettingItem("transport", "Transport", "Preferred provider transport", str(self._cfg("transport", "auto")), ["sse", "websocket", "websocket-cached", "auto"]),
            SettingItem("hide-thinking", "Hide thinking", "Hide thinking blocks", bool_value("hideThinkingBlock", bool(self._cfg("hide_thinking_block", False))), ["true", "false"]),
            SettingItem("quiet-startup", "Quiet startup", "Disable verbose startup printing", bool_value("quietStartup", bool(self._cfg("quiet_startup", False))), ["true", "false"]),
            SettingItem("double-escape-action", "Double-escape action", "Action for double escape", str(self._cfg("doubleEscapeAction", self._cfg("double_escape_action", "tree"))), ["tree", "fork", "none"]),
            SettingItem("tree-filter-mode", "Tree filter mode", "Default tree filter", str(self._cfg("treeFilterMode", self._cfg("tree_filter_mode", "default"))), ["default", "no-tools", "user-only", "labeled-only", "all"]),
            SettingItem("thinking", "Thinking level", "Reasoning depth", str(self._cfg("thinkingLevel", self._cfg("thinking_level", "medium"))), list(self._cfg("availableThinkingLevels", ["off", "minimal", "low", "medium", "high", "xhigh"]))),
            SettingItem("theme", "Theme", "Color theme", str(self._cfg("currentTheme", self._cfg("theme", "dark"))), list(self._cfg("availableThemes", ["dark", "light"]))),
        ]
        if bool(self._cfg("supportsImages", self._cfg("showImagesSupported", False))):
            items.insert(1, SettingItem("show-images", "Show images", "Render images inline", bool_value("showImages", True), ["true", "false"]))
            items.insert(2, SettingItem("image-width-cells", "Image width", "Inline image width", str(self._cfg("imageWidthCells", 80)), ["60", "80", "120"]))
        return items

    def get_settings_list(self) -> SelectList:
        return self.select_list

    def item_by_id(self, item_id: str) -> SettingItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def cycle_setting(self, item_id: str) -> str | None:
        item = self.item_by_id(item_id)
        if item is None or not item.values:
            return None
        idx = item.values.index(item.current_value) if item.current_value in item.values else -1
        new_value = item.values[(idx + 1) % len(item.values)]
        self.set_setting(item_id, new_value)
        return new_value

    def set_setting(self, item_id: str, new_value: str) -> None:
        item = self.item_by_id(item_id)
        if item is not None:
            item.current_value = new_value
        callback_map: dict[str, tuple[str, Callable[[str], Any]]] = {
            "autocompact": ("onAutoCompactChange", lambda value: value == "true"),
            "show-images": ("onShowImagesChange", lambda value: value == "true"),
            "image-width-cells": ("onImageWidthCellsChange", int),
            "auto-resize-images": ("onAutoResizeImagesChange", lambda value: value == "true"),
            "block-images": ("onBlockImagesChange", lambda value: value == "true"),
            "skill-commands": ("onEnableSkillCommandsChange", lambda value: value == "true"),
            "steering-mode": ("onSteeringModeChange", str),
            "follow-up-mode": ("onFollowUpModeChange", str),
            "transport": ("onTransportChange", str),
            "hide-thinking": ("onHideThinkingBlockChange", lambda value: value == "true"),
            "quiet-startup": ("onQuietStartupChange", lambda value: value == "true"),
            "double-escape-action": ("onDoubleEscapeActionChange", str),
            "tree-filter-mode": ("onTreeFilterModeChange", str),
            "thinking": ("onThinkingLevelChange", str),
            "theme": ("onThemeChange", str),
        }
        if item_id in callback_map:
            name, coercer = callback_map[item_id]
            self._call(name, coercer(new_value))

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)

    def render(self) -> list[str]:
        return ["Settings", *[f"{item.label}: {item.current_value}" for item in self.items]]


__all__ = ["SettingItem", "SettingsSelectorComponent", "THINKING_DESCRIPTIONS"]
