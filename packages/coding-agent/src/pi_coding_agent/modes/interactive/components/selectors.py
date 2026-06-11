"""
Selector components for interactive mode.

Mirrors the stateful behavior of TypeScript selector components without
depending on the TypeScript terminal widget classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pi_agent.types import ThinkingLevel


@dataclass
class SelectItem:
    value: Any
    label: str
    description: str | None = None


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_key(model: Any) -> tuple[str, str]:
    return str(_get_attr_or_key(model, "provider", "")), str(_get_attr_or_key(model, "id", ""))


def models_are_equal(left: Any | None, right: Any | None) -> bool:
    if left is None or right is None:
        return left is right
    return _model_key(left) == _model_key(right)


def fuzzy_score(query: str, text: str) -> int | None:
    """Simple ordered-subsequence fuzzy score. Lower is better."""
    q = query.lower().strip()
    if not q:
        return 0
    haystack = text.lower()
    pos = -1
    score = 0
    for char in q:
        next_pos = haystack.find(char, pos + 1)
        if next_pos == -1:
            return None
        score += next_pos - pos
        pos = next_pos
    return score


def fuzzy_filter(items: Iterable[Any], query: str, text_fn: Callable[[Any], str]) -> list[Any]:
    scored: list[tuple[int, Any]] = []
    for item in items:
        score = fuzzy_score(query, text_fn(item))
        if score is not None:
            scored.append((score, item))
    scored.sort(key=lambda row: row[0])
    return [item for _, item in scored]


class SelectList:
    def __init__(
        self,
        items: list[SelectItem],
        visible_count: int = 10,
        on_select: Callable[[SelectItem], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_selection_change: Callable[[SelectItem], None] | None = None,
    ) -> None:
        self.items = list(items)
        self.visible_count = visible_count
        self.selected_index = 0
        self.on_select = on_select or (lambda item: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.on_selection_change = on_selection_change or (lambda item: None)

    def set_selected_index(self, index: int) -> None:
        if not self.items:
            self.selected_index = 0
            return
        self.selected_index = max(0, min(index, len(self.items) - 1))
        self.on_selection_change(self.items[self.selected_index])

    def selected_item(self) -> SelectItem | None:
        if not self.items:
            return None
        return self.items[self.selected_index]

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"}:
            if self.items:
                self.set_selected_index((self.selected_index - 1) % len(self.items))
        elif key_data in {"down", "j", "\x1b[B"}:
            if self.items:
                self.set_selected_index((self.selected_index + 1) % len(self.items))
        elif key_data in {"\n", "enter", "return"}:
            item = self.selected_item()
            if item is not None:
                self.on_select(item)
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()

    def render(self) -> list[str]:
        if not self.items:
            return ["  No items"]
        start = max(0, min(self.selected_index - self.visible_count // 2, len(self.items) - self.visible_count))
        end = min(start + self.visible_count, len(self.items))
        lines: list[str] = []
        for index in range(start, end):
            item = self.items[index]
            prefix = "→ " if index == self.selected_index else "  "
            suffix = f" - {item.description}" if item.description else ""
            lines.append(f"{prefix}{item.label}{suffix}")
        if start > 0 or end < len(self.items):
            lines.append(f"  ({self.selected_index + 1}/{len(self.items)})")
        return lines


LEVEL_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Maximum reasoning (~32k tokens)",
}


class ThinkingSelectorComponent:
    def __init__(
        self,
        current_level: ThinkingLevel,
        available_levels: list[ThinkingLevel],
        on_select: Callable[[ThinkingLevel], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.items = [
            SelectItem(level, str(level), LEVEL_DESCRIPTIONS[str(level)])
            for level in available_levels
        ]
        self.select_list = SelectList(
            self.items,
            visible_count=len(self.items),
            on_select=lambda item: (on_select or (lambda level: None))(item.value),
            on_cancel=on_cancel,
        )
        for idx, item in enumerate(self.items):
            if item.value == current_level:
                self.select_list.set_selected_index(idx)
                break

    def get_select_list(self) -> SelectList:
        return self.select_list

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)

    def render(self) -> list[str]:
        return ["Thinking level", *self.select_list.render()]


class ThemeSelectorComponent:
    def __init__(
        self,
        current_theme: str,
        themes: list[str] | None = None,
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_preview: Callable[[str], None] | None = None,
    ) -> None:
        theme_names = themes or ["dark", "light"]
        self.items = [
            SelectItem(name, name, "(current)" if name == current_theme else None)
            for name in theme_names
        ]
        self.select_list = SelectList(
            self.items,
            visible_count=10,
            on_select=lambda item: (on_select or (lambda theme: None))(item.value),
            on_cancel=on_cancel,
            on_selection_change=lambda item: (on_preview or (lambda theme: None))(item.value),
        )
        if current_theme in theme_names:
            self.select_list.set_selected_index(theme_names.index(current_theme))

    def get_select_list(self) -> SelectList:
        return self.select_list

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)

    def render(self) -> list[str]:
        return ["Theme", *self.select_list.render()]


class ShowImagesSelectorComponent:
    def __init__(
        self,
        current_value: bool,
        on_select: Callable[[bool], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.select_list = SelectList(
            [
                SelectItem("yes", "Yes", "Show images inline in terminal"),
                SelectItem("no", "No", "Show text placeholder instead"),
            ],
            visible_count=5,
            on_select=lambda item: (on_select or (lambda show: None))(item.value == "yes"),
            on_cancel=on_cancel,
        )
        self.select_list.set_selected_index(0 if current_value else 1)

    def get_select_list(self) -> SelectList:
        return self.select_list

    def handle_input(self, key_data: str) -> None:
        self.select_list.handle_input(key_data)

    def render(self) -> list[str]:
        return ["Show images", *self.select_list.render()]


@dataclass
class ModelItem:
    provider: str
    id: str
    model: Any


def _full_model_id(model: Any) -> str:
    return f"{_get_attr_or_key(model, 'provider')}/{_get_attr_or_key(model, 'id')}"


class ModelSelectorComponent:
    def __init__(
        self,
        current_model: Any | None,
        settings_manager: Any,
        model_registry: Any,
        scoped_models: list[dict[str, Any]] | None = None,
        on_select: Callable[[Any], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        initial_search_input: str | None = None,
    ) -> None:
        self.current_model = current_model
        self.settings_manager = settings_manager
        self.model_registry = model_registry
        self.scoped_models = scoped_models or []
        self.on_select = on_select or (lambda model: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.scope = "scoped" if self.scoped_models else "all"
        self.search_value = initial_search_input or ""
        self.selected_index = 0
        self.error_message: str | None = None
        self.all_models: list[ModelItem] = []
        self.scoped_model_items: list[ModelItem] = []
        self.active_models: list[ModelItem] = []
        self.filtered_models: list[ModelItem] = []
        self.load_models()

    def load_models(self) -> None:
        refresh = getattr(self.model_registry, "refresh", None)
        if callable(refresh):
            refresh()
        get_error = getattr(self.model_registry, "get_error", None)
        if callable(get_error):
            self.error_message = get_error()
        try:
            available = self.model_registry.get_available()
        except AttributeError:
            available = self.model_registry.get_models()
        if hasattr(available, "__await__"):
            raise RuntimeError("ModelSelectorComponent.load_models requires a synchronous registry in Python tests")

        models = [
            ModelItem(str(_get_attr_or_key(model, "provider")), str(_get_attr_or_key(model, "id")), model)
            for model in available
        ]
        self.all_models = self._sort_models(models)
        self.scoped_model_items = []
        for scoped in self.scoped_models:
            model = _get_attr_or_key(scoped, "model")
            if model is not None:
                finder = getattr(self.model_registry, "find", None)
                refreshed = finder(_get_attr_or_key(model, "provider"), _get_attr_or_key(model, "id")) if callable(finder) else None
                selected_model = refreshed or model
                self.scoped_model_items.append(
                    ModelItem(str(_get_attr_or_key(selected_model, "provider")), str(_get_attr_or_key(selected_model, "id")), selected_model)
                )
        self.active_models = self.scoped_model_items if self.scope == "scoped" else self.all_models
        self.filter_models(self.search_value)
        for idx, item in enumerate(self.filtered_models):
            if models_are_equal(self.current_model, item.model):
                self.selected_index = idx
                break

    def _sort_models(self, models: list[ModelItem]) -> list[ModelItem]:
        return sorted(
            models,
            key=lambda item: (
                0 if models_are_equal(self.current_model, item.model) else 1,
                item.provider,
                item.id,
            ),
        )

    def set_scope(self, scope: str) -> None:
        if scope not in {"all", "scoped"}:
            raise ValueError("scope must be 'all' or 'scoped'")
        if scope == "scoped" and not self.scoped_model_items:
            return
        self.scope = scope
        self.active_models = self.scoped_model_items if scope == "scoped" else self.all_models
        self.selected_index = 0
        self.filter_models(self.search_value)

    def filter_models(self, query: str) -> None:
        self.search_value = query
        self.filtered_models = (
            fuzzy_filter(
                self.active_models,
                query,
                lambda item: f"{item.id} {item.provider} {item.provider}/{item.id}",
            )
            if query
            else list(self.active_models)
        )
        self.selected_index = min(self.selected_index, max(0, len(self.filtered_models) - 1))

    def handle_input(self, key_data: str) -> None:
        if key_data in {"tab", "\t"} and self.scoped_model_items:
            self.set_scope("scoped" if self.scope == "all" else "all")
        elif key_data in {"up", "k", "\x1b[A"} and self.filtered_models:
            self.selected_index = (self.selected_index - 1) % len(self.filtered_models)
        elif key_data in {"down", "j", "\x1b[B"} and self.filtered_models:
            self.selected_index = (self.selected_index + 1) % len(self.filtered_models)
        elif key_data in {"\n", "enter", "return"}:
            selected = self.selected_model()
            if selected is not None:
                self._select(selected)
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()
        elif key_data == "backspace":
            self.filter_models(self.search_value[:-1])
        else:
            self.filter_models(self.search_value + key_data)

    def selected_model(self) -> Any | None:
        if not self.filtered_models:
            return None
        return self.filtered_models[self.selected_index].model

    def _select(self, model: Any) -> None:
        setter = getattr(self.settings_manager, "set_default_model_and_provider", None)
        if callable(setter):
            setter(_get_attr_or_key(model, "provider"), _get_attr_or_key(model, "id"))
        self.on_select(model)

    def render(self) -> list[str]:
        lines = [f"Scope: {self.scope}", f"Search: {self.search_value}"]
        if self.error_message:
            lines.append(self.error_message)
        if not self.filtered_models:
            lines.append("  No matching models")
            return lines
        visible = self.filtered_models[:10]
        for idx, item in enumerate(visible):
            prefix = "→ " if idx == self.selected_index else "  "
            current = " ✓" if models_are_equal(self.current_model, item.model) else ""
            lines.append(f"{prefix}{item.id} [{item.provider}]{current}")
        selected = self.selected_model()
        if selected is not None:
            lines.append(f"  Model Name: {_get_attr_or_key(selected, 'name', _get_attr_or_key(selected, 'id'))}")
        return lines


EnabledIds = list[str] | None


def is_model_enabled(enabled_ids: EnabledIds, model_id: str) -> bool:
    return enabled_ids is None or model_id in enabled_ids


def toggle_model(enabled_ids: EnabledIds, model_id: str) -> EnabledIds:
    if enabled_ids is None:
        return [model_id]
    if model_id in enabled_ids:
        return [item for item in enabled_ids if item != model_id]
    return [*enabled_ids, model_id]


def enable_all_models(enabled_ids: EnabledIds, all_ids: list[str], target_ids: list[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        return None
    targets = target_ids or all_ids
    result = list(enabled_ids)
    for model_id in targets:
        if model_id not in result:
            result.append(model_id)
    return None if len(result) == len(all_ids) else result


def clear_models(enabled_ids: EnabledIds, all_ids: list[str], target_ids: list[str] | None = None) -> EnabledIds:
    if enabled_ids is None:
        return [model_id for model_id in all_ids if target_ids and model_id not in target_ids] if target_ids else []
    targets = set(target_ids or enabled_ids)
    return [model_id for model_id in enabled_ids if model_id not in targets]


def move_model(enabled_ids: EnabledIds, model_id: str, delta: int) -> EnabledIds:
    if enabled_ids is None:
        return None
    result = list(enabled_ids)
    try:
        index = result.index(model_id)
    except ValueError:
        return result
    new_index = index + delta
    if new_index < 0 or new_index >= len(result):
        return result
    result[index], result[new_index] = result[new_index], result[index]
    return result


def sorted_model_ids(enabled_ids: EnabledIds, all_ids: list[str]) -> list[str]:
    if enabled_ids is None:
        return list(all_ids)
    enabled_set = set(enabled_ids)
    return [*enabled_ids, *[model_id for model_id in all_ids if model_id not in enabled_set]]


@dataclass
class ScopedModelItem:
    full_id: str
    model: Any
    enabled: bool


class ScopedModelsSelectorComponent:
    """Enable, disable, order, and persist scoped models for model cycling."""

    def __init__(
        self,
        all_models: list[Any],
        enabled_model_ids: EnabledIds,
        on_change: Callable[[EnabledIds], None] | None = None,
        on_persist: Callable[[EnabledIds], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.models_by_id = {_full_model_id(model): model for model in all_models}
        self.all_ids = list(self.models_by_id.keys())
        self.enabled_ids: EnabledIds = None if enabled_model_ids is None else list(enabled_model_ids)
        self.on_change = on_change or (lambda enabled: None)
        self.on_persist = on_persist or (lambda enabled: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.search_value = ""
        self.filtered_items: list[ScopedModelItem] = []
        self.selected_index = 0
        self.is_dirty = False
        self.refresh()

    def build_items(self) -> list[ScopedModelItem]:
        return [
            ScopedModelItem(model_id, self.models_by_id[model_id], is_model_enabled(self.enabled_ids, model_id))
            for model_id in sorted_model_ids(self.enabled_ids, self.all_ids)
            if model_id in self.models_by_id
        ]

    def refresh(self) -> None:
        items = self.build_items()
        self.filtered_items = (
            fuzzy_filter(items, self.search_value, lambda item: f"{_get_attr_or_key(item.model, 'id')} {_get_attr_or_key(item.model, 'provider')}")
            if self.search_value
            else items
        )
        self.selected_index = min(self.selected_index, max(0, len(self.filtered_items) - 1))

    def _notify_change(self) -> None:
        self.on_change(None if self.enabled_ids is None else list(self.enabled_ids))

    def selected_item(self) -> ScopedModelItem | None:
        if not self.filtered_items:
            return None
        return self.filtered_items[self.selected_index]

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"} and self.filtered_items:
            self.selected_index = (self.selected_index - 1) % len(self.filtered_items)
        elif key_data in {"down", "j", "\x1b[B"} and self.filtered_items:
            self.selected_index = (self.selected_index + 1) % len(self.filtered_items)
        elif key_data in {"\n", "enter", "return"}:
            item = self.selected_item()
            if item:
                self.enabled_ids = toggle_model(self.enabled_ids, item.full_id)
                self.is_dirty = True
                self.refresh()
                self._notify_change()
        elif key_data in {"enable_all", "a"}:
            targets = [item.full_id for item in self.filtered_items] if self.search_value else None
            self.enabled_ids = enable_all_models(self.enabled_ids, self.all_ids, targets)
            self.is_dirty = True
            self.refresh()
            self._notify_change()
        elif key_data in {"clear_all", "c"}:
            targets = [item.full_id for item in self.filtered_items] if self.search_value else None
            self.enabled_ids = clear_models(self.enabled_ids, self.all_ids, targets)
            self.is_dirty = True
            self.refresh()
            self._notify_change()
        elif key_data in {"toggle_provider", "p"}:
            item = self.selected_item()
            if item:
                provider = _get_attr_or_key(item.model, "provider")
                provider_ids = [model_id for model_id, model in self.models_by_id.items() if _get_attr_or_key(model, "provider") == provider]
                all_enabled = all(is_model_enabled(self.enabled_ids, model_id) for model_id in provider_ids)
                self.enabled_ids = (
                    clear_models(self.enabled_ids, self.all_ids, provider_ids)
                    if all_enabled
                    else enable_all_models(self.enabled_ids, self.all_ids, provider_ids)
                )
                self.is_dirty = True
                self.refresh()
                self._notify_change()
        elif key_data in {"reorder_up", "["}:
            item = self.selected_item()
            if item and is_model_enabled(self.enabled_ids, item.full_id):
                self.enabled_ids = move_model(self.enabled_ids, item.full_id, -1)
                self.selected_index = max(0, self.selected_index - 1)
                self.is_dirty = True
                self.refresh()
                self._notify_change()
        elif key_data in {"reorder_down", "]"}:
            item = self.selected_item()
            if item and is_model_enabled(self.enabled_ids, item.full_id):
                self.enabled_ids = move_model(self.enabled_ids, item.full_id, 1)
                self.selected_index = min(max(0, len(self.filtered_items) - 1), self.selected_index + 1)
                self.is_dirty = True
                self.refresh()
                self._notify_change()
        elif key_data in {"save", "ctrl+s"}:
            self.on_persist(None if self.enabled_ids is None else list(self.enabled_ids))
            self.is_dirty = False
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()
        elif key_data == "backspace":
            self.search_value = self.search_value[:-1]
            self.refresh()
        else:
            self.search_value += key_data
            self.refresh()

    def render(self) -> list[str]:
        enabled_count = len(self.all_ids) if self.enabled_ids is None else len(self.enabled_ids)
        lines = [f"Model Configuration ({'all enabled' if self.enabled_ids is None else f'{enabled_count}/{len(self.all_ids)} enabled'})"]
        if not self.filtered_items:
            lines.append("  No matching models")
            return lines
        for idx, item in enumerate(self.filtered_items[:8]):
            prefix = "→ " if idx == self.selected_index else "  "
            status = "" if self.enabled_ids is None else " ✓" if item.enabled else " ✗"
            lines.append(f"{prefix}{_get_attr_or_key(item.model, 'id')} [{_get_attr_or_key(item.model, 'provider')}]{status}")
        if self.is_dirty:
            lines.append("(unsaved)")
        return lines


@dataclass
class UserMessageItem:
    id: str
    text: str
    timestamp: str | None = None


class UserMessageSelectorComponent:
    def __init__(
        self,
        messages: list[UserMessageItem | dict[str, Any]],
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        initial_selected_id: str | None = None,
    ) -> None:
        self.messages = [
            msg if isinstance(msg, UserMessageItem) else UserMessageItem(str(msg["id"]), str(msg.get("text", "")), msg.get("timestamp"))
            for msg in messages
        ]
        self.on_select = on_select or (lambda entry_id: None)
        self.on_cancel = on_cancel or (lambda: None)
        initial_index = next((idx for idx, msg in enumerate(self.messages) if msg.id == initial_selected_id), -1)
        self.selected_index = initial_index if initial_index >= 0 else max(0, len(self.messages) - 1)

    def selected_message(self) -> UserMessageItem | None:
        if not self.messages:
            return None
        return self.messages[self.selected_index]

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"} and self.messages:
            self.selected_index = (self.selected_index - 1) % len(self.messages)
        elif key_data in {"down", "j", "\x1b[B"} and self.messages:
            self.selected_index = (self.selected_index + 1) % len(self.messages)
        elif key_data in {"\n", "enter", "return"}:
            selected = self.selected_message()
            if selected:
                self.on_select(selected.id)
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()

    def render(self, width: int | None = None) -> list[str]:
        lines = ["Fork from Message"]
        if not self.messages:
            lines.append("  No user messages found")
            return lines
        for idx, msg in enumerate(self.messages[-10:]):
            absolute_idx = len(self.messages) - min(10, len(self.messages)) + idx
            prefix = "› " if absolute_idx == self.selected_index else "  "
            text = msg.text.replace("\n", " ").strip()
            if width is not None:
                text = text[: max(0, width - 2)]
            lines.append(prefix + text)
            lines.append(f"  Message {absolute_idx + 1} of {len(self.messages)}")
        return lines


__all__ = [
    "LEVEL_DESCRIPTIONS",
    "ModelItem",
    "ModelSelectorComponent",
    "ScopedModelItem",
    "ScopedModelsSelectorComponent",
    "SelectItem",
    "SelectList",
    "ShowImagesSelectorComponent",
    "ThemeSelectorComponent",
    "ThinkingSelectorComponent",
    "UserMessageItem",
    "UserMessageSelectorComponent",
    "clear_models",
    "enable_all_models",
    "fuzzy_filter",
    "fuzzy_score",
    "is_model_enabled",
    "models_are_equal",
    "move_model",
    "sorted_model_ids",
    "toggle_model",
]
