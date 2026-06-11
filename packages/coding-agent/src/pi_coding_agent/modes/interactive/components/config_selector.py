"""Resource configuration selector parity helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pi_coding_agent.core.package_manager import ResolvedPaths, ResolvedResource

from .text_input import TextInput

ResourceType = Literal["extensions", "skills", "prompts", "themes"]

RESOURCE_TYPE_LABELS: dict[str, str] = {
    "extensions": "Extensions",
    "skills": "Skills",
    "prompts": "Prompts",
    "themes": "Themes",
}


@dataclass
class ResourceItem:
    path: str
    enabled: bool
    metadata: Any
    resource_type: ResourceType
    display_name: str
    group_key: str
    subgroup_key: str


@dataclass
class ResourceSubgroup:
    type: ResourceType
    label: str
    items: list[ResourceItem] = field(default_factory=list)


@dataclass
class ResourceGroup:
    key: str
    label: str
    scope: str
    origin: str
    source: str
    subgroups: list[ResourceSubgroup] = field(default_factory=list)


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _format_base_dir(base_dir: str) -> str:
    home = str(Path.home())
    display = base_dir
    if base_dir == home:
        display = "~"
    elif base_dir.startswith(home):
        display = "~" + base_dir[len(home):]
    return display.replace("\\", "/").rstrip("/") + "/"


def get_group_label(metadata: Any) -> str:
    origin = _get_attr_or_key(metadata, "origin")
    source = _get_attr_or_key(metadata, "source")
    scope = _get_attr_or_key(metadata, "scope")
    base_dir = _get_attr_or_key(metadata, "base_dir", _get_attr_or_key(metadata, "baseDir"))
    if origin == "package":
        return f"{source} ({scope})"
    if source == "auto":
        if base_dir:
            return f"{'User' if scope == 'user' else 'Project'} ({_format_base_dir(str(base_dir))})"
        return "User (~/.pi/agent/)" if scope == "user" else "Project (.pi/)"
    return "User settings" if scope == "user" else "Project settings"


def _display_name(path: str, resource_type: str) -> str:
    file_name = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if resource_type == "extensions" and parent != "extensions":
        return f"{parent}/{file_name}"
    if resource_type == "skills" and file_name == "SKILL.md":
        return parent
    return file_name


def build_resource_groups(resolved_paths: ResolvedPaths | dict[str, list[Any]]) -> list[ResourceGroup]:
    group_map: dict[str, ResourceGroup] = {}

    def resources(resource_type: ResourceType) -> list[Any]:
        if isinstance(resolved_paths, dict):
            return list(resolved_paths.get(resource_type, []))
        return list(getattr(resolved_paths, resource_type, []))

    for resource_type in ("extensions", "skills", "prompts", "themes"):
        for resource in resources(resource_type):
            path = str(_get_attr_or_key(resource, "path"))
            metadata = _get_attr_or_key(resource, "metadata")
            enabled = bool(_get_attr_or_key(resource, "enabled"))
            origin = str(_get_attr_or_key(metadata, "origin"))
            scope = str(_get_attr_or_key(metadata, "scope"))
            source = str(_get_attr_or_key(metadata, "source"))
            base_dir = str(_get_attr_or_key(metadata, "base_dir", _get_attr_or_key(metadata, "baseDir", "")) or "")
            group_key = f"{origin}:{scope}:{source}:{base_dir}"
            group = group_map.setdefault(
                group_key,
                ResourceGroup(
                    key=group_key,
                    label=get_group_label(metadata),
                    scope=scope,
                    origin=origin,
                    source=source,
                ),
            )
            subgroup = next((sg for sg in group.subgroups if sg.type == resource_type), None)
            if subgroup is None:
                subgroup = ResourceSubgroup(resource_type, RESOURCE_TYPE_LABELS[resource_type])
                group.subgroups.append(subgroup)
            subgroup_key = f"{group_key}:{resource_type}"
            subgroup.items.append(
                ResourceItem(
                    path=path,
                    enabled=enabled,
                    metadata=metadata,
                    resource_type=resource_type,
                    display_name=_display_name(path, resource_type),
                    group_key=group_key,
                    subgroup_key=subgroup_key,
                )
            )

    groups = list(group_map.values())
    groups.sort(key=lambda group: (0 if group.origin == "package" else 1, 0 if group.scope == "user" else 1, group.source))
    type_order = {"extensions": 0, "skills": 1, "prompts": 2, "themes": 3}
    for group in groups:
        group.subgroups.sort(key=lambda sg: type_order[sg.type])
        for subgroup in group.subgroups:
            subgroup.items.sort(key=lambda item: item.display_name)
    return groups


class ConfigSelectorComponent:
    def __init__(
        self,
        resolved_paths: ResolvedPaths | dict[str, list[Any]],
        settings_manager: Any,
        cwd: str,
        agent_dir: str,
        on_close: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        request_render: Callable[[], None] | None = None,
        terminal_height: int | None = None,
    ) -> None:
        self.groups = build_resource_groups(resolved_paths)
        self.settings_manager = settings_manager
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.on_close = on_close or (lambda: None)
        self.on_exit = on_exit or (lambda: None)
        self.request_render = request_render or (lambda: None)
        self.max_visible = max(5, (terminal_height or 24) - 8)
        self.search_input = TextInput()
        self.flat_items: list[tuple[str, ResourceGroup | ResourceSubgroup | ResourceItem]] = []
        self.filtered_items: list[tuple[str, ResourceGroup | ResourceSubgroup | ResourceItem]] = []
        self.selected_index = 0
        self._build_flat_items()
        self.filter_items("")

    def _build_flat_items(self) -> None:
        self.flat_items = []
        for group in self.groups:
            self.flat_items.append(("group", group))
            for subgroup in group.subgroups:
                self.flat_items.append(("subgroup", subgroup))
                for item in subgroup.items:
                    self.flat_items.append(("item", item))

    def _select_first_item(self) -> None:
        self.selected_index = next((idx for idx, row in enumerate(self.filtered_items) if row[0] == "item"), 0)

    def filter_items(self, query: str) -> None:
        self.search_input.set_value(query)
        if not query.strip():
            self.filtered_items = list(self.flat_items)
            self._select_first_item()
            return
        lower = query.lower()
        matching_paths = {
            item.path
            for kind, item in self.flat_items
            if kind == "item"
            and isinstance(item, ResourceItem)
            and (
                lower in item.display_name.lower()
                or lower in item.resource_type.lower()
                or lower in item.path.lower()
            )
        }
        matching_groups = {item.group_key for kind, item in self.flat_items if kind == "item" and isinstance(item, ResourceItem) and item.path in matching_paths}
        matching_subgroups = {item.subgroup_key for kind, item in self.flat_items if kind == "item" and isinstance(item, ResourceItem) and item.path in matching_paths}
        result: list[tuple[str, ResourceGroup | ResourceSubgroup | ResourceItem]] = []
        current_group_key = ""
        for kind, obj in self.flat_items:
            if kind == "group" and isinstance(obj, ResourceGroup):
                current_group_key = obj.key
                if obj.key in matching_groups:
                    result.append((kind, obj))
            elif kind == "subgroup" and isinstance(obj, ResourceSubgroup):
                if f"{current_group_key}:{obj.type}" in matching_subgroups:
                    result.append((kind, obj))
            elif kind == "item" and isinstance(obj, ResourceItem) and obj.path in matching_paths:
                result.append((kind, obj))
        self.filtered_items = result
        self._select_first_item()

    def selected_item(self) -> ResourceItem | None:
        if not self.filtered_items:
            return None
        kind, obj = self.filtered_items[self.selected_index]
        return obj if kind == "item" and isinstance(obj, ResourceItem) else None

    def _find_next_item(self, direction: int) -> int:
        idx = self.selected_index + direction
        while 0 <= idx < len(self.filtered_items):
            if self.filtered_items[idx][0] == "item":
                return idx
            idx += direction
        return self.selected_index

    def toggle_selected(self) -> ResourceItem | None:
        item = self.selected_item()
        if item is None:
            return None
        item.enabled = not item.enabled
        self.request_render()
        return item

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"}:
            self.selected_index = self._find_next_item(-1)
        elif key_data in {"down", "j", "\x1b[B"}:
            self.selected_index = self._find_next_item(1)
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_close()
        elif key_data == "ctrl+c":
            self.on_exit()
        elif key_data in {" ", "\n", "enter", "return"}:
            self.toggle_selected()
        else:
            self.search_input.handle_input(key_data)
            self.filter_items(self.search_input.get_value())

    def render(self, width: int = 80) -> list[str]:
        lines = ["Resource Configuration", *self.search_input.render(width)]
        if not self.filtered_items:
            return [*lines, "No resources found"]
        for idx, (kind, obj) in enumerate(self.filtered_items[: self.max_visible]):
            if kind == "group" and isinstance(obj, ResourceGroup):
                lines.append(f"  {obj.label}")
            elif kind == "subgroup" and isinstance(obj, ResourceSubgroup):
                lines.append(f"    {obj.label}")
            elif kind == "item" and isinstance(obj, ResourceItem):
                cursor = "> " if idx == self.selected_index else "  "
                checkbox = "[x]" if obj.enabled else "[ ]"
                lines.append(f"{cursor}    {checkbox} {obj.display_name}")
        return lines


__all__ = [
    "ConfigSelectorComponent",
    "ResourceGroup",
    "ResourceItem",
    "ResourceSubgroup",
    "build_resource_groups",
    "get_group_label",
]
