"""
Session tree selector component.

Ports the deterministic navigation/filter/search contract from the TypeScript
tree selector into a Python component usable by tests and CLI glue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

FilterMode = Literal["default", "no-tools", "user-only", "labeled-only", "all"]


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _entry_id(entry: Any) -> str:
    return str(_get_attr_or_key(entry, "id", ""))


def _entry_parent_id(entry: Any) -> str | None:
    value = _get_attr_or_key(entry, "parentId", _get_attr_or_key(entry, "parent_id", None))
    return str(value) if value is not None else None


def _entry_type(entry: Any) -> str:
    return str(_get_attr_or_key(entry, "type", ""))


def _entry_message(entry: Any) -> Any:
    return _get_attr_or_key(entry, "message", {})


def _message_role(message: Any) -> str:
    return str(_get_attr_or_key(message, "role", ""))


def _extract_content(content: Any, max_len: int = 200) -> str:
    if isinstance(content, str):
        return content[:max_len]
    if not isinstance(content, list):
        return ""
    result = ""
    for item in content:
        if _get_attr_or_key(item, "type") == "text":
            result += str(_get_attr_or_key(item, "text", ""))
            if len(result) >= max_len:
                return result[:max_len]
    return result


@dataclass
class SessionTreeNode:
    entry: Any
    children: list["SessionTreeNode"] = field(default_factory=list)
    label: str | None = None
    label_timestamp: str | None = None


@dataclass
class FlatTreeNode:
    node: SessionTreeNode
    depth: int


def build_tree_from_entries(entries: list[Any]) -> list[SessionTreeNode]:
    by_id = { _entry_id(entry): SessionTreeNode(entry=entry, label=_get_attr_or_key(entry, "label", None)) for entry in entries }
    roots: list[SessionTreeNode] = []
    for entry in entries:
        node = by_id[_entry_id(entry)]
        parent_id = _entry_parent_id(entry)
        if parent_id and parent_id in by_id:
            by_id[parent_id].children.append(node)
        else:
            roots.append(node)
    return roots


class TreeSelectorComponent:
    def __init__(
        self,
        tree: list[SessionTreeNode] | list[Any],
        current_leaf_id: str | None = None,
        max_visible_lines: int = 12,
        initial_selected_id: str | None = None,
        initial_filter_mode: FilterMode = "default",
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_label_edit: Callable[[str, str | None], None] | None = None,
    ) -> None:
        if tree and not isinstance(tree[0], SessionTreeNode):
            self.roots = build_tree_from_entries(tree)  # type: ignore[arg-type]
        else:
            self.roots = tree  # type: ignore[assignment]
        self.current_leaf_id = current_leaf_id
        self.max_visible_lines = max_visible_lines
        self.filter_mode: FilterMode = initial_filter_mode
        self.search_query = ""
        self.on_select = on_select or (lambda entry_id: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.on_label_edit = on_label_edit or (lambda entry_id, label: None)
        self.flat_nodes = self._flatten()
        self.filtered_nodes: list[FlatTreeNode] = []
        self.selected_index = 0
        self.apply_filter()
        target = initial_selected_id or current_leaf_id
        if target:
            self.selected_index = self.find_nearest_visible_index(target)

    def _flatten(self) -> list[FlatTreeNode]:
        result: list[FlatTreeNode] = []
        stack = [(root, 0) for root in reversed(self.roots)]
        while stack:
            node, depth = stack.pop()
            result.append(FlatTreeNode(node=node, depth=depth))
            for child in reversed(node.children):
                stack.append((child, depth + 1))
        return result

    def _active_path_ids(self) -> set[str]:
        if not self.current_leaf_id:
            return set()
        by_id = {_entry_id(flat.node.entry): flat for flat in self.flat_nodes}
        active: set[str] = set()
        current = self.current_leaf_id
        while current:
            active.add(current)
            flat = by_id.get(current)
            if not flat:
                break
            current = _entry_parent_id(flat.node.entry)
        return active

    def _has_text_content(self, content: Any) -> bool:
        return bool(_extract_content(content).strip())

    def _passes_filter_mode(self, flat: FlatTreeNode) -> bool:
        entry = flat.node.entry
        entry_type = _entry_type(entry)
        message = _entry_message(entry)
        role = _message_role(message)
        settings_entry = entry_type in {"label", "custom", "model_change", "thinking_level_change", "session_info"}
        if self.filter_mode == "user-only":
            return entry_type == "message" and role == "user"
        if self.filter_mode == "no-tools":
            return not settings_entry and not (entry_type == "message" and role == "toolResult")
        if self.filter_mode == "labeled-only":
            return flat.node.label is not None
        if self.filter_mode == "all":
            return True
        return not settings_entry

    def searchable_text(self, node: SessionTreeNode) -> str:
        entry = node.entry
        entry_type = _entry_type(entry)
        parts = [node.label or "", entry_type]
        if entry_type == "message":
            message = _entry_message(entry)
            parts.append(_message_role(message))
            parts.append(_extract_content(_get_attr_or_key(message, "content", "")))
            if _message_role(message) == "bashExecution":
                parts.append(str(_get_attr_or_key(message, "command", "")))
        elif entry_type == "branch_summary":
            parts.append(str(_get_attr_or_key(entry, "summary", "")))
        elif entry_type == "model_change":
            parts.append(str(_get_attr_or_key(entry, "modelId", _get_attr_or_key(entry, "model_id", ""))))
        elif entry_type == "thinking_level_change":
            parts.append(str(_get_attr_or_key(entry, "thinkingLevel", _get_attr_or_key(entry, "thinking_level", ""))))
        elif entry_type == "session_info":
            parts.append(str(_get_attr_or_key(entry, "name", "")))
        return " ".join(parts)

    def apply_filter(self) -> None:
        tokens = [token for token in self.search_query.lower().split() if token]
        current_leaf = self.current_leaf_id
        filtered: list[FlatTreeNode] = []
        for flat in self.flat_nodes:
            entry = flat.node.entry
            is_current = _entry_id(entry) == current_leaf
            if _entry_type(entry) == "message" and _message_role(_entry_message(entry)) == "assistant" and not is_current:
                message = _entry_message(entry)
                stop_reason = _get_attr_or_key(message, "stopReason", _get_attr_or_key(message, "stop_reason", None))
                if not self._has_text_content(_get_attr_or_key(message, "content", "")) and stop_reason in {None, "stop", "toolUse"}:
                    continue
            if not self._passes_filter_mode(flat):
                continue
            if tokens:
                text = self.searchable_text(flat.node).lower()
                if not all(token in text for token in tokens):
                    continue
            filtered.append(flat)
        self.filtered_nodes = filtered
        self.selected_index = min(self.selected_index, max(0, len(self.filtered_nodes) - 1))

    def find_nearest_visible_index(self, entry_id: str | None) -> int:
        if not self.filtered_nodes:
            return 0
        visible = {_entry_id(flat.node.entry): idx for idx, flat in enumerate(self.filtered_nodes)}
        by_id = {_entry_id(flat.node.entry): flat for flat in self.flat_nodes}
        current = entry_id
        while current:
            if current in visible:
                return visible[current]
            flat = by_id.get(current)
            current = _entry_parent_id(flat.node.entry) if flat else None
        return len(self.filtered_nodes) - 1

    def selected_node(self) -> SessionTreeNode | None:
        if not self.filtered_nodes:
            return None
        return self.filtered_nodes[self.selected_index].node

    def update_node_label(self, entry_id: str, label: str | None, label_timestamp: str | None = None) -> None:
        for flat in self.flat_nodes:
            if _entry_id(flat.node.entry) == entry_id:
                flat.node.label = label
                flat.node.label_timestamp = label_timestamp
                break
        self.apply_filter()

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"} and self.filtered_nodes:
            self.selected_index = (self.selected_index - 1) % len(self.filtered_nodes)
        elif key_data in {"down", "j", "\x1b[B"} and self.filtered_nodes:
            self.selected_index = (self.selected_index + 1) % len(self.filtered_nodes)
        elif key_data in {"page_up"} and self.filtered_nodes:
            self.selected_index = max(0, self.selected_index - self.max_visible_lines)
        elif key_data in {"page_down"} and self.filtered_nodes:
            self.selected_index = min(len(self.filtered_nodes) - 1, self.selected_index + self.max_visible_lines)
        elif key_data in {"\n", "enter", "return"}:
            selected = self.selected_node()
            if selected:
                self.on_select(_entry_id(selected.entry))
        elif key_data in {"escape", "esc", "\x1b"}:
            if self.search_query:
                self.search_query = ""
                self.apply_filter()
            else:
                self.on_cancel()
        elif key_data in {"filter_default"}:
            self.filter_mode = "default"
            self.apply_filter()
        elif key_data in {"filter_no_tools"}:
            self.filter_mode = "default" if self.filter_mode == "no-tools" else "no-tools"
            self.apply_filter()
        elif key_data in {"filter_user_only"}:
            self.filter_mode = "default" if self.filter_mode == "user-only" else "user-only"
            self.apply_filter()
        elif key_data in {"filter_labeled_only"}:
            self.filter_mode = "default" if self.filter_mode == "labeled-only" else "labeled-only"
            self.apply_filter()
        elif key_data in {"filter_all"}:
            self.filter_mode = "default" if self.filter_mode == "all" else "all"
            self.apply_filter()
        elif key_data == "backspace":
            self.search_query = self.search_query[:-1]
            self.apply_filter()
        else:
            self.search_query += key_data
            self.apply_filter()

    def entry_display_text(self, node: SessionTreeNode) -> str:
        entry = node.entry
        entry_type = _entry_type(entry)
        if entry_type == "message":
            message = _entry_message(entry)
            role = _message_role(message)
            if role in {"user", "assistant"}:
                text = _extract_content(_get_attr_or_key(message, "content", "")).replace("\n", " ").strip()
                return f"{role}: {text}" if text else f"{role}: (no content)"
            if role == "bashExecution":
                return f"[bash]: {_get_attr_or_key(message, 'command', '')}"
            return f"[{role}]"
        if entry_type == "compaction":
            return "[compaction]"
        if entry_type == "branch_summary":
            return f"[branch summary]: {_get_attr_or_key(entry, 'summary', '')}"
        if entry_type == "model_change":
            return f"[model: {_get_attr_or_key(entry, 'modelId', _get_attr_or_key(entry, 'model_id', ''))}]"
        if entry_type == "thinking_level_change":
            return f"[thinking: {_get_attr_or_key(entry, 'thinkingLevel', _get_attr_or_key(entry, 'thinking_level', ''))}]"
        if entry_type == "session_info":
            return f"[title: {_get_attr_or_key(entry, 'name', '')}]"
        return f"[{entry_type}]"

    def render(self, width: int | None = None) -> list[str]:
        if not self.filtered_nodes:
            return ["  No entries found", f"  (0/0) [{self.filter_mode}]"]
        active = self._active_path_ids()
        start = max(0, min(self.selected_index - self.max_visible_lines // 2, len(self.filtered_nodes) - self.max_visible_lines))
        end = min(start + self.max_visible_lines, len(self.filtered_nodes))
        lines: list[str] = []
        for idx in range(start, end):
            flat = self.filtered_nodes[idx]
            entry_id = _entry_id(flat.node.entry)
            cursor = "› " if idx == self.selected_index else "  "
            path_marker = "• " if entry_id in active else ""
            label = f"[{flat.node.label}] " if flat.node.label else ""
            line = cursor + "  " * flat.depth + path_marker + label + self.entry_display_text(flat.node)
            lines.append(line[:width] if width is not None else line)
        lines.append(f"  ({self.selected_index + 1}/{len(self.filtered_nodes)}) [{self.filter_mode}]")
        return lines


__all__ = [
    "FilterMode",
    "FlatTreeNode",
    "SessionTreeNode",
    "TreeSelectorComponent",
    "build_tree_from_entries",
]
