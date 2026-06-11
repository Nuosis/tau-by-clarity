"""Session selector component parity contract."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .session_selector_search import NameFilter, SortMode, filter_and_sort_sessions, has_session_name
from .text_input import TextInput


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _modified_timestamp(session: Any) -> float:
    modified = _get_attr_or_key(session, "modified")
    if isinstance(modified, datetime):
        return modified.timestamp()
    if isinstance(modified, (int, float)):
        return float(modified)
    return 0.0


class SessionSelectorComponent:
    def __init__(
        self,
        sessions: list[Any] | None = None,
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        current_session_file_path: str | None = None,
        request_render: Callable[[], None] | None = None,
        rename_session: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        self.all_sessions = list(sessions or [])
        self.filtered_sessions: list[Any] = []
        self.selected_index = 0
        self.scope = "current"
        self.sort_mode: SortMode = "threaded"
        self.name_filter: NameFilter = "all"
        self.show_path = False
        self.confirming_delete_path: str | None = None
        self.current_session_file_path = current_session_file_path
        self.on_select = on_select or (lambda path: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.request_render = request_render or (lambda: None)
        self.rename_session = rename_session
        self.search_input = TextInput(on_submit=lambda _: self._submit())
        self.filter_sessions("")

    def filter_sessions(self, query: str) -> None:
        self.search_input.set_value(query)
        sessions = self.all_sessions
        if self.name_filter == "named":
            sessions = [session for session in sessions if has_session_name(session)]
        if not query.strip() and self.sort_mode in {"threaded", "recent"}:
            self.filtered_sessions = sorted(sessions, key=_modified_timestamp, reverse=True)
        else:
            self.filtered_sessions = filter_and_sort_sessions(sessions, query, self.sort_mode, "all")
        self.selected_index = max(0, min(self.selected_index, max(0, len(self.filtered_sessions) - 1)))

    def selected_session(self) -> Any | None:
        if not self.filtered_sessions:
            return None
        return self.filtered_sessions[self.selected_index]

    def get_selected_session_path(self) -> str | None:
        selected = self.selected_session()
        return str(_get_attr_or_key(selected, "path")) if selected is not None else None

    def _submit(self) -> None:
        path = self.get_selected_session_path()
        if path:
            self.on_select(path)

    def toggle_sort(self) -> None:
        order: list[SortMode] = ["threaded", "recent", "relevance"]
        self.sort_mode = order[(order.index(self.sort_mode) + 1) % len(order)]
        self.filter_sessions(self.search_input.get_value())

    def toggle_name_filter(self) -> None:
        self.name_filter = "named" if self.name_filter == "all" else "all"
        self.filter_sessions(self.search_input.get_value())

    def handle_input(self, key_data: str) -> None:
        if self.confirming_delete_path is not None:
            if key_data in {"\n", "enter", "return"}:
                self.confirming_delete_path = None
            elif key_data in {"escape", "esc", "\x1b"}:
                self.confirming_delete_path = None
            return
        if key_data in {"up", "k", "\x1b[A"} and self.filtered_sessions:
            self.selected_index = max(0, self.selected_index - 1)
        elif key_data in {"down", "j", "\x1b[B"} and self.filtered_sessions:
            self.selected_index = min(len(self.filtered_sessions) - 1, self.selected_index + 1)
        elif key_data == "tab":
            self.scope = "all" if self.scope == "current" else "current"
        elif key_data == "toggle_sort":
            self.toggle_sort()
        elif key_data == "toggle_named":
            self.toggle_name_filter()
        elif key_data == "toggle_path":
            self.show_path = not self.show_path
        elif key_data == "delete":
            self.confirming_delete_path = self.get_selected_session_path()
        elif key_data == "rename":
            path = self.get_selected_session_path()
            if path and self.rename_session:
                selected = self.selected_session()
                self.rename_session(path, _get_attr_or_key(selected, "name"))
        elif key_data in {"\n", "enter", "return"}:
            self._submit()
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()
        else:
            self.search_input.handle_input(key_data)
            self.filter_sessions(self.search_input.get_value())

    def render(self, width: int = 80) -> list[str]:
        title = "Resume Session (Current Folder)" if self.scope == "current" else "Resume Session (All)"
        lines = [title, *self.search_input.render(width)]
        if not self.filtered_sessions:
            lines.append("No sessions found")
            return lines
        for idx, session in enumerate(self.filtered_sessions[:10]):
            cursor = "› " if idx == self.selected_index else "  "
            label = _get_attr_or_key(session, "name") or _get_attr_or_key(session, "firstMessage", _get_attr_or_key(session, "first_message", ""))
            suffix = f" {_get_attr_or_key(session, 'path')}" if self.show_path else ""
            lines.append(f"{cursor}{label}{suffix}")
        return lines


__all__ = ["SessionSelectorComponent"]
