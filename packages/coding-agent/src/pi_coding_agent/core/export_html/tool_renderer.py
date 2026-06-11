"""Custom tool HTML renderer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .ansi_to_html import ansi_lines_to_html

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[\d;]*m")


@dataclass
class RenderedToolHtml:
    call_html: str | None = None
    result_html_collapsed: str | None = None
    result_html_expanded: str | None = None


@dataclass
class ToolRenderContext:
    args: Any
    tool_call_id: str
    cwd: str
    state: dict[str, Any] = field(default_factory=dict)
    last_component: Any | None = None
    execution_started: bool = True
    args_complete: bool = True
    is_partial: bool = False
    expanded: bool = False
    show_images: bool = False
    is_error: bool = False

    def invalidate(self) -> None:
        return


def _render_component_lines(component: Any, width: int) -> list[str]:
    rendered = component.render(width) if hasattr(component, "render") else component
    if isinstance(rendered, str):
        return rendered.splitlines() or [rendered]
    return [str(line) for line in (rendered or [])]


def _is_blank_rendered_line(line: str) -> bool:
    return len(_ANSI_ESCAPE_RE.sub("", line).strip()) == 0


def trim_rendered_result_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and _is_blank_rendered_line(lines[start]):
        start += 1
    while end > start and _is_blank_rendered_line(lines[end - 1]):
        end -= 1
    return lines[start:end]


class ToolHtmlRenderer:
    def __init__(
        self,
        *,
        get_tool_definition: Callable[[str], Any | None],
        theme: Any,
        cwd: str,
        width: int = 100,
    ) -> None:
        self.get_tool_definition = get_tool_definition
        self.theme = theme
        self.cwd = cwd
        self.width = width
        self.rendered_call_components: dict[str, Any] = {}
        self.rendered_result_components: dict[str, Any] = {}
        self.rendered_states: dict[str, dict[str, Any]] = {}
        self.rendered_args: dict[str, Any] = {}

    def _state(self, tool_call_id: str) -> dict[str, Any]:
        return self.rendered_states.setdefault(tool_call_id, {})

    def _context(
        self,
        tool_call_id: str,
        *,
        last_component: Any | None,
        expanded: bool,
        is_partial: bool,
        is_error: bool,
    ) -> ToolRenderContext:
        return ToolRenderContext(
            args=self.rendered_args.get(tool_call_id),
            tool_call_id=tool_call_id,
            cwd=self.cwd,
            last_component=last_component,
            state=self._state(tool_call_id),
            is_partial=is_partial,
            expanded=expanded,
            is_error=is_error,
        )

    def render_call(self, tool_call_id: str, tool_name: str, args: Any) -> str | None:
        try:
            self.rendered_args[tool_call_id] = args
            tool_def = self.get_tool_definition(tool_name)
            render_call = getattr(tool_def, "render_call", None) or getattr(tool_def, "renderCall", None)
            if not callable(render_call):
                return None
            component = render_call(
                args,
                self.theme,
                self._context(
                    tool_call_id,
                    last_component=self.rendered_call_components.get(tool_call_id),
                    expanded=False,
                    is_partial=True,
                    is_error=False,
                ),
            )
            self.rendered_call_components[tool_call_id] = component
            return ansi_lines_to_html(_render_component_lines(component, self.width))
        except Exception:
            return None

    def render_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: list[dict[str, Any]],
        details: Any,
        is_error: bool,
    ) -> dict[str, str] | None:
        try:
            tool_def = self.get_tool_definition(tool_name)
            render_result = getattr(tool_def, "render_result", None) or getattr(tool_def, "renderResult", None)
            if not callable(render_result):
                return None
            agent_tool_result = {"content": result, "details": details, "isError": is_error}

            collapsed_component = render_result(
                agent_tool_result,
                {"expanded": False, "isPartial": False},
                self.theme,
                self._context(
                    tool_call_id,
                    last_component=self.rendered_result_components.get(tool_call_id),
                    expanded=False,
                    is_partial=False,
                    is_error=is_error,
                ),
            )
            self.rendered_result_components[tool_call_id] = collapsed_component
            collapsed = ansi_lines_to_html(trim_rendered_result_lines(_render_component_lines(collapsed_component, self.width)))

            expanded_component = render_result(
                agent_tool_result,
                {"expanded": True, "isPartial": False},
                self.theme,
                self._context(
                    tool_call_id,
                    last_component=self.rendered_result_components.get(tool_call_id),
                    expanded=True,
                    is_partial=False,
                    is_error=is_error,
                ),
            )
            self.rendered_result_components[tool_call_id] = expanded_component
            expanded = ansi_lines_to_html(trim_rendered_result_lines(_render_component_lines(expanded_component, self.width)))

            output = {"expanded": expanded}
            if collapsed and collapsed != expanded:
                output["collapsed"] = collapsed
            return output
        except Exception:
            return None


def create_tool_html_renderer(**deps: Any) -> ToolHtmlRenderer:
    return ToolHtmlRenderer(**deps)


__all__ = [
    "RenderedToolHtml",
    "ToolHtmlRenderer",
    "ToolRenderContext",
    "create_tool_html_renderer",
    "trim_rendered_result_lines",
]
