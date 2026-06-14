"""
TUI runner for interactive mode using pi_tui.

Wires up a ProcessTerminal + TUI + Editor to give a full interactive
experience aligned with the TypeScript coding agent interactive mode.
Falls back to a readline loop if pi_tui cannot start (e.g. not a TTY).

Slash commands: /exit /clear /help /model /compact /thinking /session /tools
Footer:         model | thinking: off | ctx: 12% | tokens: 8k/64k
Ctrl+P:         cycle model
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Callable
from typing import TYPE_CHECKING

from pi_coding_agent.config import VERSION
from pi_coding_agent.core.extensions.types import ExtensionUIContext

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession


class InteractiveExtensionUIContext(ExtensionUIContext):
    """Extension UI context for the lightweight Python interactive TUI."""

    def __init__(
        self,
        *,
        append_history: Callable[[str], None],
        request_render: Callable[[], None],
        set_editor_text: Callable[[str], None],
        get_editor_text: Callable[[], str],
        set_title: Callable[[str], None] | None = None,
        set_status: Callable[[str, str | None], None] | None = None,
        set_widget: Callable[[str, list[str] | None, str], None] | None = None,
        add_autocomplete_provider: Callable[[Any], None] | None = None,
        add_terminal_input_listener: Callable[[Any], Callable[[], None]] | None = None,
        set_editor_component: Callable[[Any], None] | None = None,
        get_editor_component: Callable[[], Any] | None = None,
        show_select: Callable[[str, list[str], Any], Any] | None = None,
        show_input: Callable[[str, str | None, Any], Any] | None = None,
        show_editor: Callable[[str, str | None], Any] | None = None,
        show_custom: Callable[..., Any] | None = None,
        set_working_message: Callable[[str | None], None] | None = None,
        set_working_visible: Callable[[bool], None] | None = None,
        set_working_indicator: Callable[[Any], None] | None = None,
        set_hidden_thinking_label: Callable[[str | None], None] | None = None,
        set_footer: Callable[[Any], None] | None = None,
        set_header: Callable[[Any], None] | None = None,
        get_theme: Callable[[], Any] | None = None,
        get_all_themes: Callable[[], list[Any]] | None = None,
        get_theme_by_name: Callable[[str], Any] | None = None,
        set_theme: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        self._append_history = append_history
        self._request_render = request_render
        self._set_editor_text = set_editor_text
        self._get_editor_text = get_editor_text
        self._set_title = set_title
        self._set_status = set_status
        self._set_widget = set_widget
        self._add_autocomplete_provider = add_autocomplete_provider
        self._add_terminal_input_listener = add_terminal_input_listener
        self._set_editor_component = set_editor_component
        self._get_editor_component = get_editor_component
        self._show_select = show_select
        self._show_input = show_input
        self._show_editor = show_editor
        self._show_custom = show_custom
        self._set_working_message = set_working_message
        self._set_working_visible = set_working_visible
        self._set_working_indicator = set_working_indicator
        self._set_hidden_thinking_label = set_hidden_thinking_label
        self._set_footer = set_footer
        self._set_header = set_header
        self._get_theme = get_theme
        self._get_all_themes = get_all_themes
        self._get_theme_by_name = get_theme_by_name
        self._set_theme = set_theme

    async def select(self, title: str, options: list[str], opts: Any = None) -> str | None:
        if self._show_select:
            return await self._show_select(title, options, opts)
        self.notify(f"{title}: {', '.join(options)}")
        return None

    async def confirm(self, title: str, message: str, opts: Any = None) -> bool:
        return await self.select(f"{title}\n{message}", ["Yes", "No"], opts) == "Yes"

    async def input(self, title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
        if self._show_input:
            return await self._show_input(title, placeholder, opts)
        self.notify(title)
        return None

    def notify(self, message: str, notify_type: str | None = None) -> None:
        prefix = f"[{notify_type}] " if notify_type else ""
        self._append_history(f"{prefix}{message}")
        self._request_render()

    def on_terminal_input(self, handler: Any) -> Callable[[], None]:
        if self._add_terminal_input_listener:
            return self._add_terminal_input_listener(handler)
        return lambda: None

    def set_status(self, key: str, text: str | None) -> None:
        if self._set_status:
            self._set_status(key, text)
        self._request_render()

    def set_widget(self, key: str, content: Any, options: Any = None) -> None:
        if self._set_widget and (content is None or isinstance(content, list)):
            placement = _option_value(options, "placement", "aboveEditor")
            self._set_widget(key, content, placement)
            self._request_render()

    def set_working_message(self, message: str | None = None) -> None:
        if self._set_working_message:
            self._set_working_message(message)
        self._request_render()

    def set_working_visible(self, visible: bool) -> None:
        if self._set_working_visible:
            self._set_working_visible(visible)
        self._request_render()

    def set_working_indicator(self, options: Any = None) -> None:
        if self._set_working_indicator:
            self._set_working_indicator(options)
        self._request_render()

    def set_hidden_thinking_label(self, label: str | None = None) -> None:
        if self._set_hidden_thinking_label:
            self._set_hidden_thinking_label(label)
        self._request_render()

    def set_footer(self, factory: Any = None) -> None:
        if self._set_footer:
            self._set_footer(factory)
        self._request_render()

    def set_header(self, factory: Any = None) -> None:
        if self._set_header:
            self._set_header(factory)
        self._request_render()

    def set_title(self, title: str) -> None:
        if self._set_title:
            self._set_title(title)

    def paste_to_editor(self, text: str) -> None:
        self._set_editor_text(self._get_editor_text() + text)
        self._request_render()

    def set_editor_text(self, text: str) -> None:
        self._set_editor_text(text)
        self._request_render()

    def get_editor_text(self) -> str:
        return self._get_editor_text()

    def get_tools_expanded(self) -> bool:
        return False

    def set_tools_expanded(self, expanded: bool) -> None:
        return None

    async def custom(self, *args: Any, **kwargs: Any) -> Any:
        if self._show_custom:
            return await self._show_custom(*args, **kwargs)
        return None

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        if self._show_editor:
            return await self._show_editor(title, prefill)
        self.notify(title)
        if prefill is not None:
            self.set_editor_text(prefill)
        return None

    def add_autocomplete_provider(self, factory: Any) -> None:
        if self._add_autocomplete_provider:
            self._add_autocomplete_provider(factory)

    def set_editor_component(self, factory: Any = None) -> None:
        if self._set_editor_component:
            self._set_editor_component(factory)

    def get_editor_component(self) -> Any:
        if self._get_editor_component:
            return self._get_editor_component()
        return None

    @property
    def theme(self) -> Any:
        if self._get_theme:
            return self._get_theme()
        return {}

    def get_all_themes(self) -> list[Any]:
        if self._get_all_themes:
            return self._get_all_themes()
        return []

    def get_theme(self, name: str) -> Any:
        if self._get_theme_by_name:
            return self._get_theme_by_name(name)
        return None

    def set_theme(self, theme_val: Any) -> dict[str, Any]:
        if self._set_theme:
            return self._set_theme(theme_val)
        return {"success": False, "error": "Theme switching not available"}

    def onTerminalInput(self, handler: Any) -> Callable[[], None]:
        return self.on_terminal_input(handler)

    def setStatus(self, key: str, text: str | None) -> None:
        return self.set_status(key, text)

    def setWidget(self, key: str, content: Any, options: Any = None) -> None:
        return self.set_widget(key, content, options)

    def setWorkingMessage(self, message: str | None = None) -> None:
        return self.set_working_message(message)

    def setWorkingVisible(self, visible: bool) -> None:
        return self.set_working_visible(visible)

    def setWorkingIndicator(self, options: Any = None) -> None:
        return self.set_working_indicator(options)

    def setHiddenThinkingLabel(self, label: str | None = None) -> None:
        return self.set_hidden_thinking_label(label)

    def setFooter(self, factory: Any = None) -> None:
        return self.set_footer(factory)

    def setHeader(self, factory: Any = None) -> None:
        return self.set_header(factory)

    def setTitle(self, title: str) -> None:
        return self.set_title(title)

    def pasteToEditor(self, text: str) -> None:
        return self.paste_to_editor(text)

    def setEditorText(self, text: str) -> None:
        return self.set_editor_text(text)

    def getEditorText(self) -> str:
        return self.get_editor_text()

    def addAutocompleteProvider(self, factory: Any) -> None:
        return self.add_autocomplete_provider(factory)

    def setEditorComponent(self, factory: Any = None) -> None:
        return self.set_editor_component(factory)

    def getEditorComponent(self) -> Any:
        return self.get_editor_component()

    def getAllThemes(self) -> list[Any]:
        return self.get_all_themes()

    def getTheme(self, name: str) -> Any:
        return self.get_theme(name)

    def setTheme(self, theme_val: Any) -> dict[str, Any]:
        return self.set_theme(theme_val)

    def getToolsExpanded(self) -> bool:
        return self.get_tools_expanded()

    def setToolsExpanded(self, expanded: bool) -> None:
        return self.set_tools_expanded(expanded)


def _option_value(options: Any, key: str, default: Any = None) -> Any:
    if isinstance(options, dict):
        return options.get(key, default)
    return getattr(options, key, default)


def _path_command_argument(text: str, command: str) -> str | None:
    """Parse a path argument the same way Node interactive mode does."""
    if text == command:
        return None
    if not text.startswith(f"{command} "):
        return None
    args_string = text[len(command) + 1:].lstrip()
    if not args_string:
        return None
    first_char = args_string[0]
    if first_char in {"'", '"'}:
        closing_quote_index = args_string.find(first_char, 1)
        if closing_quote_index < 0:
            return None
        return args_string[1:closing_quote_index]
    parts = args_string.split(maxsplit=1)
    return parts[0] if parts else None


def _signal_is_aborted(signal: Any) -> bool:
    return bool(_option_value(signal, "aborted", False))


def _add_abort_listener(signal: Any, callback: Callable[[], None]) -> Callable[[], None]:
    if signal is None:
        return lambda: None
    add_event_listener = getattr(signal, "addEventListener", None) or getattr(signal, "add_event_listener", None)
    remove_event_listener = (
        getattr(signal, "removeEventListener", None)
        or getattr(signal, "remove_event_listener", None)
    )
    if callable(add_event_listener):
        try:
            add_event_listener("abort", callback, {"once": True})
        except TypeError:
            add_event_listener("abort", callback)

        def cleanup() -> None:
            if callable(remove_event_listener):
                try:
                    remove_event_listener("abort", callback)
                except TypeError:
                    remove_event_listener(callback)

        return cleanup
    listeners = getattr(signal, "listeners", None)
    if isinstance(listeners, list):
        listeners.append(callback)

        def cleanup() -> None:
            try:
                listeners.remove(callback)
            except ValueError:
                pass

        return cleanup
    return lambda: None


def _extension_command_help_lines(extension_runner: Any, color: Callable[[str], str] | None = None) -> list[str]:
    if extension_runner is None or not hasattr(extension_runner, "get_registered_commands"):
        return []
    color = color or (lambda value: value)
    lines: list[str] = []
    for command in extension_runner.get_registered_commands():
        invocation = (
            getattr(command, "invocation_name", None)
            or getattr(command, "invocationName", None)
            or getattr(command, "name", "")
        )
        if not invocation:
            continue
        description = getattr(command, "description", "") or "Extension command"
        lines.append(f"  {color('/' + invocation)} — {description}")
    return lines


def _extension_shortcut_hotkey_lines(shortcuts: dict[str, Any]) -> list[str]:
    if not shortcuts:
        return []
    lines = ["", "Extension shortcuts:"]
    for key, shortcut in sorted(shortcuts.items()):
        description = getattr(shortcut, "description", None) or getattr(shortcut, "extension_path", "")
        lines.append(f"  {key}: {description}")
    return lines


def _built_in_command_conflict_diagnostics(
    extension_runner: Any,
    built_in_names: set[str],
) -> list[dict[str, str]]:
    if extension_runner is None or not hasattr(extension_runner, "get_registered_commands"):
        return []
    diagnostics: list[dict[str, str]] = []
    for command in extension_runner.get_registered_commands():
        command_name = getattr(command, "name", "")
        if command_name not in built_in_names:
            continue
        invocation_name = (
            getattr(command, "invocation_name", None)
            or getattr(command, "invocationName", None)
            or command_name
        )
        if invocation_name == command_name:
            message = (
                f"Extension command '/{command_name}' conflicts with built-in interactive command. "
                "Skipping in autocomplete."
            )
        else:
            message = (
                f"Extension command '/{command_name}' conflicts with built-in interactive command. "
                f"Available as '/{invocation_name}'."
            )
        diagnostics.append({
            "type": "warning",
            "message": message,
            "path": getattr(command, "extension_path", "") or "",
        })
    return diagnostics


def _apply_autocomplete_provider_wrappers(base_provider: Any, wrappers: list[Any]) -> Any:
    provider = base_provider
    for wrapper in wrappers:
        provider = wrapper(provider)
    return provider


def _assistant_label(session: Any) -> str:
    """Label for assistant turns. Uses the per-instance `name` setting when set
    (e.g. "Devin:"), otherwise the generic "Assistant:"."""
    sm = getattr(session, "settings_manager", None)
    getter = getattr(sm, "get_agent_name", None) if sm is not None else None
    name = getter() if callable(getter) else None
    return f"{name}:" if name else "Assistant:"


def _item_value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        value = getattr(item, name, None)
        if value is not None:
            return value
    return default


def _loaded_resource_lines(session: Any, *, show_listing: bool = True, show_diagnostics: bool = True) -> list[str]:
    loader = getattr(session, "resource_loader", None)
    if loader is None:
        return []

    lines: list[str] = []

    def add_section(title: str, items: list[str]) -> None:
        clean_items = [item for item in items if item]
        if not clean_items:
            return
        if lines:
            lines.append("")
        lines.append(f"[{title}]")
        lines.extend(f"  {item}" for item in clean_items)

    if show_listing:
        get_agents_files = getattr(loader, "get_agents_files", None)
        if callable(get_agents_files):
            agents_result = get_agents_files() or {}
            context_files = agents_result.get("agentsFiles") or agents_result.get("agents_files") or []
            add_section("Context", [str(_item_value(item, "path", default="")) for item in context_files])

        get_skills = getattr(loader, "get_skills", None)
        if callable(get_skills):
            skills_result = get_skills() or {}
            add_section("Skills", [str(_item_value(item, "name", "file_path", "filePath", default="")) for item in skills_result.get("skills", [])])

        prompts = getattr(session, "prompt_templates", None)
        if prompts is None:
            get_prompts = getattr(loader, "get_prompts", None)
            prompts = (get_prompts() or {}).get("prompts", []) if callable(get_prompts) else []
        add_section("Prompts", [f"/{_item_value(item, 'name', default='')}" for item in prompts])

        get_extensions = getattr(loader, "get_extensions", None)
        if callable(get_extensions):
            extensions_result = get_extensions() or {}
            add_section("Extensions", [str(_item_value(item, "path", default="")) for item in extensions_result.get("extensions", [])])

        get_themes = getattr(loader, "get_themes", None)
        if callable(get_themes):
            themes_result = get_themes() or {}
            add_section("Themes", [str(_item_value(item, "name", "path", default="")) for item in themes_result.get("themes", [])])

    if show_diagnostics:
        diagnostic_sections: list[tuple[str, list[Any]]] = []
        for title, getter_name, key in (
            ("Skill conflicts", "get_skills", "diagnostics"),
            ("Prompt conflicts", "get_prompts", "diagnostics"),
            ("Theme conflicts", "get_themes", "diagnostics"),
        ):
            getter = getattr(loader, getter_name, None)
            if callable(getter):
                diagnostic_sections.append((title, list((getter() or {}).get(key, []) or [])))

        extension_diagnostics: list[Any] = []
        get_extensions = getattr(loader, "get_extensions", None)
        if callable(get_extensions):
            extensions_result = get_extensions() or {}
            extension_diagnostics.extend(extensions_result.get("diagnostics", []) or [])
            for error in extensions_result.get("errors", []) or []:
                extension_diagnostics.append({
                    "type": "error",
                    "message": _item_value(error, "error", "message", default=""),
                    "path": _item_value(error, "path", default=""),
                })
        extension_runner = getattr(session, "extension_runner", None)
        for getter_name in ("get_command_diagnostics", "getCommandDiagnostics", "get_shortcut_diagnostics", "getShortcutDiagnostics"):
            getter = getattr(extension_runner, getter_name, None)
            if callable(getter):
                extension_diagnostics.extend(getter() or [])
        extension_diagnostics.extend(_built_in_command_conflict_diagnostics(
            extension_runner,
            {
                "settings", "chat", "model", "models", "set", "scoped-models", "export", "import", "share", "copy", "name",
                "session", "changelog", "hotkeys", "fork", "clone", "tree", "trust", "login",
                "logout", "new", "compact", "resume", "reload", "quit", "exit", "clear",
                "help", "thinking", "tools",
            },
        ))
        diagnostic_sections.append(("Extension issues", extension_diagnostics))

        for title, diagnostics in diagnostic_sections:
            diag_lines = []
            for diagnostic in diagnostics:
                message = str(_item_value(diagnostic, "message", default="")).strip()
                path = str(_item_value(diagnostic, "path", default="")).strip()
                if not message and not path:
                    continue
                diag_lines.append(f"{message} ({path})" if path else message)
            add_section(title, diag_lines)

    return lines


async def run_tui(
    session: "AgentSession",
    initial_messages: list[str] | None = None,
    runtime_host: Any | None = None,
) -> None:
    """
    Run the interactive TUI using pi_tui.

    Falls back to readline if pi_tui is unavailable or not in a TTY.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        from .mode import _run_readline_fallback
        await _run_readline_fallback(session, initial_messages)
        return

    try:
        await _run_pi_tui(session, initial_messages, runtime_host=runtime_host)
    except Exception as exc:
        import traceback
        from rich.console import Console
        console = Console()
        console.print(f"[yellow]TUI error ({exc}), falling back to readline mode.[/yellow]")
        console.print(traceback.format_exc(), style="dim")
        from .mode import _run_readline_fallback
        await _run_readline_fallback(session, initial_messages)


async def _run_pi_tui(
    session: "AgentSession",
    initial_messages: list[str] | None,
    runtime_host: Any | None = None,
) -> None:
    """Set up and run the pi_tui interactive loop."""
    from pi_tui import (
        TUI,
        Editor,
        EditorTheme,
        Loader,
        Markdown,
        MarkdownTheme,
        Text,
        Spacer,
        SelectListTheme,
        ProcessTerminal,
    )
    from pi_tui import CombinedAutocompleteProvider, SlashCommand
    from pi_tui.tui import OverlayOptions
    from pi_ai.types import TextContent, UserMessage
    from pi_coding_agent.modes.interactive.components.extension_components import (
        ExtensionEditorComponent,
        ExtensionInputComponent,
        ExtensionSelectorComponent,
    )
    from pi_coding_agent.modes.interactive.theme import (
        Theme as InteractiveTheme,
        get_available_themes_with_paths,
        get_theme,
        get_theme_by_name,
        init_theme,
        set_registered_themes,
        set_theme,
        set_theme_instance,
    )

    trace_path = os.environ.get("PI_INTERACTIVE_TRACE_LOG", "").strip()

    def trace(msg: str) -> None:
        if not trace_path:
            return
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    terminal = ProcessTerminal()
    tui = TUI(terminal)
    runtime_host = runtime_host or session

    def sync_runtime_session() -> None:
        nonlocal session
        runtime_session = getattr(runtime_host, "session", None)
        if runtime_session is not None:
            session = runtime_session

    try:
        themes_result = session.resource_loader.get_themes()
        set_registered_themes(themes_result.get("themes", []))
    except Exception:
        pass
    try:
        settings_manager = getattr(session, "settings_manager", None)
        get_configured_theme = (
            getattr(settings_manager, "get_theme", None)
            or getattr(settings_manager, "getTheme", None)
            or (lambda: None)
        )
        init_theme(get_configured_theme())
    except Exception:
        init_theme()

    # ── ANSI theme helpers ────────────────────────────────────────────────────
    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def bold(s: str) -> str:
        return f"\x1b[1m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    def green(s: str) -> str:
        return f"\x1b[32m{s}\x1b[39m"

    def yellow(s: str) -> str:
        return f"\x1b[33m{s}\x1b[39m"

    def red(s: str) -> str:
        return f"\x1b[31m{s}\x1b[39m"

    def blue(s: str) -> str:
        return f"\x1b[38;5;39m{s}\x1b[39m"

    import re as _re

    _SGR_RE = _re.compile(r"\x1b\[[0-9;]*m")

    def _format_tool_call(tool_name: str, args: Any) -> str:
        """Render the tool invocation as a one-liner, like Node's `$ cmd` header."""
        if not isinstance(args, dict):
            return ""
        if tool_name == "bash":
            cmd = str(args.get("command", "")).strip()
            return f"$ {cmd}" if cmd else ""
        for key in ("path", "file_path", "filePath", "pattern", "query", "url"):
            if args.get(key):
                return f"{tool_name}({args[key]})"
        try:
            return f"{tool_name}({json.dumps(args, separators=(',', ':'))})"
        except TypeError:
            return f"{tool_name}({args})"

    def _tool_box(header: str, call: str, response_lines: list[str], bg: str) -> str:
        """Full-width 256-color block: green=success, red=error. The call (header
        + command) is bright; the response is gray. 256-color so the fill renders
        in Terminal.app (truecolor bg is dropped there)."""
        cols = max(24, int(getattr(terminal, "columns", 80) or 80) - 2)

        def row(text: str, fg: str) -> str:
            text = _SGR_RE.sub("", text)
            if len(text) > cols:  # truncate, don't wrap
                text = text[: cols - 1] + "…"
            return f"\x1b[48;5;{bg};38;5;{fg}m{text.ljust(cols)}\x1b[0m"

        out = [row(header, "231")]            # bright white header (✓/✗ tool)
        if call:
            out.append(row("", "231"))
            out.append(row(call, "231"))      # the call, bright
        if response_lines:
            out.append(row("", "231"))
            out.extend(row(ln, "250") for ln in response_lines)  # response, gray
        return "\n".join(out)

    def _user_box(text: str) -> str:
        """Gray full-width box for a submitted user message — visually separates
        the user's turn from the assistant's reply."""
        cols = max(24, int(getattr(terminal, "columns", 80) or 80) - 2)

        def row(t: str) -> str:
            t = _SGR_RE.sub("", t)
            if len(t) > cols:
                t = t[: cols - 1] + "…"
            return f"\x1b[48;5;238;38;5;253m{t.ljust(cols)}\x1b[0m"

        out = [row("")]
        out.extend(row(ln) for ln in text.split("\n"))
        out.append(row(""))
        return "\n".join(out)

    select_theme = SelectListTheme(
        selected_text=cyan,
        description=dim,
        scroll_info=dim,
        no_match=dim,
    )

    # Input box: gray background fill (256-color, renders in Terminal.app) with
    # the existing top/bottom border lines.
    def _input_bg(s: str) -> str:
        return f"\x1b[48;5;238m{s}\x1b[49m"

    editor_theme = EditorTheme(
        border_color=lambda s: f"\x1b[48;5;238m\x1b[38;5;244m{s}\x1b[0m",
        bg_color=_input_bg,
        select_list=select_theme,
    )

    def _md_heading(s: str) -> str:
        # Headings render as heading(bold(text)); drop bold's color so the
        # heading colour (yellow) wins, keep the bold weight.
        s = s.replace("\x1b[38;5;39m", "")
        return f"\x1b[1m\x1b[33m{s}\x1b[22m"

    def _md_bold(s: str) -> str:
        return f"\x1b[1m\x1b[38;5;39m{s}\x1b[22m\x1b[39m"  # blue (matches tool calls) + bold

    def _md_code(s: str) -> str:
        return f"\x1b[38;5;51m{s}\x1b[39m"  # cyan (backticks stripped by parser)

    markdown_theme = MarkdownTheme(
        heading=_md_heading,
        bold=_md_bold,
        code=_md_code,
        code_block=_md_code,
        code_block_border=dim,
        list_bullet=cyan,
    )
    # padding_x=0: the history widget already pads, and we prefix "Assistant:".
    _md_component = Markdown("", 0, 0, markdown_theme)

    def render_markdown(text: str) -> str:
        """Render assistant markdown to ANSI lines via the shared Markdown component.

        Trailing padding is stripped per line (the component pads to width) so the
        body sits cleanly after the "Assistant:" label and does not overflow.
        """
        try:
            _md_component.set_text(text)
            width = max(20, int(getattr(terminal, "columns", 80) or 80) - 2)
            lines = [ln.rstrip() for ln in _md_component.render(width)]
            return "\n".join(lines).strip("\n")
        except Exception:
            return text

    # ── Output area ──────────────────────────────────────────────────────────
    header_text = Text("", padding_x=1, padding_y=0)
    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    widget_text = Text("", padding_x=1, padding_y=0)
    footer_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(header_text)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(widget_text)
    tui.add_child(Spacer(1))

    def append_history(line: str) -> None:
        """Append a completed line to history."""
        trace(f"append_history: {line[:120]!r}")
        current = history_text._text
        history_text.set_text((current + "\n" + line).lstrip("\n"))
        history_text.invalidate()

    def assistant_text_from_message(message: object) -> str:
        """
        Extract full assistant text from an assistant message snapshot.
        Mirrors TS behavior where UI updates from full message state, not only deltas.
        """
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    def message_text_from_any(message: Any) -> tuple[str, str]:
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = message.get("content", "")
        else:
            role = str(getattr(message, "role", ""))
            content = getattr(message, "content", "")
        if isinstance(content, str):
            return role, content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                elif getattr(item, "type", None) == "text":
                    parts.append(str(getattr(item, "text", "")))
            return role, "".join(parts)
        return role, ""

    def rebuild_history_from_current_session() -> None:
        getter = getattr(session, "get_messages", None) or getattr(session, "getMessages", None)
        if callable(getter):
            messages = getter()
        else:
            messages = getattr(session, "messages", [])
        lines: list[str] = []
        for message in messages or []:
            role, body = message_text_from_any(message)
            body = body.strip()
            if not body:
                continue
            label = "You" if role == "user" else "Assistant" if role == "assistant" else role or "Message"
            lines.append(f"{label}: {body}")
        history_text.set_text("\n".join(lines))
        history_text.invalidate()

    def set_stream(text: str) -> None:
        """Update the current streaming response line."""
        trace(f"set_stream: {text[:120]!r}")
        stream_text.set_text(text)
        stream_text.invalidate()
        tui.request_render()

    # ── Footer ────────────────────────────────────────────────────────────────
    # NOTE: footer_text is added to the TUI *after* the editor (see below) so it
    # renders at the very bottom, under the input — like the Node pi footer.
    extension_statuses: dict[str, str] = {}
    extension_widgets: dict[str, tuple[list[str], str]] = {}
    working_message: str | None = None
    working_visible = True
    working_indicator_options: Any = None
    hidden_thinking_label = "Thinking..."

    def render_extension_surface(factory: Any, *args: Any) -> str:
        if factory is None:
            return ""
        value = factory(*args) if callable(factory) else factory
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        render = getattr(value, "render", None)
        if callable(render):
            try:
                rendered = render(terminal.columns)
            except TypeError:
                rendered = render()
            if isinstance(rendered, list):
                return "\n".join(str(item) for item in rendered)
            return str(rendered)
        return str(value)

    def component_from_value(value: Any) -> Any:
        if value is None:
            raise ValueError("custom UI factory must return a component")
        if isinstance(value, str):
            return Text(value, padding_x=1, padding_y=0)
        if isinstance(value, list):
            return Text("\n".join(str(item) for item in value), padding_x=1, padding_y=0)
        return value

    def render_extension_widgets() -> None:
        lines: list[str] = []
        for placement in ("aboveEditor", "belowEditor"):
            for _key, (widget_lines, widget_placement) in extension_widgets.items():
                if widget_placement == placement:
                    lines.extend(widget_lines[:10])
        widget_text.set_text("\n".join(lines))
        widget_text.invalidate()

    def set_extension_status(key: str, text: str | None) -> None:
        if text is None:
            extension_statuses.pop(key, None)
        else:
            extension_statuses[key] = text
        update_footer()

    def set_extension_widget(key: str, lines: list[str] | None, placement: str) -> None:
        if lines is None:
            extension_widgets.pop(key, None)
        else:
            extension_widgets[key] = (list(lines), placement)
        render_extension_widgets()

    def set_terminal_title(title: str) -> None:
        terminal_set_title = getattr(terminal, "set_title", None) or getattr(terminal, "setTitle", None)
        if callable(terminal_set_title):
            terminal_set_title(title)

    def persist_theme_name(theme_name: str) -> None:
        settings_manager = getattr(session, "settings_manager", None)
        if settings_manager is None:
            return
        setter = getattr(settings_manager, "set_theme", None) or getattr(settings_manager, "setTheme", None)
        if callable(setter):
            setter(theme_name)
            return
        save_project = getattr(settings_manager, "save_project", None) or getattr(settings_manager, "saveProject", None)
        if callable(save_project):
            try:
                save_project("theme", theme_name)
                return
            except Exception:
                pass
        apply_overrides = getattr(settings_manager, "apply_overrides", None) or getattr(
            settings_manager,
            "applyOverrides",
            None,
        )
        if callable(apply_overrides):
            apply_overrides({"theme": theme_name})

    def set_interactive_theme(theme_or_name: Any) -> dict[str, Any]:
        if isinstance(theme_or_name, InteractiveTheme):
            set_theme_instance(theme_or_name)
            tui.request_render()
            return {"success": True}
        result = set_theme(str(theme_or_name), True)
        if result.get("success"):
            persist_theme_name(str(theme_or_name))
            tui.request_render()
        return result

    def set_extension_header(factory: Any = None) -> None:
        header_text.set_text(render_extension_surface(factory, tui, get_theme()))
        header_text.invalidate()

    def set_extension_footer(factory: Any = None) -> None:
        if factory is None:
            update_footer()
        else:
            footer_text.set_text(render_extension_surface(factory, tui, get_theme(), {}))
            footer_text.invalidate()

    def set_extension_working_message(message: str | None = None) -> None:
        nonlocal working_message
        working_message = message
        update_footer()

    def set_extension_working_visible(visible: bool) -> None:
        nonlocal working_visible
        working_visible = bool(visible)
        update_footer()

    def set_extension_working_indicator(options: Any = None) -> None:
        nonlocal working_indicator_options
        working_indicator_options = options
        update_footer()

    def set_extension_hidden_thinking_label(label: str | None = None) -> None:
        nonlocal hidden_thinking_label
        hidden_thinking_label = label or "Thinking..."

    def _fmt_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n // 1000}k"
        return str(n)

    def update_footer() -> None:
        """Refresh footer: model | thinking: off | ctx: 12% | tokens: 8k/64k"""
        model = session.model
        model_str = model.id if model else "no model"
        thinking = getattr(session, "thinking_level", "off") or "off"
        parts = [model_str, f"thinking: {thinking}"]
        ctx = session.get_context_usage()
        if ctx and ctx.get("percent") is not None:
            pct = ctx["percent"]
            tkn = _fmt_tokens(ctx.get("tokens", 0))
            cw = _fmt_tokens(ctx.get("contextWindow", 0))
            parts.append(f"ctx: {pct:.0f}% ({tkn}/{cw})")
        parts.append(f"v{VERSION}")
        for key, text in sorted(extension_statuses.items()):
            parts.append(f"{key}: {text}")
        if working_visible and working_message:
            indicator_label = _option_value(working_indicator_options, "label") or _option_value(
                working_indicator_options,
                "text",
            )
            working_part = f"working: {working_message}"
            if indicator_label:
                working_part += f" ({indicator_label})"
            parts.append(working_part)
        footer_text.set_text(dim("  " + " | ".join(parts)))
        footer_text.invalidate()

    update_footer()

    # ── Editor ───────────────────────────────────────────────────────────────
    editor = Editor(tui, editor_theme)
    default_editor = editor
    editor_component_factory: Any = None
    extension_runner = getattr(session, "extension_runner", None)
    try:
        from pi_coding_agent.core.keybindings import KeybindingsManager

        keybindings = KeybindingsManager.create()
    except Exception:
        keybindings = None
    extension_shortcuts: dict[str, Any] = {}
    editor_runtime_callbacks: dict[str, Any] = {"on_submit": None, "on_keydown": None}
    built_in_slash_specs = [
        ("settings", "Open settings menu"),
        ("chat", "Render chat transcript"),
        ("goal", "Show, set, or clear the current session goal"),
        ("model", "Select provider and model strength"),
        ("models", "Alias for /model"),
        ("set", "Set provider tier model mapping"),
        ("scoped-models", "Enable/disable scoped models"),
        ("export", "Export session as HTML"),
        ("import", "Import and resume a session"),
        ("share", "Share session"),
        ("copy", "Copy last assistant message"),
        ("name", "Set session display name"),
        ("session", "Show session statistics"),
        ("changelog", "Show changelog entries"),
        ("hotkeys", "Show keyboard shortcuts"),
        ("fork", "Create a fork from a previous user message"),
        ("clone", "Duplicate current session"),
        ("tree", "Navigate session tree"),
        ("trust", "Save project trust decision"),
        ("login", "Configure provider authentication"),
        ("logout", "Remove provider authentication"),
        ("new", "Start a new session"),
        ("compact", "Compact conversation context"),
        ("kill", "Kill running tau sessions and subagents"),
        ("resume", "Resume a different session"),
        ("reload", "Reload resources"),
        ("quit", "Quit the agent"),
        ("exit", "Exit the agent"),
        ("clear", "Clear conversation history"),
        ("help", "Show help"),
        ("thinking", "Cycle thinking level"),
        ("tools", "List active tools"),
    ]
    built_in_slash_names = {name for name, _description in built_in_slash_specs}

    def build_slash_commands(current_extension_runner: Any) -> list[Any]:
        commands = [SlashCommand(name=name, description=description) for name, description in built_in_slash_specs]
        get_registered_commands = getattr(current_extension_runner, "get_registered_commands", None) or getattr(
            current_extension_runner,
            "getRegisteredCommands",
            None,
        )
        if callable(get_registered_commands):
            for command in get_registered_commands():
                command_name = getattr(command, "name", "")
                if command_name in built_in_slash_names:
                    continue
                invocation = (
                    getattr(command, "invocation_name", None)
                    or getattr(command, "invocationName", None)
                    or command_name
                )
                if invocation:
                    commands.append(
                        SlashCommand(
                            name=invocation,
                            description=getattr(command, "description", "") or "Extension command",
                        )
                    )
        return commands

    slash_commands = build_slash_commands(extension_runner)
    base_autocomplete = CombinedAutocompleteProvider(commands=slash_commands)
    autocomplete_provider_wrappers: list[Any] = []

    def apply_autocomplete_provider() -> None:
        provider = _apply_autocomplete_provider_wrappers(
            base_autocomplete,
            autocomplete_provider_wrappers,
        )
        set_provider = getattr(editor, "set_autocomplete_provider", None)
        if callable(set_provider):
            set_provider(provider)
        tui.request_render()

    def add_autocomplete_provider(factory: Any) -> None:
        autocomplete_provider_wrappers.append(factory)
        apply_autocomplete_provider()

    def refresh_extension_runtime_state() -> None:
        nonlocal extension_runner, base_autocomplete, extension_shortcuts
        extension_runner = getattr(session, "extension_runner", None)
        base_autocomplete = CombinedAutocompleteProvider(commands=build_slash_commands(extension_runner))
        get_shortcuts = getattr(extension_runner, "get_shortcuts", None) or getattr(extension_runner, "getShortcuts", None)
        extension_shortcuts = (
            get_shortcuts(keybindings.get_effective_config())
            if keybindings is not None and callable(get_shortcuts)
            else {}
        )
        apply_autocomplete_provider()

    def get_current_editor_text() -> str:
        get_text = getattr(editor, "get_text", None)
        if callable(get_text):
            return get_text()
        return getattr(editor, "_text", "")

    def set_current_editor_text(text: str) -> None:
        set_text = getattr(editor, "set_text", None)
        if callable(set_text):
            set_text(text)

    def sync_editor_runtime_callbacks() -> None:
        for attr, callback in editor_runtime_callbacks.items():
            if callback is not None:
                setattr(editor, attr, callback)

    def set_custom_editor_component(factory: Any = None) -> None:
        nonlocal editor, editor_component_factory
        editor_component_factory = factory
        current_text = get_current_editor_text()
        next_editor = default_editor if factory is None else factory(tui, editor_theme, keybindings)
        if next_editor is None:
            raise ValueError("setEditorComponent factory must return an editor component")
        next_set_text = getattr(next_editor, "set_text", None)
        if callable(next_set_text):
            next_set_text(current_text)
        sync_old = editor
        try:
            index = tui.children.index(sync_old)
            tui.children[index] = next_editor
        except ValueError:
            tui.add_child(next_editor)
        if sync_old is not next_editor:
            dispose = getattr(sync_old, "dispose", None)
            if callable(dispose):
                dispose()
        editor = next_editor
        sync_editor_runtime_callbacks()
        apply_autocomplete_provider()
        tui.set_focus(editor)
        tui.request_render()

    def show_prompt_component(component: Any) -> None:
        try:
            index = tui.children.index(editor)
            tui.children[index] = component
        except ValueError:
            tui.add_child(component)
        tui.set_focus(component)
        tui.request_render()

    def hide_prompt_component(component: Any) -> None:
        dispose = getattr(component, "dispose", None)
        if callable(dispose):
            dispose()
        try:
            index = tui.children.index(component)
            tui.children[index] = editor
        except ValueError:
            if editor not in tui.children:
                tui.add_child(editor)
        tui.set_focus(editor)
        tui.request_render()

    async def show_extension_selector(title: str, options: list[str], opts: Any = None) -> str | None:
        signal = _option_value(opts, "signal")
        if _signal_is_aborted(signal):
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        cleanup_abort_listener: Callable[[], None] = lambda: None

        def resolve(value: str | None) -> None:
            def _resolve() -> None:
                if future.done():
                    return
                cleanup_abort_listener()
                hide_prompt_component(component)
                future.set_result(value)

            loop.call_soon_threadsafe(_resolve)

        component = ExtensionSelectorComponent(
            title,
            options,
            on_select=lambda option: resolve(option),
            on_cancel=lambda: resolve(None),
            opts={"timeout": _option_value(opts, "timeout")},
        )
        cleanup_abort_listener = _add_abort_listener(signal, lambda: resolve(None))
        show_prompt_component(component)
        return await future

    async def show_extension_input(title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
        signal = _option_value(opts, "signal")
        if _signal_is_aborted(signal):
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        cleanup_abort_listener: Callable[[], None] = lambda: None

        def resolve(value: str | None) -> None:
            def _resolve() -> None:
                if future.done():
                    return
                cleanup_abort_listener()
                hide_prompt_component(component)
                future.set_result(value)

            loop.call_soon_threadsafe(_resolve)

        component = ExtensionInputComponent(
            title,
            placeholder,
            on_submit=lambda value: resolve(value),
            on_cancel=lambda: resolve(None),
            opts={
                "timeout": _option_value(opts, "timeout"),
                "secret": bool(_option_value(opts, "secret", False)),
            },
        )
        cleanup_abort_listener = _add_abort_listener(signal, lambda: resolve(None))
        show_prompt_component(component)
        return await future

    async def show_extension_editor(title: str, prefill: str | None = None, opts: Any = None) -> str | None:
        signal = _option_value(opts, "signal")
        if _signal_is_aborted(signal):
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        cleanup_abort_listener: Callable[[], None] = lambda: None

        def resolve(value: str | None) -> None:
            def _resolve() -> None:
                if future.done():
                    return
                cleanup_abort_listener()
                hide_prompt_component(component)
                future.set_result(value)

            loop.call_soon_threadsafe(_resolve)

        component = ExtensionEditorComponent(
            title,
            prefill,
            on_submit=lambda value: resolve(value),
            on_cancel=lambda: resolve(None),
            tui=tui,
            keybindings=keybindings,
        )
        cleanup_abort_listener = _add_abort_listener(signal, lambda: resolve(None))
        show_prompt_component(component)
        return await future

    async def show_extension_custom(factory: Any, options: Any = None) -> Any:
        if not callable(factory):
            raise ValueError("custom UI factory must be callable")
        is_overlay = bool(_option_value(options, "overlay", False))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        saved_text = get_current_editor_text()
        component: Any = None
        overlay_handle: Any = None
        closed = False

        def close(result: Any = None) -> None:
            def _close() -> None:
                nonlocal closed
                if closed:
                    return
                closed = True
                if is_overlay:
                    if overlay_handle is not None:
                        overlay_handle.hide()
                elif component is not None:
                    hide_prompt_component(component)
                    set_current_editor_text(saved_text)
                dispose = getattr(component, "dispose", None)
                if callable(dispose):
                    dispose()
                if not future.done():
                    future.set_result(result)

            loop.call_soon_threadsafe(_close)

        try:
            created = factory(tui, get_theme(), keybindings, close)
            if asyncio.iscoroutine(created):
                created = await created
            component = component_from_value(created)
            if is_overlay:
                overlay_options = _option_value(options, "overlayOptions")
                if callable(overlay_options):
                    overlay_options = overlay_options()
                if isinstance(overlay_options, dict):
                    overlay_options = OverlayOptions(
                        width=overlay_options.get("width"),
                        min_width=overlay_options.get("minWidth") or overlay_options.get("min_width"),
                        max_height=overlay_options.get("maxHeight") or overlay_options.get("max_height"),
                        anchor=overlay_options.get("anchor", "center"),
                        offset_x=overlay_options.get("offsetX") or overlay_options.get("offset_x", 0),
                        offset_y=overlay_options.get("offsetY") or overlay_options.get("offset_y", 0),
                    )
                overlay_handle = tui.show_overlay(component, overlay_options)
                on_handle = _option_value(options, "onHandle")
                if callable(on_handle):
                    on_handle(overlay_handle)
            else:
                show_prompt_component(component)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        return await future

    apply_autocomplete_provider()
    tui.add_child(editor)
    # Footer goes below the input, at the very bottom (Node-style).
    tui.add_child(footer_text)
    tui.set_focus(editor)

    extension_ui_context = InteractiveExtensionUIContext(
        append_history=append_history,
        request_render=tui.request_render,
        set_editor_text=set_current_editor_text,
        get_editor_text=get_current_editor_text,
        set_title=set_terminal_title,
        set_status=set_extension_status,
        set_widget=set_extension_widget,
        add_autocomplete_provider=add_autocomplete_provider,
        add_terminal_input_listener=tui.add_input_listener,
        set_editor_component=set_custom_editor_component,
        get_editor_component=lambda: editor_component_factory,
        show_select=show_extension_selector,
        show_input=show_extension_input,
        show_editor=show_extension_editor,
        show_custom=show_extension_custom,
        set_working_message=set_extension_working_message,
        set_working_visible=set_extension_working_visible,
        set_working_indicator=set_extension_working_indicator,
        set_hidden_thinking_label=set_extension_hidden_thinking_label,
        set_footer=set_extension_footer,
        set_header=set_extension_header,
        get_theme=get_theme,
        get_all_themes=get_available_themes_with_paths,
        get_theme_by_name=get_theme_by_name,
        set_theme=set_interactive_theme,
    )
    bind_extensions = getattr(session, "bind_extensions", None)

    refresh_extension_runtime_state()

    def _persist_defaults(updates: dict[str, Any]) -> str:
        """Persist default* keys with the right scope:

        - If no GLOBAL base default exists yet (first run), write GLOBAL so every
          new agent inherits it.
        - Once a global base exists, write PROJECT-local only — so changing this
          agent's model/thinking never touches other agents.
        """
        sm = getattr(session, "settings_manager", None)
        if sm is None:
            return "nowhere"
        import json as _json

        from pi_coding_agent.config import get_settings_path

        global_base_set = False
        try:
            data = _json.load(open(get_settings_path(), encoding="utf-8"))
            global_base_set = bool(data.get("defaultProvider") or data.get("defaultModel"))
        except Exception:
            global_base_set = False

        writer = getattr(sm, "save_project" if global_base_set else "save_global", None)
        scope = "this agent" if global_base_set else "global default"
        if callable(writer):
            for k, v in updates.items():
                try:
                    writer(k, v)
                except Exception:
                    pass
        return scope

    async def _execute_extension_shortcut(key: str) -> bool:
        if not extension_shortcuts:
            return False
        try:
            extension_runner = getattr(session, "extension_runner", None)
            if extension_runner is None:
                return False
            handled = await extension_runner.execute_shortcut(
                key,
                keybindings.get_effective_config() if keybindings else {},
            )
            if handled:
                tui.request_render()
            return handled
        except Exception as exc:
            append_history(f"{red('Shortcut handler error:')} {exc}")
            tui.request_render()
            return True

    # ── Submit handler ────────────────────────────────────────────────────────
    is_busy = False
    working_loader: Any = None  # Loader component shown while the agent works

    def _start_working_loader() -> None:
        """Show the animated 'Working…' loader (Node-style) below the transcript."""
        nonlocal working_loader
        if working_loader is not None:
            return
        working_loader = Loader(tui, blue, dim, "Working…")
        try:  # place it right after the streaming line
            idx = tui.children.index(stream_text)
            tui.children.insert(idx + 1, working_loader)
        except (ValueError, AttributeError):
            tui.add_child(working_loader)
        tui.request_render()

    def _stop_working_loader() -> None:
        nonlocal working_loader
        if working_loader is None:
            return
        try:
            working_loader.stop()
        except Exception:
            pass
        try:
            tui.children.remove(working_loader)
        except (ValueError, AttributeError):
            pass
        working_loader = None
        tui.request_render()

    async def handle_submit(text: str) -> None:
        nonlocal is_busy
        trace(f"handle_submit: raw={text!r}")
        stripped = text.strip()
        if not stripped:
            trace("handle_submit: empty, return")
            return

        # ── Slash commands ────────────────────────────────────────────────────
        if stripped in ("/exit", "/quit", "exit", "quit"):
            trace("handle_submit: exit command")
            tui.stop()
            return

        if stripped == "/clear":
            trace("handle_submit: clear command")
            history_text.set_text("")
            history_text.invalidate()
            set_stream("")
            return

        if stripped == "/chat":
            trace("handle_submit: chat command")
            history_text.set_text("")
            history_text.invalidate()
            set_stream("")
            rebuild_history_from_current_session()
            tui.request_render()
            return

        if stripped == "/goal" or stripped.startswith("/goal "):
            arg = stripped[6:].strip() if stripped.startswith("/goal ") else ""
            getter = getattr(session, "get_current_goal", None)
            setter = getattr(session, "set_current_goal", None)
            if arg.lower() in {"clear", "unset", "reset", "off"}:
                if callable(setter):
                    setter(None)
                append_history(dim("Goal cleared."))
            elif arg:
                if callable(setter):
                    setter(arg)
                    append_history(f"{green('Goal:')} {arg}")
                else:
                    append_history(dim("This session does not support goals."))
            else:
                current = getter() if callable(getter) else None
                append_history(f"{cyan('Goal:')} {current}" if current else dim("No goal set."))
            update_footer()
            tui.request_render()
            return

        if stripped == "/kill" or stripped.startswith("/kill "):
            target = stripped[6:].strip() if stripped.startswith("/kill ") else None
            try:
                from pi_coding_agent.core.runtime_registry import kill_processes

                killed = kill_processes(target or None, root=session.cwd)
                if target and not killed:
                    append_history(dim(f"No running tau session matched: {target}"))
                elif not killed:
                    append_history(dim("No running tau sessions."))
                else:
                    lines = [green(f"Killed {len(killed)} tau session(s).")]
                    for entry in killed[:20]:
                        lines.append(
                            "  "
                            + dim(
                                f"{entry.get('kind')} {entry.get('session_id')} "
                                f"(pid {entry.get('pid')})"
                            )
                        )
                    if len(killed) > 20:
                        lines.append(dim(f"  ... {len(killed) - 20} more"))
                    append_history("\n".join(lines))
            except Exception as exc:
                append_history(f"{red('Kill failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/help":
            lines = [
                bold("Available commands:"),
                f"  {cyan('/exit')}     — Exit the agent",
                f"  {cyan('/clear')}    — Clear conversation history",
                f"  {cyan('/chat')}     — Render the session chat transcript",
                f"  {cyan('/goal')}     — Show, set, or clear current session goal",
                f"  {cyan('/model')}    — List available models / switch model",
                f"  {cyan('/copy')}     — Copy last assistant message",
                f"  {cyan('/name')}     — Set session display name",
                f"  {cyan('/export')}   — Export session as HTML or JSONL",
                f"  {cyan('/new')}      — Clear conversation and start a fresh session view",
                f"  {cyan('/reload')}   — Reload settings and resources",
                f"  {cyan('/compact')}  — Compact context to free tokens",
                f"  {cyan('/kill')}     — Kill running tau sessions and subagents",
                f"  {cyan('/thinking')} — Cycle thinking level",
                f"  {cyan('/session')}  — Show session statistics",
                f"  {cyan('/tools')}    — List active tools",
                f"  {cyan('Ctrl+P')}    — Cycle to next model",
            ]
            extension_lines = _extension_command_help_lines(extension_runner, cyan)
            if extension_lines:
                lines.extend(["", bold("Extension commands:"), *extension_lines])
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/copy":
            try:
                from pi_coding_agent.utils.clipboard import copy_to_clipboard

                text = session.get_last_assistant_text()
                if text:
                    copy_to_clipboard(text)
                    append_history(green("Copied last assistant message."))
                else:
                    append_history(dim("No assistant message to copy."))
            except Exception as exc:
                append_history(f"{red('Copy failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/new":
            try:
                new_session = getattr(runtime_host, "new_session", None) or getattr(runtime_host, "newSession", None)
                if callable(new_session):
                    result = await new_session()
                    if result.get("cancelled"):
                        append_history(dim("New session cancelled."))
                        tui.request_render()
                        return
                    sync_runtime_session()
                    refresh_extension_runtime_state()
                    update_footer()
                else:
                    await session.new_session()
                history_text.set_text("")
                history_text.invalidate()
                set_stream("")
                append_history(dim("Started a new session."))
            except Exception as exc:
                append_history(f"{red('New session failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/clone":
            try:
                session_manager = getattr(session, "session_manager", None) or getattr(session, "sessionManager", None)
                leaf_id = None
                if session_manager is not None:
                    get_leaf_id = getattr(session_manager, "get_leaf_id", None) or getattr(session_manager, "getLeafId", None)
                    if callable(get_leaf_id):
                        leaf_id = get_leaf_id()
                runtime_fork = getattr(runtime_host, "fork", None)
                if callable(runtime_fork) and leaf_id:
                    result = await runtime_fork(leaf_id, {"position": "at"})
                elif callable(runtime_fork) and not leaf_id:
                    append_history(dim("Nothing to clone yet"))
                    tui.request_render()
                    return
                else:
                    result = await session.clone_session()
                if result.get("cancelled"):
                    append_history(dim("Clone cancelled."))
                else:
                    append_history(green("Cloned to new session"))
            except Exception as exc:
                append_history(f"{red('Clone failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/reload":
            try:
                await session.reload()
                refresh_extension_runtime_state()
                resource_lines = _loaded_resource_lines(
                    session,
                    show_listing=not quiet_startup,
                    show_diagnostics=True,
                )
                if resource_lines:
                    append_history("\n".join(resource_lines))
                append_history(green("Reloaded keybindings, extensions, skills, prompts, themes."))
            except Exception as exc:
                append_history(f"{red('Reload failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/export" or stripped.startswith("/export "):
            output_path = _path_command_argument(stripped, "/export")
            try:
                if output_path and output_path.endswith(".jsonl"):
                    export_jsonl = getattr(session, "export_to_jsonl", None) or getattr(session, "exportToJsonl", None)
                    if not callable(export_jsonl):
                        raise RuntimeError("JSONL export is not supported by this session")
                    exported = export_jsonl(output_path)
                else:
                    exported = await session.export_to_html(output_path or None)
                append_history(f"{green('Exported session:')} {exported}")
            except Exception as exc:
                append_history(f"{red('Export failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/name" or stripped.startswith("/name "):
            name = stripped[6:].strip() if stripped.startswith("/name ") else ""
            if not name:
                append_history(dim("Usage: /name <session name>"))
            else:
                session.set_session_name(name)
                append_history(f"{green('Session name:')} {name}")
            tui.request_render()
            return

        if stripped == "/fork" or stripped.startswith("/fork "):
            entry_id = stripped[6:].strip() if stripped.startswith("/fork ") else ""
            if not entry_id:
                messages = session.get_user_messages_for_forking()
                if not messages:
                    append_history(dim("No user messages to fork from."))
                else:
                    lines = [bold("Forkable user messages:")]
                    for item in messages[-12:]:
                        text = item.get("text", "")
                        preview = text[:80] + "..." if len(text) > 80 else text
                        lines.append(f"  {item.get('entry_id')}: {preview}")
                    lines.append(dim("Run /fork <entry_id> to switch to a fork."))
                    append_history("\n".join(lines))
            else:
                try:
                    runtime_fork = getattr(runtime_host, "fork", None)
                    if callable(runtime_fork):
                        result = await runtime_fork(entry_id)
                        if not result.get("cancelled"):
                            sync_runtime_session()
                            refresh_extension_runtime_state()
                    else:
                        result = await session.fork_session(entry_id)
                    if result.get("cancelled"):
                        append_history(dim("Fork cancelled."))
                    else:
                        selected = result.get("selectedText") or ""
                        append_history(green("Forked to new session"))
                        if selected:
                            editor.set_text(selected)
                except Exception as exc:
                    append_history(f"{red('Fork failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/tree" or stripped.startswith("/tree "):
            entry_id = stripped[6:].strip() if stripped.startswith("/tree ") else ""
            if not entry_id:
                entries = session.get_session_tree_entries()
                if not entries:
                    append_history(dim("Session tree is empty."))
                else:
                    lines = [bold("Session tree entries:")]
                    for item in entries[-20:]:
                        label = f" [{item['label']}]" if item.get("label") else ""
                        text = item.get("text", "")
                        preview = text[:70] + "..." if len(text) > 70 else text
                        lines.append(f"  {item['entry_id']} {item['type']}{label}: {preview}")
                    lines.append(dim("Run /tree <entry_id> to navigate to an entry."))
                    append_history("\n".join(lines))
            else:
                try:
                    result = await session.navigate_tree(entry_id)
                    if result.get("cancelled"):
                        append_history(dim("Tree navigation cancelled."))
                    else:
                        editor_text = result.get("editorText")
                        if editor_text:
                            editor.set_text(editor_text)
                        rebuild_history_from_current_session()
                        append_history(green("Navigated session tree."))
                except Exception as exc:
                    append_history(f"{red('Tree navigation failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/resume" or stripped.startswith("/resume "):
            session_path = stripped[8:].strip() if stripped.startswith("/resume ") else ""
            if not session_path:
                append_history(dim("Usage: /resume <session_path>"))
            else:
                try:
                    switch_session = getattr(runtime_host, "switch_session", None) or getattr(runtime_host, "switchSession", None)
                    if callable(switch_session):
                        result = await switch_session(session_path)
                        if result.get("cancelled"):
                            append_history(dim("Resume cancelled."))
                        else:
                            sync_runtime_session()
                            refresh_extension_runtime_state()
                            update_footer()
                            append_history(green("Resumed session."))
                    else:
                        await session.switch_session(session_path)
                        append_history(green("Resumed session."))
                except Exception as exc:
                    append_history(f"{red('Resume failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/login" or stripped.startswith("/login "):
            await _handle_login_command(
                stripped, session, append_history, update_footer, tui,
                show_extension_selector, show_extension_input,
                cyan, dim, red, green, _persist_defaults,
            )
            return

        if stripped == "/logout" or stripped.startswith("/logout "):
            await _handle_logout_command(
                stripped, session, append_history, update_footer, tui,
                show_extension_selector, dim, red, green,
            )
            return

        if stripped == "/hotkeys":
            lines = [
                bold("Hotkeys:"),
                "  Enter: Submit",
                "  Shift+Enter: New line",
                "  Ctrl+P: Cycle model",
                "  Esc: Cancel current input/operation where supported",
                "  Arrow keys: Move cursor",
            ]
            lines.extend(_extension_shortcut_hotkey_lines(extension_shortcuts))
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/changelog":
            try:
                from pathlib import Path
                from pi_coding_agent.utils.changelog import parse_changelog

                candidates = [
                    Path(__file__).resolve().parents[4] / "CHANGELOG.md",
                    Path.cwd() / "CHANGELOG.md",
                ]
                changelog_text = ""
                for candidate in candidates:
                    if candidate.exists():
                        changelog_text = candidate.read_text(encoding="utf-8", errors="replace")
                        break
                entries = parse_changelog(changelog_text) if changelog_text else []
                if entries:
                    latest = entries[-3:]
                    lines = [bold("Changelog:")]
                    for entry in latest:
                        date = f" - {entry.date}" if entry.date else ""
                        content = entry.content.strip()
                        preview = content[:500] + "..." if len(content) > 500 else content
                        lines.append(f"## {entry.version}{date}\n{preview}")
                    append_history("\n\n".join(lines))
                else:
                    append_history(dim("No changelog entries found."))
            except Exception as exc:
                append_history(f"{red('Changelog failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/settings":
            try:
                settings = session.settings_manager.get_merged_raw()
                lines = [
                    bold("Settings:"),
                    f"  defaultProvider: {settings.get('defaultProvider') or settings.get('default_provider') or '-'}",
                    f"  defaultModel: {settings.get('defaultModel') or settings.get('default_model') or '-'}",
                    f"  defaultThinkingLevel: {settings.get('defaultThinkingLevel') or settings.get('default_thinking_level') or '-'}",
                    f"  theme: {settings.get('theme') or '-'}",
                    f"  transport: {settings.get('transport') or 'sse'}",
                    f"  enabledModels: {', '.join(session.settings_manager.get_enabled_models() or []) or '-'}",
                    f"  extensions: {len(session.settings_manager.get_extensions())}",
                    f"  skills: {len(session.settings_manager.get_skills())}",
                    f"  prompts: {len(session.settings_manager.get_prompts())}",
                    f"  themes: {len(session.settings_manager.get_themes())}",
                ]
                append_history("\n".join(lines))
            except Exception as exc:
                append_history(f"{red('Settings failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/scoped-models":
            scoped = getattr(session, "_scoped_models", None) or []
            enabled = session.settings_manager.get_enabled_models() or []
            lines = [bold("Scoped models:")]
            if scoped:
                for item in scoped:
                    model = item.get("model") if isinstance(item, dict) else None
                    if model:
                        lines.append(f"  {model.provider}/{model.id}")
            elif enabled:
                lines.extend(f"  {pattern}" for pattern in enabled)
            else:
                lines.append("  all available models")
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/import" or stripped.startswith("/import "):
            session_path = _path_command_argument(stripped, "/import") or ""
            if not session_path:
                append_history(dim("Usage: /import <path.jsonl>"))
            else:
                try:
                    importer = (
                        getattr(runtime_host, "import_from_jsonl", None)
                        or getattr(runtime_host, "importFromJsonl", None)
                    )
                    if callable(importer):
                        result = await importer(session_path)
                        if result.get("cancelled"):
                            append_history(dim("Import cancelled."))
                        else:
                            append_history(green("Imported session."))
                    else:
                        await session.switch_session(session_path)
                        append_history(green("Imported session."))
                except Exception as exc:
                    append_history(f"{red('Import failed:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/share":
            append_history(dim("Creating secret gist..."))
            tui.request_render()
            try:
                result = await session.share_session()
                append_history(f"{green('Share URL:')} {result['share_url']}\nGist: {result['gist_url']}")
            except Exception as exc:
                append_history(f"{red('Share failed:')} {exc}")
            tui.request_render()
            return

        if stripped == "/trust" or stripped.startswith("/trust "):
            arg = stripped[7:].strip().lower() if stripped.startswith("/trust ") else "yes"
            if arg in {"yes", "true", "trust", "trusted"}:
                session.set_project_trust(True)
                append_history(green("Project trusted."))
            elif arg in {"no", "false", "untrust", "untrusted"}:
                session.set_project_trust(False)
                append_history(yellow("Project marked untrusted."))
            elif arg in {"clear", "unset", "reset"}:
                session.set_project_trust(None)
                append_history(dim("Project trust decision cleared."))
            else:
                append_history(dim("Usage: /trust [yes|no|clear]"))
            tui.request_render()
            return

        if stripped == "/tools":
            names = session.get_active_tool_names()
            if names:
                append_history(bold("Active tools:") + "\n" + "\n".join(f"  - {n}" for n in names))
            else:
                append_history(dim("No active tools."))
            tui.request_render()
            return

        if stripped == "/session":
            stats = session.get_session_stats()
            lines = [
                bold("Session stats:"),
                f"  Session ID:   {stats.get('sessionId', '?')}",
                f"  User msgs:    {stats.get('userMessages', 0)}",
                f"  Asst msgs:    {stats.get('assistantMessages', 0)}",
                f"  Tool calls:   {stats.get('toolCalls', 0)}",
                f"  Total tokens: {_fmt_tokens(stats.get('tokens', {}).get('total', 0))}",
                f"  Cost:         ${stats.get('cost', 0.0):.4f}",
            ]
            append_history("\n".join(lines))
            tui.request_render()
            return

        if stripped == "/thinking":
            new_level = session.cycle_thinking_level()
            if new_level:
                scope = _persist_defaults({"defaultThinkingLevel": new_level})
                append_history(f"{cyan('Thinking level:')} {new_level}  {dim('· saved as ' + scope)}")
            else:
                append_history(dim("Thinking not supported by current model."))
            update_footer()
            tui.request_render()
            return

        if stripped == "/compact" or stripped.startswith("/compact "):
            custom_instructions = stripped[9:].strip() if stripped.startswith("/compact ") else None
            append_history(dim("Compacting context..."))
            tui.request_render()
            try:
                summary = await session.compact(custom_instructions)
                if summary:
                    short = summary[:400] + "..." if len(summary) > 400 else summary
                    append_history(f"{green('Compaction complete.')}\n{dim(short)}")
                else:
                    append_history(dim("Compaction complete (nothing to summarize)."))
            except Exception as exc:
                append_history(f"{red('Compaction error:')} {exc}")
            update_footer()
            tui.request_render()
            return

        if stripped == "/model" or stripped.startswith("/model ") or stripped == "/models" or stripped.startswith("/models "):
            normalized_stripped = "/model" + stripped[len("/models"):] if stripped.startswith("/models") else stripped
            await _handle_model_command(normalized_stripped, session,
                                        append_history, update_footer, tui,
                                        cyan, dim, red, bold, green,
                                        _persist_defaults,
                                        show_extension_selector)
            return

        if stripped == "/set" or stripped.startswith("/set "):
            await _handle_set_command(
                stripped, session, append_history, update_footer, tui,
                show_extension_selector, show_extension_input,
                cyan, dim, red, green, _persist_defaults,
            )
            return

        # ── Busy guard ────────────────────────────────────────────────────────
        if is_busy:
            trace("handle_submit: busy, queue follow-up")
            queued = UserMessage(
                role="user",
                content=[TextContent(type="text", text=stripped)],
                timestamp=int(time.time() * 1000),
            )
            await session.follow_up(queued)
            append_history(f"{dim('Queued follow-up:')} {stripped}")
            tui.request_render()
            return

        is_busy = True
        trace("handle_submit: busy=true")
        _start_working_loader()

        # Show user message in history
        append_history(_user_box(stripped))
        tui.request_render()

        # Collect streaming text and done signal
        collected: list[str] = []
        rendered_response = ""
        response_is_error = False
        pending_tools: dict[str, Any] = {}
        current_run_state = "idle"
        current_run_phase = None
        def on_event(event) -> None:
            """
            Handle AgentEvent (Pydantic models) aligned with TS lifecycle:
              message_start → initialize assistant streaming row
              message_update → render from full assistant snapshot when possible
              message_end → fallback finalization (for non-delta providers)
              agent_end → complete request
              turn_end → surface error message
              auto_retry_start/end → show retry indicator
              auto_compaction_start/end → show compaction indicator
            """
            nonlocal rendered_response, response_is_error, current_run_state, current_run_phase
            try:
                if isinstance(event, dict):
                    etype = event.get("type", "") or ""
                else:
                    etype = getattr(event, "type", None) or ""
                trace(f"on_event: {etype}")

                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) == "assistant":
                        if not stream_text._text:
                            set_stream(f"{bold(_assistant_label(session))} ")

                elif etype == "message_update":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) != "assistant":
                        return

                    snapshot_text = assistant_text_from_message(msg)
                    if snapshot_text:
                        if snapshot_text != rendered_response:
                            trace(f"on_event: snapshot_text len={len(snapshot_text)}")
                            rendered_response = snapshot_text
                            set_stream(f"{bold(_assistant_label(session))} {render_markdown(snapshot_text)}")
                        return

                    ae = getattr(event, "assistant_message_event", None)
                    if getattr(ae, "type", None) == "text_delta":
                        collected.append(getattr(ae, "delta", ""))
                        response_so_far = "".join(collected)
                        if response_so_far != rendered_response:
                            trace(f"on_event: delta_text len={len(response_so_far)}")
                            rendered_response = response_so_far
                            set_stream(f"{bold(_assistant_label(session))} {render_markdown(response_so_far)}")

                elif etype == "message_end":
                    msg = getattr(event, "message", None)
                    if getattr(msg, "role", None) == "assistant":
                        final_text = assistant_text_from_message(msg)
                        if final_text and final_text != rendered_response:
                            trace(f"on_event: final_text len={len(final_text)}")
                            rendered_response = final_text
                            set_stream(f"{bold(_assistant_label(session))} {render_markdown(final_text)}")

                elif etype == "agent_end":
                    if not rendered_response:
                        terminal_messages = getattr(event, "messages", None)
                        if isinstance(terminal_messages, list):
                            for msg in terminal_messages:
                                if getattr(msg, "role", None) != "assistant":
                                    continue
                                err = getattr(msg, "error_message", None)
                                if isinstance(err, str) and err.strip():
                                    trace(f"on_event: agent_end error={err!r}")
                                    set_stream(f"{red('Error:')} {err}")
                                    rendered_response = err
                                    response_is_error = True
                                    break
                                fallback_text = assistant_text_from_message(msg)
                                if fallback_text:
                                    trace(f"on_event: agent_end fallback_text len={len(fallback_text)}")
                                    set_stream(f"{bold(_assistant_label(session))} {render_markdown(fallback_text)}")
                                    rendered_response = fallback_text
                                    break

                elif etype == "turn_end":
                    msg = getattr(event, "message", None)
                    err = getattr(msg, "error_message", None)
                    if err:
                        trace(f"on_event: turn_end error={err!r}")
                        set_stream(f"{red('Error:')} {err}")
                        response_is_error = True

                elif etype == "tool_execution_start":
                    tool_call_id = getattr(event, "tool_call_id", "")
                    tool_name = getattr(event, "tool_name", "tool")
                    args = getattr(event, "args", None)
                    pending_tools[tool_call_id] = (tool_name, args)
                    # Active tool: blue marker (resolves to a green/red box on end).
                    call = _format_tool_call(tool_name, args)
                    append_history(blue(f"⏵ {tool_name}" + (f"  {call}" if call else "")))
                    tui.request_render()

                elif etype == "tool_execution_end":
                    tool_call_id = getattr(event, "tool_call_id", "")
                    entry = pending_tools.pop(tool_call_id, None)
                    if isinstance(entry, tuple):
                        tool_name, args = entry
                    else:
                        tool_name, args = getattr(event, "tool_name", "tool"), None
                    is_error = bool(getattr(event, "is_error", False))
                    result = getattr(event, "result", None)
                    bg = "52" if is_error else "22"  # 256-color: dark red / dark green
                    marker = "✗" if is_error else "✓"
                    header = f"{marker} {tool_name}" + ("  (error)" if is_error else "")
                    call = _format_tool_call(tool_name, args)
                    response_lines: list[str] = []
                    content = getattr(result, "content", None)
                    if isinstance(content, list):
                        text_parts: list[str] = []
                        for c in content:
                            if getattr(c, "type", None) == "text":
                                t = getattr(c, "text", "")
                                if isinstance(t, str) and t:
                                    text_parts.append(t)
                        full = "\n".join(text_parts).rstrip()
                        if full:
                            raw = full.split("\n")
                            if len(raw) > 6:  # truncate: note + last lines
                                response_lines = [f"… (+{len(raw) - 6} earlier lines)"] + raw[-6:]
                            else:
                                response_lines = raw
                    append_history(_tool_box(header, call, response_lines, bg))
                    tui.request_render()

                elif etype == "run_state":
                    current_run_state = (
                        event.get("state", "unknown")
                        if isinstance(event, dict)
                        else getattr(event, "state", "unknown")
                    )
                    current_run_phase = (
                        event.get("phase")
                        if isinstance(event, dict)
                        else getattr(event, "phase", None)
                    )
                    trace(f"on_event: run_state={current_run_state} phase={current_run_phase}")

                elif etype == "auto_retry_start":
                    attempt = event.get("attempt", 0) if isinstance(event, dict) else getattr(event, "attempt", 0)
                    max_a = event.get("maxAttempts", 3) if isinstance(event, dict) else getattr(event, "maxAttempts", 3)
                    delay = event.get("delayMs", 0) if isinstance(event, dict) else getattr(event, "delayMs", 0)
                    err = event.get("errorMessage", "") if isinstance(event, dict) else getattr(event, "errorMessage", "")
                    append_history(f"{yellow(f'Retry {attempt}/{max_a}:')} {dim(str(err))} (wait {delay // 1000}s)")
                    tui.request_render()

                elif etype == "auto_retry_end":
                    success = event.get("success", True) if isinstance(event, dict) else getattr(event, "success", True)
                    if not success:
                        err = event.get("finalError", "") if isinstance(event, dict) else getattr(event, "finalError", "")
                        append_history(f"{red('Retry failed:')} {err}")
                        tui.request_render()

                elif etype == "auto_compaction_start":
                    append_history(dim("Auto-compacting context..."))
                    tui.request_render()

                elif etype == "auto_compaction_end":
                    result = event.get("result") if isinstance(event, dict) else getattr(event, "result", None)
                    err_msg = event.get("errorMessage") if isinstance(event, dict) else getattr(event, "errorMessage", None)
                    if err_msg:
                        append_history(f"{red('Compaction error:')} {err_msg}")
                    else:
                        append_history(dim("Context compacted."))
                    update_footer()
                    tui.request_render()

            except Exception as exc:
                trace(f"on_event: exception={exc!r}")

        unsub = session.subscribe(on_event)
        try:
            trace("handle_submit: before session.prompt")
            await session.prompt(stripped, source="interactive")
            trace("handle_submit: after session.prompt")
        except Exception as exc:
            trace(f"handle_submit: exception={exc!r}")
            phase = f" while {current_run_phase}" if current_run_phase else ""
            set_stream(f"{red('Error:')} {exc}{dim(f' [{current_run_state}{phase}]')}")
        finally:
            trace("handle_submit: finally begin")
            unsub()
            # Move the completed response from stream_text to history. A normal
            # assistant body is rendered through the shared Markdown component
            # (headings yellow, inline code cyan, bold light-cyan); errors and
            # other stream content are moved across verbatim.
            if rendered_response and not response_is_error:
                append_history(f"{bold(_assistant_label(session))} {render_markdown(rendered_response)}")
                set_stream("")
            elif stream_text._text:
                append_history(stream_text._text)
                set_stream("")
            append_history("")  # one line of margin between turns
            is_busy = False
            _stop_working_loader()
            update_footer()
            trace("handle_submit: busy=false")
            tui.request_render()

    # ── Capture main loop for cross-thread callbacks ──────────────────────────
    main_loop = asyncio.get_running_loop()

    async def _cycle_model_async() -> None:
        """Ctrl+P handler: cycle to next available model."""
        try:
            result = await session.cycle_model("forward")
            if result:
                m = result["model"]
                append_history(f"{cyan('Model:')} {m.id} ({m.provider})")
            else:
                append_history(dim("Only one model available."))
            update_footer()
            tui.request_render()
        except Exception as exc:
            append_history(f"{red('Model switch failed:')} {exc}")
            tui.request_render()

    def on_submit_sync(text: str) -> None:
        asyncio.run_coroutine_threadsafe(handle_submit(text), main_loop)

    def on_keydown_sync(key: str) -> None:
        """Handle special key sequences (called from terminal thread)."""
        if key.lower() in extension_shortcuts:
            asyncio.run_coroutine_threadsafe(_execute_extension_shortcut(key), main_loop)
            return
        # Ctrl+P = '\x10'
        if key == "\x10":
            asyncio.run_coroutine_threadsafe(_cycle_model_async(), main_loop)

    editor_runtime_callbacks["on_submit"] = on_submit_sync
    editor_runtime_callbacks["on_keydown"] = on_keydown_sync
    sync_editor_runtime_callbacks()

    # ── ESC interrupts a running agent turn ───────────────────────────────────
    async def _abort_turn() -> None:
        trace("abort: ESC interrupt")
        try:
            if hasattr(session, "abort"):
                await session.abort()
        except Exception as exc:
            trace(f"abort error: {exc}")
        append_history(yellow("⏹ Interrupted."))
        set_stream("")
        tui.request_render()

    def _esc_interrupt_listener(data: str) -> Any:
        # A lone ESC while the agent is working aborts the turn; consume the key
        # so the editor doesn't also act on it. (Escape *sequences* like arrows
        # arrive as "\x1b[..." and are left untouched.)
        if data == "\x1b" and is_busy:
            asyncio.run_coroutine_threadsafe(_abort_turn(), main_loop)
            return {"consume": True}
        return None

    tui.add_input_listener(_esc_interrupt_listener)

    # ── Start TUI ─────────────────────────────────────────────────────────────
    trace("tui: start")
    tui.start()

    if callable(bind_extensions):
        await bind_extensions({"uiContext": extension_ui_context, "mode": "tui"})
    settings_manager = getattr(session, "settings_manager", None)
    get_quiet_startup = getattr(settings_manager, "get_quiet_startup", None) or getattr(settings_manager, "getQuietStartup", None)
    quiet_startup = bool(get_quiet_startup()) if callable(get_quiet_startup) else False
    resource_lines = _loaded_resource_lines(
        session,
        show_listing=not quiet_startup,
        show_diagnostics=True,
    )
    if resource_lines:
        append_history("\n".join(resource_lines))
        tui.request_render()

    # Branded startup banner — rendered after the [Extension issues] section,
    # with one blank line of space before it.
    if not quiet_startup:
        from pi_coding_agent.utils.banner import render_banner

        append_history("\n" + render_banner(color=True).rstrip("\n"))
        tui.request_render()

    # First-run guidance: no global default provider configured yet.
    try:
        get_default_provider = getattr(settings_manager, "get_default_provider", None)
        has_default = bool(get_default_provider()) if callable(get_default_provider) else False
        auth = getattr(session, "auth_storage", None)
        if auth is None or not hasattr(auth, "list_stored_providers"):
            from pi_coding_agent.core.auth_storage import AuthStorage
            auth = AuthStorage()
        stored = auth.list_stored_providers() if hasattr(auth, "list_stored_providers") else []
        if not has_default:
            if stored:
                append_history("\n" + yellow(
                    f"No default provider set. Run /login or /model to choose one "
                    f"(saved globally). You have credentials for: {', '.join(stored)}."
                ))
            else:
                append_history("\n" + yellow(
                    "No provider configured yet. Run /login to set up a provider and "
                    "credentials (saved globally to ~/.tau/agent)."
                ))
            tui.request_render()
    except Exception:
        pass

    if initial_messages:
        trace(f"tui: initial_messages={initial_messages!r}")
        for msg in initial_messages:
            await handle_submit(msg)
            # Yield to event loop so pending render ticks can fire before the
            # next message is processed.
            await asyncio.sleep(0)

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while not tui.stopped:
            await asyncio.sleep(0.05)
    except (KeyboardInterrupt, asyncio.CancelledError):
        trace("tui: keyboard/cancelled")
        pass
    finally:
        if not tui.stopped:
            trace("tui: stop in finally")
            tui.stop()


async def _handle_model_command(
    stripped: str,
    session: "AgentSession",
    append_history,
    update_footer,
    tui,
    cyan, dim, red, bold, green,
    persist_defaults=None,
    show_select=None,
) -> None:
    """Handle /model and /model <id> commands."""
    parts = stripped.split(None, 1)
    model_arg = parts[1].strip() if len(parts) > 1 else None

    if model_arg:
        # /model <id> — switch to named model
        from pi_coding_agent.core.model_resolver import find_exact_model_reference_match

        available = await session.model_registry.get_available()
        target = find_exact_model_reference_match(model_arg, available)
        if target is None:
            append_history(f"{red('Unknown model:')} {model_arg}")
            tui.request_render()
            return
        try:
            await session.set_model(target)
            scope = persist_defaults({"defaultProvider": target.provider, "defaultModel": target.id}) if persist_defaults else "default"
            append_history(f"{cyan('Switched to model:')} {target.id} ({target.provider})  {dim('· saved as ' + scope)}")
            update_footer()
        except Exception as exc:
            append_history(f"{red('Model switch failed:')} {exc}")
        tui.request_render()
        return

    if show_select is None:
        append_history(dim("Use /model <provider/model> or /set <provider> <tier> <model>."))
        tui.request_render()
        return

    selection = await _select_provider_and_strength(show_select)
    if selection is None:
        append_history(dim("Model selection cancelled."))
        tui.request_render()
        return
    provider, strength = selection
    if provider in {"openai-compatible", "anthropic-compatible"}:
        configured_provider = await _select_configured_compatible_provider(provider, show_select)
        if configured_provider is None:
            append_history(_compatible_provider_login_reminder(provider))
            tui.request_render()
            return
        provider = configured_provider
    await _apply_profile_model(
        session, provider, strength, None, append_history, update_footer, tui,
        cyan, dim, red, green, persist_defaults,
    )
    tui.request_render()


async def _handle_set_command(
    stripped: str,
    session: "AgentSession",
    append_history,
    update_footer,
    tui,
    show_select,
    show_input,
    cyan, dim, red, green,
    persist_defaults=None,
) -> None:
    """Handle /set and /set <provider> <tier> <model> commands."""
    from pi_coding_agent.core.provider_profiles import STRENGTHS, normalize_provider_id

    parts = stripped.split()
    if len(parts) >= 4:
        provider = normalize_provider_id(parts[1])
        strength = parts[2].strip().lower()
        if strength not in STRENGTHS:
            append_history(f"{red('Invalid tier:')} {parts[2]} {dim('(use strong, standard, or weak)')}")
            tui.request_render()
            return
        model_id = " ".join(parts[3:]).strip()
        if provider in {"openai-compatible", "anthropic-compatible"}:
            configured_provider = await _select_configured_compatible_provider(provider, show_select)
            if configured_provider is None:
                append_history(_compatible_provider_login_reminder(provider))
                tui.request_render()
                return
            provider = configured_provider
        thinking_level = await _prompt_reasoning_level(show_select, show_input)
        if thinking_level is None and show_select is not None and show_input is not None:
            append_history(dim("Set cancelled."))
            tui.request_render()
            return
        _store_tier_config(provider, strength, model_id, thinking_level)
        reload_registry = getattr(session.model_registry, "reload", None)
        if callable(reload_registry):
            reload_registry()
        append_history(green(f"Set {provider} {strength} to {model_id} (thinking {thinking_level or 'off'})."))
        tui.request_render()
        return

    if show_select is None or show_input is None:
        append_history(dim("Usage: /set <provider> <tier> <model>"))
        tui.request_render()
        return

    selection = await _select_provider_and_strength(show_select)
    if selection is None:
        append_history(dim("Set cancelled."))
        tui.request_render()
        return
    provider, strength = selection
    if provider in {"openai-compatible", "anthropic-compatible"}:
        configured_provider = await _select_configured_compatible_provider(provider, show_select)
        if configured_provider is None:
            append_history(_compatible_provider_login_reminder(provider))
            tui.request_render()
            return
        provider = configured_provider
    default_model = _default_model_for(provider, strength) or ""
    model_id = await show_input("Model ID", default_model, None)
    if not model_id or not str(model_id).strip():
        append_history(dim("Set cancelled."))
        tui.request_render()
        return
    thinking_level = await _prompt_reasoning_level(show_select, show_input)
    if thinking_level is None:
        append_history(dim("Set cancelled."))
        tui.request_render()
        return
    _store_tier_config(provider, strength, str(model_id).strip(), thinking_level)
    reload_registry = getattr(session.model_registry, "reload", None)
    if callable(reload_registry):
        reload_registry()
    append_history(green(f"Set {provider} {strength} to {str(model_id).strip()} (thinking {thinking_level})."))
    tui.request_render()


async def _handle_login_command(
    stripped: str,
    session: "AgentSession",
    append_history,
    update_footer,
    tui,
    show_select,
    show_input,
    cyan, dim, red, green,
    persist_defaults=None,
) -> None:
    """Handle /login provider selection, auth method selection, and direct API-key login."""
    from pi_coding_agent.core.provider_profiles import get_provider_profile, normalize_provider_id

    parts = stripped.split(maxsplit=2)
    if len(parts) >= 3:
        provider = normalize_provider_id(parts[1])
        try:
            session.login_api_key(provider, parts[2])
            scope = persist_defaults({"defaultProvider": provider}) if persist_defaults else "default"
            append_history(green(f"Stored API key for {provider}  {dim('· saved as ' + scope)}"))
        except Exception as exc:
            append_history(f"{red('Login failed:')} {exc}")
        tui.request_render()
        return

    if show_select is None or show_input is None:
        append_history(dim("Usage: /login <provider> <api_key>"))
        tui.request_render()
        return

    provider = normalize_provider_id(parts[1]) if len(parts) == 2 else None
    profile = get_provider_profile(provider) if provider else None
    if profile is None:
        chosen_provider = await _select_provider(show_select, "Login provider")
        if chosen_provider is None:
            append_history(dim("Login cancelled."))
            tui.request_render()
            return
        profile = get_provider_profile(chosen_provider)
    if profile is None:
        append_history(f"{red('Unknown provider:')} {provider}")
        tui.request_render()
        return

    method = "api_key"
    if len(profile.auth_methods) > 1:
        label_by_method = {"subscription": "subscription", "api_key": "api_key"}
        selected = await show_select("Auth method", [label_by_method[m] for m in profile.auth_methods], None)
        if selected is None:
            append_history(dim("Login cancelled."))
            tui.request_render()
            return
        method = str(selected)

    try:
        if method == "subscription":
            await _subscription_login(profile.id, session, append_history, show_input)
            append_history(green(f"Subscription login stored for {profile.label}."))
        elif profile.id in {"openai-compatible", "anthropic-compatible"}:
            provider_id, provider_label = await _compatible_provider_login(
                profile.id,
                session,
                show_input,
            )
            append_history(green(f"{provider_label} stored as {provider_id}."))
            if persist_defaults:
                persist_defaults({"defaultProvider": provider_id})
            update_footer()
            tui.request_render()
            return
        else:
            key = await show_input(
                f"{profile.label} API key (paste, then Enter)",
                "",
                {"secret": True},
            )
            if not key or not str(key).strip():
                append_history(dim("Login cancelled."))
                tui.request_render()
                return
            session.login_api_key(profile.id, str(key).strip())
            append_history(green(f"API key stored for {profile.label}."))
        if persist_defaults:
            persist_defaults({"defaultProvider": profile.id})
        update_footer()
    except Exception as exc:
        append_history(f"{red('Login failed:')} {exc}")
    tui.request_render()


async def _handle_logout_command(
    stripped: str,
    session: "AgentSession",
    append_history,
    update_footer,
    tui,
    show_select,
    dim, red, green,
) -> None:
    """Handle /logout provider selection and direct provider logout."""
    from pi_coding_agent.core.provider_display_names import get_provider_display_name
    from pi_coding_agent.core.provider_profiles import normalize_provider_id

    parts = stripped.split(maxsplit=2)
    provider = normalize_provider_id(parts[1]) if len(parts) >= 2 else None
    credential_type = _normalize_logout_credential_type(parts[2]) if len(parts) >= 3 else None

    auth = getattr(session, "auth_storage", None) or getattr(session, "_auth_storage", None)
    if provider is None:
        stored = auth.list_stored_providers() if auth is not None and hasattr(auth, "list_stored_providers") else []
        if not stored:
            append_history(dim("No stored credentials."))
            tui.request_render()
            return
        if show_select is None:
            append_history(dim("Usage: /logout <provider>"))
            tui.request_render()
            return
        choices = [(str(provider_id), get_provider_display_name(str(provider_id))) for provider_id in sorted(stored)]
        labels = [f"{label} ({provider_id})" for provider_id, label in choices]
        selected = await show_select("Logout provider", labels, None)
        if selected is None:
            append_history(dim("Logout cancelled."))
            tui.request_render()
            return
        selected_label = str(selected)
        for provider_id, label in choices:
            if selected_label == f"{label} ({provider_id})":
                provider = provider_id
                break

    if not provider:
        append_history(dim("Logout cancelled."))
        tui.request_render()
        return
    if credential_type is None:
        choices = _logout_credential_choices(auth, provider)
        if not choices:
            append_history(dim(f"No stored credentials for {provider}."))
            tui.request_render()
            return
        if len(choices) == 1:
            credential_type = choices[0]
        elif show_select is None:
            append_history(dim("Usage: /logout <provider> <api_key|token>"))
            tui.request_render()
            return
        else:
            selected = await show_select("Credential type", choices, None)
            if selected is None:
                append_history(dim("Logout cancelled."))
                tui.request_render()
                return
            credential_type = _normalize_logout_credential_type(str(selected))
    if credential_type not in {"api_key", "token"}:
        append_history(f"{red('Invalid credential type:')} {credential_type} {dim('(use api_key or token)')}")
        tui.request_render()
        return

    try:
        session.logout_provider(provider, credential_type)
        append_history(green(f"Removed stored {credential_type} for {provider}."))
        update_footer()
    except Exception as exc:
        append_history(f"{red('Logout failed:')} {exc}")
    tui.request_render()


def _normalize_logout_credential_type(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower().replace("-", "_")
    if cleaned in {"api_key", "key", "apikey"}:
        return "api_key"
    if cleaned in {"token", "oauth", "subscription"}:
        return "token"
    return cleaned


def _logout_credential_choices(auth: Any, provider: str) -> list[str]:
    choices: list[str] = []
    if auth is not None and hasattr(auth, "get_api_key") and auth.get_api_key(provider):
        choices.append("api_key")
    if auth is not None and hasattr(auth, "get_oauth_token") and auth.get_oauth_token(provider):
        choices.append("token")
    return choices


async def _select_provider_and_strength(show_select) -> tuple[str, str] | None:
    provider = await _select_provider(show_select, "Provider")
    if provider is None:
        return None
    strength = await show_select("Model strength", ["strong", "standard", "weak"], None)
    if strength is None:
        return None
    return provider, str(strength)


async def _select_provider(show_select, title: str) -> str | None:
    from pi_coding_agent.core.provider_profiles import provider_profile_choices

    choices = provider_profile_choices()
    labels = [label for _provider_id, label in choices]
    selected = await show_select(title, labels, None)
    if selected is None:
        return None
    selected_label = str(selected)
    for provider_id, label in choices:
        if label == selected_label:
            return provider_id
    return None


def _default_model_for(provider: str, strength: str) -> str | None:
    from pi_coding_agent.core.provider_profiles import default_model_for

    tier = _tier_config_for(provider, strength)
    if tier and tier.get("model"):
        return str(tier["model"])
    return default_model_for(provider, strength)


def _tier_config_for(provider: str, strength: str) -> dict[str, Any] | None:
    from pi_coding_agent.config import get_models_path

    config = _read_models_config(get_models_path())
    provider_config = config.get("providers", {}).get(provider)
    if not isinstance(provider_config, dict):
        return None
    tiers = provider_config.get("tiers")
    if not isinstance(tiers, dict):
        return None
    tier = tiers.get(strength)
    return tier if isinstance(tier, dict) else None


def _store_tier_config(provider: str, strength: str, model_id: str, thinking_level: str | None) -> None:
    from pi_coding_agent.config import get_models_path
    from pi_coding_agent.core.provider_profiles import get_provider_profile

    models_path = get_models_path()
    config = _read_models_config(models_path)
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        config["providers"] = providers
    provider_config = providers.get(provider)
    if not isinstance(provider_config, dict):
        profile = get_provider_profile(provider)
        provider_config = {"name": profile.label} if profile else {"name": provider}
        providers[provider] = provider_config
    tiers = provider_config.setdefault("tiers", {})
    if not isinstance(tiers, dict):
        tiers = {}
        provider_config["tiers"] = tiers
    tiers[strength] = {
        "model": model_id,
        "thinkingLevel": thinking_level or "off",
    }
    _write_models_config(models_path, config)


def _thinking_level_for_tier(provider: str, strength: str, model: Any) -> str | None:
    tier = _tier_config_for(provider, strength)
    if tier and tier.get("thinkingLevel"):
        return str(tier["thinkingLevel"])
    if bool(getattr(model, "reasoning", False)):
        return "medium"
    return "off"


def _configured_compatible_provider_choices(template_provider: str) -> list[tuple[str, str]]:
    from pi_coding_agent.config import get_models_path
    from pi_coding_agent.core.provider_profiles import get_provider_profile

    profile = get_provider_profile(template_provider)
    if profile is None:
        return []
    config = _read_models_config(get_models_path())
    choices: list[tuple[str, str]] = []
    for provider_id, provider_config in config.get("providers", {}).items():
        if not isinstance(provider_config, dict):
            continue
        if provider_config.get("api") != profile.api:
            continue
        label = str(provider_config.get("name") or provider_id)
        choices.append((str(provider_id), label))
    return sorted(choices, key=lambda item: item[1].lower())


async def _select_configured_compatible_provider(template_provider: str, show_select) -> str | None:
    choices = _configured_compatible_provider_choices(template_provider)
    if not choices or show_select is None:
        return None
    labels = [label for _provider_id, label in choices]
    selected = await show_select("Configured provider", labels, None)
    if selected is None:
        return None
    selected_label = str(selected)
    for provider_id, label in choices:
        if label == selected_label:
            return provider_id
    return None


def _compatible_provider_login_reminder(template_provider: str) -> str:
    label = "OpenAI Compatible" if template_provider == "openai-compatible" else "Anthropic Compatible"
    return f"No {label} providers configured. Run /login and choose {label} first."


async def _prompt_reasoning_level(show_select, show_input) -> str | None:
    if show_select is None or show_input is None:
        return None
    reasoning = await show_select("Reasoning", ["yes", "no"], None)
    if reasoning is None:
        return None
    if str(reasoning).strip().lower() != "yes":
        return "off"
    level = await show_input("Thinking level", "medium", None)
    if level is None:
        return None
    cleaned = str(level).strip()
    return cleaned or "medium"


async def _apply_profile_model(
    session: "AgentSession",
    provider: str,
    strength: str,
    model_id: str | None,
    append_history,
    update_footer,
    tui,
    cyan, dim, red, green,
    persist_defaults=None,
    thinking_level: str | None = None,
) -> None:
    from pi_coding_agent.core.provider_profiles import normalize_provider_id, synthetic_model

    normalized_provider = normalize_provider_id(provider)
    selected_model_id = model_id or _default_model_for(normalized_provider, strength)
    if not selected_model_id:
        append_history(f"{red('Unknown provider:')} {provider}")
        return
    model = None
    finder = getattr(session.model_registry, "find", None)
    if callable(finder):
        model = finder(normalized_provider, selected_model_id)
    if model is None:
        model = synthetic_model(normalized_provider, selected_model_id)
    if model is None:
        append_history(f"{red('Unknown model:')} {normalized_provider}/{selected_model_id}")
        return
    try:
        await session.set_model(model)
    except Exception:
        pass
    updates = {"defaultProvider": normalized_provider, "defaultModel": selected_model_id}
    if thinking_level is not None:
        effective_thinking = thinking_level
    else:
        effective_thinking = _thinking_level_for_tier(normalized_provider, strength, model)
    if effective_thinking is not None:
        set_thinking = getattr(session, "set_thinking_level", None)
        if callable(set_thinking):
            set_thinking(effective_thinking)
        updates["defaultThinkingLevel"] = effective_thinking
    scope = persist_defaults(updates) if persist_defaults else "default"
    append_history(
        f"{cyan('Model:')} {selected_model_id} ({normalized_provider}, {strength}, thinking {updates.get('defaultThinkingLevel', 'off')})  "
        f"{dim('· saved as ' + scope)}"
    )
    update_footer()


def _slug_provider_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "compatible-provider"


def _normalize_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    if not cleaned:
        return cleaned
    if not re.match(r"^https?://", cleaned, re.I):
        cleaned = "https://" + cleaned
    return cleaned


def _read_models_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"providers": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        data["providers"] = {}
    return data


def _write_models_config(path: str, config: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


async def _compatible_provider_login(
    template_provider: str,
    session: "AgentSession",
    show_input,
) -> tuple[str, str]:
    from pi_coding_agent.config import get_models_path
    from pi_coding_agent.core.provider_profiles import get_provider_profile

    profile = get_provider_profile(template_provider)
    if profile is None:
        raise ValueError(f"Unknown compatible provider template: {template_provider}")

    label = await show_input("Provider display name", "MiniMax", None)
    if not label or not str(label).strip():
        raise RuntimeError("Compatible provider setup cancelled")
    provider_label = str(label).strip()
    provider_id = _slug_provider_name(provider_label)

    url = await show_input(f"{provider_label} base URL", profile.base_url, None)
    if not url or not str(url).strip():
        raise RuntimeError("Compatible provider setup cancelled")
    base_url = _normalize_base_url(str(url))

    key = await show_input(
        f"{provider_label} API key (paste, then Enter)",
        "",
        {"secret": True},
    )
    if not key or not str(key).strip():
        raise RuntimeError("Compatible provider setup cancelled")

    auth = getattr(session, "auth_storage", None) or getattr(session, "_auth_storage", None)
    if auth is None or not hasattr(auth, "set_api_key"):
        raise RuntimeError("Session auth storage does not support API keys")
    auth.set_api_key(provider_id, str(key).strip())

    models_path = get_models_path()
    config = _read_models_config(models_path)
    config["providers"][provider_id] = {
        **(config["providers"].get(provider_id) if isinstance(config["providers"].get(provider_id), dict) else {}),
        "name": provider_label,
        "api": profile.api,
        "baseUrl": base_url,
        "models": config["providers"].get(provider_id, {}).get("models", []) if isinstance(config["providers"].get(provider_id), dict) else [],
    }
    _write_models_config(models_path, config)

    reload_session = getattr(session, "reload", None)
    if callable(reload_session):
        await reload_session()
    return provider_id, provider_label


async def _subscription_login(provider: str, session: "AgentSession", append_history, show_input) -> None:
    import webbrowser

    from pi_ai.utils.oauth.types import OAuthLoginCallbacks

    if provider == "openai":
        from pi_ai.utils.oauth.openai_codex import openai_codex_oauth_provider as oauth_provider
    elif provider == "anthropic":
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider as oauth_provider
    elif provider == "google":
        from pi_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider as oauth_provider
    else:
        raise ValueError(f"{provider} does not support subscription login")

    def on_auth(info) -> None:
        opened = False
        try:
            opened = bool(webbrowser.open(info.url))
        except Exception:
            pass
        if opened:
            append_history("Opened browser for subscription login.")
        else:
            append_history(f"Could not open browser automatically. Authorize here:\n{info.url}")

    async def on_prompt(prompt) -> str:
        value = await show_input(prompt.message, getattr(prompt, "placeholder", None), None)
        if value is None and not getattr(prompt, "allow_empty", False):
            raise RuntimeError("OAuth login cancelled")
        return "" if value is None else str(value)

    credentials = await oauth_provider.login(
        OAuthLoginCallbacks(
            on_auth=on_auth,
            on_prompt=on_prompt,
            on_progress=lambda message: append_history(str(message)),
        )
    )
    token = {
        "access_token": credentials.access,
        "refresh_token": credentials.refresh,
        "expires_at": credentials.expires / 1000 if credentials.expires else 0,
        "oauth_provider": getattr(oauth_provider, "id", provider),
        **dict(credentials.extra),
    }
    auth = getattr(session, "auth_storage", None)
    if auth is None:
        auth = getattr(session, "_auth_storage", None)
    if auth is None or not hasattr(auth, "set_oauth_token"):
        raise RuntimeError("Session auth storage does not support OAuth tokens")
    auth.set_oauth_token(provider, token)
    if provider == "google":
        auth.set_oauth_token("google-gemini-cli", token)
