"""
Extension types — mirrors packages/coding-agent/src/core/extensions/types.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from pi_coding_agent.core.source_info import SourceInfo, create_synthetic_source_info


# ============================================================================
# Manifest
# ============================================================================

@dataclass
class PiManifest:
    """Pi manifest from package.json / pyproject.toml."""
    extensions: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)


# ============================================================================
# Tool Types
# ============================================================================

@dataclass
class ToolDefinition:
    """Tool definition for registerTool()."""
    name: str
    label: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    execution_mode: str | None = None
    execution_policy: dict[str, Any] | None = None
    execute: Callable | None = None
    render_call: Callable | None = None
    render_result: Callable | None = None
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] | None = None


# Alias for backward compatibility
@dataclass
class RegisteredTool:
    """A tool registered by an extension."""
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    execute: Callable[..., Awaitable[Any]] | None = None
    extension_path: str = ""


# ============================================================================
# Command & Flag Types
# ============================================================================

@dataclass
class RegisteredCommand:
    """A command registered by an extension."""
    name: str
    description: str = ""
    handler: Callable[..., Any] | None = None
    get_argument_completions: Callable[[str], list[Any] | None] | None = None
    extension_path: str = ""
    source_info: SourceInfo | None = None


@dataclass
class ExtensionFlag:
    """A feature flag registered by an extension."""
    name: str
    description: str = ""
    type: str = "boolean"  # "boolean" | "string"
    default: bool | str | None = False
    extension_path: str = ""


@dataclass
class ExtensionShortcut:
    """A keyboard shortcut registered by an extension."""
    shortcut: str
    description: str = ""
    handler: Callable[..., Any] | None = None
    extension_path: str = ""

    @property
    def key(self) -> str:
        return self.shortcut


# ============================================================================
# Session Events
# ============================================================================

@dataclass
class SessionStartEvent:
    type: str = "session_start"


@dataclass
class SessionBeforeSwitchEvent:
    type: str = "session_before_switch"
    reason: str = "new"
    target_session_file: str | None = None


@dataclass
class SessionSwitchEvent:
    type: str = "session_switch"
    reason: str = "new"
    previous_session_file: str | None = None


@dataclass
class SessionBeforeForkEvent:
    type: str = "session_before_fork"
    entry_id: str = ""


@dataclass
class SessionForkEvent:
    type: str = "session_fork"
    previous_session_file: str | None = None


@dataclass
class SessionBeforeCompactEvent:
    type: str = "session_before_compact"
    preparation: Any = None
    branch_entries: list[Any] = field(default_factory=list)
    custom_instructions: str | None = None
    signal: Any = None


@dataclass
class SessionCompactEvent:
    type: str = "session_compact"
    compaction_entry: Any = None
    from_extension: bool = False


@dataclass
class SessionShutdownEvent:
    type: str = "session_shutdown"


@dataclass
class SessionBeforeTreeEvent:
    type: str = "session_before_tree"
    preparation: Any = None
    signal: Any = None


@dataclass
class SessionTreeEvent:
    type: str = "session_tree"
    new_leaf_id: str | None = None
    old_leaf_id: str | None = None
    summary_entry: Any = None
    from_extension: bool = False


# ============================================================================
# Input Events
# ============================================================================

@dataclass
class InputEvent:
    text: str = ""
    source: str = "interactive"
    images: list[Any] = field(default_factory=list)
    type: str = "input"


@dataclass
class InputEventResult:
    action: str = "continue"  # "continue" | "transform" | "handled"
    text: str | None = None
    images: list[Any] | None = None


# ============================================================================
# Tool Events
# ============================================================================

@dataclass
class ToolCallEvent:
    """Fired before a tool executes."""
    type: str = "tool_call"
    tool_call_id: str = ""
    tool_name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEventResult:
    block: bool = False
    reason: str | None = None
    # Optional: rewrite the tool-call arguments before execution. When a handler
    # returns "arguments" (or "input"), the runtime uses it in place of the
    # original args (e.g. a PII filter reapplying real values for tokens).
    arguments: dict[str, Any] | None = None
    input: dict[str, Any] | None = None


@dataclass
class ToolResultEvent:
    """Fired after a tool executes."""
    type: str = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    content: list[Any] = field(default_factory=list)
    is_error: bool = False
    details: Any = None


@dataclass
class ToolResultEventResult:
    content: list[Any] | None = None
    details: Any = None
    is_error: bool | None = None


# ============================================================================
# Agent Events
# ============================================================================

@dataclass
class BeforeAgentStartEvent:
    type: str = "before_agent_start"
    messages: list[Any] = field(default_factory=list)


@dataclass
class BeforeAgentStartEventResult:
    message: Any = None
    system_prompt: str | None = None


@dataclass
class AgentEndEvent:
    type: str = "agent_end"
    messages: list[Any] = field(default_factory=list)


@dataclass
class ResourcesDiscoverEvent:
    type: str = "resources_discover"


@dataclass
class ResourcesDiscoverResult:
    skill_paths: list[str] | None = None
    prompt_paths: list[str] | None = None
    theme_paths: list[str] | None = None


# ============================================================================
# Context Usage
# ============================================================================

@dataclass
class ContextUsage:
    tokens: int | None = None
    context_window: int = 0
    percent: float | None = None


# ============================================================================
# Extension Container
# ============================================================================

HandlerFn = Callable[..., Any | Awaitable[Any]]


@dataclass
class Extension:
    """A loaded extension."""
    path: str
    resolved_path: str
    handlers: dict[str, list[HandlerFn]] = field(default_factory=dict)
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    message_renderers: dict[str, Callable] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, ExtensionShortcut] = field(default_factory=dict)
    source_info: SourceInfo | None = None


# ============================================================================
# Extension API
# ============================================================================

class ExtensionUIContext:
    """UI methods exposed on ctx.ui for extensions."""

    async def select(self, title: str, options: list[str], opts: Any = None) -> str | None:
        return None

    async def confirm(self, title: str, message: str, opts: Any = None) -> bool:
        return False

    async def input(self, title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
        return None

    def notify(self, message: str, notify_type: str | None = None) -> None:
        return None

    def on_terminal_input(self, handler: Any) -> Callable[[], None]:
        return lambda: None

    def set_status(self, key: str, text: str | None) -> None:
        return None

    def set_working_message(self, message: str | None = None) -> None:
        return None

    def set_working_visible(self, visible: bool) -> None:
        return None

    def set_working_indicator(self, options: Any = None) -> None:
        return None

    def set_hidden_thinking_label(self, label: str | None = None) -> None:
        return None

    def set_widget(self, key: str, content: Any, options: Any = None) -> None:
        return None

    def set_footer(self, factory: Any) -> None:
        return None

    def set_header(self, factory: Any) -> None:
        return None

    def set_title(self, title: str) -> None:
        return None

    async def custom(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def paste_to_editor(self, text: str) -> None:
        self.set_editor_text(text)

    def set_editor_text(self, text: str) -> None:
        return None

    def get_editor_text(self) -> str:
        return ""

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        return None

    def add_autocomplete_provider(self, factory: Any) -> None:
        return None

    def set_editor_component(self, factory: Any = None) -> None:
        return None

    def get_editor_component(self) -> Any:
        return None

    @property
    def theme(self) -> Any:
        return {}

    def get_all_themes(self) -> list[Any]:
        return []

    def get_theme(self, name: str) -> Any:
        return None

    def set_theme(self, theme_val: Any) -> dict[str, Any]:
        return {"success": False, "error": "Theme switching not supported"}

    def get_tools_expanded(self) -> bool:
        return False

    def set_tools_expanded(self, expanded: bool) -> None:
        return None

    def onTerminalInput(self, handler: Any) -> Callable[[], None]:
        return self.on_terminal_input(handler)

    def setStatus(self, key: str, text: str | None) -> None:
        return self.set_status(key, text)

    def setWorkingMessage(self, message: str | None = None) -> None:
        return self.set_working_message(message)

    def setWorkingVisible(self, visible: bool) -> None:
        return self.set_working_visible(visible)

    def setWorkingIndicator(self, options: Any = None) -> None:
        return self.set_working_indicator(options)

    def setHiddenThinkingLabel(self, label: str | None = None) -> None:
        return self.set_hidden_thinking_label(label)

    def setWidget(self, key: str, content: Any, options: Any = None) -> None:
        return self.set_widget(key, content, options)

    def setFooter(self, factory: Any) -> None:
        return self.set_footer(factory)

    def setHeader(self, factory: Any) -> None:
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


class ExtensionContext:
    """Context passed to extension handlers."""

    def __init__(
        self,
        cwd: str = "",
        session_id: str = "",
        model: Any = None,
        messages: list[Any] | None = None,
        stale_message_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._stale_message_getter = stale_message_getter
        self.cwd = cwd
        self.session_id = session_id
        self.model = model
        self.messages = messages or []

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_") or name in {"__class__", "__dict__", "__setattr__", "__getattribute__"}:
            return object.__getattribute__(self, name)
        getter = object.__getattribute__(self, "_stale_message_getter")
        if getter is not None:
            message = getter()
            if message:
                raise RuntimeError(message)
        return object.__getattribute__(self, name)


class ExtensionAPI:
    """
    API exposed to extension factory functions.
    Mirrors the pi object passed to extension factories in TS.
    """

    def __init__(self, extension: Extension, runtime: dict[str, Any] | None = None) -> None:
        self._extension = extension
        self._runtime = runtime if runtime is not None else {"flagValues": {}}

    def on(self, event_type: str, handler: HandlerFn) -> None:
        """Register an event handler."""
        if event_type not in self._extension.handlers:
            self._extension.handlers[event_type] = []
        self._extension.handlers[event_type].append(handler)

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: Callable[..., Awaitable[Any]],
        label: str = "",
        prompt_snippet: str | None = None,
        prompt_guidelines: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a custom tool."""
        self._extension.tools[name] = ToolDefinition(
            name=name,
            label=label or name,
            description=description,
            parameters=parameters,
            execute=execute,
            prompt_snippet=prompt_snippet,
            prompt_guidelines=prompt_guidelines,
            execution_mode=kwargs.get("execution_mode", kwargs.get("executionMode")),
            execution_policy=kwargs.get("execution_policy", kwargs.get("executionPolicy")),
        )

    def register_command(
        self,
        name: str,
        description: str | dict[str, Any] = "",
        handler: Callable[..., Any] | None = None,
        get_argument_completions: Callable[[str], list[Any] | None] | None = None,
    ) -> None:
        """Register a slash command."""
        if isinstance(description, dict):
            options = description
            description = str(options.get("description") or "")
            handler = options.get("handler", handler)
            get_argument_completions = (
                options.get("getArgumentCompletions")
                or options.get("get_argument_completions")
                or get_argument_completions
            )
        self._extension.commands[name] = RegisteredCommand(
            name=name,
            description=description,
            handler=handler,
            get_argument_completions=get_argument_completions,
            extension_path=self._extension.path,
            source_info=self._extension.source_info
            or create_synthetic_source_info(self._extension.path, source="local"),
        )

    registerCommand = register_command

    def register_message_renderer(
        self,
        custom_type: str,
        renderer: Callable,
    ) -> None:
        """Register a custom message renderer."""
        self._extension.message_renderers[custom_type] = renderer

    def register_flag(
        self,
        name: str,
        description: str | dict[str, Any] = "",
        type: str = "boolean",
        default: bool | str | None = False,
    ) -> None:
        """Register a feature flag."""
        if isinstance(description, dict):
            options = description
            description = str(options.get("description") or "")
            type = str(options.get("type") or type)
            default = options.get("default", default)
        self._extension.flags[name] = ExtensionFlag(
            name=name,
            description=description,
            type=type,
            default=default,
            extension_path=self._extension.path,
        )
        flag_values = self._runtime.setdefault("flagValues", {})
        if default is not None and name not in flag_values:
            flag_values[name] = default

    def get_flag(self, name: str) -> bool | str | None:
        """Get an extension flag value for a flag registered by this extension."""
        if name not in self._extension.flags:
            return None
        return self._runtime.get("flagValues", {}).get(name)

    def register_provider(self, name: str, config: dict[str, Any]) -> None:
        """Queue a model provider registration from this extension."""
        self._runtime.setdefault("pendingProviderRegistrations", []).append(
            {
                "name": name,
                "config": config,
                "extensionPath": self._extension.path,
            }
        )

    def unregister_provider(self, name: str) -> None:
        """Remove a pending provider registration by name."""
        pending = self._runtime.setdefault("pendingProviderRegistrations", [])
        self._runtime["pendingProviderRegistrations"] = [
            item for item in pending if item.get("name") != name
        ]

    registerFlag = register_flag
    getFlag = get_flag
    registerProvider = register_provider
    unregisterProvider = unregister_provider

    def register_shortcut(
        self,
        key: str,
        description: str | dict[str, Any] = "",
        handler: Callable | None = None,
    ) -> None:
        """Register a keyboard shortcut."""
        if isinstance(description, dict):
            options = description
            description = str(options.get("description") or "")
            handler = options.get("handler", handler)
        self._extension.shortcuts[key] = ExtensionShortcut(
            shortcut=key,
            description=description,
            handler=handler,
            extension_path=self._extension.path,
        )

    registerShortcut = register_shortcut

    # ── Core session/tool/command queries ─────────────────────────────────────
    # These delegate to callables injected by ExtensionRunner.bind_core(). They
    # return empty / no-op values before the core is bound (e.g. when called at
    # factory time, before the agent session is ready). Mirrors the runtime-
    # delegated methods on the TS ExtensionAPI (getCommands/getAllTools/…).

    def get_active_tools(self) -> list[str]:
        """Names of the tools currently active for the agent."""
        action = self._runtime.get("getActiveTools")
        return list(action()) if callable(action) else []

    def get_all_tools(self) -> list[dict[str, Any]]:
        """All registered tools (built-in + extension) with metadata."""
        action = self._runtime.get("getAllTools")
        return list(action()) if callable(action) else []

    def get_all_tool_names(self) -> list[str]:
        """Names of all registered tools, whether active or not."""
        action = self._runtime.get("getAllToolNames")
        return list(action()) if callable(action) else []

    def set_active_tools(self, tool_names: list[str]) -> None:
        """Replace the active tool set by name (rebuilds the system prompt)."""
        action = self._runtime.get("setActiveTools")
        if callable(action):
            action(list(tool_names))

    def get_commands(self) -> list[dict[str, Any]]:
        """All slash commands (extension + prompt + skill) with their source."""
        action = self._runtime.get("getCommands")
        return list(action()) if callable(action) else []

    def append_entry(self, custom_type: str, data: Any = None) -> str:
        """Append a custom entry to the session (extension state, not LLM context)."""
        action = self._runtime.get("appendEntry")
        return action(custom_type, data) if callable(action) else ""

    getActiveTools = get_active_tools
    getAllTools = get_all_tools
    getAllToolNames = get_all_tool_names
    setActiveTools = set_active_tools
    getCommands = get_commands
    appendEntry = append_entry


ExtensionFactory = Callable[[ExtensionAPI], None | Awaitable[None]]
