"""
Extension types — mirrors packages/coding-agent/src/core/extensions/types.ts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


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
    key: str
    description: str = ""
    handler: Callable[..., Any] | None = None
    extension_path: str = ""


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


# ============================================================================
# Extension API
# ============================================================================

class ExtensionContext:
    """Context passed to extension handlers."""

    def __init__(
        self,
        cwd: str = "",
        session_id: str = "",
        model: Any = None,
        messages: list[Any] | None = None,
    ) -> None:
        self.cwd = cwd
        self.session_id = session_id
        self.model = model
        self.messages = messages or []


class ExtensionAPI:
    """
    API exposed to extension factory functions.
    Mirrors the pi object passed to extension factories in TS.
    """

    def __init__(self, extension: Extension) -> None:
        self._extension = extension

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
        )

    def register_command(
        self,
        name: str,
        description: str = "",
        handler: Callable[..., Any] | None = None,
        get_argument_completions: Callable[[str], list[Any] | None] | None = None,
    ) -> None:
        """Register a slash command."""
        self._extension.commands[name] = RegisteredCommand(
            name=name,
            description=description,
            handler=handler,
            get_argument_completions=get_argument_completions,
            extension_path=self._extension.path,
        )

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
        description: str = "",
        type: str = "boolean",
        default: bool | str | None = False,
    ) -> None:
        """Register a feature flag."""
        self._extension.flags[name] = ExtensionFlag(
            name=name,
            description=description,
            type=type,
            default=default,
            extension_path=self._extension.path,
        )

    def register_shortcut(
        self,
        key: str,
        description: str = "",
        handler: Callable | None = None,
    ) -> None:
        """Register a keyboard shortcut."""
        self._extension.shortcuts[key] = ExtensionShortcut(
            key=key,
            description=description,
            handler=handler,
            extension_path=self._extension.path,
        )


ExtensionFactory = Callable[[ExtensionAPI], None | Awaitable[None]]
