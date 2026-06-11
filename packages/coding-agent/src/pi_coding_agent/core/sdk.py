"""
SDK factory — mirrors packages/coding-agent/src/core/sdk.ts

Public API for creating AgentSession instances programmatically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pi_agent.types import AgentTool, ThinkingLevel
from pi_ai.types import Model

from .agent_session import AgentSession
from .auth_storage import AuthStorage
from .extensions import ToolDefinition
from .model_registry import ModelRegistry
from .session_manager import SessionManager
from .settings_manager import Settings, SettingsManager


class ResourceLoader(Protocol):
    """Protocol for resource loaders."""
    def get_extensions(self) -> dict[str, Any] | None: ...
    def get_skills(self) -> dict[str, Any]: ...
    def get_agents_files(self) -> dict[str, Any]: ...
    def get_system_prompt(self) -> str | None: ...
    def get_append_system_prompt(self) -> list[str]: ...
    async def reload(self) -> None: ...


@dataclass
class CreateAgentSessionOptions:
    """
    Options for creating an AgentSession.
    Mirrors CreateAgentSessionOptions in TypeScript.
    """
    # Working directory for project-local discovery. Default: os.getcwd()
    cwd: str | None = None
    # Global config directory. Default: ~/.pi/agent
    agent_dir: str | None = None
    
    # Auth storage for credentials
    auth_storage: AuthStorage | None = None
    # Model registry
    model_registry: ModelRegistry | None = None
    
    # Model to use. Default: from settings, else first available
    model: Model | None = None
    # Thinking level. Default: from settings, else 'medium' (clamped to model capabilities)
    thinking_level: ThinkingLevel | None = None
    # Models available for cycling (Ctrl+P in interactive mode)
    scoped_models: list[dict[str, Model | ThinkingLevel | None]] | None = None
    
    # Optional allowlist of tool names. Legacy AgentTool lists are also accepted.
    tools: list[str] | list[AgentTool] | None = None
    # Optional denylist of tool names to disable after the allowlist/default.
    exclude_tools: list[str] | None = None
    # Optional default tool suppression mode. True is treated as "all".
    no_tools: Literal["all", "builtin"] | bool | None = None
    # Custom tools to register (in addition to built-in tools)
    custom_tools: list[ToolDefinition] | None = None
    
    # Resource loader. When omitted, DefaultResourceLoader is used
    resource_loader: ResourceLoader | None = None
    
    # Session manager
    session_manager: SessionManager | None = None
    
    # Settings manager
    settings_manager: SettingsManager | None = None
    # Per-session variables substituted into prompt text, e.g. {ACTIVE_PATH}.
    session_vars: dict[str, str] | None = None
    # Session start event metadata for extension runtime startup.
    session_start_event: dict[str, Any] | None = None


@dataclass
class CreateAgentSessionResult:
    """
    Result from create_agent_session.
    Mirrors CreateAgentSessionResult in TypeScript.
    """
    # The created session
    session: AgentSession
    # Extensions result (for UI context setup in interactive mode)
    extensions_result: dict[str, Any] | None = None
    # Warning if session was restored with a different model than saved
    model_fallback_message: str | None = None


async def create_agent_session(
    options: CreateAgentSessionOptions | None = None
) -> CreateAgentSessionResult:
    """
    Create an AgentSession with the specified options.
    Mirrors createAgentSession() in TypeScript.
    
    Example:
        # Minimal - uses defaults
        result = await create_agent_session()
        session = result.session
        
        # With explicit model
        from pi_ai import get_model
        result = await create_agent_session(
            CreateAgentSessionOptions(
                model=get_model('anthropic', 'claude-opus-4-5'),
                thinking_level='high',
            )
        )
    """
    if options is None:
        options = CreateAgentSessionOptions()
    
    cwd = options.cwd or os.getcwd()
    agent_dir = options.agent_dir  # Will use default in components if None
    
    # Use provided or create AuthStorage and ModelRegistry
    auth_storage = options.auth_storage or AuthStorage()
    model_registry = options.model_registry or ModelRegistry()
    
    settings_manager = options.settings_manager or SettingsManager.create(cwd=cwd, agent_dir=agent_dir)
    session_manager = options.session_manager or SessionManager.create(cwd=cwd)
    
    resource_loader = options.resource_loader
    if not resource_loader:
        # Import here to avoid circular dependency
        from .resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
        loader_opts = DefaultResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            settings_manager=settings_manager,
        )
        resource_loader = DefaultResourceLoader(loader_opts)
        await resource_loader.reload()
    
    # Check if session has existing data to restore
    existing_session = session_manager.build_context()
    has_existing_session = len(existing_session.messages) > 0
    
    model = options.model
    model_fallback_message: str | None = None
    
    # If session has data, try to restore model from it
    if not model and has_existing_session:
        saved_model = existing_session.model
        if saved_model:
            try:
                from pi_ai import get_model
                restored_model = get_model(saved_model["provider"], saved_model["model_id"])
                if model_registry.get_api_key(saved_model["provider"]):
                    model = restored_model
            except Exception:
                model_fallback_message = f"Could not restore model {saved_model['provider']}/{saved_model['model_id']}"
    
    # If still no model, use default
    if not model:
        try:
            model = model_registry.resolve_model(
                model_id=settings_manager.get_default_model(),
                provider=settings_manager.get_default_provider(),
            )
        except Exception:
            pass
    
    thinking_level = options.thinking_level
    if thinking_level is None:
        # Restore from session or use settings default
        if has_existing_session:
            thinking_level = existing_session.thinking_level or "medium"
        else:
            thinking_level = settings_manager.get_default_thinking_level() or "medium"
    
    # Clamp to model capabilities
    if not model or not getattr(model, "reasoning", False):
        thinking_level = "off"
    
    settings = settings_manager.get()
    settings.thinking_level = thinking_level
    settings.model_id = model.id if model else None
    settings.provider = model.provider if model else None
    if options.session_vars:
        merged_session_vars = dict(settings.session_vars or {})
        merged_session_vars.update({str(k): str(v) for k, v in options.session_vars.items()})
        settings.session_vars = merged_session_vars

    configured_tools = options.tools
    tools_from_settings = False
    if configured_tools is None and not options.no_tools:
        get_tools = getattr(settings_manager, "get_tools", None)
        configured_tools = get_tools() if callable(get_tools) else None
        tools_from_settings = configured_tools is not None
    
    initial_active_tool_names = _resolve_initial_active_tool_names(
        tools=configured_tools,
        no_tools=options.no_tools,
        exclude_tools=options.exclude_tools,
    )

    custom_tool_names = _extract_tool_names(options.custom_tools)
    if custom_tool_names and (configured_tools is None or tools_from_settings) and not options.no_tools:
        initial_active_tool_names.extend(
            name for name in custom_tool_names if name not in initial_active_tool_names
        )
    extension_tool_names = _extract_extension_tool_names(resource_loader)
    if extension_tool_names and (configured_tools is None or tools_from_settings) and not options.no_tools:
        excluded = set(options.exclude_tools or [])
        initial_active_tool_names.extend(
            name
            for name in extension_tool_names
            if name not in excluded and name not in initial_active_tool_names
        )

    # Create session with all options
    session = AgentSession(
        cwd=cwd,
        model=model,
        settings=settings,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        custom_tools=options.custom_tools,
        initial_active_tool_names=initial_active_tool_names,
        session_start_event=options.session_start_event,
    )
    
    # Apply scoped models if provided
    if options.scoped_models:
        session.set_scoped_models(options.scoped_models)
    
    extensions_result = resource_loader.get_extensions() if hasattr(resource_loader, "get_extensions") else None
    
    return CreateAgentSessionResult(
        session=session,
        extensions_result=extensions_result,
        model_fallback_message=model_fallback_message,
    )


def _extract_tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if isinstance(tool, str):
            names.append(tool)
        else:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.append(name)
    return names


def _extract_extension_tool_names(resource_loader: Any) -> list[str]:
    get_extensions = getattr(resource_loader, "get_extensions", None)
    if not callable(get_extensions):
        return []
    result = get_extensions() or {}
    extensions = result.get("extensions") if isinstance(result, dict) else getattr(result, "extensions", [])
    names: list[str] = []
    for extension in extensions or []:
        tools = getattr(extension, "tools", {}) or {}
        tool_values = tools.values() if isinstance(tools, dict) else tools
        for tool in tool_values or []:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def _resolve_initial_active_tool_names(
    *,
    tools: list[str] | list[AgentTool] | None,
    no_tools: Literal["all", "builtin"] | bool | None,
    exclude_tools: list[str] | None,
) -> list[str]:
    default_active = ["read", "bash", "edit", "write"]
    if tools is not None:
        active = _extract_tool_names(tools)
    elif no_tools:
        active = []
    else:
        active = list(default_active)

    excluded = set(exclude_tools or [])
    return [name for name in active if name not in excluded]


from .agent_session_services import (  # noqa: E402
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)
from .agent_session_runtime import (  # noqa: E402
    AgentSessionRuntime,
    CreateAgentSessionRuntimeFactory,
    CreateAgentSessionRuntimeResult,
    SessionImportFileNotFoundError,
    create_agent_session_runtime,
)
from .tools import (  # noqa: E402
    create_all_tools,
    create_bash_tool_definition,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool_definition,
    create_edit_tool,
    create_find_tool_definition,
    create_find_tool,
    create_grep_tool_definition,
    create_grep_tool,
    create_ls_tool_definition,
    create_ls_tool,
    create_read_tool_definition,
    create_read_only_tools,
    create_read_tool,
    create_write_tool_definition,
    create_write_tool,
    with_file_mutation_queue,
)

createAllTools = create_all_tools
createBashToolDefinition = create_bash_tool_definition
createBashTool = create_bash_tool
createCodingTools = create_coding_tools
createEditToolDefinition = create_edit_tool_definition
createEditTool = create_edit_tool
createFindToolDefinition = create_find_tool_definition
createFindTool = create_find_tool
createGrepToolDefinition = create_grep_tool_definition
createGrepTool = create_grep_tool
createLsToolDefinition = create_ls_tool_definition
createLsTool = create_ls_tool
createReadToolDefinition = create_read_tool_definition
createReadOnlyTools = create_read_only_tools
createReadTool = create_read_tool
createWriteToolDefinition = create_write_tool_definition
createWriteTool = create_write_tool
withFileMutationQueue = with_file_mutation_queue
