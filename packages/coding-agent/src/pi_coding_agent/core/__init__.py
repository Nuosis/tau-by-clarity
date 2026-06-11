"""Public core API surface for the coding agent package.

Imports are resolved lazily so package-level imports do not create cycles while
``pi_coding_agent`` is still importing its core submodules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentSession": (".agent_session", "AgentSession"),
    "AgentSessionRuntime": (".agent_session_runtime", "AgentSessionRuntime"),
    "AgentSessionRuntimeDiagnostic": (".agent_session_services", "AgentSessionRuntimeDiagnostic"),
    "AgentSessionServices": (".agent_session_services", "AgentSessionServices"),
    "AuthStorage": (".auth_storage", "AuthStorage"),
    "AuthStorageBackend": (".auth_storage", "AuthStorageBackend"),
    "BashResult": (".bash_executor", "BashResult"),
    "CURRENT_SESSION_VERSION": (".session_manager", "CURRENT_SESSION_VERSION"),
    "CompactionSettings": (".settings_manager", "CompactionSettings"),
    "CreateAgentSessionFromServicesOptions": (".agent_session_services", "CreateAgentSessionFromServicesOptions"),
    "CreateAgentSessionOptions": (".sdk", "CreateAgentSessionOptions"),
    "CreateAgentSessionResult": (".sdk", "CreateAgentSessionResult"),
    "CreateAgentSessionRuntimeFactory": (".agent_session_runtime", "CreateAgentSessionRuntimeFactory"),
    "CreateAgentSessionRuntimeResult": (".agent_session_runtime", "CreateAgentSessionRuntimeResult"),
    "CreateAgentSessionServicesOptions": (".agent_session_services", "CreateAgentSessionServicesOptions"),
    "ConfiguredPackage": (".package_manager", "ConfiguredPackage"),
    "DefaultResourceLoader": (".resource_loader", "DefaultResourceLoader"),
    "DefaultResourceLoaderOptions": (".resource_loader", "DefaultResourceLoaderOptions"),
    "EventBus": (".event_bus", "EventBus"),
    "FileAuthStorageBackend": (".auth_storage", "FileAuthStorageBackend"),
    "FileSettingsStorage": (".settings_manager", "FileSettingsStorage"),
    "Extension": (".extensions", "Extension"),
    "ExtensionAPI": (".extensions", "ExtensionAPI"),
    "ExtensionContext": (".extensions", "ExtensionContext"),
    "ExtensionFactory": (".extensions", "ExtensionFactory"),
    "ExtensionRunner": (".extensions", "ExtensionRunner"),
    "ImageSettings": (".settings_manager", "ImageSettings"),
    "InMemoryAuthStorageBackend": (".auth_storage", "InMemoryAuthStorageBackend"),
    "InMemorySettingsStorage": (".settings_manager", "InMemorySettingsStorage"),
    "ModelRegistry": (".model_registry", "ModelRegistry"),
    "PiManifest": (".extensions", "PiManifest"),
    "RegisteredCommand": (".extensions", "RegisteredCommand"),
    "RegisteredTool": (".extensions", "RegisteredTool"),
    "ResourceLoader": (".sdk", "ResourceLoader"),
    "RetrySettings": (".settings_manager", "RetrySettings"),
    "SessionImportFileNotFoundError": (".agent_session_runtime", "SessionImportFileNotFoundError"),
    "SessionManager": (".session_manager", "SessionManager"),
    "Settings": (".settings_manager", "Settings"),
    "SettingsManager": (".settings_manager", "SettingsManager"),
    "SettingsStorage": (".settings_manager", "SettingsStorage"),
    "SourceInfo": (".source_info", "SourceInfo"),
    "SourceOrigin": (".source_info", "SourceOrigin"),
    "SourceScope": (".source_info", "SourceScope"),
    "ToolDefinition": (".extensions", "ToolDefinition"),
    "build_session_context": (".session_manager", "build_session_context"),
    "compact_context": (".compaction", "compact_context"),
    "create_agent_session": (".sdk", "create_agent_session"),
    "create_agent_session_from_services": (".agent_session_services", "create_agent_session_from_services"),
    "create_agent_session_runtime": (".agent_session_runtime", "create_agent_session_runtime"),
    "create_agent_session_services": (".agent_session_services", "create_agent_session_services"),
    "create_all_tools": (".tools", "create_all_tools"),
    "create_bash_tool": (".tools", "create_bash_tool"),
    "create_bash_tool_definition": (".tools", "create_bash_tool_definition"),
    "create_event_bus": (".event_bus", "create_event_bus"),
    "create_coding_tools": (".tools", "create_coding_tools"),
    "create_edit_tool": (".tools", "create_edit_tool"),
    "create_edit_tool_definition": (".tools", "create_edit_tool_definition"),
    "create_find_tool": (".tools", "create_find_tool"),
    "create_find_tool_definition": (".tools", "create_find_tool_definition"),
    "create_grep_tool": (".tools", "create_grep_tool"),
    "create_grep_tool_definition": (".tools", "create_grep_tool_definition"),
    "create_ls_tool": (".tools", "create_ls_tool"),
    "create_ls_tool_definition": (".tools", "create_ls_tool_definition"),
    "create_read_tool": (".tools", "create_read_tool"),
    "create_read_tool_definition": (".tools", "create_read_tool_definition"),
    "create_read_only_tools": (".tools", "create_read_only_tools"),
    "create_tool": (".tools", "create_tool"),
    "create_tool_definition": (".tools", "create_tool_definition"),
    "create_write_tool": (".tools", "create_write_tool"),
    "create_write_tool_definition": (".tools", "create_write_tool_definition"),
    "create_source_info": (".source_info", "create_source_info"),
    "create_synthetic_source_info": (".source_info", "create_synthetic_source_info"),
    "discover_and_load_extensions": (".extensions", "discover_and_load_extensions"),
    "execute_bash": (".bash_executor", "execute_bash"),
    "load_extensions": (".extensions", "load_extensions"),
    "should_compact": (".compaction", "should_compact"),
    "source_info_to_dict": (".source_info", "source_info_to_dict"),
}

_ALIASES = {
    "createAgentSession": "create_agent_session",
    "createAgentSessionRuntime": "create_agent_session_runtime",
    "createAgentSessionServices": "create_agent_session_services",
    "createAgentSessionFromServices": "create_agent_session_from_services",
    "createEventBus": "create_event_bus",
    "createAllTools": "create_all_tools",
    "createBashTool": "create_bash_tool",
    "createBashToolDefinition": "create_bash_tool_definition",
    "createCodingTools": "create_coding_tools",
    "createEditTool": "create_edit_tool",
    "createEditToolDefinition": "create_edit_tool_definition",
    "createFindTool": "create_find_tool",
    "createFindToolDefinition": "create_find_tool_definition",
    "createGrepTool": "create_grep_tool",
    "createGrepToolDefinition": "create_grep_tool_definition",
    "createLsTool": "create_ls_tool",
    "createLsToolDefinition": "create_ls_tool_definition",
    "createReadTool": "create_read_tool",
    "createReadToolDefinition": "create_read_tool_definition",
    "createReadOnlyTools": "create_read_only_tools",
    "createTool": "create_tool",
    "createToolDefinition": "create_tool_definition",
    "createWriteTool": "create_write_tool",
    "createWriteToolDefinition": "create_write_tool_definition",
    "createSourceInfo": "create_source_info",
    "createSyntheticSourceInfo": "create_synthetic_source_info",
}


def __getattr__(name: str) -> Any:
    target = _ALIASES.get(name)
    if target is not None:
        value = __getattr__(target)
        globals()[name] = value
        return value

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted([*_EXPORTS, *_ALIASES])
