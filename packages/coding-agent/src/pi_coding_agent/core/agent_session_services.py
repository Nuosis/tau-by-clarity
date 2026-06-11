"""Runtime service factory for cwd-bound agent session infrastructure."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.config import get_agent_dir
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_registry import ModelRegistry
from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
from pi_coding_agent.core.sdk import (
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    create_agent_session,
)
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


@dataclass
class AgentSessionRuntimeDiagnostic:
    type: str
    message: str


@dataclass
class CreateAgentSessionServicesOptions:
    cwd: str
    agent_dir: str | None = None
    auth_storage: AuthStorage | None = None
    settings_manager: SettingsManager | None = None
    model_registry: ModelRegistry | None = None
    extension_flag_values: dict[str, bool | str] | None = None
    resource_loader_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSessionServices:
    cwd: str
    agent_dir: str
    auth_storage: AuthStorage
    settings_manager: SettingsManager
    model_registry: ModelRegistry
    resource_loader: Any
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


@dataclass
class CreateAgentSessionFromServicesOptions:
    services: AgentSessionServices
    session_manager: SessionManager
    session_start_event: dict[str, Any] | None = None
    model: Any = None
    thinking_level: Any = None
    scoped_models: list[dict[str, Any]] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: str | bool | None = None
    custom_tools: list[Any] | None = None


def _extension_iterable(extensions_result: Any) -> list[Any]:
    if isinstance(extensions_result, dict):
        return list(extensions_result.get("extensions") or [])
    return list(getattr(extensions_result, "extensions", []) or [])


def _apply_extension_flag_values(
    resource_loader: Any,
    extension_flag_values: dict[str, bool | str] | None,
) -> list[AgentSessionRuntimeDiagnostic]:
    if not extension_flag_values:
        return []

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    registered_flags: dict[str, str] = {}

    for extension in _extension_iterable(extensions_result):
        flags = getattr(extension, "flags", {})
        flag_values = flags.values() if isinstance(flags, dict) else flags
        for flag in flag_values or []:
            name = getattr(flag, "name", None)
            if isinstance(name, str):
                registered_flags[name] = getattr(flag, "type", "boolean")

    unknown: list[str] = []
    runtime = None
    if isinstance(extensions_result, dict):
        runtime = extensions_result.setdefault("runtime", {})
        runtime.setdefault("flagValues", {})

    for name, value in extension_flag_values.items():
        flag_type = registered_flags.get(name)
        if flag_type is None:
            unknown.append(name)
            continue
        if flag_type == "boolean":
            effective_value: bool | str = True
        elif isinstance(value, str):
            effective_value = value
        else:
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type="error",
                    message=f'Extension flag "--{name}" requires a value',
                )
            )
            continue
        if isinstance(runtime, dict):
            runtime["flagValues"][name] = effective_value

    if unknown:
        label = "Unknown option" if len(unknown) == 1 else "Unknown options"
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type="error",
                message=f"{label}: {', '.join(f'--{name}' for name in unknown)}",
            )
        )

    return diagnostics


def _apply_pending_provider_registrations(
    resource_loader: Any,
    model_registry: ModelRegistry,
) -> list[AgentSessionRuntimeDiagnostic]:
    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    runtime = extensions_result.get("runtime") if isinstance(extensions_result, dict) else None
    if not isinstance(runtime, dict):
        return diagnostics

    pending = list(runtime.get("pendingProviderRegistrations") or [])
    for registration in pending:
        if not isinstance(registration, dict):
            continue
        name = registration.get("name")
        config = registration.get("config")
        extension_path = registration.get("extensionPath") or "<unknown>"
        if not isinstance(name, str) or not isinstance(config, dict):
            continue
        try:
            model_registry.register_provider(name, config)
        except Exception as error:
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type="error",
                    message=f'Extension "{extension_path}" error: {error}',
                )
            )

    runtime["pendingProviderRegistrations"] = []
    return diagnostics


async def create_agent_session_services(
    options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices:
    cwd = os.path.abspath(os.path.expanduser(options.cwd))
    agent_dir = os.path.abspath(os.path.expanduser(options.agent_dir or get_agent_dir()))
    auth_storage = options.auth_storage or AuthStorage()
    settings_manager = options.settings_manager or SettingsManager.create(cwd=cwd, agent_dir=agent_dir)
    model_registry = options.model_registry or ModelRegistry(auth_storage=auth_storage)

    loader_options = dict(options.resource_loader_options or {})
    resource_loader = loader_options.pop("resource_loader", None)
    if resource_loader is None:
        resource_loader = DefaultResourceLoader(
            DefaultResourceLoaderOptions(
                **loader_options,
                cwd=cwd,
                agent_dir=agent_dir,
                settings_manager=settings_manager,
            )
        )
    await resource_loader.reload()

    diagnostics = _apply_pending_provider_registrations(resource_loader, model_registry)
    diagnostics.extend(_apply_extension_flag_values(resource_loader, options.extension_flag_values))
    return AgentSessionServices(
        cwd=cwd,
        agent_dir=agent_dir,
        auth_storage=auth_storage,
        settings_manager=settings_manager,
        model_registry=model_registry,
        resource_loader=resource_loader,
        diagnostics=diagnostics,
    )


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> CreateAgentSessionResult:
    return await create_agent_session(
        CreateAgentSessionOptions(
            cwd=options.services.cwd,
            agent_dir=options.services.agent_dir,
            auth_storage=options.services.auth_storage,
            settings_manager=options.services.settings_manager,
            model_registry=options.services.model_registry,
            resource_loader=options.services.resource_loader,
            session_manager=options.session_manager,
            model=options.model,
            thinking_level=options.thinking_level,
            scoped_models=options.scoped_models,
            tools=options.tools,
            exclude_tools=options.exclude_tools,
            no_tools=options.no_tools,
            custom_tools=options.custom_tools,
            session_start_event=options.session_start_event,
        )
    )


__all__ = [
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionServicesOptions",
    "create_agent_session_from_services",
    "create_agent_session_services",
]
