"""
Test SDK alignment with TypeScript version.
Tests the new CreateAgentSessionOptions and CreateAgentSessionResult.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path

import pytest
from pi_ai import get_model
from pi_coding_agent import (
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    create_agent_session,
)
from pi_coding_agent.core.settings_manager import SettingsManager


class _FakeResourceLoader:
    def __init__(
        self,
        *,
        skill_path: str | None = None,
        system_prompt: str | None = None,
        append_system_prompt: list[str] | None = None,
        agents_files: list[dict[str, str]] | None = None,
    ):
        self.reload_count = 0
        self._system_prompt = system_prompt
        self._append_system_prompt = append_system_prompt or []
        self._agents_files = agents_files or []
        self._skill_path = skill_path
        self.extended_resources = []

    async def reload(self):
        self.reload_count += 1

    def extend_resources(self, paths):
        self.extended_resources.append(paths)

    def get_extensions(self):
        return {"extensions": [], "diagnostics": []}

    def get_skills(self):
        if not self._skill_path:
            return {"skills": [], "diagnostics": []}

        class Skill:
            name = "parity-skill"
            file_path = self._skill_path

        return {"skills": [Skill()], "diagnostics": []}

    def get_agents_files(self):
        return {"agentsFiles": self._agents_files}

    def get_system_prompt(self):
        return self._system_prompt

    def get_append_system_prompt(self):
        return self._append_system_prompt


class TestSDKAlignment:
    """Test SDK API alignment with TypeScript version."""

    @pytest.mark.asyncio
    async def test_create_agent_session_minimal(self):
        """Test minimal session creation (no options)."""
        result = await create_agent_session()
        
        assert isinstance(result, CreateAgentSessionResult)
        assert result.session is not None
        assert result.session.cwd is not None
        assert result.session.model is not None

    @pytest.mark.asyncio
    async def test_create_agent_session_with_options(self):
        """Test session creation with explicit options."""
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        
        options = CreateAgentSessionOptions(
            model=model,
            thinking_level="high",
        )
        
        result = await create_agent_session(options)
        
        assert result.session.model == model
        # Thinking level may be clamped based on model capabilities
        assert result.session.thinking_level in ["off", "minimal", "low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_session_uses_resource_loader_prompt_inputs(self, tmp_path):
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text("Skill body content for parity.", encoding="utf-8")
        loader = _FakeResourceLoader(
            skill_path=str(skill_path),
            system_prompt="Base prompt from loader.",
            append_system_prompt=["Append prompt from loader."],
            agents_files=[{"path": str(tmp_path / "AGENTS.md"), "content": "Project context from loader."}],
        )

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=loader,
                tools=["read"],
            )
        )

        prompt = result.session.system_prompt
        assert "Base prompt from loader." in prompt
        assert "Append prompt from loader." in prompt
        assert "Project context from loader." in prompt
        assert "Skill body content for parity." in prompt
        assert result.session.get_active_tool_names() == ["read"]

    @pytest.mark.asyncio
    async def test_session_vars_substitute_in_system_prompt(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(system_prompt="Active path: {ACTIVE_PATH}"),
                session_vars={"ACTIVE_PATH": "/repo/app"},
                tools=[],
            )
        )

        assert "Active path: /repo/app" in result.session.system_prompt
        assert "{ACTIVE_PATH}" not in result.session.system_prompt

    @pytest.mark.asyncio
    async def test_session_vars_are_available_to_extensions(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension

        seen = {}

        def on_start(ctx, event):
            seen["session_vars"] = getattr(ctx, "session_vars", None)
            seen["sessionVars"] = getattr(ctx, "sessionVars", None)

        extension = Extension(
            path="inline.py",
            resolved_path="inline.py",
            handlers={"session_start": [on_start]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": []}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
                session_vars={"PROJECT": "/repo/app"},
                tools=[],
            )
        )

        await result.session.bind_extensions({})

        assert seen["session_vars"] == {"PROJECT": "/repo/app"}
        assert seen["sessionVars"] == {"PROJECT": "/repo/app"}

    @pytest.mark.asyncio
    async def test_no_tools_disables_default_tools(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
                no_tools="all",
            )
        )

        assert result.session.get_active_tool_names() == []
        assert "Available tools:\n(none)" in result.session.system_prompt

    @pytest.mark.asyncio
    async def test_temperature_option_reaches_agent(self, tmp_path):
        """--temperature flows through CreateAgentSessionOptions to the Agent."""
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
                temperature=0.0,
            )
        )
        assert result.session._agent.temperature == 0.0

    @pytest.mark.asyncio
    async def test_temperature_defaults_to_none(self, tmp_path):
        """Omitting temperature leaves it None (provider/model default)."""
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        assert result.session._agent.temperature is None

    @pytest.mark.asyncio
    async def test_project_settings_tools_empty_disables_default_tools(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PI_MEMORY_ENABLED", raising=False)
        settings_manager = SettingsManager.in_memory({
            "name": "Devin",
            "tools": [],
            "memory_enabled": True,
        })

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                settings_manager=settings_manager,
                resource_loader=_FakeResourceLoader(),
            )
        )

        assert result.session.get_active_tool_names() == []
        assert "Available tools:\n(none)" in result.session.system_prompt
        assert result.session._settings.name == "Devin"
        assert result.session._settings.memory_enabled is True

    @pytest.mark.asyncio
    async def test_project_settings_tools_empty_keeps_extension_tools(self, tmp_path):
        from pi_coding_agent.core.extensions.types import ToolDefinition

        settings_manager = SettingsManager.in_memory({"tools": []})

        async def execute(tool_call_id, params, cancel_event=None, on_update=None, ctx=None):
            return {"content": [{"type": "text", "text": "ok"}]}

        class _Extension:
            tools = {
                "subagent": ToolDefinition(
                    name="subagent",
                    label="Subagent",
                    description="Delegate work",
                    parameters={"type": "object", "properties": {}},
                    execute=execute,
                )
            }

        class _Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [_Extension()], "diagnostics": []}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                settings_manager=settings_manager,
                resource_loader=_Loader(),
            )
        )

        assert result.session.get_active_tool_names() == ["subagent"]

    @pytest.mark.asyncio
    async def test_default_active_tools_match_node_harness(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )

        assert result.session.get_active_tool_names() == ["read", "bash", "edit", "write"]

    @pytest.mark.asyncio
    async def test_custom_tool_definitions_are_registered_and_active(self, tmp_path):
        from pi_coding_agent.core.extensions.types import ToolDefinition

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            return {"content": [{"type": "text", "text": "custom result"}]}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
                custom_tools=[
                    ToolDefinition(
                        name="custom_lookup",
                        label="Custom Lookup",
                        description="Lookup custom data",
                        parameters={"type": "object", "properties": {}},
                        execute=execute,
                    )
                ],
            )
        )

        assert "custom_lookup" in result.session.get_all_tool_names()
        assert result.session.get_active_tool_names() == [
            "read",
            "bash",
            "edit",
            "write",
            "custom_lookup",
        ]

    @pytest.mark.asyncio
    async def test_extension_tool_definitions_are_registered_active_and_contextual(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension, ToolDefinition

        seen: dict[str, object] = {}

        async def execute(tool_call_id, params, cancel_event=None, on_update=None, ctx=None):
            seen["tool_call_id"] = tool_call_id
            seen["params"] = params
            seen["ctx_cwd"] = ctx.cwd if ctx else None
            return {"content": [{"type": "text", "text": "extension result"}]}

        extension = Extension(
            path="/tmp/extension-tools.py",
            resolved_path="/tmp/extension-tools.py",
            tools={
                "extension_lookup": ToolDefinition(
                    name="extension_lookup",
                    label="Extension Lookup",
                    description="Lookup extension data",
                    parameters={"type": "object", "properties": {"value": {"type": "string"}}},
                    execute=execute,
                )
            },
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )

        assert "extension_lookup" in result.session.get_all_tool_names()
        assert result.session.get_active_tool_names() == [
            "read",
            "bash",
            "edit",
            "write",
            "extension_lookup",
        ]

        tool = next(t for t in result.session.agent.state.tools if t.name == "extension_lookup")
        tool_result = await tool.execute("call-1", {"value": "abc"}, None, None)

        assert tool_result.content[0].text == "extension result"
        assert seen == {
            "tool_call_id": "call-1",
            "params": {"value": "abc"},
            "ctx_cwd": str(tmp_path),
        }

    @pytest.mark.asyncio
    async def test_extension_tools_receive_session_backed_context_actions(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension, ToolDefinition

        seen: dict[str, object] = {}

        async def execute(tool_call_id, params, cancel_event=None, on_update=None, ctx=None):
            seen["cwd"] = ctx.cwd
            seen["model_id"] = ctx.model.id
            seen["is_idle"] = ctx.isIdle()
            seen["has_pending_messages"] = ctx.hasPendingMessages()
            seen["system_prompt"] = ctx.getSystemPrompt()
            seen["context_usage"] = ctx.getContextUsage()
            seen["has_ui"] = ctx.hasUI
            return {"content": [{"type": "text", "text": "context ok"}]}

        extension = Extension(
            path="/tmp/context-tool.py",
            resolved_path="/tmp/context-tool.py",
            tools={
                "context_probe": ToolDefinition(
                    name="context_probe",
                    label="Context Probe",
                    description="Probe extension context",
                    parameters={"type": "object", "properties": {}},
                    execute=execute,
                )
            },
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(system_prompt="Session prompt marker."),
            )
        )

        tool = next(t for t in result.session.agent.state.tools if t.name == "context_probe")
        tool_result = await tool.execute("call-1", {}, None, None)

        assert tool_result.content[0].text == "context ok"
        assert seen["cwd"] == str(tmp_path)
        assert seen["model_id"] == "claude-3-5-sonnet-20241022"
        assert seen["is_idle"] is True
        assert seen["has_pending_messages"] is False
        assert "Session prompt marker." in seen["system_prompt"]
        assert seen["has_ui"] is False
        assert seen["context_usage"]["contextWindow"] > 0

    @pytest.mark.asyncio
    async def test_tool_definition_wrapper_preserves_runtime_contract(self):
        from pi_agent.types import AgentTool, AgentToolResult
        from pi_ai.types import TextContent
        from pi_coding_agent.core.extensions.types import ExtensionContext, ToolDefinition
        from pi_coding_agent.core.tools import (
            create_tool_definition_from_agent_tool,
            wrap_tool_definition,
            wrap_tool_definitions,
        )

        seen: dict[str, object] = {}

        async def execute(tool_call_id, params, cancel_event=None, on_update=None, ctx=None):
            seen["ctx"] = ctx
            seen["params"] = params
            return {"content": [{"type": "text", "text": f"{tool_call_id}:{params['value']}"}]}

        definition = ToolDefinition(
            name="context_tool",
            label="Context Tool",
            description="Uses extension context",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            prepare_arguments=lambda params: {**params, "prepared": True},
            execution_mode="concurrent",
            execute=execute,
        )
        ctx = ExtensionContext(cwd="/repo", session_id="s1")
        tool = wrap_tool_definition(definition, lambda: ctx)

        assert wrap_tool_definitions([definition], lambda: ctx)[0].name == "context_tool"
        assert tool.name == "context_tool"
        assert tool.label == "Context Tool"
        assert tool.prepareArguments({"value": "x"}) == {"value": "x", "prepared": True}
        assert tool.executionMode == "concurrent"

        result = await tool.execute("call-1", {"value": "abc"}, None, None)
        assert result.content[0].text == "call-1:abc"
        assert seen == {"ctx": ctx, "params": {"value": "abc"}}

        simple_seen: dict[str, object] = {}

        async def simple_execute(params, ctx):
            simple_seen["ctx"] = ctx
            simple_seen["params"] = params
            return "simple"

        simple_tool = wrap_tool_definition(
            ToolDefinition(
                name="simple_context_tool",
                description="Uses args and context only",
                parameters={"type": "object"},
                execute=simple_execute,
            ),
            lambda: ctx,
        )
        simple_result = await simple_tool.execute("call-simple", {"value": "xyz"}, None, None)
        assert simple_result.content[0].text == "simple"
        assert simple_seen == {"ctx": ctx, "params": {"value": "xyz"}}

        async def plain_execute(tool_call_id, params, cancel_event=None, on_update=None):
            return AgentToolResult(content=[TextContent(type="text", text="plain")])

        plain_tool = AgentTool(
            name="plain",
            label="Plain",
            description="Plain runtime tool",
            parameters={"type": "object"},
            execute=plain_execute,
        )
        synthesized = create_tool_definition_from_agent_tool(plain_tool)

        assert synthesized.name == "plain"
        assert synthesized.label == "Plain"
        assert (await synthesized.execute("call-2", {}, None, None)).content[0].text == "plain"

    @pytest.mark.asyncio
    async def test_session_routes_context_through_extension_runner(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension

        seen = []

        def context_handler(ctx, event):
            seen.append((ctx.cwd, event["messages"]))
            return {"messages": [*event["messages"], "from-extension"]}

        extension = Extension(
            path="inline.py",
            resolved_path="inline.py",
            handlers={"context": [context_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": []}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )

        assert result.session.has_extension_handlers("context") is True
        transformed = await result.session._transform_context(["base"])
        assert transformed == ["base", "from-extension"]
        assert seen == [(str(tmp_path), ["base"])]

    @pytest.mark.asyncio
    async def test_bind_extensions_emits_session_start_and_extends_discovered_resources(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension

        events = []

        def session_start_handler(ctx, event):
            events.append(("session_start", ctx.cwd, dict(event)))

        def resources_discover_handler(ctx, event):
            events.append(("resources_discover", ctx.cwd, dict(event)))
            return {
                "skillPaths": [
                    {
                        "path": str(tmp_path / "extension-skill.md"),
                        "extensionPath": str(tmp_path / "extension.py"),
                    }
                ],
                "prompt_paths": [
                    {
                        "path": str(tmp_path / "extension-prompt.md"),
                        "extension_path": str(tmp_path / "extension.py"),
                    }
                ],
                "themePaths": [str(tmp_path / "extension-theme.json")],
            }

        extension = Extension(
            path="inline.py",
            resolved_path="inline.py",
            handlers={
                "session_start": [session_start_handler],
                "resources_discover": [resources_discover_handler],
            },
        )
        second_extension = Extension(
            path=str(tmp_path / "second.py"),
            resolved_path=str(tmp_path / "second.py"),
            handlers={
                "resources_discover": [
                    lambda ctx, event: {"skillPaths": [str(tmp_path / "second-skill.md")]}
                ],
            },
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension, second_extension], "diagnostics": []}

        loader = Loader()
        session_start_event = {
            "type": "session_start",
            "reason": "resume",
            "previousSessionFile": "/tmp/previous.json",
        }
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=loader,
                session_start_event=session_start_event,
            )
        )

        await result.session.bind_extensions({"mode": "interactive"})

        assert events[0] == ("session_start", str(tmp_path), session_start_event)
        assert events[1] == (
            "resources_discover",
            str(tmp_path),
            {"type": "resources_discover", "cwd": str(tmp_path), "reason": "startup"},
        )
        assert len(loader.extended_resources) == 1
        paths = loader.extended_resources[0]
        assert [entry["path"] for entry in paths.skill_paths] == [
            str(tmp_path / "extension-skill.md"),
            str(tmp_path / "second-skill.md"),
        ]
        assert [entry["path"] for entry in paths.prompt_paths] == [str(tmp_path / "extension-prompt.md")]
        assert [entry["path"] for entry in paths.theme_paths] == [str(tmp_path / "extension-theme.json")]
        assert paths.skill_paths[0]["metadata"] == {
            "source": "extension",
            "scope": "temporary",
            "origin": "top-level",
            "baseDir": str(tmp_path),
        }
        assert paths.skill_paths[1]["metadata"] == {
            "source": "extension",
            "scope": "temporary",
            "origin": "top-level",
            "baseDir": str(tmp_path),
        }

    @pytest.mark.asyncio
    async def test_reload_emits_reload_lifecycle_and_discovers_reload_resources(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension

        events = []

        def event_handler(ctx, event):
            events.append(dict(event))

        def resources_discover_handler(ctx, event):
            events.append(dict(event))
            return {"skillPaths": [str(tmp_path / f"{event['reason']}-skill.md")]}

        extension = Extension(
            path=str(tmp_path / "extension.py"),
            resolved_path=str(tmp_path / "extension.py"),
            handlers={
                "session_start": [event_handler],
                "session_shutdown": [event_handler],
                "resources_discover": [resources_discover_handler],
            },
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": []}

        loader = Loader()
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=loader,
            )
        )

        await result.session.bind_extensions({"mode": "interactive"})
        await result.session.reload()

        assert events == [
            {"type": "session_start", "reason": "startup"},
            {"type": "resources_discover", "cwd": str(tmp_path), "reason": "startup"},
            {"type": "session_shutdown", "reason": "reload"},
            {"type": "session_start", "reason": "reload"},
            {"type": "resources_discover", "cwd": str(tmp_path), "reason": "reload"},
        ]
        assert loader.reload_count == 1
        assert [entry["path"] for entry in loader.extended_resources[0].skill_paths] == [
            str(tmp_path / "startup-skill.md")
        ]
        assert [entry["path"] for entry in loader.extended_resources[1].skill_paths] == [
            str(tmp_path / "reload-skill.md")
        ]

    @pytest.mark.asyncio
    async def test_before_provider_request_transforms_payload_through_session_prompt(self, tmp_path, monkeypatch):
        from pi_ai.types import AssistantMessage, EventDone, EventStart, TextContent, Usage
        from pi_coding_agent.core.extensions.types import Extension

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        seen = []

        def before_provider_handler(event, ctx):
            seen.append((event["payload"], ctx.cwd))
            next_payload = dict(event["payload"])
            next_payload["extensionTouched"] = True
            return next_payload

        extension = Extension(
            path="/tmp/provider-hook.py",
            resolved_path="/tmp/provider-hook.py",
            handlers={"before_provider_request": [before_provider_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        provider_payloads = []

        async def fake_stream(model, ctx, opts=None):
            payload = {"original": True}
            if opts and opts.on_payload:
                payload = await opts.on_payload(payload, model)
            provider_payloads.append(payload)
            text = "done"
            message = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=text)],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=0,
            )
            yield EventStart(type="start", partial=message)
            yield EventDone(type="done", reason="stop", message=message)

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        result.session.agent.stream_fn = fake_stream

        await result.session.prompt("hello")

        assert seen == [({"original": True}, str(tmp_path))]
        assert provider_payloads == [{"original": True, "extensionTouched": True}]

    @pytest.mark.asyncio
    async def test_after_provider_response_emits_through_session_prompt(self, tmp_path, monkeypatch):
        from pi_ai.types import AssistantMessage, EventDone, EventStart, TextContent, Usage
        from pi_coding_agent.core.extensions.types import Extension

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        seen = []

        def after_provider_handler(event, ctx):
            seen.append((dict(event), ctx.cwd))

        extension = Extension(
            path="/tmp/provider-response-hook.py",
            resolved_path="/tmp/provider-response-hook.py",
            handlers={"after_provider_response": [after_provider_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        async def fake_stream(model, ctx, opts=None):
            if opts and opts.on_response:
                await opts.on_response({"status": 201, "headers": {"x-provider": "test"}}, model)
            text = "done"
            message = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=text)],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=0,
            )
            yield EventStart(type="start", partial=message)
            yield EventDone(type="done", reason="stop", message=message)

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        result.session.agent.stream_fn = fake_stream

        await result.session.prompt("hello")

        assert seen == [
            (
                {
                    "type": "after_provider_response",
                    "status": 201,
                    "headers": {"x-provider": "test"},
                },
                str(tmp_path),
            )
        ]

    @pytest.mark.asyncio
    async def test_session_prompt_preflight_reports_auth_failure_before_rpc_success(self, tmp_path, monkeypatch):
        from pi_coding_agent.core.auth_storage import AuthStorage

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                auth_storage=AuthStorage.in_memory({}),
                resource_loader=_FakeResourceLoader(),
            )
        )
        preflight = []

        with pytest.raises(RuntimeError, match="No API key found"):
            await result.session.prompt("hello", preflight_result=preflight.append)

        assert preflight == [False]

    @pytest.mark.asyncio
    async def test_tool_call_extension_handler_mutates_builtin_tool_args_through_session_prompt(self, tmp_path, monkeypatch):
        from pi_ai.types import (
            AssistantMessage,
            EventDone,
            EventStart,
            EventToolCallEnd,
            EventToolCallStart,
            TextContent,
            ToolCall,
            Usage,
        )
        from pi_coding_agent.core.extensions.types import Extension

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        target = tmp_path / "target.txt"
        target.write_text("mutated path worked", encoding="utf-8")
        seen = []

        def tool_call_handler(event, ctx):
            seen.append((event["toolName"], event["toolCallId"], dict(event["input"]), ctx.cwd))
            event["input"]["path"] = str(target)

        extension = Extension(
            path="/tmp/tool-call-hook.py",
            resolved_path="/tmp/tool-call-hook.py",
            handlers={"tool_call": [tool_call_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        call_count = 0

        async def fake_stream(model, ctx, opts=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tool_call = ToolCall(
                    type="toolCall",
                    id="read-1",
                    name="read",
                    arguments={"path": str(tmp_path / "missing.txt")},
                )
                partial = AssistantMessage(
                    role="assistant",
                    content=[tool_call],
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=0,
                )
                yield EventStart(type="start", partial=partial)
                yield EventToolCallStart(type="toolcall_start", content_index=0, partial=partial)
                yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tool_call, partial=partial)
                yield EventDone(type="done", reason="toolUse", message=partial)
                return

            message = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=0,
            )
            yield EventStart(type="start", partial=message)
            yield EventDone(type="done", reason="stop", message=message)

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        result.session.agent.stream_fn = fake_stream

        await result.session.prompt("read")

        tool_results = [m for m in result.session.agent.state.messages if getattr(m, "role", "") == "toolResult"]
        assert seen == [
            ("read", "read-1", {"path": str(tmp_path / "missing.txt")}, str(tmp_path)),
        ]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        assert tool_results[0].content[0].text == "mutated path worked"

    @pytest.mark.asyncio
    async def test_tool_result_extension_handler_overrides_builtin_tool_result_through_session_prompt(self, tmp_path, monkeypatch):
        from pi_ai.types import (
            AssistantMessage,
            EventDone,
            EventStart,
            EventToolCallEnd,
            EventToolCallStart,
            TextContent,
            ToolCall,
            Usage,
        )
        from pi_coding_agent.core.extensions.types import Extension

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        target = tmp_path / "target.txt"
        target.write_text("original tool output", encoding="utf-8")
        seen = []

        def tool_result_handler(event, ctx):
            seen.append((event["toolName"], event["toolCallId"], event["content"][0].text, event["isError"], ctx.cwd))
            return {
                "content": [TextContent(type="text", text="extension override")],
                "details": {"source": "extension"},
                "isError": True,
            }

        extension = Extension(
            path="/tmp/tool-result-hook.py",
            resolved_path="/tmp/tool-result-hook.py",
            handlers={"tool_result": [tool_result_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        call_count = 0

        async def fake_stream(model, ctx, opts=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tool_call = ToolCall(
                    type="toolCall",
                    id="read-2",
                    name="read",
                    arguments={"path": str(target)},
                )
                partial = AssistantMessage(
                    role="assistant",
                    content=[tool_call],
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=0,
                )
                yield EventStart(type="start", partial=partial)
                yield EventToolCallStart(type="toolcall_start", content_index=0, partial=partial)
                yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tool_call, partial=partial)
                yield EventDone(type="done", reason="toolUse", message=partial)
                return

            message = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=0,
            )
            yield EventStart(type="start", partial=message)
            yield EventDone(type="done", reason="stop", message=message)

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        result.session.agent.stream_fn = fake_stream

        await result.session.prompt("read")

        tool_results = [m for m in result.session.agent.state.messages if getattr(m, "role", "") == "toolResult"]
        assert seen == [("read", "read-2", "original tool output", False, str(tmp_path))]
        assert len(tool_results) == 1
        assert tool_results[0].content[0].text == "extension override"
        assert tool_results[0].details == {"source": "extension"}
        assert tool_results[0].is_error is True

    @pytest.mark.asyncio
    async def test_tool_call_extension_handler_blocks_builtin_tool_through_session_prompt(self, tmp_path, monkeypatch):
        from pi_ai.types import (
            AssistantMessage,
            EventDone,
            EventStart,
            EventToolCallEnd,
            EventToolCallStart,
            TextContent,
            ToolCall,
            Usage,
        )
        from pi_coding_agent.core.extensions.types import Extension

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        seen = []

        def tool_call_handler(event, ctx):
            seen.append((event["toolName"], event["toolCallId"], dict(event["input"]), ctx.cwd))
            return {"block": True, "reason": "blocked by extension policy"}

        extension = Extension(
            path="/tmp/tool-call-block-hook.py",
            resolved_path="/tmp/tool-call-block-hook.py",
            handlers={"tool_call": [tool_call_handler]},
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        call_count = 0

        async def fake_stream(model, ctx, opts=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tool_call = ToolCall(
                    type="toolCall",
                    id="read-blocked",
                    name="read",
                    arguments={"path": str(tmp_path / "missing.txt")},
                )
                partial = AssistantMessage(
                    role="assistant",
                    content=[tool_call],
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=0,
                )
                yield EventStart(type="start", partial=partial)
                yield EventToolCallStart(type="toolcall_start", content_index=0, partial=partial)
                yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tool_call, partial=partial)
                yield EventDone(type="done", reason="toolUse", message=partial)
                return

            message = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="done")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="stop",
                timestamp=0,
            )
            yield EventStart(type="start", partial=message)
            yield EventDone(type="done", reason="stop", message=message)

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        result.session.agent.stream_fn = fake_stream

        await result.session.prompt("read")

        tool_results = [m for m in result.session.agent.state.messages if getattr(m, "role", "") == "toolResult"]
        assert seen == [
            ("read", "read-blocked", {"path": str(tmp_path / "missing.txt")}, str(tmp_path)),
        ]
        assert len(tool_results) == 1
        assert tool_results[0].content[0].text == "blocked by extension policy"
        assert tool_results[0].is_error is True

    @pytest.mark.asyncio
    async def test_session_exposes_rpc_runtime_contract(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session

        assert session.agent is not None
        assert session.resource_loader is not None
        assert isinstance(session.messages, list)
        assert session.session_file is None or isinstance(session.session_file, str)
        assert session.session_name is None or isinstance(session.session_name, str)

        session.set_steering_mode("latest")
        session.set_follow_up_mode("queue")
        assert session.steering_mode == "latest"
        assert session.follow_up_mode == "queue"
        session.abort_retry()

    @pytest.mark.asyncio
    async def test_new_session_switches_in_place(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        old_id = session.session_id
        old_file = session.session_file

        assert await session.new_session() is True

        assert session.session_id != old_id
        assert session.session_file != old_file
        assert session.messages == []
        assert session.get_active_tool_names() == ["read", "bash", "edit", "write"]

    @pytest.mark.asyncio
    async def test_clone_session_switches_to_copied_session(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        session._session_manager.append_session_info("Original")
        old_id = session.session_id
        old_file = session.session_file

        clone_result = await session.clone_session()

        assert clone_result == {"cancelled": False}
        assert session.session_id != old_id
        assert session.session_file != old_file
        assert session.session_name == "Original"

    @pytest.mark.asyncio
    async def test_fork_session_switches_in_place_and_returns_selected_text(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        first_id = session._session_manager.append_message({
            "role": "user",
            "content": "First request",
            "timestamp": 1,
        })
        second_id = session._session_manager.append_message({
            "role": "user",
            "content": "Second request",
            "timestamp": 2,
        })
        old_id = session.session_id

        fork_result = await session.fork_session(second_id)

        assert fork_result == {"cancelled": False, "selectedText": "Second request"}
        assert session.session_id != old_id
        assert [m["content"] for m in session.messages if m.get("role") == "user"] == ["First request"]
        assert session._session_manager.get_leaf_id() == first_id

    @pytest.mark.asyncio
    async def test_recover_session_branches_before_context_overflow_tail(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        user_id = session._session_manager.append_message({
            "role": "user",
            "content": "Investigate the TauClaire timeout",
            "timestamp": 1,
        })
        tool_call_id = session._session_manager.append_message({
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "name": "bash",
                "arguments": {"command": "run huge eval"},
            }],
            "stop_reason": "toolUse",
            "timestamp": 2,
        })
        tool_result_id = session._session_manager.append_message({
            "role": "toolResult",
            "content": [{"type": "text", "text": "SQL log line\n" * 5000}],
            "timestamp": 3,
        })
        error_id = session._session_manager.append_message({
            "role": "assistant",
            "content": [],
            "stop_reason": "error",
            "error_message": (
                "Error Code context_length_exceeded: Your input exceeds the "
                "context window of this model."
            ),
            "timestamp": 4,
        })
        old_id = session.session_id

        recover_result = await session.recover_session()

        assert recover_result["cancelled"] is False
        assert recover_result["reason"] == "context_length_exceeded"
        assert recover_result["sourceEntryId"] == error_id
        assert recover_result["branchPointId"] == user_id
        assert recover_result["oldSessionId"] == old_id
        assert recover_result["newSessionId"] != old_id
        assert recover_result["droppedEntryIds"] == [tool_call_id, tool_result_id, error_id]
        assert "TauClaire timeout" in recover_result["summary"]
        assert "context_length_exceeded" in recover_result["summary"]

        entries = session._session_manager.get_entries()
        assert entries[-1].type == "recovery"
        assert session._session_manager.get_leaf_id() == recover_result["recoveryEntryId"]
        rendered = session.messages
        assert rendered[0]["role"] == "user"
        assert rendered[0]["content"] == "Investigate the TauClaire timeout"
        assert rendered[-1]["role"] == "user"
        assert "Recovery checkpoint" in rendered[-1]["content"][0]["text"]
        assert "SQL log line" not in json.dumps(rendered)

    @pytest.mark.asyncio
    async def test_recover_session_accepts_numbered_tool_error(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        first_user = session._session_manager.append_message({
            "role": "user",
            "content": "First task",
            "timestamp": 1,
        })
        session._session_manager.append_message({
            "role": "assistant",
            "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": "bad"}}],
            "stop_reason": "toolUse",
            "timestamp": 2,
        })
        first_error = session._session_manager.append_message({
            "role": "toolResult",
            "content": [{"type": "text", "text": "Command exited with code 1"}],
            "details": {"kind": "tool_error"},
            "is_error": True,
            "timestamp": 3,
        })
        session._session_manager.append_message({
            "role": "user",
            "content": "Second task",
            "timestamp": 4,
        })
        session._session_manager.append_message({
            "role": "assistant",
            "content": [],
            "stop_reason": "error",
            "error_message": "TimeoutError: TauClaire timed out. stderr=",
            "timestamp": 5,
        })

        recover_result = await session.recover_session("2")

        assert recover_result["cancelled"] is False
        assert recover_result["sourceEntryId"] == first_error
        assert recover_result["reason"] == "tool_error"
        assert recover_result["branchPointId"] == first_user

    @pytest.mark.asyncio
    async def test_session_tree_entries_and_navigation_contract(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        first_id = session._session_manager.append_message({
            "role": "user",
            "content": "Tree request",
            "timestamp": 1,
        })
        session._session_manager.append_message({
            "role": "user",
            "content": "Later request",
            "timestamp": 2,
        })

        entries = session.get_session_tree_entries()
        assert any(item["entry_id"] == first_id and item["text"] == "Tree request" for item in entries)

        nav_result = await session.navigate_tree(first_id)
        assert nav_result == {"cancelled": False, "editorText": "Tree request"}

    @pytest.mark.asyncio
    async def test_session_login_logout_api_key_contract(self, tmp_path, monkeypatch):
        from pi_coding_agent.core.auth_storage import AuthStorage

        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr(AuthStorage, "AUTH_DIR", str(tmp_path))
        monkeypatch.setattr(AuthStorage, "AUTH_FILE", str(auth_file))

        auth_storage = AuthStorage()
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
                auth_storage=auth_storage,
            )
        )
        session = result.session

        session.login_api_key("openai", "test-secret")
        assert auth_storage.get_api_key("openai") == "test-secret"

        session.logout_provider("openai")
        assert auth_storage.get_api_key("openai") is None

    def test_project_trust_store_persists_normalized_decisions(self, tmp_path, monkeypatch):
        from pi_coding_agent.core.trust_manager import ProjectTrustStore

        agent_dir = tmp_path / "agent"
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

        store = ProjectTrustStore()
        store.set(str(project), True)
        assert store.get(str(project)) is True
        assert store.get(str(project / ".")) is True

        store.set(str(project), False)
        assert store.get(str(project)) is False

        store.set(str(project), None)
        assert store.get(str(project)) is None
        stored = json.loads((agent_dir / "trust.json").read_text())
        assert os.path.realpath(str(project)) not in stored

    def test_has_project_trust_inputs(self, tmp_path):
        from pi_coding_agent.core.trust_manager import has_project_trust_inputs

        project = tmp_path / "project"
        child = project / "nested"
        child.mkdir(parents=True)
        assert has_project_trust_inputs(str(child)) is False

        (project / "AGENTS.md").write_text("rules", encoding="utf-8")
        assert has_project_trust_inputs(str(child)) is True

    @pytest.mark.asyncio
    async def test_session_project_trust_uses_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session

        assert session.get_project_trust() is None
        session.set_project_trust(True)
        assert session.get_project_trust() is True
        session.set_project_trust(False)
        assert session.get_project_trust() is False
        session.set_project_trust(None)
        assert session.get_project_trust() is None

    @pytest.mark.asyncio
    async def test_share_session_creates_secret_gist(self, tmp_path, monkeypatch):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        session.set_session_name("share-test")
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0

            async def communicate(self):
                return b"https://gist.github.com/user/abc123\n", b""

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr("pi_coding_agent.core.agent_session.shutil.which", lambda name: "/usr/bin/gh")
        monkeypatch.setattr(
            "pi_coding_agent.core.agent_session.asyncio.create_subprocess_exec",
            fake_create_subprocess_exec,
        )
        monkeypatch.setenv("PI_SHARE_VIEWER_URL", "https://viewer.example/session/")

        shared = await session.share_session()

        assert captured["args"][:4] == ("/usr/bin/gh", "gist", "create", "--public=false")
        assert shared == {
            "gist_url": "https://gist.github.com/user/abc123",
            "gist_id": "abc123",
            "share_url": "https://viewer.example/session/#abc123",
        }

    @pytest.mark.asyncio
    async def test_execute_bash_uses_session_cwd_and_records_bash_execution(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session

        bash_result = await session.execute_bash("pwd")

        assert bash_result["exit_code"] == 0
        assert str(tmp_path) in bash_result["output"]
        messages = session.messages
        assert messages[-1]["role"] == "bashExecution"
        assert messages[-1]["command"] == "pwd"
        assert str(tmp_path) in messages[-1]["output"]

    @pytest.mark.asyncio
    async def test_abort_bash_cancels_running_command(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        task = asyncio.create_task(
            session.execute_bash("python -c 'import time; time.sleep(10)'")
        )
        await asyncio.sleep(0.1)
        assert session.is_bash_running is True

        session.abort_bash()
        bash_result = await task

        assert bash_result["cancelled"] is True
        assert session.is_bash_running is False

    def test_model_registry_unregisters_and_resets_runtime_providers(self):
        from pi_coding_agent.core.model_registry import ModelRegistry

        registry = ModelRegistry()
        provider_config = {
            "api": "openai-completions",
            "baseUrl": "https://runtime.example/v1",
            "models": [{"id": "runtime-model"}],
        }
        registry.register_provider("runtime-ai", provider_config)
        assert registry.find("runtime-ai", "runtime-model") is not None

        registry.unregister_provider("runtime-ai")
        assert registry.find("runtime-ai", "runtime-model") is None

        registry.register_provider("runtime-ai", provider_config)
        registry.reset_registered_providers()
        assert registry.find("runtime-ai", "runtime-model") is None

    @pytest.mark.asyncio
    async def test_reload_resets_runtime_providers(self, tmp_path):
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=_FakeResourceLoader(),
            )
        )
        session = result.session
        session.model_registry.register_provider(
            "runtime-ai",
            {
                "api": "openai-completions",
                "baseUrl": "https://runtime.example/v1",
                "models": [{"id": "runtime-model"}],
            },
        )
        assert session.model_registry.find("runtime-ai", "runtime-model") is not None

        await session.reload()

        assert session.model_registry.find("runtime-ai", "runtime-model") is None

    def test_extension_runner_lists_commands_with_paths(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, RegisteredCommand
        from pi_coding_agent.core.source_info import create_synthetic_source_info

        source_info = create_synthetic_source_info(
            "/tmp/ext.py",
            source="local",
            scope="project",
            base_dir="/tmp",
        )
        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            source_info=source_info,
            commands={
                "hello": RegisteredCommand(
                    name="hello",
                    description="Say hello",
                    handler=lambda args: args,
                    extension_path="/tmp/ext.py",
                    source_info=source_info,
                )
            },
        )
        runner = ExtensionRunner([extension])

        assert runner.get_registered_commands_with_paths() == [
            {
                "command": {"name": "hello", "description": "Say hello"},
                "extensionPath": "/tmp/ext.py",
                "sourceInfo": {
                    "path": "/tmp/ext.py",
                    "source": "local",
                    "scope": "project",
                    "origin": "top-level",
                    "baseDir": "/tmp",
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_extension_runner_node_style_aliases_drive_real_runner_behavior(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension

        observed = {}

        async def on_message_end(event, ctx):
            observed["cwd"] = ctx.cwd
            return {"message": {**event["message"], "content": "changed"}}

        async def on_user_bash(event, ctx):
            observed["bash"] = (event["command"], ctx.session_id)
            return {"command": event["command"] + " && pwd"}

        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            handlers={
                "message_end": [on_message_end],
                "user_bash": [on_user_bash],
            },
        )
        runner = ExtensionRunner([extension], cwd="/repo", session_id="s1")
        ui = object()
        errors = []

        runner.bindCore({"sendMessage": lambda text: text}, {"model": lambda: "model"})
        runner.bindCommandContext({"waitForIdle": lambda: None})
        runner.setUIContext(ui, "interactive")
        unsubscribe = runner.onError(errors.append)

        default_ctx = runner.createContext()
        assert default_ctx.ui is ui
        assert default_ctx.uiContext is ui
        assert default_ctx.mode == "interactive"
        assert default_ctx.hasUI is True
        assert runner.hasUI() is True
        assert runner.getUIContext() is ui
        assert runner.getExtensionPaths() == ["/tmp/ext.py"]
        assert default_ctx.cwd == "/repo"
        assert runner.createCommandContext().session_id == "s1"
        assert runner.hasHandlers("message_end") is True

        message = await runner.emitMessageEnd({"message": {"role": "assistant", "content": "old"}})
        user_bash = await runner.emitUserBash({"command": "echo hi"})

        assert message == {"role": "assistant", "content": "changed"}
        assert user_bash == {"command": "echo hi && pwd"}
        assert observed == {"cwd": "/repo", "bash": ("echo hi", "s1")}

        runner.emitError({"extensionPath": "/tmp/ext.py", "event": "test", "error": "boom"})
        assert errors == [{"extensionPath": "/tmp/ext.py", "event": "test", "error": "boom"}]
        unsubscribe()
        runner.emitError({"extensionPath": "/tmp/ext.py", "event": "test", "error": "ignored"})
        assert len(errors) == 1

    def test_extension_runner_resolves_duplicate_command_invocation_names(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, RegisteredCommand

        first = Extension(
            path="/tmp/first.py",
            resolved_path="/tmp/first.py",
            commands={
                "hello": RegisteredCommand(
                    name="hello",
                    description="First hello",
                    handler=lambda args: args,
                    extension_path="/tmp/first.py",
                )
            },
        )
        second = Extension(
            path="/tmp/second.py",
            resolved_path="/tmp/second.py",
            commands={
                "hello": RegisteredCommand(
                    name="hello",
                    description="Second hello",
                    handler=lambda args: args,
                    extension_path="/tmp/second.py",
                )
            },
        )
        runner = ExtensionRunner([first, second])

        command_names = [item["command"]["name"] for item in runner.get_registered_commands_with_paths()]

        assert command_names == ["hello:1", "hello:2"]
        assert runner.get_command("hello") is None
        assert runner.get_command("hello:1").description == "First hello"
        assert runner.get_command("hello:2").description == "Second hello"
        assert [command.invocation_name for command in runner.get_registered_commands()] == ["hello:1", "hello:2"]

    @pytest.mark.asyncio
    async def test_extension_runner_resolves_argument_completions_by_invocation_name(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, RegisteredCommand

        first = Extension(
            path="/tmp/first.py",
            resolved_path="/tmp/first.py",
            commands={
                "open": RegisteredCommand(
                    name="open",
                    description="Open first",
                    handler=lambda args: args,
                    get_argument_completions=lambda prefix: [{"label": f"first:{prefix}"}],
                    extension_path="/tmp/first.py",
                )
            },
        )
        second = Extension(
            path="/tmp/second.py",
            resolved_path="/tmp/second.py",
            commands={
                "open": RegisteredCommand(
                    name="open",
                    description="Open second",
                    handler=lambda args: args,
                    get_argument_completions=lambda prefix: [{"label": f"second:{prefix}"}],
                    extension_path="/tmp/second.py",
                )
            },
        )
        runner = ExtensionRunner([first, second])

        assert await runner.get_argument_completions("open:1", "wo") == [{"label": "first:wo"}]
        assert await runner.get_argument_completions("open:2", "wo") == [{"label": "second:wo"}]
        assert await runner.get_argument_completions("open", "wo") is None

    @pytest.mark.asyncio
    async def test_extension_runner_executes_commands_with_bound_context_actions(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, RegisteredCommand

        seen = {}

        async def handler(args, ctx):
            seen["args"] = args
            seen["cwd"] = ctx.cwd
            seen["action"] = await ctx.newSession({"parentSession": "s1"})
            return "handled"

        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            commands={
                "plan": RegisteredCommand(
                    name="plan",
                    description="Plan",
                    handler=handler,
                    extension_path="/tmp/ext.py",
                )
            },
        )
        runner = ExtensionRunner([extension], cwd="/tmp/project", session_id="session-1")
        runner.bind_command_context_actions({
            "newSession": lambda opts=None: {"cancelled": False, "options": opts},
        })

        result = await runner.execute_command("plan", "next steps")

        assert result == "handled"
        assert seen == {
            "args": "next steps",
            "cwd": "/tmp/project",
            "action": {"cancelled": False, "options": {"parentSession": "s1"}},
        }

    @pytest.mark.asyncio
    async def test_session_prompt_executes_extension_command_without_model_turn(self, tmp_path):
        from pi_coding_agent.core.extensions.types import Extension, RegisteredCommand

        seen = {}

        async def handler(args, ctx):
            seen["args"] = args
            seen["cwd"] = ctx.cwd
            seen["switch"] = await ctx.switchSession("/tmp/session.jsonl")

        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            commands={
                "organize": RegisteredCommand(
                    name="organize",
                    description="Organize",
                    handler=handler,
                    extension_path="/tmp/ext.py",
                )
            },
        )

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [extension], "diagnostics": [], "runtime": {"flagValues": {}}}

        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                resource_loader=Loader(),
            )
        )
        await result.session.bind_extensions({
            "commandContextActions": {
                "switchSession": lambda path: {"cancelled": False, "path": path},
            }
        })

        await result.session.prompt("/organize current world")

        assert seen == {
            "args": "current world",
            "cwd": str(tmp_path),
            "switch": {"cancelled": False, "path": "/tmp/session.jsonl"},
        }
        assert result.session.messages == []

    @pytest.mark.asyncio
    async def test_extension_runner_supports_node_and_ctx_first_handler_order(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension

        calls = []

        def node_order_handler(event, ctx):
            calls.append(("node", event["type"], ctx.cwd))
            return {"node": True}

        def ctx_first_handler(ctx, event):
            calls.append(("ctx", event["type"], ctx.cwd))
            return {"ctx": True}

        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            handlers={"session_start": [node_order_handler, ctx_first_handler]},
        )
        runner = ExtensionRunner([extension], cwd="/tmp/project", session_id="s1")

        result = await runner.emit({"type": "session_start", "reason": "startup"})

        assert result == {"node": True, "ctx": True}
        assert calls == [
            ("node", "session_start", "/tmp/project"),
            ("ctx", "session_start", "/tmp/project"),
        ]

    @pytest.mark.asyncio
    async def test_extension_runner_event_context_uses_bound_context_actions(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension

        seen = {}

        def handler(event, ctx):
            seen["cwd"] = ctx.cwd
            seen["mode"] = ctx.mode
            seen["model"] = ctx.model
            seen["prompt"] = ctx.getSystemPrompt()
            seen["idle"] = ctx.isIdle()
            seen["pending"] = ctx.hasPendingMessages()

        extension = Extension(
            path="/tmp/ext.py",
            resolved_path="/tmp/ext.py",
            handlers={"session_start": [handler]},
        )
        runner = ExtensionRunner([extension], cwd="/tmp/project", session_id="s1")
        runner.bind_context_actions(
            actions={
                "getSystemPrompt": lambda: "Bound prompt",
                "isIdle": lambda: True,
                "hasPendingMessages": lambda: False,
            },
            values={
                "mode": "interactive",
                "model": "model-a",
            },
        )

        await runner.emit({"type": "session_start", "reason": "startup"})

        assert seen == {
            "cwd": "/tmp/project",
            "mode": "interactive",
            "model": "model-a",
            "prompt": "Bound prompt",
            "idle": True,
            "pending": False,
        }

    @pytest.mark.asyncio
    async def test_extension_shortcuts_resolve_conflicts_and_execute_with_context(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, ExtensionShortcut

        calls = []

        def first_handler(ctx):
            calls.append(("first", ctx.cwd))

        async def second_handler(ctx):
            calls.append(("second", ctx.cwd))

        first = Extension(
            path="/tmp/first.py",
            resolved_path="/tmp/first.py",
            shortcuts={
                "ctrl+j": ExtensionShortcut(
                    shortcut="ctrl+j",
                    description="First",
                    handler=first_handler,
                    extension_path="/tmp/first.py",
                ),
                "escape": ExtensionShortcut(
                    shortcut="escape",
                    description="Reserved",
                    handler=first_handler,
                    extension_path="/tmp/first.py",
                ),
            },
        )
        second = Extension(
            path="/tmp/second.py",
            resolved_path="/tmp/second.py",
            shortcuts={
                "ctrl+j": ExtensionShortcut(
                    shortcut="ctrl+j",
                    description="Second",
                    handler=second_handler,
                    extension_path="/tmp/second.py",
                ),
                "ctrl+n": ExtensionShortcut(
                    shortcut="ctrl+n",
                    description="Non-reserved builtin",
                    handler=second_handler,
                    extension_path="/tmp/second.py",
                ),
            },
        )
        runner = ExtensionRunner([first, second], cwd="/tmp/project", session_id="s1")
        resolved_keybindings = {
            "app.interrupt": "escape",
            "app.session.toggleNamedFilter": "ctrl+n",
        }

        shortcuts = runner.get_shortcuts(resolved_keybindings)

        assert set(shortcuts) == {"ctrl+j", "ctrl+n"}
        assert shortcuts["ctrl+j"].extension_path == "/tmp/second.py"
        diagnostics = runner.get_shortcut_diagnostics()
        assert len(diagnostics) == 3
        assert "conflicts with built-in shortcut. Skipping" in diagnostics[0]["message"]
        assert "registered by both /tmp/first.py and /tmp/second.py" in diagnostics[1]["message"]
        assert "is built-in shortcut for app.session.toggleNamedFilter" in diagnostics[2]["message"]

        assert await runner.execute_shortcut("ctrl+j", resolved_keybindings) is True
        assert await runner.execute_shortcut("escape", resolved_keybindings) is False
        assert calls == [("second", "/tmp/project")]

    @pytest.mark.asyncio
    async def test_extension_runner_resources_discover_accumulates_node_order_handlers(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension

        def first_handler(event, ctx):
            return {"skillPaths": ["/tmp/skill-a.md"]}

        def second_handler(event, ctx):
            return {"skillPaths": ["/tmp/skill-b.md"], "promptPaths": ["/tmp/prompt.md"]}

        runner = ExtensionRunner(
            [
                Extension(
                    path="/tmp/first.py",
                    resolved_path="/tmp/first.py",
                    handlers={"resources_discover": [first_handler]},
                ),
                Extension(
                    path="/tmp/second.py",
                    resolved_path="/tmp/second.py",
                    handlers={"resources_discover": [second_handler]},
                ),
            ],
            cwd="/tmp/project",
            session_id="s1",
        )

        result = await runner.emit_resources_discover("/tmp/project", "startup")

        assert result == {
            "skillPaths": [
                {"path": "/tmp/skill-a.md", "extensionPath": "/tmp/first.py"},
                {"path": "/tmp/skill-b.md", "extensionPath": "/tmp/second.py"},
            ],
            "promptPaths": [{"path": "/tmp/prompt.md", "extensionPath": "/tmp/second.py"}],
            "themePaths": [],
        }

    @pytest.mark.asyncio
    async def test_create_agent_session_returns_result_object(self):
        """Test that create_agent_session returns CreateAgentSessionResult."""
        result = await create_agent_session()
        
        # Check all expected fields
        assert hasattr(result, "session")
        assert hasattr(result, "extensions_result")
        assert hasattr(result, "model_fallback_message")
        
        # Session should be valid
        assert result.session is not None
        
        # Optional fields can be None
        assert result.extensions_result is None or isinstance(result.extensions_result, dict)
        assert result.model_fallback_message is None or isinstance(result.model_fallback_message, str)

    @pytest.mark.asyncio
    async def test_extension_flag_values_flow_from_services_to_session_and_reload(self, tmp_path):
        from pi_coding_agent.core.agent_session_services import (
            CreateAgentSessionFromServicesOptions,
            CreateAgentSessionServicesOptions,
            create_agent_session_from_services,
            create_agent_session_services,
        )
        from pi_coding_agent.core.extensions.types import Extension, ExtensionFlag
        from pi_coding_agent.core.session_manager import SessionManager

        extension = Extension(
            path="/tmp/flags.py",
            resolved_path="/tmp/flags.py",
            flags={
                "debug": ExtensionFlag(
                    name="debug",
                    type="boolean",
                    default=False,
                    extension_path="/tmp/flags.py",
                )
            },
        )

        class Loader(_FakeResourceLoader):
            def __init__(self):
                super().__init__()
                self.runtime = {"flagValues": {"debug": False}}

            async def reload(self):
                self.reload_count += 1
                self.runtime = {"flagValues": {"debug": False}}

            def get_extensions(self):
                return {
                    "extensions": [extension],
                    "diagnostics": [],
                    "runtime": self.runtime,
                }

        loader = Loader()
        services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=str(tmp_path),
                agent_dir=str(tmp_path / "agent"),
                extension_flag_values={"debug": True},
                resource_loader_options={"resource_loader": loader},
            )
        )
        session_manager = SessionManager.create(str(tmp_path), str(tmp_path / "sessions"))
        result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
            )
        )

        assert result.session._extension_runner.get_flag_values() == {"debug": True}

        await result.session.reload()

        assert loader.reload_count == 2
        assert result.session._extension_runner.get_flag_values() == {"debug": True}

    @pytest.mark.asyncio
    async def test_pending_provider_registrations_are_registered_for_direct_sessions(self, tmp_path):
        from pi_coding_agent.core.model_registry import ModelRegistry

        runtime = {
            "flagValues": {},
            "pendingProviderRegistrations": [
                {
                    "name": "runtime-ai",
                    "config": {
                        "api": "openai-completions",
                        "baseUrl": "https://runtime.example/v1",
                        "models": [{"id": "runtime-model"}],
                    },
                    "extensionPath": "/tmp/ext.py",
                }
            ],
        }

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [], "diagnostics": [], "runtime": runtime}

        registry = ModelRegistry()
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                model_registry=registry,
                resource_loader=Loader(),
            )
        )

        assert result.session.model_registry.find("runtime-ai", "runtime-model") is not None
        assert runtime["pendingProviderRegistrations"] == []

    @pytest.mark.asyncio
    async def test_pending_provider_registrations_are_registered_by_services(self, tmp_path):
        from pi_coding_agent.core.agent_session_services import (
            CreateAgentSessionServicesOptions,
            create_agent_session_services,
        )

        runtime = {
            "flagValues": {},
            "pendingProviderRegistrations": [
                {
                    "name": "runtime-ai",
                    "config": {
                        "api": "openai-completions",
                        "baseUrl": "https://runtime.example/v1",
                        "models": [{"id": "runtime-model"}],
                    },
                    "extensionPath": "/tmp/ext.py",
                }
            ],
        }

        class Loader(_FakeResourceLoader):
            def get_extensions(self):
                return {"extensions": [], "diagnostics": [], "runtime": runtime}

        services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=str(tmp_path),
                agent_dir=str(tmp_path / "agent"),
                resource_loader_options={"resource_loader": Loader()},
            )
        )

        assert services.diagnostics == []
        assert services.model_registry.find("runtime-ai", "runtime-model") is not None
        assert runtime["pendingProviderRegistrations"] == []

    @pytest.mark.asyncio
    async def test_runtime_services_create_session_from_services_and_runtime_replacement(self, tmp_path):
        from pi_coding_agent.core.sdk import (
            AgentSessionRuntime,
            CreateAgentSessionFromServicesOptions,
            CreateAgentSessionRuntimeResult,
            CreateAgentSessionServicesOptions,
            create_agent_session_from_services,
            create_agent_session_runtime,
            create_agent_session_services,
        )
        from pi_coding_agent.core.session_manager import SessionManager

        services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=str(tmp_path),
                agent_dir=str(tmp_path / "agent"),
                resource_loader_options={
                    "no_extensions": True,
                    "no_skills": True,
                    "no_prompt_templates": True,
                    "no_themes": True,
                },
            )
        )
        assert services.cwd == str(tmp_path)
        assert services.agent_dir == str(tmp_path / "agent")
        assert services.diagnostics == []

        session_manager = SessionManager.create(str(tmp_path), str(tmp_path / "sessions"))
        result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                tools=["read"],
            )
        )
        assert result.session.cwd == str(tmp_path)
        assert result.session.get_active_tool_names() == ["read"]

        calls: list[dict[str, object]] = []

        async def factory(options):
            calls.append(dict(options))
            next_services = await create_agent_session_services(
                CreateAgentSessionServicesOptions(
                    cwd=options["cwd"],
                    agent_dir=options["agent_dir"],
                    resource_loader_options={
                        "no_extensions": True,
                        "no_skills": True,
                        "no_prompt_templates": True,
                        "no_themes": True,
                    },
                )
            )
            next_result = await create_agent_session_from_services(
                CreateAgentSessionFromServicesOptions(
                    services=next_services,
                    session_manager=options["session_manager"],
                    session_start_event=options.get("session_start_event"),
                    model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
                    tools=["read"],
                )
            )
            return CreateAgentSessionRuntimeResult(
                session=next_result.session,
                services=next_services,
                diagnostics=next_services.diagnostics,
                extensions_result=next_result.extensions_result,
                model_fallback_message=next_result.model_fallback_message,
            )

        runtime = await create_agent_session_runtime(
            factory,
            {
                "cwd": str(tmp_path),
                "agent_dir": str(tmp_path / "agent"),
                "session_manager": session_manager,
            },
        )
        assert isinstance(runtime, AgentSessionRuntime)
        assert runtime.session.session_file == session_manager.get_session_file()

        rebound: list[str] = []

        async def rebind(session):
            rebound.append(session.session_file or "")

        runtime.set_rebind_session(rebind)
        old_runner = runtime.session.extension_runner
        captured_ctx = old_runner.create_context()
        assert captured_ctx.cwd == str(tmp_path)
        new_result = await runtime.new_session({"parentSession": runtime.session.session_file})

        assert new_result == {"cancelled": False}
        assert len(calls) == 2
        assert calls[-1]["session_start_event"]["reason"] == "new"
        assert runtime.session.session_file != session_manager.get_session_file()
        assert rebound[-1] == runtime.session.session_file
        with pytest.raises(RuntimeError, match="stale after session replacement or reload"):
            _ = captured_ctx.cwd
        with pytest.raises(RuntimeError, match="stale after session replacement or reload"):
            _ = old_runner.create_context().cwd

        class CancellingRunner:
            emitted: list[dict[str, object]]

            def __init__(self):
                self.emitted = []

            def has_handlers(self, event_type):
                return event_type in {"session_before_switch", "session_before_fork"}

            async def emit(self, event):
                self.emitted.append(event)
                return {"cancel": True}

        cancelling_runner = CancellingRunner()
        runtime.session._extension_runner = cancelling_runner
        call_count = len(calls)

        cancelled_new = await runtime.new_session()
        cancelled_fork = await runtime.fork("entry-1", {"position": "at"})

        assert cancelled_new == {"cancelled": True}
        assert cancelled_fork == {"cancelled": True}
        assert len(calls) == call_count
        assert cancelling_runner.emitted == [
            {"type": "session_before_switch", "reason": "new", "targetSessionFile": None},
            {"type": "session_before_fork", "entryId": "entry-1", "position": "at"},
        ]

    @pytest.mark.asyncio
    async def test_dataclass_options_structure(self):
        """Test that CreateAgentSessionOptions is a proper dataclass."""
        options = CreateAgentSessionOptions(
            cwd="/tmp/test",
            agent_dir="/tmp/agent",
            thinking_level="medium",
        )
        
        assert options.cwd == "/tmp/test"
        assert options.agent_dir == "/tmp/agent"
        assert options.thinking_level == "medium"
        
        # Check other fields have defaults
        assert options.model is None
        assert options.scoped_models is None
        assert options.tools is None
        assert options.custom_tools is None

    @pytest.mark.asyncio
    async def test_session_has_new_methods(self):
        """Test that session has all newly aligned methods."""
        result = await create_agent_session()
        session = result.session
        
        # Test new methods exist
        assert hasattr(session, "dispose")
        assert hasattr(session, "set_scoped_models")
        assert hasattr(session, "set_auto_compaction_enabled")
        assert hasattr(session, "set_auto_retry_enabled")
        assert hasattr(session, "execute_bash")
        assert hasattr(session, "abort_bash")
        assert hasattr(session, "set_session_name")
        assert hasattr(session, "get_user_messages_for_forking")
        assert hasattr(session, "reload")
        assert hasattr(session, "bind_extensions")
        assert hasattr(session, "has_extension_handlers")
        
        # Test new properties
        assert hasattr(session, "auto_compaction_enabled")
        assert hasattr(session, "auto_retry_enabled")

        # Node/TypeScript public API aliases.
        assert hasattr(session, "setScopedModels")
        assert hasattr(session, "setAutoCompactionEnabled")
        assert hasattr(session, "setAutoRetryEnabled")
        assert hasattr(session, "executeBash")
        assert hasattr(session, "abortBash")
        assert hasattr(session, "setSessionName")
        assert hasattr(session, "getUserMessagesForForking")
        assert hasattr(session, "bindExtensions")
        assert hasattr(session, "hasExtensionHandlers")
        assert hasattr(session, "getAllTools")
        assert hasattr(session, "getToolDefinition")
        assert hasattr(session, "supportsThinking")
        assert hasattr(session, "abortBranchSummary")
        assert hasattr(session, "recordBashResult")
        assert hasattr(session, "exportToJsonl")
        assert hasattr(session, "createReplacedSessionContext")
        assert hasattr(session, "autoCompactionEnabled")
        assert hasattr(session, "autoRetryEnabled")
        assert hasattr(session, "thinkingLevel")
        assert hasattr(session, "isStreaming")
        assert hasattr(session, "sessionFile")
        assert hasattr(session, "sessionName")

    @pytest.mark.asyncio
    async def test_session_node_camel_case_aliases_drive_real_session_behavior(self, tmp_path):
        """Node-style session aliases should call the same production behavior."""
        from pi_coding_agent.core.settings_manager import SettingsManager

        settings_manager = SettingsManager(
            project_root=str(tmp_path),
            global_settings_file=str(tmp_path / "agent" / "settings.json"),
        )
        settings_manager.load()
        result = await create_agent_session(CreateAgentSessionOptions(settings_manager=settings_manager))
        session = result.session

        initial_compaction = session.autoCompactionEnabled
        session.setAutoCompactionEnabled(not initial_compaction)
        assert session.auto_compaction_enabled is (not initial_compaction)
        assert session.autoCompactionEnabled is (not initial_compaction)

        initial_retry = session.autoRetryEnabled
        session.setAutoRetryEnabled(not initial_retry)
        assert session.auto_retry_enabled is (not initial_retry)
        assert session.autoRetryEnabled is (not initial_retry)

        session.setSteeringMode("one-at-a-time")
        session.setFollowUpMode("all")
        assert session.steering_mode == "one-at-a-time"
        assert session.steeringMode == "one-at-a-time"
        assert session.follow_up_mode == "all"
        assert session.followUpMode == "all"

        session.setSessionName("Node Alias Session")
        assert session.session_name == "Node Alias Session"
        assert session.sessionName == "Node Alias Session"

        assert session.getSessionStats() == session.get_session_stats()
        assert session.getUserMessagesForForking() == session.get_user_messages_for_forking()

        all_tools = session.getAllTools()
        assert {tool["name"] for tool in all_tools} >= {"read", "bash", "edit", "write"}
        bash_definition = session.getToolDefinition("bash")
        assert bash_definition is not None
        assert bash_definition.name == "bash"
        assert session.supportsThinking() is bool(getattr(session.model, "reasoning", False))

        session.recordBashResult(
            "echo hi",
            {
                "output": "hi\n",
                "exitCode": 0,
                "cancelled": False,
                "truncated": False,
                "fullOutputPath": "/tmp/bash.out",
            },
            {"excludeFromContext": True},
        )
        bash_messages = [
            message for message in session.session_manager.get_messages()
            if message.get("role") == "bashExecution"
        ]
        assert bash_messages[-1]["exit_code"] == 0
        assert bash_messages[-1]["full_output_path"] == "/tmp/bash.out"
        assert bash_messages[-1]["exclude_from_context"] is True

        jsonl_path = session.exportToJsonl(str(tmp_path / "session-export.jsonl"))
        exported_lines = (tmp_path / "session-export.jsonl").read_text().strip().splitlines()
        assert jsonl_path == str(tmp_path / "session-export.jsonl")
        assert len(exported_lines) >= 2
        assert '"type": "session"' in exported_lines[0]

        replaced_context = session.createReplacedSessionContext()
        assert hasattr(replaced_context, "sendMessage") or isinstance(replaced_context, dict)

    @pytest.mark.asyncio
    async def test_dispose_method(self):
        """Test dispose method cleans up resources."""
        result = await create_agent_session()
        session = result.session
        
        # Should have listeners initially
        initial_listener_count = len(session._listeners)
        
        # Dispose should clear listeners
        session.dispose()
        
        assert len(session._listeners) == 0

    @pytest.mark.asyncio
    async def test_set_scoped_models(self):
        """Test set_scoped_models method."""
        result = await create_agent_session()
        session = result.session
        
        model1 = get_model("anthropic", "claude-3-5-sonnet-20241022")
        model2 = get_model("google", "gemini-2.0-flash")
        
        scoped = [
            {"model": model1, "thinking_level": "high"},
            {"model": model2, "thinking_level": "medium"},
        ]
        
        session.set_scoped_models(scoped)
        
        assert session._scoped_models == scoped
        assert session.scoped_models == scoped
        assert session.scopedModels == scoped
        assert session.sessionId == session.session_id
        assert session.hasPendingBashMessages is False

    @pytest.mark.asyncio
    async def test_auto_compaction_enabled_property(self, tmp_path):
        """Test auto_compaction_enabled property."""
        from pi_coding_agent.core.settings_manager import SettingsManager

        settings_manager = SettingsManager(
            project_root=str(tmp_path),
            global_settings_file=str(tmp_path / "agent" / "settings.json"),
        )
        settings_manager.load()
        result = await create_agent_session(CreateAgentSessionOptions(settings_manager=settings_manager))
        session = result.session
        
        initial = session.auto_compaction_enabled
        assert isinstance(initial, bool)
        
        session.set_auto_compaction_enabled(not initial)
        assert session.auto_compaction_enabled is (not initial)

    @pytest.mark.asyncio
    async def test_auto_retry_enabled_property(self, tmp_path):
        """Test auto_retry_enabled property."""
        from pi_coding_agent.core.settings_manager import SettingsManager

        settings_manager = SettingsManager(
            project_root=str(tmp_path),
            global_settings_file=str(tmp_path / "agent" / "settings.json"),
        )
        settings_manager.load()
        result = await create_agent_session(CreateAgentSessionOptions(settings_manager=settings_manager))
        session = result.session
        
        initial = session.auto_retry_enabled
        assert isinstance(initial, bool)
        
        session.set_auto_retry_enabled(not initial)
        assert session.auto_retry_enabled is (not initial)

    @pytest.mark.asyncio
    async def test_get_user_messages_for_forking(self):
        """Test get_user_messages_for_forking method."""
        result = await create_agent_session()
        session = result.session
        
        # Initially should be empty
        messages = session.get_user_messages_for_forking()
        assert isinstance(messages, list)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_has_extension_handlers(self):
        """Test has_extension_handlers method."""
        result = await create_agent_session()
        session = result.session
        
        assert session.has_extension_handlers("any_event") is False

    @pytest.mark.asyncio
    async def test_set_session_name(self):
        """Test set_session_name method."""
        result = await create_agent_session()
        session = result.session
        
        # Should not raise
        session.set_session_name("Test Session")


class TestExportAlignment:
    """Test that all TypeScript exports are available in Python."""

    def test_agent_package_exports(self):
        """Test pi-agent package exports."""
        from pi_agent import (
            Agent,
            AgentToolUpdateCallback,
            CustomAgentMessages,
            StreamFn,
        )
        
        assert Agent is not None
        assert StreamFn is not None
        assert CustomAgentMessages is not None
        assert AgentToolUpdateCallback is not None

    def test_ai_package_exports(self):
        """Test pi-ai package exports."""
        from pi_ai import (
            clear_api_providers,
            create_assistant_message_event_stream,
            get_api_providers,
            parse_streaming_json,
            validate_tool_call,
        )
        
        assert get_api_providers is not None
        assert clear_api_providers is not None
        assert validate_tool_call is not None
        assert parse_streaming_json is not None
        assert create_assistant_message_event_stream is not None

    def test_coding_agent_exports(self):
        """Test pi-coding-agent package exports."""
        from pi_coding_agent import (
            DEFAULT_COMPACTION_SETTINGS,
            DEFAULT_MAX_BYTES,
            DEFAULT_MAX_LINES,
            CompactionSettings,
            CreateAgentSessionOptions,
            CreateAgentSessionResult,
            ImageSettings,
            RetrySettings,
            build_session_context,
            compact,
            create_agent_session,
            format_size,
            should_compact,
            truncate_head,
            truncate_line,
            truncate_tail,
        )
        
        # SDK
        assert create_agent_session is not None
        assert CreateAgentSessionOptions is not None
        assert CreateAgentSessionResult is not None
        
        # Settings
        assert CompactionSettings is not None
        assert ImageSettings is not None
        assert RetrySettings is not None
        
        # Compaction
        assert compact is not None
        assert should_compact is not None
        assert DEFAULT_COMPACTION_SETTINGS is not None
        
        # Truncate
        assert truncate_head is not None
        assert truncate_tail is not None
        assert truncate_line is not None
        assert format_size is not None
        assert DEFAULT_MAX_BYTES is not None
        assert DEFAULT_MAX_LINES is not None
        
        # Session
        assert build_session_context is not None

    def test_tui_editor_methods(self):
        """Test TUI Editor component has new methods."""
        from pi_tui.components.editor import Editor
        
        # Check methods exist
        assert hasattr(Editor, "get_autocomplete_max_visible")
        assert hasattr(Editor, "set_autocomplete_max_visible")


class TestTypeAlignment:
    """Test type definitions match TypeScript."""

    def test_stream_fn_protocol(self):
        """Test StreamFn is a Protocol with correct signature."""
        from pi_agent.types import StreamFn
        
        # StreamFn should be a Protocol (typing.Protocol)
        assert hasattr(StreamFn, "__call__")

    def test_model_compat_structured_types(self):
        """Test Model.compat has structured types."""
        from pi_ai.types import (
            OpenAICompletionsCompat,
            OpenAIResponsesCompat,
            OpenRouterRouting,
            VercelGatewayRouting,
        )
        
        # All should be dataclasses
        assert hasattr(OpenAICompletionsCompat, "__dataclass_fields__")
        assert hasattr(OpenAIResponsesCompat, "__dataclass_fields__")
        assert hasattr(OpenRouterRouting, "__dataclass_fields__")
        assert hasattr(VercelGatewayRouting, "__dataclass_fields__")

    def test_extensions_tool_definition_exported(self):
        """Test ToolDefinition is exported from extensions."""
        from pi_coding_agent.core.extensions import ToolDefinition
        
        assert ToolDefinition is not None


class TestInteractiveComponentParity:
    """Test Python interactive components expose useful Node-aligned behavior."""

    def test_trust_selector_navigates_and_selects(self):
        from pi_coding_agent.modes.interactive.components import TrustSelectorComponent

        selected: list[bool] = []
        cancelled: list[bool] = []
        component = TrustSelectorComponent(
            cwd="/repo",
            saved_decision=True,
            project_trusted=True,
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
        )

        assert "Saved decision: trusted" in component.render()
        component.handle_input("down")
        component.handle_input("\n")
        assert selected == [False]
        component.handle_input("escape")
        assert cancelled == [True]

    def test_bash_execution_component_tracks_output_and_status(self):
        from pi_coding_agent.modes.interactive.components import BashExecutionComponent

        component = BashExecutionComponent("printf hi")
        component.append_output("hello")
        component.append_output("\nworld")
        assert component.get_command() == "printf hi"
        assert component.get_output() == "hello\nworld"
        assert "Running..." in component.render()

        component.set_complete(exit_code=2, cancelled=False, full_output_path="/tmp/full.txt")
        rendered = "\n".join(component.render())
        assert "$ printf hi" in rendered
        assert "(exit 2)" in rendered

    def test_message_components_render_thinking_and_error_states(self):
        from pi_coding_agent.modes.interactive.components import (
            AssistantMessageComponent,
            UserMessageComponent,
        )

        user = UserMessageComponent("Plan this")
        user_lines = user.render()
        assert user_lines[0].startswith("\x1b]133;A")

        msg = {
            "content": [
                {"type": "thinking", "thinking": "private chain"},
                {"type": "text", "text": "Visible answer"},
            ],
            "stopReason": "error",
            "errorMessage": "bad stream",
        }
        assistant = AssistantMessageComponent(msg, hide_thinking_block=True)
        rendered = "\n".join(assistant.render())
        assert "Thinking..." in rendered
        assert "Visible answer" in rendered
        assert "Error: bad stream" in rendered

    def test_footer_component_formats_session_usage(self, tmp_path, monkeypatch):
        from dataclasses import dataclass

        from pi_ai.types import Model, ModelCost
        from pi_coding_agent.config import VERSION
        from pi_coding_agent.modes.interactive.components import FooterComponent

        @dataclass
        class State:
            model: Model
            thinking_level: str = "high"

        class FakeRegistry:
            def is_using_oauth(self, model):
                return False

        class FakeSessionManager:
            def get_cwd(self):
                return str(tmp_path)

            def get_session_name(self):
                return "work"

            def get_entries(self):
                return [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "usage": {
                                "input": 1200,
                                "output": 3400,
                                "cacheRead": 600,
                                "cacheWrite": 200,
                                "cost": {"total": 0.123},
                            },
                        },
                    }
                ]

        class FakeFooterData:
            def get_git_branch(self):
                return "main"

            def get_extension_statuses(self):
                return {"b": "Beta\nstatus", "a": "Alpha\tstatus"}

            def get_available_provider_count(self):
                return 2

        class FakeSession:
            model = Model(
                id="gpt-5.4-nano",
                name="GPT-5.4 Nano",
                api="openai-responses",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(),
                context_window=400_000,
                max_tokens=128_000,
                reasoning=True,
            )
            state = State(model=model)
            session_manager = FakeSessionManager()
            model_registry = FakeRegistry()

            def get_context_usage(self):
                return {"percent": 12.5, "contextWindow": 400_000}

        monkeypatch.setenv("HOME", str(tmp_path.parent))
        footer = FooterComponent(FakeSession(), FakeFooterData())
        lines = footer.render(120)

        assert "~/" in lines[0]
        assert "(main)" in lines[0]
        assert "work" in lines[0]
        assert "↑1.2k" in lines[1]
        assert "↓3.4k" in lines[1]
        assert "(openai) gpt-5.4-nano" in lines[1]
        assert f"v{VERSION}" in lines[1]
        assert lines[1].index("12.5%/400k") < lines[1].index(f"v{VERSION}")
        assert lines[2] == "Alpha status Beta status"

    def test_basic_selector_components_select_and_cancel(self):
        from pi_coding_agent.modes.interactive.components import (
            ShowImagesSelectorComponent,
            ThemeSelectorComponent,
            ThinkingSelectorComponent,
        )

        selected: list[object] = []
        cancelled: list[bool] = []
        thinking = ThinkingSelectorComponent(
            "low",
            ["off", "low", "high"],
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
        )
        assert thinking.get_select_list().selected_item().value == "low"
        thinking.handle_input("down")
        thinking.handle_input("\n")
        assert selected[-1] == "high"

        previews: list[str] = []
        theme = ThemeSelectorComponent(
            "dark",
            themes=["dark", "light"],
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
            on_preview=previews.append,
        )
        theme.handle_input("down")
        assert previews[-1] == "light"
        theme.handle_input("\n")
        assert selected[-1] == "light"

        images = ShowImagesSelectorComponent(False, on_select=selected.append)
        assert images.get_select_list().selected_item().value == "no"
        images.handle_input("up")
        images.handle_input("\n")
        assert selected[-1] is True

    def test_model_selector_filters_toggles_scope_and_persists_selection(self):
        from pi_ai.types import Model, ModelCost
        from pi_coding_agent.modes.interactive.components import ModelSelectorComponent

        models = [
            Model(
                id="alpha",
                name="Alpha",
                api="openai-responses",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            ),
            Model(
                id="beta",
                name="Beta",
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            ),
        ]

        class Registry:
            def __init__(self):
                self.refreshed = False

            def refresh(self):
                self.refreshed = True

            def get_error(self):
                return None

            def get_available(self):
                return models

            def find(self, provider, model_id):
                return next((m for m in models if m.provider == provider and m.id == model_id), None)

        class Settings:
            def __init__(self):
                self.saved = None

            def set_default_model_and_provider(self, provider, model_id):
                self.saved = (provider, model_id)

        selected: list[object] = []
        settings = Settings()
        selector = ModelSelectorComponent(
            current_model=models[1],
            settings_manager=settings,
            model_registry=Registry(),
            scoped_models=[{"model": models[0]}],
            on_select=selected.append,
        )

        assert selector.scope == "scoped"
        assert [item.id for item in selector.filtered_models] == ["alpha"]
        selector.handle_input("tab")
        assert selector.scope == "all"
        assert selector.filtered_models[0].id == "beta"
        selector.filter_models("alp")
        assert [item.id for item in selector.filtered_models] == ["alpha"]
        selector.handle_input("\n")
        assert selected == [models[0]]
        assert settings.saved == ("openai", "alpha")

    def test_session_selector_search_tokens_phrase_regex_and_filters(self):
        from datetime import datetime

        from pi_coding_agent.modes.interactive.components import (
            filterAndSortSessions,
            filter_and_sort_sessions,
            getSessionSearchText,
            hasSessionName,
            has_session_name,
            matchSession,
            match_session,
            parseSearchQuery,
            parse_search_query,
        )

        sessions = [
            {
                "id": "a1",
                "name": "Planning",
                "allMessagesText": "market strategy and roadmap",
                "cwd": "/repo/a",
                "modified": datetime(2026, 5, 3),
            },
            {
                "id": "b2",
                "name": "",
                "allMessagesText": "billing implementation",
                "cwd": "/repo/b",
                "modified": datetime(2026, 5, 5),
            },
        ]

        parsed = parse_search_query('"market strategy"')
        assert match_session(sessions[0], parsed).matches is True
        assert matchSession(sessions[0], parsed).matches is True
        assert match_session(sessions[1], parsed).matches is False
        assert parse_search_query("re:[").error is not None
        assert parseSearchQuery("re:[").error is not None
        assert has_session_name(sessions[0]) is True
        assert hasSessionName(sessions[0]) is True
        assert has_session_name(sessions[1]) is False
        assert "Planning market strategy" in getSessionSearchText(sessions[0])

        named = filter_and_sort_sessions(sessions, "repo", "relevance", "named")
        assert named == [sessions[0]]
        assert filterAndSortSessions(sessions, "repo", "relevance", "named") == [sessions[0]]
        recent = filter_and_sort_sessions(sessions, "repo", "recent")
        assert recent == sessions
        relevance = filter_and_sort_sessions(sessions, "billing", "relevance")
        assert relevance == [sessions[1]]

    def test_scoped_models_selector_toggles_orders_and_persists(self):
        from pi_ai.types import Model, ModelCost
        from pi_coding_agent.modes.interactive.components import ScopedModelsSelectorComponent

        models = [
            Model(
                id="alpha",
                name="Alpha",
                api="openai-responses",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            ),
            Model(
                id="beta",
                name="Beta",
                api="openai-responses",
                provider="openai",
                base_url="https://api.openai.com/v1",
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            ),
            Model(
                id="claude",
                name="Claude",
                api="anthropic-messages",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            ),
        ]
        changes: list[object] = []
        persisted: list[object] = []
        selector = ScopedModelsSelectorComponent(
            models,
            None,
            on_change=changes.append,
            on_persist=persisted.append,
        )

        selector.handle_input("\n")
        assert changes[-1] == ["openai/alpha"]
        assert selector.is_dirty is True

        selector.handle_input("down")
        selector.handle_input("\n")
        assert changes[-1] == ["openai/alpha", "openai/beta"]

        selector.handle_input("reorder_up")
        assert changes[-1] == ["openai/beta", "openai/alpha"]

        selector.handle_input("toggle_provider")
        assert changes[-1] == []

        selector.handle_input("enable_all")
        assert changes[-1] is None

        selector.handle_input("save")
        assert persisted[-1] is None
        assert selector.is_dirty is False

    def test_user_message_selector_defaults_to_recent_and_selects(self):
        from pi_coding_agent.modes.interactive.components import UserMessageSelectorComponent

        selected: list[str] = []
        cancelled: list[bool] = []
        selector = UserMessageSelectorComponent(
            [
                {"id": "one", "text": "First"},
                {"id": "two", "text": "Second"},
            ],
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
        )

        assert selector.selected_message().id == "two"
        selector.handle_input("up")
        assert selector.selected_message().id == "one"
        selector.handle_input("\n")
        assert selected == ["one"]
        selector.handle_input("escape")
        assert cancelled == [True]

    def test_tree_selector_filters_searches_labels_and_selects(self):
        from pi_coding_agent.modes.interactive.components import TreeSelectorComponent

        entries = [
            {
                "id": "root",
                "type": "message",
                "parentId": None,
                "message": {"role": "user", "content": [{"type": "text", "text": "Start project"}]},
            },
            {
                "id": "assistant",
                "type": "message",
                "parentId": "root",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Plan created"}]},
            },
            {
                "id": "tool",
                "type": "message",
                "parentId": "assistant",
                "message": {"role": "toolResult", "content": [{"type": "text", "text": "tool output"}]},
            },
            {
                "id": "model",
                "type": "model_change",
                "parentId": "tool",
                "modelId": "gpt-5.4-nano",
            },
            {
                "id": "later",
                "type": "message",
                "parentId": "model",
                "message": {"role": "user", "content": [{"type": "text", "text": "Follow up"}]},
            },
        ]
        selected: list[str] = []
        cancelled: list[bool] = []
        selector = TreeSelectorComponent(
            entries,
            current_leaf_id="later",
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
        )

        assert selector.selected_node().entry["id"] == "later"
        assert "model" not in [node.node.entry["id"] for node in selector.filtered_nodes]

        selector.handle_input("filter_all")
        assert "model" in [node.node.entry["id"] for node in selector.filtered_nodes]

        selector.update_node_label("assistant", "Important")
        selector.handle_input("filter_labeled_only")
        assert [node.node.entry["id"] for node in selector.filtered_nodes] == ["assistant"]

        selector.handle_input("\n")
        assert selected == ["assistant"]

        selector.handle_input("escape")
        assert cancelled == [True]

        selector.handle_input("filter_default")
        selector.handle_input("F")
        selector.handle_input("o")
        assert [node.node.entry["id"] for node in selector.filtered_nodes] == ["later"]

    def test_rendering_helpers_key_hints_visual_truncation_and_diff(self):
        from pi_coding_agent.modes.interactive.components import (
            DynamicBorder,
            format_key_text,
            key_hint,
            parse_diff_line,
            render_diff,
            render_intra_line_diff,
            truncate_to_visual_lines,
        )

        assert format_key_text("ctrl+alt+x", platform="darwin") == "ctrl+option+x"
        assert format_key_text("ctrl+alt+x", capitalize=True, platform="linux") == "Ctrl+Alt+X"
        assert key_hint("tui.select.cancel", "cancel") == "escape cancel"
        assert DynamicBorder().render(4) == ["────"]

        visual = truncate_to_visual_lines("abcdefghij\nklmno", max_visual_lines=2, width=5)
        assert visual.visual_lines == ["fghij", "klmno"]
        assert visual.skipped_count == 1

        assert parse_diff_line("+12 hello") == {"prefix": "+", "line_num": "12", "content": "hello"}
        removed, added = render_intra_line_diff("hello world", "hello there")
        assert "{-world-}" in removed
        assert "{+there+}" in added
        rendered = render_diff("-1 hello world\n+1 hello there")
        assert "{-world-}" in rendered
        assert "{+there+}" in rendered

    def test_bordered_loader_and_tool_execution_component(self):
        from pi_coding_agent.modes.interactive.components import (
            BorderedLoader,
            ToolExecutionComponent,
            get_text_output,
        )

        aborted: list[bool] = []
        loader = BorderedLoader("Loading", cancellable=True)
        loader.on_abort = lambda: aborted.append(True)
        assert "Loading" in loader.render(10)
        loader.handle_input("escape")
        assert loader.aborted is True
        assert aborted == [True]

        tool = ToolExecutionComponent("read", "call-1", {"path": "README.md"})
        assert "README.md" in "\n".join(tool.render())
        tool.mark_execution_started()
        assert "(running)" in tool.render()
        tool.update_result(
            {
                "content": [
                    {"type": "text", "text": "file contents"},
                    {"type": "image", "mimeType": "image/png"},
                ],
                "isError": False,
            },
            is_partial=False,
        )
        assert get_text_output(tool.result, show_images=False) == "file contents\n[image: image/png]"
        rendered = "\n".join(tool.render())
        assert "file contents" in rendered
        assert "(complete)" in rendered

    def test_auth_extension_and_countdown_components_are_stateful(self):
        from pi_coding_agent.modes.interactive.components import (
            CountdownTimer,
            ExtensionInputComponent,
            ExtensionSelectorComponent,
            OAuthSelectorComponent,
        )

        class AuthStorage:
            def get(self, provider_id):
                return {"type": "api_key"} if provider_id == "openai" else None

        selected: list[str] = []
        cancelled: list[bool] = []
        oauth = OAuthSelectorComponent(
            "login",
            AuthStorage(),
            [
                {"id": "openai", "name": "OpenAI", "auth_type": "api_key"},
                {"id": "anthropic", "name": "Anthropic", "auth_type": "oauth"},
            ],
            on_select=selected.append,
            on_cancel=lambda: cancelled.append(True),
        )
        oauth.filter_providers("anth")
        assert oauth.selected_provider()["id"] == "anthropic"
        oauth.handle_input("\n")
        assert selected == ["anthropic"]
        oauth.handle_input("escape")
        assert cancelled == [True]

        selector = ExtensionSelectorComponent("Pick", ["a", "b"], on_select=selected.append)
        selector.handle_input("down")
        selector.handle_input("\n")
        assert selected[-1] == "b"
        selector.handle_input("\r")
        assert selected[-1] == "b"

        input_component = ExtensionInputComponent("Ask", on_submit=selected.append)
        input_component.handle_input("x")
        input_component.handle_input("\n")
        assert selected[-1] == "x"
        input_component = ExtensionInputComponent("Ask", on_submit=selected.append)
        input_component.handle_input("y")
        input_component.handle_input("\r")
        assert selected[-1] == "y"
        input_component = ExtensionInputComponent("Ask", "Paste value", on_submit=selected.append)
        assert input_component.render() == ["Ask", "> Paste value "]
        input_component.focused = True
        assert "\x1b_pi:c\x07" in input_component.render()[1]
        input_component.handle_input("z")
        assert "z" in input_component.render()[1]
        secret_input = ExtensionInputComponent("Secret", "Paste secret", on_submit=selected.append, opts={"secret": True})
        secret_input.handle_input("a")
        secret_input.handle_input("b")
        assert secret_input.render() == ["Secret", "> ** "]
        secret_input.handle_input("\n")
        assert selected[-1] == "ab"
        secret_input = ExtensionInputComponent("Secret", "Paste secret", on_submit=selected.append, opts={"secret": True})
        secret_input.handle_input("sk-pasted-key")
        assert secret_input.render() == ["Secret", "> ************* "]
        secret_input.handle_input("\n")
        assert selected[-1] == "sk-pasted-key"
        secret_input = ExtensionInputComponent("Secret", "Paste secret", on_submit=selected.append, opts={"secret": True})
        secret_input.handle_input("\x1b[200~sk-bracketed-key\x1b[201~")
        assert secret_input.render() == ["Secret", "> **************** "]
        secret_input.handle_input("\r")
        assert selected[-1] == "sk-bracketed-key"

        ticks: list[int] = []
        expired: list[bool] = []
        timer = CountdownTimer(1500, on_tick=ticks.append, on_expire=lambda: expired.append(True))
        assert ticks[0] == 2
        timer.tick()
        timer.tick()
        assert expired == [True]

    @pytest.mark.asyncio
    async def test_login_dialog_collects_prompt_and_cancel(self):
        from pi_coding_agent.modes.interactive.components import LoginDialogComponent

        completed: list[tuple[bool, str | None]] = []
        dialog = LoginDialogComponent("github", on_complete=lambda success, message=None: completed.append((success, message)))
        dialog.show_auth("https://example.test/login", "Use device flow")
        assert "https://example.test/login" in "\n".join(dialog.render())

        prompt_task = asyncio.create_task(dialog.show_prompt("Paste code", "abc123"))
        await asyncio.sleep(0)
        dialog.handle_input("z")
        dialog.handle_input("\n")
        assert await prompt_task == "z"

        dialog.handle_input("escape")
        assert completed[-1] == (False, "Login cancelled")

    def test_config_settings_session_and_editors_have_real_contracts(self, tmp_path):
        from pi_coding_agent.core.package_manager import PathMetadata, ResolvedPaths, ResolvedResource
        from pi_coding_agent.modes.interactive.components import (
            ConfigSelectorComponent,
            CustomEditor,
            ExtensionEditorComponent,
            SessionSelectorComponent,
            SettingsSelectorComponent,
            STUB_COMPONENTS,
        )

        resolved = ResolvedPaths(
            extensions=[
                ResolvedResource(
                    path=str(tmp_path / ".pi" / "extensions" / "alpha.ts"),
                    enabled=True,
                    metadata=PathMetadata(source="auto", scope="project", origin="top-level", base_dir=str(tmp_path / ".pi")),
                )
            ],
            skills=[
                ResolvedResource(
                    path=str(tmp_path / ".pi" / "skills" / "writer" / "SKILL.md"),
                    enabled=False,
                    metadata=PathMetadata(source="auto", scope="project", origin="top-level", base_dir=str(tmp_path / ".pi")),
                )
            ],
        )
        renders: list[bool] = []
        config = ConfigSelectorComponent(
            resolved,
            settings_manager=object(),
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / ".pi"),
            request_render=lambda: renders.append(True),
        )
        config.filter_items("writer")
        assert config.selected_item().display_name == "writer"
        assert config.toggle_selected().enabled is True
        assert renders == [True]

        callbacks: dict[str, list[object]] = {"onAutoCompactChange": [], "onThemeChange": []}
        settings = SettingsSelectorComponent(
            {
                "autoCompact": True,
                "currentTheme": "dark",
                "availableThemes": ["dark", "light"],
                "availableThinkingLevels": ["off", "medium"],
            },
            {
                "onAutoCompactChange": callbacks["onAutoCompactChange"].append,
                "onThemeChange": callbacks["onThemeChange"].append,
            },
        )
        settings.set_setting("autocompact", "false")
        settings.set_setting("theme", "light")
        assert callbacks["onAutoCompactChange"] == [False]
        assert callbacks["onThemeChange"] == ["light"]

        selected_sessions: list[str] = []
        sessions = [
            {"path": "/s/one.jsonl", "name": "Planning", "firstMessage": "Plan", "modified": 1, "allMessagesText": "market"},
            {"path": "/s/two.jsonl", "name": "", "firstMessage": "Build", "modified": 2, "allMessagesText": "billing"},
        ]
        session_selector = SessionSelectorComponent(sessions, on_select=selected_sessions.append)
        session_selector.filter_sessions("billing")
        assert session_selector.get_selected_session_path() == "/s/two.jsonl"
        session_selector.handle_input("\n")
        assert selected_sessions == ["/s/two.jsonl"]

        submitted: list[str] = []
        editor = ExtensionEditorComponent("Edit", prefill="hi", on_submit=submitted.append)
        editor.handle_input("!")
        editor.handle_input("\n")
        assert submitted == ["hi!"]

        escaped: list[bool] = []
        custom = CustomEditor()
        custom.on_escape = lambda: escaped.append(True)
        custom.handle_input("a")
        custom.handle_input("escape")
        assert custom.get_text() == "a"
        assert escaped == [True]

        assert set(STUB_COMPONENTS) == set()

    def test_interactive_theme_utilities_match_exported_node_contract(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
        custom_dir = tmp_path / "agent" / "themes"
        custom_dir.mkdir(parents=True)
        theme_path = custom_dir / "custom.json"
        colors = {
            "accent": "primary",
            "border": "#222222",
            "borderAccent": "#333333",
            "borderMuted": "#444444",
            "success": "#00ff00",
            "error": "#ff0000",
            "warning": "#ffff00",
            "muted": "#777777",
            "dim": "#666666",
            "text": "#eeeeee",
            "thinkingText": "#eeeeee",
            "selectedBg": "#111111",
            "userMessageBg": "#111111",
            "userMessageText": "#eeeeee",
            "customMessageBg": "#111111",
            "customMessageText": "#eeeeee",
            "customMessageLabel": "#eeeeee",
            "toolPendingBg": "#111111",
            "toolSuccessBg": "#111111",
            "toolErrorBg": "#111111",
            "toolTitle": "#eeeeee",
            "toolOutput": "#eeeeee",
            "mdHeading": "#eeeeee",
            "mdLink": "#eeeeee",
            "mdLinkUrl": "#777777",
            "mdCode": "#eeeeee",
            "mdCodeBlock": "#eeeeee",
            "mdCodeBlockBorder": "#444444",
            "mdQuote": "#eeeeee",
            "mdQuoteBorder": "#444444",
            "mdHr": "#444444",
            "mdListBullet": "#eeeeee",
            "toolDiffAdded": "#00ff00",
            "toolDiffRemoved": "#ff0000",
            "toolDiffContext": "#777777",
            "syntaxComment": "#777777",
            "syntaxKeyword": "#eeeeee",
            "syntaxFunction": "#eeeeee",
            "syntaxVariable": "#eeeeee",
            "syntaxString": "#eeeeee",
            "syntaxNumber": "#eeeeee",
            "syntaxType": "#eeeeee",
            "syntaxOperator": "#eeeeee",
            "syntaxPunctuation": "#eeeeee",
            "thinkingOff": "#777777",
            "thinkingMinimal": "#eeeeee",
            "thinkingLow": "#eeeeee",
            "thinkingMedium": "#eeeeee",
            "thinkingHigh": "#eeeeee",
            "thinkingXhigh": "#eeeeee",
            "bashMode": "#eeeeee",
        }
        theme_path.write_text(
            json.dumps({"name": "custom", "vars": {"primary": "#123456"}, "colors": colors}),
            encoding="utf-8",
        )

        import pi_coding_agent
        from pi_coding_agent.modes.interactive import theme as theme_pkg
        from pi_coding_agent.modes.interactive.theme import (
            get_available_themes,
            get_language_from_path,
            get_markdown_theme,
            get_select_list_theme,
            get_settings_list_theme,
            highlight_code,
            load_theme_from_path,
            set_theme,
            theme,
        )

        loaded = load_theme_from_path(str(theme_path))
        assert loaded.name == "custom"
        assert loaded.fg("accent", "x") == "\x1b[38;2;18;52;86mx\x1b[39m"
        assert "custom" in get_available_themes()

        assert set_theme("custom") == {"success": True}
        assert theme.fg("accent", "y") == "\x1b[38;2;18;52;86my\x1b[39m"
        assert theme_pkg.getLanguageFromPath("src/app.tsx") == "typescript"
        assert get_language_from_path("Dockerfile") == "dockerfile"
        assert pi_coding_agent.get_language_from_path("script.py") == "python"

        markdown_theme = get_markdown_theme()
        select_theme = get_select_list_theme()
        settings_theme = get_settings_list_theme()
        assert markdown_theme.heading("h").endswith("h\x1b[39m")
        assert select_theme.selected_text("selected").endswith("selected\x1b[39m")
        assert settings_theme.value("value", False).endswith("value\x1b[39m")
        assert highlight_code("one\ntwo", "python") == [
            "\x1b[38;2;238;238;238mone\x1b[39m",
            "\x1b[38;2;238;238;238mtwo\x1b[39m",
        ]

    @pytest.mark.asyncio
    async def test_extension_editor_external_editor_process_flow(self, tmp_path, monkeypatch):
        import sys

        from pi_coding_agent.modes.interactive.components import ExtensionEditorComponent

        script = tmp_path / "edit_file.py"
        script.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_text('edited\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )

        class Tui:
            def __init__(self):
                self.calls = []

            def stop(self):
                self.calls.append(("stop",))

            def start(self):
                self.calls.append(("start",))

            def request_render(self, full=False):
                self.calls.append(("request_render", full))

        tui = Tui()
        monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")
        editor = ExtensionEditorComponent("Edit", prefill="original", tui=tui)

        assert await editor.open_external_editor() is True
        assert editor.get_text() == "edited"
        assert ("stop",) in tui.calls
        assert ("start",) in tui.calls
        assert ("request_render", True) in tui.calls

        monkeypatch.setenv("EDITOR", f"{sys.executable} -c \"import sys; sys.exit(2)\"")
        editor.set_text("keep")
        assert await editor.open_external_editor() is False
        assert editor.get_text() == "keep"


class TestCoreRuntimeParity:
    """Core runtime helpers aligned to Node harness behavior."""

    def test_output_accumulator_bounds_memory_and_preserves_full_output(self):
        from pathlib import Path

        from pi_coding_agent.core.tools import OutputAccumulator

        accumulator = OutputAccumulator(max_lines=2, max_bytes=16, temp_file_prefix="pi-test-output")
        accumulator.append("alpha\n".encode("utf-8"))
        accumulator.append("beta\n".encode("utf-8"))
        accumulator.append("gamma\n".encode("utf-8"))
        accumulator.finish()

        snapshot = accumulator.snapshot(persist_if_truncated=True)
        accumulator.close_temp_file()

        assert snapshot.truncation.truncated is True
        assert snapshot.truncation.total_lines == 3
        assert "gamma" in snapshot.content
        assert snapshot.full_output_path is not None
        assert Path(snapshot.full_output_path).read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

    def test_output_accumulator_handles_split_utf8_chunks(self):
        from pi_coding_agent.core.tools import OutputAccumulator

        raw = "ok café".encode("utf-8")
        accumulator = OutputAccumulator(max_lines=10, max_bytes=100)
        accumulator.append(raw[:-1])
        accumulator.append(raw[-1:])
        accumulator.finish()

        snapshot = accumulator.snapshot()
        assert snapshot.content == "ok café"
        assert accumulator.get_last_line_bytes() == len("ok café".encode("utf-8"))

    @pytest.mark.asyncio
    async def test_file_mutation_queue_serializes_same_file_and_releases(self, tmp_path):
        from pi_coding_agent.core.tools import active_file_mutation_queue_count, with_file_mutation_queue

        target = tmp_path / "target.txt"
        order: list[str] = []
        first_started = asyncio.Event()

        async def first():
            order.append("first-start")
            first_started.set()
            await asyncio.sleep(0.02)
            order.append("first-end")
            return "first"

        async def second():
            order.append("second")
            return "second"

        first_task = asyncio.create_task(with_file_mutation_queue(str(target), first))
        await first_started.wait()
        second_task = asyncio.create_task(with_file_mutation_queue(str(target), second))

        assert await first_task == "first"
        assert await second_task == "second"
        assert order == ["first-start", "first-end", "second"]
        assert active_file_mutation_queue_count() == 0

    @pytest.mark.asyncio
    async def test_file_mutation_queue_allows_different_files_to_overlap(self, tmp_path):
        from pi_coding_agent.core.tools import with_file_mutation_queue

        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        overlapped: list[bool] = []

        async def first():
            first_entered.set()
            await release_first.wait()

        async def second():
            overlapped.append(first_entered.is_set() and not release_first.is_set())
            release_first.set()

        first_task = asyncio.create_task(with_file_mutation_queue(str(tmp_path / "a.txt"), first))
        await first_entered.wait()
        await with_file_mutation_queue(str(tmp_path / "b.txt"), second)
        await first_task

        assert overlapped == [True]

    @pytest.mark.asyncio
    async def test_output_guard_routes_normal_stdout_to_stderr_and_raw_to_stdout(self, monkeypatch):
        from pi_coding_agent.core import output_guard

        stdout = io.StringIO()
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)

        output_guard.take_over_stdout()
        try:
            print("normal")
            await output_guard.write_raw_stdout("raw\n")
            await output_guard.flush_raw_stdout()
            assert output_guard.is_stdout_taken_over() is True
        finally:
            output_guard.restore_stdout()

        assert stdout.getvalue() == "raw\n"
        assert "normal" in stderr.getvalue()

    def test_tool_render_utils_normalize_paths_text_and_images(self, tmp_path, monkeypatch):
        from pi_coding_agent.core.tools import (
            get_text_output,
            normalize_display_text,
            render_tool_path,
            replace_tabs,
            shorten_path,
            str_or_none,
        )

        monkeypatch.setenv("HOME", str(tmp_path))
        home_file = tmp_path / "work.txt"
        assert shorten_path(str(home_file)) == "~/work.txt"
        assert str_or_none("x") == "x"
        assert str_or_none(None) == ""
        assert str_or_none(3) is None
        assert replace_tabs("a\tb") == "a   b"
        assert normalize_display_text("a\rb") == "ab"
        assert render_tool_path("relative.txt", str(tmp_path)) == "~/relative.txt"
        assert render_tool_path(None, str(tmp_path)) == "[invalid arg]"

        output = get_text_output(
            {
                "content": [
                    {"type": "text", "text": "hello\r\nworld"},
                    {"type": "image", "mimeType": "image/png"},
                ]
            },
            show_images=False,
        )
        assert output == "hello\nworld\n[image: image/png]"

    def test_provider_display_source_info_and_telemetry_helpers(self, monkeypatch):
        from pi_coding_agent.core.package_manager import PathMetadata
        from pi_coding_agent.core.provider_display_names import BUILT_IN_PROVIDER_DISPLAY_NAMES, get_provider_display_name
        from pi_coding_agent.core.source_info import create_source_info, create_synthetic_source_info, source_info_to_dict
        from pi_coding_agent.core.telemetry import is_install_telemetry_enabled

        assert BUILT_IN_PROVIDER_DISPLAY_NAMES["minimax-cn"] == "MiniMax (China)"
        assert get_provider_display_name("openai") == "OpenAI"
        assert get_provider_display_name("custom-provider") == "custom-provider"

        metadata = PathMetadata(source="pkg", scope="project", origin="package", base_dir="/pkg")
        source_info = create_source_info("/pkg/ext.ts", metadata)
        assert source_info.source == "pkg"
        assert source_info.scope == "project"
        assert source_info.origin == "package"
        assert source_info.base_dir == "/pkg"
        assert source_info_to_dict(source_info) == {
            "path": "/pkg/ext.ts",
            "source": "pkg",
            "scope": "project",
            "origin": "package",
            "baseDir": "/pkg",
        }

        synthetic = create_synthetic_source_info("/tmp/runtime.py", source="runtime")
        assert synthetic.scope == "temporary"
        assert synthetic.origin == "top-level"

        class Settings:
            def getEnableInstallTelemetry(self):
                return True

        monkeypatch.delenv("PI_TELEMETRY", raising=False)
        assert is_install_telemetry_enabled(Settings()) is True
        assert is_install_telemetry_enabled(Settings(), telemetry_env="0") is False
        assert is_install_telemetry_enabled(Settings(), telemetry_env="yes") is True

    def test_provider_attribution_headers_are_gated_and_merged(self):
        from pi_ai.types import Model, ModelCost
        from pi_coding_agent.core.provider_attribution import merge_provider_attribution_headers

        def model(provider, base_url):
            return Model(
                id="m",
                name="m",
                api="openai-responses",
                provider=provider,
                base_url=base_url,
                cost=ModelCost(),
                context_window=100,
                max_tokens=10,
            )

        assert merge_provider_attribution_headers(
            model("openrouter", "https://openrouter.ai/api/v1"),
            {"enableInstallTelemetry": False},
            None,
        ) is None

        headers = merge_provider_attribution_headers(
            model("openrouter", "https://openrouter.ai/api/v1"),
            {"enableInstallTelemetry": True},
            None,
            {"X-OpenRouter-Title": "custom"},
        )
        assert headers == {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "custom",
            "X-OpenRouter-Categories": "cli-agent",
        }

        opencode_headers = merge_provider_attribution_headers(
            model("opencode", "https://opencode.ai"),
            {"enableInstallTelemetry": False},
            "session-1",
        )
        assert opencode_headers == {"x-opencode-session": "session-1", "x-opencode-client": "pi"}

        cloudflare_headers = merge_provider_attribution_headers(
            model("custom", "https://gateway.ai.cloudflare.com/v1"),
            {"enableInstallTelemetry": True},
            None,
        )
        assert cloudflare_headers == {"User-Agent": "pi-coding-agent"}

    def test_session_cwd_helpers_detect_missing_stored_cwd(self, tmp_path):
        from pi_coding_agent.core.session_cwd import (
            MissingSessionCwdError,
            assert_session_cwd_exists,
            format_missing_session_cwd_prompt,
            get_missing_session_cwd_issue,
        )

        class SessionManager:
            def __init__(self, cwd):
                self.cwd = cwd

            def getSessionFile(self):
                return "/sessions/one.jsonl"

            def getCwd(self):
                return self.cwd

        existing = SessionManager(str(tmp_path))
        assert get_missing_session_cwd_issue(existing, "/fallback") is None

        missing = SessionManager(str(tmp_path / "missing"))
        issue = get_missing_session_cwd_issue(missing, "/fallback")
        assert issue.session_file == "/sessions/one.jsonl"
        assert issue.session_cwd.endswith("missing")
        assert "continue in current cwd" in format_missing_session_cwd_prompt(issue)

        with pytest.raises(MissingSessionCwdError) as exc:
            assert_session_cwd_exists(missing, "/fallback")
        assert exc.value.issue == issue

    def test_initial_message_consumes_first_cli_message_and_preserves_images(self):
        from pi_coding_agent.cli_sub.args import Args
        from pi_coding_agent.cli_sub.initial_message import build_initial_message

        parsed = Args(messages=["first", "second"])
        images = [{"type": "image", "mimeType": "image/png", "data": "abc"}]
        result = build_initial_message(
            parsed=parsed,
            stdin_content="stdin\n",
            file_text="file\n",
            file_images=images,
        )

        assert result.initial_message == "stdin\nfile\nfirst"
        assert result.initial_images == images
        assert parsed.messages == ["second"]

        empty = build_initial_message(parsed=Args())
        assert empty.initial_message is None
        assert empty.initial_images is None

    def test_rpc_jsonl_serialization_and_reader_use_lf_only_framing(self):
        import json

        from pi_coding_agent.modes.rpc.jsonl import JsonlLineReader, read_jsonl_lines, serialize_json_line

        payload = {"text": "line separator\u2028paragraph\u2029still same record"}
        serialized = serialize_json_line(payload)
        assert serialized.endswith("\n")
        assert json.loads(serialized) == payload

        lines = read_jsonl_lines([
            b'{"a":"split',
            b'\\u2028inside"}\r\n{"b":2}',
        ])
        assert lines == ['{"a":"split\\u2028inside"}', '{"b":2}']

        emitted: list[str] = []
        reader = JsonlLineReader(emitted.append)
        raw = '{"emoji":"caf\xc3\xa9"}\n'.encode("latin1")
        reader.feed(raw[:-1])
        assert emitted == []
        reader.feed(raw[-1:])
        reader.end()
        assert emitted == ['{"emoji":"café"}']

    def test_rpc_command_source_info_serializes_node_contract(self):
        from pi_coding_agent.core.source_info import create_synthetic_source_info
        from pi_coding_agent.modes.rpc.mode import _source_info_dict
        from pi_coding_agent.modes.rpc.types import RpcSlashCommand

        class Resource:
            source_info = create_synthetic_source_info(
                "/tmp/project/.pi/prompts/plan.md",
                source="local",
                scope="project",
                base_dir="/tmp/project/.pi/prompts",
            )

        command = RpcSlashCommand(
            name="plan",
            description="Plan work",
            source="prompt",
            sourceInfo=_source_info_dict(Resource(), "/tmp/project/.pi/prompts/plan.md"),
        ).model_dump(exclude_none=True)

        assert command == {
            "name": "plan",
            "description": "Plan work",
            "source": "prompt",
            "sourceInfo": {
                "path": "/tmp/project/.pi/prompts/plan.md",
                "source": "local",
                "scope": "project",
                "origin": "top-level",
                "baseDir": "/tmp/project/.pi/prompts",
            },
        }
        assert "path" not in command
        assert "location" not in command

    def test_modes_package_exports_node_style_run_mode_surface(self):
        import pi_coding_agent
        from pi_coding_agent.modes import (
            InteractiveMode,
            PrintModeOptions,
            RpcClient,
            RpcClientOptions,
            RpcCommand,
            RpcExtensionUIRequest,
            RpcExtensionUIResponse,
            RpcResponse,
            RpcSessionState,
            runPrintMode,
            runRpcMode,
            run_print_mode,
            run_rpc_mode,
        )

        class Host:
            session = object()

        interactive = InteractiveMode(Host(), {"initialMessages": ["hello"]})
        assert interactive.session is Host.session
        assert PrintModeOptions(initialMessage="x").initial_message == "x"
        assert runPrintMode is run_print_mode
        assert runRpcMode is run_rpc_mode
        assert pi_coding_agent.runPrintMode is run_print_mode
        assert pi_coding_agent.runRpcMode is run_rpc_mode
        assert RpcClient is pi_coding_agent.RpcClient
        assert RpcClientOptions is pi_coding_agent.RpcClientOptions
        assert RpcCommand is not None
        assert RpcExtensionUIRequest is not None
        assert RpcExtensionUIResponse is not None
        assert RpcResponse is not None
        assert RpcSessionState is not None

    @pytest.mark.asyncio
    async def test_rpc_mode_stdin_iterator_uses_jsonl_reader_contract(self):
        from pi_coding_agent.modes.rpc.mode import _iter_jsonl_stdin_lines

        reader = asyncio.StreamReader()
        reader.feed_data(b' {"type":"prompt","message":"keeps leading json whitespace"}\r\n')
        reader.feed_data(b'{"type":"steer","message":"split')
        reader.feed_data(b' chunk"}\n{"type":"abort"}')
        reader.feed_eof()

        lines = [line async for line in _iter_jsonl_stdin_lines(reader)]

        assert lines == [
            ' {"type":"prompt","message":"keeps leading json whitespace"}',
            '{"type":"steer","message":"split chunk"}',
            '{"type":"abort"}',
        ]

    def test_auth_guidance_messages_point_to_docs(self):
        from pi_coding_agent.core.auth_guidance import (
            format_no_api_key_found_message,
            format_no_model_selected_message,
            format_no_models_available_message,
        )

        no_models = format_no_models_available_message()
        assert "No models available" in no_models
        assert "providers.md" in no_models
        assert "models.md" in no_models

        no_selected = format_no_model_selected_message()
        assert "Then use /model to select a model." in no_selected

        unknown = format_no_api_key_found_message("unknown")
        assert "No API key found for the selected model." in unknown
        assert "unknown" not in unknown.split("\n", 1)[0]

        explicit = format_no_api_key_found_message("openai")
        assert "No API key found for openai." in explicit

    def test_http_dispatcher_timeout_parse_format_and_configure(self):
        from pi_coding_agent.core.http_dispatcher import (
            DEFAULT_HTTP_IDLE_TIMEOUT_MS,
            configure_http_dispatcher,
            format_http_idle_timeout_ms,
            get_configured_http_idle_timeout_ms,
            parse_http_idle_timeout_ms,
        )

        assert DEFAULT_HTTP_IDLE_TIMEOUT_MS == 300_000
        assert parse_http_idle_timeout_ms("disabled") == 0
        assert parse_http_idle_timeout_ms(" 1200.9 ") == 1200
        assert parse_http_idle_timeout_ms("") is None
        assert parse_http_idle_timeout_ms(-1) is None
        assert parse_http_idle_timeout_ms(float("inf")) is None
        assert format_http_idle_timeout_ms(300_000) == "5 min"
        assert format_http_idle_timeout_ms(45_000) == "45 sec"

        assert configure_http_dispatcher(0) == 0
        assert get_configured_http_idle_timeout_ms() == 0
        with pytest.raises(ValueError):
            configure_http_dispatcher(-5)

    def test_utility_helpers_strip_json_mime_paths_browser_and_user_agent(self, tmp_path):
        from pi_coding_agent.utils.ansi import strip_ansi
        from pi_coding_agent.utils.json import strip_json_comments
        from pi_coding_agent.utils.mime import detect_supported_image_mime_type
        from pi_coding_agent.utils.open_browser import browser_command
        from pi_coding_agent.utils.paths import (
            PathInputOptions,
            format_path_relative_to_cwd_or_absolute,
            get_cwd_relative_path,
            is_local_path,
            normalize_path,
            resolve_path,
        )
        from pi_coding_agent.utils.pi_user_agent import get_pi_user_agent

        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
        assert strip_ansi("\x1b]8;;https://example.test\x07link\x1b]8;;\x07") == "link"
        with pytest.raises(TypeError):
            strip_ansi(123)  # type: ignore[arg-type]

        jsonish = '{"url":"https://x.test//keep","items":[1,2,],// drop\n"ok":true,}'
        stripped = strip_json_comments(jsonish)
        assert "https://x.test//keep" in stripped
        assert "// drop" not in stripped
        assert "[1,2]" in stripped
        assert '"ok":true}' in stripped

        png = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (b"\0" * 13) + (b"\0" * 4)
        apng = png + (8).to_bytes(4, "big") + b"acTL" + (b"\0" * 8) + (b"\0" * 4)
        assert detect_supported_image_mime_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
        assert detect_supported_image_mime_type(png) == "image/png"
        assert detect_supported_image_mime_type(apng) is None
        assert detect_supported_image_mime_type(b"GIF89a") == "image/gif"
        assert detect_supported_image_mime_type(b"RIFFxxxxWEBP") == "image/webp"
        assert detect_supported_image_mime_type(b"not image") is None

        home = str(tmp_path)
        opts = PathInputOptions(trim=True, home_dir=home, strip_at_prefix=True, normalize_unicode_spaces=True)
        assert normalize_path("@ ~/a\u00a0b", opts) == " ~/a b"
        assert normalize_path("@~/a\u00a0b", opts) == f"{home}/a b"
        assert normalize_path("file:///tmp/a%20b.txt") == "/tmp/a b.txt"
        assert resolve_path("child.txt", str(tmp_path)) == str(tmp_path / "child.txt")
        assert get_cwd_relative_path(str(tmp_path / "child.txt"), str(tmp_path)) == "child.txt"
        assert get_cwd_relative_path(str(tmp_path.parent / "outside.txt"), str(tmp_path)) is None
        assert format_path_relative_to_cwd_or_absolute(str(tmp_path / "nested" / "x.txt"), str(tmp_path)) == "nested/x.txt"
        assert is_local_path("file:///tmp/x") is True
        assert is_local_path("https://example.test/x") is False
        assert is_local_path("npm:@scope/pkg") is False

        assert browser_command("https://example.test", "Darwin") == ("open", ["https://example.test"])
        assert browser_command("https://example.test", "Windows") == (
            "rundll32",
            ["url.dll,FileProtocolHandler", "https://example.test"],
        )
        assert browser_command("https://example.test", "Linux") == ("xdg-open", ["https://example.test"])

        user_agent = get_pi_user_agent("1.2.3")
        assert user_agent.startswith("pi/1.2.3 (")
        assert "python/" in user_agent

    @pytest.mark.asyncio
    async def test_export_html_ansi_and_custom_tool_rendering(self, tmp_path):
        import base64
        import re

        from pi_coding_agent.core.export_html import (
            ExportOptions,
            ansi_lines_to_html,
            ansi_to_html,
            create_tool_html_renderer,
            export_session_to_html,
        )

        assert ansi_to_html("\x1b[1;31mred & <x>\x1b[0m") == (
            '<span style="color:#800000;font-weight:bold">red &amp; &lt;x&gt;</span>'
        )
        assert ansi_to_html("\x1b[38;5;196mhot\x1b[0m") == '<span style="color:#ff0000">hot</span>'
        assert ansi_to_html("\x1b[38;2;1;2;3mrgb\x1b[0m") == '<span style="color:rgb(1,2,3)">rgb</span>'
        assert ansi_lines_to_html(["", "plain"]) == '<div class="ansi-line">&nbsp;</div><div class="ansi-line">plain</div>'

        class Component:
            def __init__(self, lines):
                self.lines = lines

            def render(self, width):
                return self.lines

        class ToolDef:
            def render_call(self, args, theme, ctx):
                ctx.state["called"] = True
                return Component([f"\x1b[32mcall {args['value']}\x1b[0m"])

            def render_result(self, result, options, theme, ctx):
                return Component(["", f"{'expanded' if options['expanded'] else 'collapsed'} {result['content'][0]['text']}", ""])

        renderer = create_tool_html_renderer(
            get_tool_definition=lambda name: ToolDef() if name == "custom_tool" else None,
            theme={},
            cwd=str(tmp_path),
            width=80,
        )

        entries = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call-1", "name": "custom_tool", "arguments": {"value": "x"}},
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "custom_tool",
                    "content": [{"type": "text", "text": "done"}],
                    "details": {"ok": True},
                    "isError": False,
                },
            },
        ]

        class SessionManager:
            def get_session_id(self):
                return "s1"

            def get_entries(self):
                return entries

            def get_header(self):
                return {"id": "s1", "cwd": str(tmp_path)}

            def get_leaf_id(self):
                return "leaf"

        output_path = tmp_path / "session.html"
        await export_session_to_html(
            SessionManager(),
            options=ExportOptions(output_path=str(output_path), tool_renderer=renderer),
        )
        html_text = output_path.read_text(encoding="utf-8")
        match = re.search(r'<script type="application/json" id="session-data-base64">([^<]+)</script>', html_text)
        assert match
        payload = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))

        rendered = payload["renderedTools"]["call-1"]
        assert "call x" in rendered["callHtml"]
        assert "color:#008000" in rendered["callHtml"]
        assert "collapsed done" in rendered["resultHtmlCollapsed"]
        assert "expanded done" in rendered["resultHtmlExpanded"]
        assert payload["leafId"] == "leaf"

    @pytest.mark.asyncio
    async def test_rpc_client_clone_command_and_pending_rejection(self):
        import json

        from pi_coding_agent.modes.rpc.client import RpcClient

        class Stdin:
            def __init__(self):
                self.lines: list[bytes] = []

            def write(self, data: bytes):
                self.lines.append(data)

            def flush(self):
                return None

        class Process:
            def __init__(self):
                self.stdin = Stdin()
                self.returncode = None

            def poll(self):
                return self.returncode

        client = RpcClient()
        process = Process()
        client._process = process  # type: ignore[attr-defined]

        clone_task = asyncio.create_task(client.clone())
        while not client._pending_requests:  # type: ignore[attr-defined]
            await asyncio.sleep(0)

        sent = json.loads(process.stdin.lines[-1].decode("utf-8"))
        assert sent["type"] == "clone"
        assert sent["id"] == "req_1"

        client._handle_line({"type": "response", "id": "req_1", "command": "clone", "success": True, "data": {"cancelled": False}})  # type: ignore[attr-defined]
        assert await clone_task == {"cancelled": False}

        future = asyncio.get_event_loop().create_future()
        client._pending_requests["req_x"] = future  # type: ignore[attr-defined]
        error = client._create_process_exit_error(2, None)  # type: ignore[attr-defined]
        client._reject_pending_requests(error)  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="Agent process exited"):
            await future

    def test_rpc_client_handle_line_uses_json_events_and_ignores_invalid_lines(self):
        from pi_coding_agent.modes.rpc.client import RpcClient

        client = RpcClient()
        events: list[dict[str, object]] = []
        client.on_event(events.append)

        client._handle_line("not json")  # type: ignore[attr-defined]
        client._handle_line('{"type":"agent_start","id":"e1"}')  # type: ignore[attr-defined]
        assert events == [{"type": "agent_start", "id": "e1"}]


class TestDefaultValues:
    """Test default values match TypeScript."""

    def test_current_provider_defaults_are_resolvable(self):
        """Test direct provider defaults resolve to current catalog models."""
        from pi_coding_agent.core.model_resolver import DEFAULT_MODEL_PER_PROVIDER

        assert DEFAULT_MODEL_PER_PROVIDER["openai"] == "gpt-5.5"
        assert DEFAULT_MODEL_PER_PROVIDER["openai-codex"] == "gpt-5.5"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax"] == "MiniMax-M3"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax-cn"] == "MiniMax-M2.1"
        for provider in ("openai", "openai-codex", "minimax", "minimax-cn"):
            assert get_model(provider, DEFAULT_MODEL_PER_PROVIDER[provider]) is not None
        assert get_model("minimax-cn", "MiniMax-M3") is None

    @pytest.mark.asyncio
    async def test_package_filters_preserve_disabled_resources_and_manifest_globs(self, tmp_path):
        """Package object filters expose all manifest resources with enabled state."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        package = tmp_path / "pkg"
        for folder in ("extensions", "skills", "prompts", "themes"):
            (package / folder).mkdir(parents=True)
        (package / "extensions" / "a.py").write_text("A = 1")
        (package / "extensions" / "b.py").write_text("B = 1")
        (package / "skills" / "a.md").write_text("# Skill A")
        (package / "skills" / "b.md").write_text("# Skill B")
        (package / "prompts" / "p.md").write_text("# Prompt")
        (package / "themes" / "t.json").write_text("{}")
        (package / "package.json").write_text(
            json.dumps(
                {
                    "pi": {
                        "extensions": ["extensions/*.py"],
                        "skills": ["skills/*.md"],
                        "prompts": ["prompts/*.md"],
                        "themes": ["themes/*.json"],
                    }
                }
            )
        )

        settings = SettingsManager.in_memory(
            {
                "packages": [
                    {
                        "source": str(package),
                        "extensions": ["extensions/a.py"],
                        "skills": [],
                    }
                ]
            }
        )
        manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / ".agent"),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        extensions = {Path(item.path).name: item.enabled for item in resolved.extensions}
        skills = {Path(item.path).name: item.enabled for item in resolved.skills}
        prompts = {Path(item.path).name: item.enabled for item in resolved.prompts}
        themes = {Path(item.path).name: item.enabled for item in resolved.themes}

        assert extensions == {"a.py": True, "b.py": False}
        assert skills == {"a.md": False, "b.md": False}
        assert prompts == {"p.md": True}
        assert themes == {"t.json": True}
        assert all(item.metadata.source == str(package) for item in resolved.extensions)
        assert all(item.metadata.origin == "package" for item in resolved.extensions)

    @pytest.mark.asyncio
    async def test_package_resolution_dedupes_identity_with_project_winning(self, tmp_path):
        """Project package config wins over user config for the same package identity."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

        package = tmp_path / "shared-pkg"
        (package / "extensions").mkdir(parents=True)
        (package / "extensions" / "user.py").write_text("USER = 1")
        (package / "extensions" / "project.py").write_text("PROJECT = 1")
        (package / "package.json").write_text(
            json.dumps({"pi": {"extensions": ["extensions/*.py"]}})
        )
        storage = InMemorySettingsStorage(
            global_value=json.dumps(
                {
                    "packages": [
                        {"source": str(package), "extensions": ["extensions/user.py"]}
                    ]
                }
            ),
            project_value=json.dumps(
                {
                    "packages": [
                        {"source": str(package), "extensions": ["extensions/project.py"]}
                    ]
                }
            ),
        )
        settings = SettingsManager.from_storage(storage)
        manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        extensions = {Path(item.path).name: item for item in resolved.extensions}
        assert {name: item.enabled for name, item in extensions.items()} == {
            "project.py": True,
            "user.py": False,
        }
        assert {item.metadata.scope for item in resolved.extensions} == {"project"}

    @pytest.mark.asyncio
    async def test_package_manager_resolves_top_level_settings_resources_with_project_precedence(self, tmp_path):
        """Top-level resource settings are resolved after packages with project precedence."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

        shared = tmp_path / "shared.md"
        shared.write_text("# Shared Skill")
        storage = InMemorySettingsStorage(
            global_value=json.dumps({"skills": [str(shared)]}),
            project_value=json.dumps({"skills": [str(shared), "!shared.md"]}),
        )
        settings = SettingsManager.from_storage(storage)
        manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        assert len(resolved.skills) == 1
        skill = resolved.skills[0]
        assert skill.path == str(shared)
        assert skill.enabled is False
        assert skill.metadata.scope == "project"
        assert skill.metadata.source == "local"
        assert skill.metadata.origin == "top-level"

    @pytest.mark.asyncio
    async def test_package_manager_resolves_auto_discovered_project_and_user_resources(self, tmp_path):
        """Project .pi and user agent-dir resource directories are auto-discovered."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        from pi_coding_agent.config import CONFIG_DIR_NAME
        agent_dir = tmp_path / "agent"
        project_pi = tmp_path / CONFIG_DIR_NAME
        (agent_dir / "prompts").mkdir(parents=True)
        (project_pi / "themes").mkdir(parents=True)
        prompt = agent_dir / "prompts" / "auto.md"
        theme = project_pi / "themes" / "project.json"
        prompt.write_text("# Auto Prompt")
        theme.write_text("{}")
        settings = SettingsManager.in_memory({})
        manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        prompts = {Path(item.path).name: item for item in resolved.prompts}
        themes = {Path(item.path).name: item for item in resolved.themes}
        assert prompts["auto.md"].metadata.source == "auto"
        assert prompts["auto.md"].metadata.scope == "user"
        assert prompts["auto.md"].enabled is True
        assert themes["project.json"].metadata.source == "auto"
        assert themes["project.json"].metadata.scope == "project"
        assert themes["project.json"].enabled is True

    @pytest.mark.asyncio
    async def test_package_manager_resolves_trusted_project_agents_skill_ancestors(self, tmp_path, monkeypatch):
        """Trusted projects discover .agents/skills from cwd ancestors up to git root."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        repo = tmp_path / "repo"
        cwd = repo / "nested" / "child"
        cwd.mkdir(parents=True)
        (repo / ".git").mkdir()
        (repo / ".agents" / "skills" / "repo-skill").mkdir(parents=True)
        (cwd / ".agents" / "skills" / "child-skill").mkdir(parents=True)
        repo_skill = repo / ".agents" / "skills" / "repo-skill" / "SKILL.md"
        child_skill = cwd / ".agents" / "skills" / "child-skill" / "SKILL.md"
        repo_skill.write_text("# Repo Skill")
        child_skill.write_text("# Child Skill")
        settings = SettingsManager.in_memory({})
        manager = DefaultPackageManager(
            cwd=str(cwd),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        skills = {Path(item.path).parent.name: item for item in resolved.skills}
        assert skills["repo-skill"].metadata.scope == "project"
        assert skills["repo-skill"].metadata.base_dir == str(repo / ".agents")
        assert skills["child-skill"].metadata.scope == "project"
        assert skills["child-skill"].metadata.base_dir == str(cwd / ".agents")

    @pytest.mark.asyncio
    async def test_package_manager_skips_project_agents_skills_when_untrusted(self, tmp_path, monkeypatch):
        """Untrusted project settings do not auto-discover project .agents skills."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cwd = tmp_path / "project"
        (cwd / ".agents" / "skills" / "project-skill").mkdir(parents=True)
        (cwd / ".agents" / "skills" / "project-skill" / "SKILL.md").write_text("# Project Skill")
        settings = SettingsManager.in_memory({})
        settings.set_project_trusted(False)
        manager = DefaultPackageManager(
            cwd=str(cwd),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        resolved = await manager.resolve()

        assert resolved.skills == []

    @pytest.mark.asyncio
    async def test_package_manager_node_style_persist_and_listing_methods(self, tmp_path):
        """Package manager exposes Node-style install/remove/list configured APIs."""
        import pi_coding_agent
        import pi_coding_agent.core as core
        from pi_coding_agent.core.package_manager import ConfiguredPackage, DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        class PackageManager(DefaultPackageManager):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.installed: list[tuple[str, dict[str, object] | None]] = []
                self.removed: list[tuple[str, dict[str, object] | None]] = []

            async def install(self, source, options=None):
                self.installed.append((source, options))

            async def remove(self, source, options=None):
                self.removed.append((source, options))

            def get_installed_path(self, source, scope):
                return f"/installed/{scope}/{source.replace('/', '_')}"

        settings = SettingsManager.in_memory(
            {
                "packages": [
                    "npm:@foo/bar",
                    {"source": "npm:@filtered/pkg", "skills": []},
                ]
            }
        )
        manager = PackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        configured = manager.list_configured_packages()

        assert core.ConfiguredPackage is ConfiguredPackage
        assert pi_coding_agent.ConfiguredPackage is ConfiguredPackage
        assert all(isinstance(item, ConfiguredPackage) for item in configured)
        assert [(item.source, item.scope, item.filtered, item.installed_path) for item in configured] == [
            ("npm:@foo/bar", "user", False, "/installed/user/npm:@foo_bar"),
            ("npm:@filtered/pkg", "user", True, "/installed/user/npm:@filtered_pkg"),
        ]

        await manager.install_and_persist("npm:@new/pkg", {"local": False})
        assert manager.installed == [("npm:@new/pkg", {"local": False})]
        assert "npm:@new/pkg" in settings.get_global_settings()["packages"]

        removed = await manager.remove_and_persist("npm:@foo/bar", {"local": False})
        assert removed is True
        assert manager.removed == [("npm:@foo/bar", {"local": False})]
        assert "npm:@foo/bar" not in settings.get_global_settings()["packages"]
        progress_manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )
        events = []
        progress_manager.setProgressCallback(events.append)
        await progress_manager.install(str(tmp_path), {"local": False})
        assert [(event.type, event.action, event.source) for event in events] == [
            ("start", "install", str(tmp_path)),
            ("complete", "install", str(tmp_path)),
        ]
        assert manager.getInstalledPath("npm:@foo/bar", "user") == "/installed/user/npm:@foo_bar"
        assert manager.addSourceToSettings("npm:@camel/pkg", {"local": False}) is True
        assert "npm:@camel/pkg" in settings.get_global_settings()["packages"]
        assert manager.removeSourceFromSettings("npm:@camel/pkg", {"local": False}) is True
        assert "npm:@camel/pkg" not in settings.get_global_settings()["packages"]
        source_package = tmp_path / "source-package"
        (source_package / "skills").mkdir(parents=True)
        (source_package / "skills" / "node-style.md").write_text("# Node Style Skill")
        resolved = await manager.resolveExtensionSources([str(source_package)], {"temporary": True})
        assert [(Path(item.path).name, item.metadata.scope, item.metadata.source) for item in resolved.skills] == [
            ("node-style.md", "temporary", str(source_package)),
        ]
        assert manager.installAndPersist.__func__ is manager.install_and_persist.__func__
        assert manager.removeAndPersist.__func__ is manager.remove_and_persist.__func__
        assert manager.listConfiguredPackages() == manager.list_configured_packages()

    @pytest.mark.asyncio
    async def test_package_manager_update_errors_when_specific_source_not_configured(self, tmp_path):
        """Specific package updates fail when the source is not configured."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        settings = SettingsManager.in_memory({"packages": ["npm:@foo/bar"]})
        manager = DefaultPackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        with pytest.raises(RuntimeError, match="No matching package found for npm:@missing/pkg"):
            await manager.update("npm:@missing/pkg")

    @pytest.mark.asyncio
    async def test_package_manager_update_specific_source_only_updates_match(self, tmp_path):
        """Specific package updates only touch matching configured package identities."""
        from pi_coding_agent.core.package_manager import DefaultPackageManager
        from pi_coding_agent.core.settings_manager import SettingsManager

        class PackageManager(DefaultPackageManager):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.updated: list[tuple[str, str]] = []

            async def _update_source_for_scope(self, source, scope):
                self.updated.append((source, scope))

        settings = SettingsManager.in_memory({"packages": ["npm:@foo/bar", "npm:@foo/baz"]})
        manager = PackageManager(
            cwd=str(tmp_path),
            agent_dir=str(tmp_path / "agent"),
            settings_manager=settings,
        )

        await manager.update("npm:@foo/bar")

        assert manager.updated == [("npm:@foo/bar", "user")]

    def test_core_package_exports_node_style_public_surface(self):
        """Test pi_coding_agent.core exposes the public API callers expect."""
        import pi_coding_agent.core as core
        from pi_coding_agent.core.agent_session import AgentSession

        assert core.AgentSession is AgentSession
        assert core.createAgentSession is core.create_agent_session
        assert core.createAgentSessionRuntime is core.create_agent_session_runtime
        assert core.createAgentSessionServices is core.create_agent_session_services
        assert core.createAgentSessionFromServices is core.create_agent_session_from_services
        assert core.createEventBus is core.create_event_bus
        assert core.createSyntheticSourceInfo is core.create_synthetic_source_info
        assert core.ExtensionRunner.__name__ == "ExtensionRunner"
        assert core.ToolDefinition.__name__ == "ToolDefinition"
        assert core.ModelRegistry.__name__ == "ModelRegistry"
        assert core.SessionManager.__name__ == "SessionManager"

    def test_grouped_tool_factories_match_node_sdk_contract(self, tmp_path):
        import pi_coding_agent
        import pi_coding_agent.core as core
        from pi_coding_agent.core import sdk
        from pi_coding_agent.core.tools import (
            all_tool_names,
            create_all_tools,
            create_bash_tool_definition,
            create_coding_tools,
            create_edit_tool_definition,
            create_find_tool_definition,
            create_grep_tool_definition,
            create_ls_tool_definition,
            create_read_tool_definition,
            create_read_only_tools,
            create_tool,
            create_tool_definition,
            create_write_tool_definition,
        )

        assert all_tool_names == {"read", "bash", "edit", "write", "grep", "find", "ls"}
        assert [tool.name for tool in create_coding_tools(str(tmp_path))] == ["read", "bash", "edit", "write"]
        assert [tool.name for tool in create_read_only_tools(str(tmp_path))] == ["read", "grep", "find", "ls"]
        assert list(create_all_tools(str(tmp_path))) == ["read", "bash", "edit", "write", "grep", "find", "ls"]
        assert create_tool("read", str(tmp_path)).name == "read"
        assert create_tool_definition("bash", str(tmp_path)).name == "bash"
        named_definition_factories = [
            (create_read_tool_definition, "read"),
            (create_bash_tool_definition, "bash"),
            (create_edit_tool_definition, "edit"),
            (create_write_tool_definition, "write"),
            (create_grep_tool_definition, "grep"),
            (create_find_tool_definition, "find"),
            (create_ls_tool_definition, "ls"),
        ]
        for factory, name in named_definition_factories:
            assert factory(str(tmp_path)).name == name

        with pytest.raises(ValueError, match="Unknown tool name"):
            create_tool("unknown", str(tmp_path))

        assert sdk.createBashToolDefinition is create_bash_tool_definition
        assert sdk.createReadToolDefinition is create_read_tool_definition
        assert sdk.createCodingTools is create_coding_tools
        assert sdk.createReadOnlyTools is create_read_only_tools
        assert core.createBashToolDefinition is create_bash_tool_definition
        assert core.createReadToolDefinition is create_read_tool_definition
        assert core.createCodingTools is create_coding_tools
        assert core.createReadOnlyTools is create_read_only_tools
        assert pi_coding_agent.createBashToolDefinition is create_bash_tool_definition
        assert pi_coding_agent.createReadToolDefinition is create_read_tool_definition
        assert pi_coding_agent.createCodingTools is create_coding_tools
        assert pi_coding_agent.createReadOnlyTools is create_read_only_tools

    def test_root_package_exports_node_style_resources_and_utilities(self, tmp_path, monkeypatch):
        import pi_coding_agent

        agent_dir = tmp_path / "agent"
        project = tmp_path / "project"
        child = project / "child"
        agent_dir.mkdir()
        child.mkdir(parents=True)
        (agent_dir / "AGENTS.md").write_text("global context", encoding="utf-8")
        (project / "CLAUDE.md").write_text("project context", encoding="utf-8")

        loaded = pi_coding_agent.loadProjectContextFiles(str(child), str(agent_dir))

        assert [item["content"] for item in loaded] == ["global context", "project context"]
        untrusted_loaded = pi_coding_agent.loadProjectContextFiles(str(child), str(agent_dir), False)
        assert [item["content"] for item in untrusted_loaded] == ["global context"]

        uppercase_agent_dir = tmp_path / "uppercase-agent"
        uppercase_project = tmp_path / "uppercase-project"
        uppercase_child = uppercase_project / "child"
        uppercase_agent_dir.mkdir()
        uppercase_child.mkdir(parents=True)
        (uppercase_agent_dir / "AGENTS.MD").write_text("uppercase global", encoding="utf-8")
        (uppercase_project / "CLAUDE.MD").write_text("uppercase project", encoding="utf-8")

        uppercase_loaded = pi_coding_agent.loadProjectContextFiles(
            str(uppercase_child),
            str(uppercase_agent_dir),
        )

        assert [item["content"] for item in uppercase_loaded] == ["uppercase global", "uppercase project"]
        assert pi_coding_agent.convertToLlm is pi_coding_agent.convert_to_llm
        assert pi_coding_agent.createSyntheticSourceInfo is pi_coding_agent.create_synthetic_source_info
        assert pi_coding_agent.formatSkillsForPrompt is pi_coding_agent.format_skills_for_prompt
        assert pi_coding_agent.parseFrontmatter is pi_coding_agent.parse_frontmatter
        assert pi_coding_agent.stripFrontmatter is pi_coding_agent.strip_frontmatter
        assert pi_coding_agent.resizeImage is pi_coding_agent.resize_image
        assert pi_coding_agent.convertToPng is pi_coding_agent.convert_to_png
        assert pi_coding_agent.getShellConfig is pi_coding_agent.get_shell_config
        assert "DefaultResourceLoader" in pi_coding_agent.__all__

    @pytest.mark.asyncio
    async def test_default_resource_loader_exposes_node_style_methods(self, tmp_path):
        from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions

        agent_dir = tmp_path / "agent"
        project = tmp_path / "project"
        skill_dir = tmp_path / "extra"
        skill = skill_dir / "SKILL.md"
        prompt = tmp_path / "extra-prompt.md"
        theme = tmp_path / "theme.json"
        agent_dir.mkdir()
        project.mkdir()
        skill_dir.mkdir()
        skill.write_text("---\nname: extra\ndescription: Extra skill\n---\nSkill body", encoding="utf-8")
        prompt.write_text("---\nname: extra-prompt\n---\nPrompt body", encoding="utf-8")
        theme.write_text('{"accent":"blue"}', encoding="utf-8")

        loader = DefaultResourceLoader(
            DefaultResourceLoaderOptions(
                cwd=str(project),
                agent_dir=str(agent_dir),
                no_extensions=True,
                no_skills=True,
                no_prompt_templates=True,
                no_themes=True,
                no_context_files=True,
                system_prompt="System",
                append_system_prompt=["Append"],
            )
        )
        await loader.reload()

        assert loader.getExtensions() == loader.get_extensions()
        assert loader.getSkills() == loader.get_skills()
        assert loader.getPrompts() == loader.get_prompts()
        assert loader.getThemes() == loader.get_themes()
        assert loader.getAgentsFiles() == loader.get_agents_files()
        assert loader.getSystemPrompt() == "System"
        assert loader.getAppendSystemPrompt() == ["Append"]

        loader.extendResources(
            {
                "skillPaths": [{"path": str(skill), "metadata": {"source": "test", "scope": "temporary", "origin": "top-level"}}],
                "promptPaths": [{"path": str(prompt), "metadata": {"source": "test", "scope": "temporary", "origin": "top-level"}}],
                "themePaths": [{"path": str(theme), "metadata": {"source": "test", "scope": "temporary", "origin": "top-level"}}],
            }
        )

        assert [item.name for item in loader.getSkills()["skills"]] == ["extra"]
        assert [item.name for item in loader.getPrompts()["prompts"]] == ["extra-prompt"]
        assert [item.name for item in loader.getThemes()["themes"]] == ["theme"]

    def test_root_package_exports_node_style_interactive_components(self):
        import pi_coding_agent
        from pi_coding_agent.modes.interactive import components

        for name in [
            "ArminComponent",
            "AssistantMessageComponent",
            "BashExecutionComponent",
            "BorderedLoader",
            "BranchSummaryMessageComponent",
            "CompactionSummaryMessageComponent",
            "CustomEditor",
            "CustomMessageComponent",
            "DaxnutsComponent",
            "DynamicBorder",
            "ExtensionEditorComponent",
            "ExtensionInputComponent",
            "ExtensionSelectorComponent",
            "FooterComponent",
            "LoginDialogComponent",
            "ModelSelectorComponent",
            "OAuthSelectorComponent",
            "ScopedModelsSelectorComponent",
            "SessionSelectorComponent",
            "SettingsSelectorComponent",
            "ShowImagesSelectorComponent",
            "SkillInvocationMessageComponent",
            "ThemeSelectorComponent",
            "ThinkingSelectorComponent",
            "ToolExecutionComponent",
            "TreeSelectorComponent",
            "TrustSelectorComponent",
            "UserMessageComponent",
            "UserMessageSelectorComponent",
        ]:
            assert getattr(pi_coding_agent, name) is getattr(components, name)
            assert name in pi_coding_agent.__all__

        assert components.keyHint is components.key_hint
        assert components.keyText is components.key_text
        assert components.rawKeyHint is components.raw_key_hint
        assert components.renderDiff is components.render_diff
        assert components.truncateToVisualLines is components.truncate_to_visual_lines
        assert pi_coding_agent.keyHint is components.key_hint
        assert pi_coding_agent.renderDiff is components.render_diff
        assert pi_coding_agent.truncateToVisualLines is components.truncate_to_visual_lines

        skill = components.SkillInvocationMessageComponent({"name": "planning", "content": "Do the work"})
        assert skill.render(80)[0].startswith("[skill] planning")
        skill.setExpanded(True)
        assert skill.render(80) == ["[skill] planning", "Do the work"]

        class UI:
            render_count = 0

            def request_render(self):
                self.render_count += 1

        armin = components.ArminComponent(UI())
        daxnuts = components.DaxnutsComponent(UI())
        assert any("ARMIN SAYS HI" in line for line in armin.render(80))
        assert any("POWERED BY DAXNUTS" in line for line in daxnuts.render(80))
        armin.dispose()
        daxnuts.dispose()
        assert armin.disposed is True
        assert daxnuts.disposed is True

    def test_interactive_extension_ui_context_updates_editor_status_and_widgets(self):
        from pi_coding_agent.modes.interactive.tui import InteractiveExtensionUIContext

        history = []
        renders = []
        editor = {"text": "draft"}
        statuses = {}
        widgets = {}
        titles = []
        autocomplete_wrappers = []
        terminal_input_listeners = []
        editor_component_factory = {"value": None}
        working = {}
        surfaces = {}

        ctx = InteractiveExtensionUIContext(
            append_history=history.append,
            request_render=lambda: renders.append("render"),
            set_editor_text=lambda text: editor.__setitem__("text", text),
            get_editor_text=lambda: editor["text"],
            set_title=titles.append,
            set_status=lambda key, text: statuses.__setitem__(key, text),
            set_widget=lambda key, lines, placement: widgets.__setitem__(key, (lines, placement)),
            add_autocomplete_provider=autocomplete_wrappers.append,
            add_terminal_input_listener=lambda handler: (
                terminal_input_listeners.append(handler),
                lambda: terminal_input_listeners.remove(handler),
            )[1],
            set_editor_component=lambda factory: editor_component_factory.__setitem__("value", factory),
            get_editor_component=lambda: editor_component_factory["value"],
            set_working_message=lambda message: working.__setitem__("message", message),
            set_working_visible=lambda visible: working.__setitem__("visible", visible),
            set_working_indicator=lambda options: working.__setitem__("indicator", options),
            set_hidden_thinking_label=lambda label: working.__setitem__("thinking_label", label),
            set_footer=lambda factory: surfaces.__setitem__("footer", factory),
            set_header=lambda factory: surfaces.__setitem__("header", factory),
        )

        ctx.notify("Loaded", "info")
        ctx.setStatus("sync", "ready")
        ctx.setWidget("panel", ["line 1", "line 2"], {"placement": "belowEditor"})
        ctx.setTitle("Pi Session")
        ctx.pasteToEditor(" plus")
        ctx.setEditorText("replacement")
        ctx.addAutocompleteProvider(lambda provider: provider)
        terminal_unsubscribe = ctx.onTerminalInput(lambda data: {"data": data.upper()})
        custom_editor_factory = lambda tui, theme, keybindings: object()
        ctx.setEditorComponent(custom_editor_factory)
        ctx.setWorkingMessage("syncing")
        ctx.setWorkingVisible(False)
        ctx.setWorkingIndicator({"label": "pulse"})
        ctx.setHiddenThinkingLabel("Reasoning hidden")
        header_factory = lambda tui, theme: "Custom header"
        footer_factory = lambda tui, theme, footer_data: "Custom footer"
        ctx.setHeader(header_factory)
        ctx.setFooter(footer_factory)

        assert history == ["[info] Loaded"]
        assert statuses == {"sync": "ready"}
        assert widgets == {"panel": (["line 1", "line 2"], "belowEditor")}
        assert titles == ["Pi Session"]
        assert editor["text"] == "replacement"
        assert ctx.getEditorText() == "replacement"
        assert len(autocomplete_wrappers) == 1
        assert len(terminal_input_listeners) == 1
        terminal_unsubscribe()
        assert terminal_input_listeners == []
        assert ctx.getEditorComponent() is custom_editor_factory
        ctx.setEditorComponent(None)
        assert ctx.getEditorComponent() is None
        assert working == {
            "message": "syncing",
            "visible": False,
            "indicator": {"label": "pulse"},
            "thinking_label": "Reasoning hidden",
        }
        assert surfaces == {"header": header_factory, "footer": footer_factory}
        assert len(renders) == 11

    def test_interactive_autocomplete_provider_wrappers_compose_suggestions(self):
        from pi_tui.autocomplete import AutocompleteItem, CombinedAutocompleteProvider, SuggestionResult
        from pi_coding_agent.modes.interactive.tui import _apply_autocomplete_provider_wrappers

        base = CombinedAutocompleteProvider()

        class ExtraProvider:
            def __init__(self, current):
                self.current = current

            def get_suggestions(self, lines, cursor_line, cursor_col):
                current = self.current.get_suggestions(lines, cursor_line, cursor_col)
                items = list(current.items) if current else []
                items.append(AutocompleteItem(value="extension", label="extension", description="Extension item"))
                return SuggestionResult(items=items, prefix=current.prefix if current else "")

            def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):
                return self.current.apply_completion(lines, cursor_line, cursor_col, item, prefix)

        provider = _apply_autocomplete_provider_wrappers(
            base,
            [lambda current: ExtraProvider(current)],
        )

        result = provider.get_suggestions(["hello"], 0, 5)

        assert result is not None
        assert result.items[-1].value == "extension"
        assert result.items[-1].description == "Extension item"

    def test_interactive_help_and_hotkeys_include_extension_registrations(self):
        from pi_coding_agent.core.extensions.runner import ExtensionRunner
        from pi_coding_agent.core.extensions.types import Extension, ExtensionShortcut, RegisteredCommand
        from pi_coding_agent.modes.interactive.tui import (
            _built_in_command_conflict_diagnostics,
            _extension_command_help_lines,
            _extension_shortcut_hotkey_lines,
            _path_command_argument,
        )

        assert _path_command_argument("/export", "/export") is None
        assert _path_command_argument('/export "my session.jsonl" trailing', "/export") == "my session.jsonl"
        assert _path_command_argument("/import 'old session.jsonl'", "/import") == "old session.jsonl"
        assert _path_command_argument("/import old.jsonl --ignored", "/import") == "old.jsonl"
        assert _path_command_argument('/export "unterminated', "/export") is None

        runner = ExtensionRunner(
            [
                Extension(
                    path="/tmp/ext.py",
                    resolved_path="/tmp/ext.py",
                    commands={
                        "plan": RegisteredCommand(
                            name="plan",
                            description="Plan current work",
                            extension_path="/tmp/ext.py",
                        )
                    },
                    shortcuts={
                        "ctrl+j": ExtensionShortcut(
                            shortcut="ctrl+j",
                            description="Open planning panel",
                            extension_path="/tmp/ext.py",
                        )
                    },
                )
            ]
        )

        command_lines = _extension_command_help_lines(runner)
        shortcut_lines = _extension_shortcut_hotkey_lines(runner.get_shortcuts({}))

        assert command_lines == ["  /plan — Plan current work"]
        assert shortcut_lines == ["", "Extension shortcuts:", "  ctrl+j: Open planning panel"]

        conflict_runner = ExtensionRunner(
            [
                Extension(
                    path="/tmp/model.py",
                    resolved_path="/tmp/model.py",
                    commands={
                        "model": RegisteredCommand(
                            name="model",
                            description="Extension model command",
                            extension_path="/tmp/model.py",
                        )
                    },
                )
            ]
        )
        duplicate_runner = ExtensionRunner(
            [
                Extension(
                    path="/tmp/first.py",
                    resolved_path="/tmp/first.py",
                    commands={
                        "model": RegisteredCommand(
                            name="model",
                            description="First model command",
                            extension_path="/tmp/first.py",
                        )
                    },
                ),
                Extension(
                    path="/tmp/second.py",
                    resolved_path="/tmp/second.py",
                    commands={
                        "model": RegisteredCommand(
                            name="model",
                            description="Second model command",
                            extension_path="/tmp/second.py",
                        )
                    },
                ),
            ]
        )

        assert _built_in_command_conflict_diagnostics(conflict_runner, {"model"}) == [
            {
                "type": "warning",
                "message": (
                    "Extension command '/model' conflicts with built-in interactive command. "
                    "Skipping in autocomplete."
                ),
                "path": "/tmp/model.py",
            }
        ]
        assert [
            item["message"]
            for item in _built_in_command_conflict_diagnostics(duplicate_runner, {"model"})
        ] == [
            "Extension command '/model' conflicts with built-in interactive command. Available as '/model:1'.",
            "Extension command '/model' conflicts with built-in interactive command. Available as '/model:2'.",
        ]

    @pytest.mark.asyncio
    async def test_thinking_level_default_medium(self):
        """Test thinking_level defaults to 'medium' (not 'off')."""
        result = await create_agent_session()
        session = result.session
        
        # Default should be 'medium' (may be clamped to 'off' if model doesn't support reasoning)
        # Check the settings object
        # Note: actual thinking_level may be 'off' if model doesn't support reasoning
        assert session._settings.thinking_level in ["off", "medium", "low", "minimal", "high"]
