"""Routing activation through the UI command, session, provider adapter and tools.

The HTTP fixture supplies prescribed metadata: these are orchestration checks,
not model-quality or metadata-classifier evals.
"""
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from jsonschema import Draft202012Validator
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_registry import ModelRegistry
from pi_coding_agent.core.model_router import routing_level
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings, SettingsManager
from pi_coding_agent.modes.interactive.tui import _footer_model_parts, _handle_model_command


def metadata(work="implement", method="specified", interpretation="direct"):
    return {"work_kind": work, "method_certainty": method,
            "interpretation_required": interpretation, "constraint_interaction": "independent",
            "classification_explanations": {key: "Fixture classification for boundary testing."
                for key in ("work_kind", "method_certainty", "interpretation_required", "constraint_interaction")}}


def envelope(actions, next_metadata):
    return {"text": "Processing fixture." if actions else "Finished.", "dependency_probe": None,
            "response": {"tool_calls": actions, "next_invocation": next_metadata}}


class Loader:
    def get_extensions(self): return {"extensions": [], "diagnostics": []}
    def get_skills(self): return {"skills": [], "diagnostics": []}
    def get_agents_files(self): return {"agentsFiles": []}
    def get_system_prompt(self): return "Complete the requested fixture operation."
    def get_append_system_prompt(self): return []


async def model_command(session, text, select=None):
    history, footer = [], []
    await _handle_model_command(text, session, history.append,
        lambda: footer.append(" | ".join(_footer_model_parts(session))),
        SimpleNamespace(request_render=lambda: None), str, str, str, str, str, show_select=select)
    return history, footer


def make_session(tmp_path, monkeypatch, base_url):
    from pi_coding_agent.core.router_config import ROUTER_LEVELS
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"router-fixture": {
        "api": "openai-responses", "baseUrl": base_url,
        "models": [{"id": level, "name": level, "reasoning": True} for level in ROUTER_LEVELS],
    }}, "router": {"levels": {level: {"provider": "router-fixture", "model": level,
                                      "reasoning": "low" if level == "ultra-light" else "high"}
                              for level in ROUTER_LEVELS}}}))
    monkeypatch.setattr("pi_coding_agent.config.get_models_path", lambda: str(path))
    auth = AuthStorage.in_memory({"router-fixture": {"type": "api_key", "key": "test-key"}})
    registry = ModelRegistry(auth_storage=auth, models_json_path=str(path))
    session = AgentSession(cwd=str(tmp_path), model=registry.find("router-fixture", "default"),
        auth_storage=auth, model_registry=registry, settings=Settings(auto_compact=False),
        settings_manager=SettingsManager.in_memory({"compaction": {"enabled": False}}),
        session_manager=SessionManager.create(str(tmp_path), str(tmp_path / "sessions")),
        resource_loader=Loader(), initial_active_tool_names=["read", "edit"])
    return session, path


@pytest.mark.asyncio
async def test_router_activation_http_tools_footer_and_fixed_model(tmp_path, monkeypatch):
    requests, traces = [], []
    monkeypatch.setattr("pi_coding_agent.core.agent_session._instr_emit",
                        lambda name, **kwargs: traces.append((name, kwargs)))
    file = tmp_path / "fixture.txt"
    file.write_text("before\npreserve\n")

    async def responses(request):
        body = await request.json()
        requests.append(body)
        step = len(requests)
        if step <= 4:
            assert body["tool_choice"] == {"type": "function", "name": "submit_response"}
            assert body["tools"][0]["strict"] is True
            assert body["parallel_tool_calls"] is False
            if step == 1:
                wire = envelope([{"name": "read", "arguments": {"path": str(file), "offset": None, "limit": None}}], metadata())
            elif step == 2:
                assert any(item.get("type") == "function_call_output" and "before" in item["output"] for item in body["input"])
                wire = envelope([{"name": "edit", "arguments": {"path": str(file), "oldText": "before", "newText": "after"}}], metadata("plan", "needs_design"))
            else:
                assert file.read_text() == "after\npreserve\n"
                wire = envelope([], None)
            Draft202012Validator(body["tools"][0]["parameters"]).validate(wire)
            item = {"type": "function_call", "id": f"fc_{step}", "call_id": f"call_{step}",
                    "name": "submit_response", "arguments": json.dumps(wire), "status": "completed"}
            events = [
                {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}},
                {"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": 0, "delta": item["arguments"]},
                {"type": "response.function_call_arguments.done", "item_id": item["id"], "output_index": 0, "arguments": item["arguments"]},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
            ]
        else:
            assert all(tool["name"] != "submit_response" for tool in body["tools"])
            item = {"type": "message", "id": "msg_fixed", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Fixed model reply.", "annotations": []}]}
            events = [{"type": "response.output_item.added", "output_index": 0, "item": item},
                      {"type": "response.output_item.done", "output_index": 0, "item": item}]
        events.append({"type": "response.completed", "response": {
            "id": f"resp_{step}", "status": "completed", "output": [item],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}})
        return web.Response(text="".join("data: " + json.dumps(event) + "\n\n" for event in events),
                            content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/responses", responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        session, _ = make_session(tmp_path, monkeypatch, f"http://127.0.0.1:{port}")
        assert "router on" not in _footer_model_parts(session)
        history, footer = await model_command(session, "/model router")
        assert history == ["Router on."] and footer == ["router on"]
        await session.prompt("Read fixture.txt and replace before with after, preserving the other line.")
        assert session.agent.state.error is None
        assert [body["model"] for body in requests] == ["default", "ultra-light", "max"]
        assert [body["reasoning"]["effort"] for body in requests] == ["high", "low", "high"]
        assert file.read_text() == "after\npreserve\n"
        assert not any(getattr(block, "name", None) == "submit_response"
                       for msg in session.agent.state.messages for block in getattr(msg, "content", []) if not isinstance(block, str))
        selected = [kw["metadata"]["level"] for name, kw in traces if name == "tau.router_selection"]
        assert selected == ["default", "ultra-light", "max"]
        await session.prompt("Report completion again.")
        assert requests[-1]["model"] == "default"
        assert session.router_enabled and session.agent.state.error is None
        history, footer = await model_command(session, "/model router-fixture/light")
        assert not session.router_enabled and footer[-1].startswith("light | thinking:")
        await session.prompt("Reply using the selected fixed model.")
        assert requests[-1]["model"] == "light"
        assert session.agent.state.error is None
        reopened = SessionManager.open(session._session_manager.get_session_file())
        entries = [{**entry.data, "type": entry.type} for entry in reopened.get_entries()]
        decisions = [entry["data"] for entry in entries
                     if entry.get("customType") == "tau.router_selection"]
        assert [row["level"] for row in decisions] == ["default", "ultra-light", "max", "default"]
        assert [row["reasoning"] for row in decisions] == ["high", "low", "high", "high"]
        assert decisions[2]["rule"] == "planning"
        assert decisions[2]["next_invocation"]["classification_explanations"]
        assert len([e for e in entries if e.get("customType") == "tau.router_metadata"]) == 4
        from pi_coding_agent.core.model_stats import model_stats, render_model_stats
        stats = model_stats(entries)
        assert stats == session.get_model_stats()
        assert stats["total"] == 5
        assert {r["model"]: r["percent"] for r in stats["models"]} == {
            "default": 40, "ultra-light": 20, "max": 20, "light": 20}
        assert "40.0%  (2)" in render_model_stats(stats)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_router_menu_and_failed_activation_keep_footer_honest(tmp_path, monkeypatch):
    session, path = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    config = json.loads(path.read_text())
    del config["router"]["levels"]["max"]
    path.write_text(json.dumps(config))
    history, footer = await model_command(session, "/model router")
    assert history[0].startswith("Could not enable router:")
    assert not session.router_enabled and "router on" not in footer
    config["router"]["levels"]["max"] = config["router"]["levels"]["default"]
    path.write_text(json.dumps(config))

    async def select(title, choices, options):
        assert "Router" in choices
        return "Router"

    history, footer = await model_command(session, "/model", select)
    assert history == ["Router on."] and footer == ["router on"]


@pytest.mark.parametrize("meta,probe,expected", [
    (None, None, "default"), (metadata(), None, "ultra-light"),
    (metadata(method="adaptation_required"), None, "light"),
    (metadata(interpretation="bounded"), None, "light"),
    (metadata(method="needs_design"), None, "default"),
    (metadata(interpretation="inferential"), None, "default"),
    (metadata(work="plan"), None, "max"),
    (metadata(interpretation="competing_explanations"), None, "max"),
    (metadata(), {"a_unresolved": True,"b_unresolved": True,"a_affects_b": True,"b_affects_a": False}, "light"),
    (metadata(), {"a_unresolved": True,"b_unresolved": True,"a_affects_b": True,"b_affects_a": True}, "max"),
    (metadata(), {"a_unresolved": False,"b_unresolved": True,"a_affects_b": True,"b_affects_a": True}, "ultra-light"),
])
def test_sketch_policy(meta, probe, expected):
    assert routing_level(meta, probe)[0] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_kind", ["pending_null", "unknown", "missing_carrier"])
async def test_invalid_routed_output_cannot_execute_tools(tmp_path, monkeypatch, bad_kind):
    from pi_ai.types import AssistantMessage, EventDone, TextContent, ToolCall, Usage

    session, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    file = tmp_path / "protected.txt"
    file.write_text("preserve")

    async def provider(model, context, options):
        meta = metadata()
        if bad_kind == "unknown":
            meta["method_certainty"] = "unknown"
        wire = envelope([{"name": "edit", "arguments": {"path": str(file), "oldText": "preserve", "newText": "changed"}}],
                        None if bad_kind == "pending_null" else meta)
        content = ([TextContent(text="No carrier")] if bad_kind == "missing_carrier" else
                   [ToolCall(id="bad", name="submit_response", arguments=wire)])
        final = AssistantMessage(content=content, api=model.api, provider=model.provider,
                                 model=model.id, usage=Usage(input=100, output=20, total_tokens=120), timestamp=0)
        yield EventDone(reason="stop", message=final)

    session._provider_stream = provider
    await model_command(session, "/model router")
    await session.prompt("Edit the fixture")
    assert session.agent.state.error
    assert file.read_text() == "preserve"
    assert session.agent.state.messages[-1].usage.total_tokens == 120
    assert session._router.next_metadata is None


def test_nested_optional_tool_arguments_survive_strict_transport():
    from pi_ai.types import Tool
    from pi_coding_agent.core.model_router import native_arguments, response_schema

    parameters = {"type": "object", "required": ["items"], "properties": {
        "items": {"type": "array", "items": {"$ref": "#/$defs/row"}}},
        "$defs": {"row": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "optional_count": {"type": "integer"}}}}}
    schema = response_schema([Tool(name="nested", description="Nested tool", parameters=parameters)])
    wire = envelope([{"name": "nested", "arguments": {"items": [{"name": "keep", "optional_count": None}]}}], metadata())
    Draft202012Validator(schema).validate(wire)
    arguments = native_arguments(wire["response"]["tool_calls"][0]["arguments"], parameters)
    Draft202012Validator(parameters).validate(arguments)
    assert arguments == {"items": [{"name": "keep"}]}


@pytest.mark.asyncio
async def test_router_activates_with_real_a2a_catalog(tmp_path, monkeypatch):
    from pi_ai.types import Tool
    from pi_coding_agent.a2a.extension import extension_factory

    catalog = []
    def register(name, *, description, parameters, execute):
        catalog.append(Tool(name=name, description=description, parameters=parameters))
    extension_factory(SimpleNamespace(register_tool=register))
    assert len(catalog) == 7
    session, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    session.agent.state.tools.extend(catalog)
    history, footer = await model_command(session, "/model router")
    assert history == ["Router on."] and footer == ["router on"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,valid", [
    ('{"nested":{"arbitrary key":[1,true,null,{"x":"é"}]},"empty":{}}', True),
    ('{"broken":', False), ('[1,2]', False), ('{"x":NaN}', False),
])
async def test_a2a_router_stream_decodes_and_validates_without_sending(tmp_path, monkeypatch, payload, valid):
    from pi_ai.types import AssistantMessage, Context, EventDone, Tool, ToolCall, Usage
    from pi_coding_agent.a2a.extension import SendParams

    session, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    tool = Tool(name="a2a_send_message", description="Send", parameters=SendParams.model_json_schema())
    session.agent.state.tools.append(tool)
    await model_command(session, "/model router")
    args = {key: None for key in tool.parameters["properties"]}
    args.update(to_agent="fixture", message="fixture", payload=payload, metadata='{"preserve":null}')
    wire = envelope([{"name": tool.name, "arguments": args}], metadata())
    async def provider(model, context, options):
        Draft202012Validator(context.tools[0].parameters).validate(wire)
        yield EventDone(reason="toolUse", message=AssistantMessage(
            content=[ToolCall(id="carrier", name="submit_response", arguments=wire)],
            api=model.api, provider=model.provider, model=model.id,
            usage=Usage(input=100, output=20, total_tokens=120), timestamp=0))
    events = [event async for event in session._router.stream(
        provider, session.model, Context(messages=[], tools=[tool]), {})]
    if valid:
        call = events[-1].message.content[-1]
        assert call.name == tool.name
        parsed = SendParams.model_validate(call.arguments)
        assert parsed.payload == json.loads(payload)
        assert parsed.metadata == {"preserve": None}
        assert "thread_id" not in call.arguments
    else:
        assert events[-1].type == "error"
        assert events[-1].error.content == []
        assert session._router.next_metadata is None


@pytest.mark.parametrize("subschema,value", [
    ({"type": "object", "additionalProperties": {"type": "integer"}}, {"a": 3}),
    ({"type": "object"}, {"a": {"nested": None}}),
    ({}, [1, None, {"arbitrary": True}]),
])
def test_free_form_argument_transport(subschema, value):
    from pi_ai.types import Tool
    from pi_coding_agent.core.model_router import native_arguments, response_schema
    parameters = {"type": "object", "properties": {"data": subschema}, "required": ["data"]}
    schema = response_schema([Tool(name="free_form", description="JSON", parameters=parameters)])
    args = {"data": json.dumps(value)}
    Draft202012Validator(schema).validate(envelope([{"name": "free_form", "arguments": args}], metadata()))
    decoded = native_arguments(args, parameters)
    Draft202012Validator(parameters).validate(decoded)
    assert decoded == {"data": value}


@pytest.mark.asyncio
async def test_router_preference_survives_new_session_and_fixed_selection(tmp_path, monkeypatch):
    def settings():
        return SettingsManager(project_root=str(tmp_path), global_settings_file=str(tmp_path / "settings.json"))
    first, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    first._settings_manager = settings()
    await model_command(first, "/model router")
    assert settings().get_router_enabled()
    second, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
    second = (await create_agent_session(CreateAgentSessionOptions(
        cwd=str(tmp_path), model=second.model, auth_storage=second._auth_storage,
        model_registry=second._model_registry, settings_manager=settings(),
        session_manager=second._session_manager, resource_loader=Loader(), tools=["read", "edit"],
    ))).session
    assert second.router_enabled and _footer_model_parts(second) == ["router on"]
    await model_command(second, "/model router-fixture/light")
    third, _ = make_session(tmp_path, monkeypatch, "http://unused.invalid")
    third._settings_manager = settings()
    await third.restore_router()
    assert not third.router_enabled and not settings().get_router_enabled()
