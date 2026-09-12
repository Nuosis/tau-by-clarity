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
