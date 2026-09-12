"""Persisted /set router mappings and configuration-time resolution."""
import json
from types import SimpleNamespace

import pytest
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_registry import ModelRegistry
from pi_coding_agent.core.router_config import (
    ROUTER_LEVELS,
    RouterAssignment,
    load_router_selections,
    read_router_assignments,
    store_router_assignment,
)
from pi_coding_agent.modes.interactive.tui import _handle_set_command, _set_argument_completions


@pytest.fixture
def router_env(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"sentinel": {"preserve": True}, "providers": {
        "test-router": {"api": "openai-responses", "baseUrl": "http://unused.invalid",
                        "models": [{"id": "reasoner", "name": "Reasoner", "reasoning": True},
                                   {"id": "plain", "name": "Plain", "reasoning": False}]}
    }}))
    monkeypatch.setattr("pi_coding_agent.config.get_models_path", lambda: str(path))
    registry = ModelRegistry(auth_storage=AuthStorage.in_memory(), models_json_path=str(path))
    assert registry.find("test-router", "reasoner") is not None
    return path, registry


async def command(text, registry, show_input=None, show_select=None):
    history = []
    await _handle_set_command(
        text, SimpleNamespace(model_registry=registry), history.append, lambda: None,
        SimpleNamespace(request_render=lambda: None), show_select, show_input,
        str, str, str, str,
    )
    return history


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ROUTER_LEVELS)
async def test_set_router_persists_and_resolves_each_tier(router_env, level):
    path, registry = router_env
    before = json.loads(path.read_text())
    history = await command(f"/set router {level} test-router/reasoner low", registry)
    assert history == [f"Set router {level} to test-router/reasoner (thinking low)."]
    stored = json.loads(path.read_text())
    assert stored["providers"] == before["providers"] and stored["sentinel"] == before["sentinel"]
    # A fresh registry + loader must recover the assignment from disk.
    fresh = ModelRegistry(auth_storage=AuthStorage.in_memory(), models_json_path=str(path))
    selection = load_router_selections(str(path), fresh)[level]
    assert selection.model.id == "reasoner" and selection.reasoning == "low"
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_set_router_level_opens_editor_and_cancel_preserves_file(router_env):
    path, registry = router_env
    answers = iter(["test-router/reasoner", "high"])
    prompts = []

    async def show_input(title, default, options):
        prompts.append((title, default))
        return next(answers)

    await command("/set router light", registry, show_input=show_input)
    assert prompts == [("Provider/model", "openai/gpt-5.6-luna"),
                       ("Reasoning (off disables)", "high")]
    assert read_router_assignments(str(path))["light"].reasoning == "high"
    saved = path.read_bytes()

    async def cancel(*args):
        return None

    assert await command("/set router light", registry, show_input=cancel) == ["Set cancelled."]
    assert path.read_bytes() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "/set router unknown test-router/reasoner low",
    "/set router max unknown/not-registered medium",
    "/set router light test-router/plain high",
    "/set router light broken low",
])
async def test_invalid_router_configuration_does_not_write(router_env, text):
    path, registry = router_env
    before = path.read_bytes()
    history = await command(text, registry)
    assert history[0].startswith("Could not set router:")
    assert path.read_bytes() == before


def test_router_config_preserves_other_levels_and_rejects_malformed_file(router_env):
    path, registry = router_env
    for level in ["light", "max"]:
        store_router_assignment(str(path), level, RouterAssignment(
            provider="test-router", model="reasoner", reasoning="high"), registry)
    assert set(load_router_selections(str(path), registry)) == {"light", "max"}
    path.write_text('{broken')
    with pytest.raises(ValueError):
        store_router_assignment(str(path), "light", RouterAssignment(
            provider="test-router", model="reasoner", reasoning=None), registry)
    assert path.read_text() == '{broken'


def test_set_router_argument_completion():
    assert [item.value for item in _set_argument_completions("router ")] == [
        f"router {level}" for level in ROUTER_LEVELS]
    assert [item.value for item in _set_argument_completions("router ul")] == ["router ultra-light"]


@pytest.mark.asyncio
async def test_saved_router_assignment_reaches_agent_provider_boundary(router_env):
    from pi_agent import Agent, AgentOptions
    from pi_ai.types import AssistantMessage, EventDone, EventStart, TextContent, Usage

    path, registry = router_env
    await command("/set router ultra-light test-router/reasoner low", registry)
    selections = load_router_selections(str(path), registry)
    calls = []

    async def provider(model, context, opts):
        calls.append((model.provider, model.id, opts.reasoning))
        message = AssistantMessage(content=[TextContent(text="done")], api=model.api,
                                   provider=model.provider, model=model.id, usage=Usage(),
                                   stop_reason="stop", timestamp=0)
        yield EventStart(partial=message)
        yield EventDone(reason="stop", message=message)

    # No registry lookup occurs inside this selection callback.
    agent = Agent(AgentOptions(stream_fn=provider,
                               before_model_invocation=lambda invocation: selections["ultra-light"]))
    agent.set_model(registry.find("test-router", "plain"))
    await agent.prompt("hello")
    assert calls == [("test-router", "reasoner", "low")]
    assert agent.state.model.id == "plain"


@pytest.mark.asyncio
async def test_set_menu_offers_router_configuration(router_env):
    path, registry = router_env

    async def select(title, choices, options):
        wanted = "Router tier mapping" if title == "Provider" else "max"
        assert wanted in choices
        return wanted

    answers = iter(["test-router/reasoner", "medium"])

    async def enter(*args):
        return next(answers)

    await command("/set", registry, show_input=enter, show_select=select)
    assert load_router_selections(str(path), registry)["max"].reasoning == "medium"
