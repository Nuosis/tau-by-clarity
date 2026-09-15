"""
Tests for AgentSession — mirrors packages/coding-agent/test/ agent session tests.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai import get_model
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.extensions.types import Extension
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings
from pi_agent import AgentOptions


def _ts():
    return int(time.time() * 1000)


async def _mock_stream_fn(model, context, opts=None):
    partial = AssistantMessage(
        role="assistant", content=[], api=model.api, provider=model.provider,
        model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventStart(type="start", partial=partial)
    text = "I can help you with that!"
    with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=with_text)
    with_full = with_text.model_copy(update={"content": [TextContent(type="text", text=text)]})
    yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_full)
    final = AssistantMessage(
        role="assistant", content=[TextContent(type="text", text=text)],
        api=model.api, provider=model.provider, model=model.id,
        usage=Usage(), stop_reason="stop", timestamp=_ts(),
    )
    yield EventDone(type="done", reason="stop", message=final)


@pytest.fixture
def session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def agent_session(session_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    settings = Settings(auto_compact=False)
    # Use new factory API: create a per-session manager
    session_manager = SessionManager.create(cwd=session_dir, session_dir=session_dir)

    session = AgentSession(
        cwd=session_dir,
        model=model,
        settings=settings,
        session_manager=session_manager,
    )
    # Inject mock stream function
    session._agent.stream_fn = _mock_stream_fn
    return session


def test_agent_session_restores_supplied_session_context(session_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    manager = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    manager.append_message(
        {"role": "user", "content": "previous turn", "timestamp": _ts()}
    )
    reopened = SessionManager.open(manager.get_session_file())

    session = AgentSession(
        cwd=session_dir,
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
        settings=Settings(auto_compact=False),
        session_manager=reopened,
    )

    assert len(session.state.messages) == 1
    assert session.extension_runner.create_context().messages == session.state.messages


@pytest.mark.asyncio
async def test_agent_session_creates_session_id(agent_session):
    assert agent_session.session_id
    # Session IDs are 8-char hex strings (matching TypeScript generate_id)
    assert len(agent_session.session_id) >= 8


@pytest.mark.asyncio
async def test_agent_session_prompt(agent_session):
    await agent_session.prompt("Hello!")

    state = agent_session.state
    assert len(state.messages) >= 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_session_persists_messages(agent_session, session_dir):
    await agent_session.prompt("Hello!")

    # Check that messages were persisted via the session manager attached to the session
    messages = agent_session._session_manager.get_messages()
    assert len(messages) > 0


@pytest.mark.asyncio
async def test_extension_context_receives_live_agent_messages(agent_session):
    await agent_session.prompt("Hello!")

    context = agent_session.extension_runner.create_context()

    assert context.messages == agent_session.state.messages
    assert [message.role for message in context.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_agent_session_subscribe_events(agent_session):
    events = []
    unsub = agent_session.subscribe(events.append)
    await agent_session.prompt("Hello!")
    unsub()

    event_types = [e.type for e in events]
    assert "agent_start" in event_types
    assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_turn_end_extension_can_continue_same_agent_loop(agent_session):
    calls = 0

    async def two_response_stream(model, context, opts=None):
        nonlocal calls
        calls += 1
        text = "premature" if calls == 1 else "completed after guard"
        partial = AssistantMessage(
            role="assistant", content=[], api=model.api, provider=model.provider,
            model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
        yield EventTextStart(type="text_start", content_index=0, partial=with_text)
        final = partial.model_copy(update={
            "content": [TextContent(type="text", text=text)],
            "stop_reason": "stop",
        })
        yield EventTextEnd(type="text_end", content_index=0, content=text, partial=final)
        yield EventDone(type="done", reason="stop", message=final)

    continued = False
    handler_errors = []
    streaming_states = []
    queued_states = []

    async def on_turn_end(event, ctx):
        nonlocal continued
        if continued:
            return
        continued = True
        streaming_states.append(agent_session.is_streaming)
        try:
            await ctx.sendUserMessage(
                "[workflow guard] Required workflow step is missing; continue before answering.",
                {"deliverAs": "followUp"},
            )
            queued_states.append(agent_session._agent.has_queued_messages())
        except Exception as exc:
            handler_errors.append(exc)
            raise

    agent_session.extension_runner.extensions.append(Extension(
        path="turn-guard",
        resolved_path="turn-guard",
        handlers={"turn_end": [on_turn_end]},
    ))
    agent_session._agent.stream_fn = two_response_stream

    await agent_session.prompt("continue guarded workflow")

    assert continued
    assert not handler_errors
    assert streaming_states == [True]
    assert queued_states == [True]
    assert not agent_session._agent.has_queued_messages()
    assert calls == 2
    assistant_texts = [
        block.text
        for message in agent_session.state.messages
        if getattr(message, "role", "") == "assistant"
        for block in getattr(message, "content", [])
        if isinstance(block, TextContent)
    ]
    assert assistant_texts == ["premature", "completed after guard"]


@pytest.mark.asyncio
async def test_turn_end_hook_finalizes_only_the_finished_answer(agent_session, tmp_path):
    """A TurnEnd hook owns the user-facing answer, after ordinary work ends.

    This is a lifecycle contract, not evidence that a model follows the voice
    profile. The behavioral claim is covered by a live-model replay.
    """
    hook_dir = Path(agent_session.cwd) / ".tau"
    hook_dir.mkdir(exist_ok=True)
    turn_end_input = tmp_path / "turn-end-input.json"
    stop_input = tmp_path / "stop-input.json"
    finalizer = tmp_path / "finalizer.py"
    finalizer.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "payload = json.load(sys.stdin)",
                f"pathlib.Path({str(turn_end_input)!r}).write_text(json.dumps(payload))",
                "print(json.dumps({'hookSpecificOutput': {",
                "    'hookEventName': 'TurnEnd',",
                "    'additionalContext': 'FINAL VOICE RULES',",
                "    'replacementPrompt': 'Rewrite the draft for delivery.',",
                "}}))",
            ]
        ),
        encoding="utf-8",
    )
    auditor = tmp_path / "audit.py"
    auditor.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "payload = json.load(sys.stdin)",
                f"pathlib.Path({str(stop_input)!r}).write_text(json.dumps(payload))",
                "print(json.dumps({'suppressOutput': True}))",
            ]
        ),
        encoding="utf-8",
    )
    (hook_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "TurnEnd": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"{sys.executable} {finalizer}",
                                    "timeout": 5,
                                }
                            ]
                        }
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"{sys.executable} {auditor}",
                                    "timeout": 5,
                                }
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []
    extension_turns: list[str] = []

    async def observe_candidate(event, ctx):
        del ctx
        extension_turns.extend(
            block.text
            for block in event["message"].content
            if isinstance(block, TextContent)
        )

    agent_session.extension_runner.extensions.append(
        Extension(
            path="candidate-observer",
            resolved_path="candidate-observer",
            handlers={"turn_end": [observe_candidate]},
        )
    )

    async def two_response_stream(model, context, opts=None):
        calls.append(
            {
                "system": context.system_prompt,
                "tools": [tool.name for tool in context.tools],
                "last_role": context.messages[-1].role,
                "last_text": str(context.messages[-1].content),
            }
        )
        text = (
            "**Project**: `task-deadbeef`. Which option do you want?"
            if len(calls) == 1
            else "The project record is stale. I checked the code and found the gap."
        )
        partial = AssistantMessage(
            role="assistant", content=[], api=model.api, provider=model.provider,
            model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        yield EventStart(type="start", partial=partial)
        final = partial.model_copy(
            update={"content": [TextContent(type="text", text=text)]}
        )
        yield EventDone(type="done", reason="stop", message=final)

    agent_session._agent.stream_fn = two_response_stream
    await agent_session.prompt("Check the code, then tell me what changed.")

    assert len(calls) == 2
    assert "FINAL VOICE RULES" not in str(calls[0]["system"])
    assert "**Project**: `task-deadbeef`" in str(calls[1]["last_text"]), calls
    assert "FINAL VOICE RULES" in str(calls[1]["system"])
    assert calls[0]["tools"]
    assert calls[1]["tools"] == []
    assert calls[1]["last_role"] == "user"
    assert extension_turns == ["**Project**: `task-deadbeef`. Which option do you want?"]

    turn_payload = json.loads(turn_end_input.read_text(encoding="utf-8"))
    assert turn_payload["last_assistant_message"].startswith("**Project**")
    stop_payload = json.loads(stop_input.read_text(encoding="utf-8"))
    assert stop_payload["last_assistant_message"] == (
        "The project record is stale. I checked the code and found the gap."
    )
    assert agent_session._agent.state.system_prompt == agent_session._base_system_prompt
    assert agent_session._agent.state.tools


@pytest.mark.asyncio
async def test_native_before_final_output_hook_emits_only_the_probe_word_response(
    session_dir,
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hook_dir = Path(session_dir) / ".tau"
    hook_dir.mkdir()
    (hook_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "BeforeFinalOutput": [
                        {
                            "hooks": [
                                {
                                    "type": "builtin",
                                    "name": "final_output_probe",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    session = AgentSession(
        cwd=session_dir,
        model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
        settings=Settings(auto_compact=False),
        session_manager=SessionManager.create(
            cwd=session_dir,
            session_dir=session_dir,
        ),
    )
    session._agent.stream_fn = _mock_stream_fn
    events = []
    session.subscribe(events.append)

    await session.prompt("Prove the source hook fires.")

    assistant_events = [
        event
        for event in events
        if event.type in {"message_start", "message_update", "message_end"}
        and getattr(getattr(event, "message", None), "role", None) == "assistant"
    ]
    assistant_messages = [
        message
        for message in session._session_manager.get_messages()
        if message.get("role") == "assistant"
    ]
    assert [event.type for event in assistant_events] == [
        "message_start",
        "message_end",
    ]
    assert assistant_messages[-1]["content"][-1]["text"] == (
        "PRUEBA: el borrador interno fue reemplazado. TURN_END_HOOK_FIRED"
    )


@pytest.mark.asyncio
async def test_agent_session_set_model(agent_session):
    new_model = get_model("openai", "gpt-5.4-nano")
    # set_model is now async (validates API key); use _agent.set_model for unit tests
    agent_session._agent.set_model(new_model)
    assert agent_session.state.model.id == "gpt-5.4-nano"


@pytest.mark.asyncio
async def test_agent_session_set_thinking_level(agent_session):
    agent_session.set_thinking_level("high")
    assert agent_session.state.thinking_level == "high"

    # Verify it was persisted via the session manager
    entries = agent_session._session_manager.load_entries()
    level_entries = [e for e in entries if e.type == "thinking_level_change"]
    assert any(e.data.get("thinkingLevel") == "high" for e in level_entries)


@pytest.mark.asyncio
async def test_agent_session_allows_custom_thinking_level_for_reasoning_model(agent_session):
    agent_session._agent.set_model(agent_session.state.model.model_copy(update={"reasoning": True}))
    agent_session.set_thinking_level("adaptive")
    assert agent_session.state.thinking_level == "adaptive"


@pytest.mark.asyncio
async def test_agent_session_clamps_custom_thinking_level_for_non_reasoning_model(agent_session):
    agent_session._agent.set_model(agent_session.state.model.model_copy(update={"reasoning": False}))
    agent_session.set_thinking_level("adaptive")
    assert agent_session.state.thinking_level == "off"


@pytest.mark.asyncio
async def test_agent_session_fork(agent_session):
    await agent_session.prompt("Hello!")
    original_count = len(agent_session.state.messages)  # 2: user + assistant

    forked = await agent_session.fork()
    assert forked.session_id != agent_session.session_id
    # Fork branches from leaf's parent, so it drops the assistant leaf entry
    assert len(forked.state.messages) <= original_count


@pytest.mark.asyncio
async def test_agent_session_get_session_info(agent_session):
    info = agent_session.get_session_info()
    assert "session_id" in info
    assert "cwd" in info
    assert "model" in info
    assert "message_count" in info


@pytest.mark.asyncio
async def test_agent_session_abort(agent_session):
    # Abort should not raise if not streaming
    await agent_session.abort()
    assert not agent_session.state.is_streaming


def test_default_model_falls_back_when_pinned_model_has_no_auth(monkeypatch, session_dir):
    """
    If settings pin an unauthenticated model/provider (e.g. stale bedrock config),
    AgentSession should fall back to an authenticated default provider.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(
        auto_compact=False,
        provider="amazon-bedrock",
        model_id="amazon.nova-2-lite-v1:0",
    )
    session_manager = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    session = AgentSession(
        cwd=session_dir,
        settings=settings,
        session_manager=session_manager,
    )
    assert session.model is not None
    assert session.model.provider == "google"


# ── New tests for Group 2 features ────────────────────────────────────────────

class TestSessionPersistenceOnMessageEnd:
    """2a: Per-message persistence using message_end (not agent_end)."""

    @pytest.mark.asyncio
    async def test_messages_persisted_after_prompt(self, agent_session):
        await agent_session.prompt("Hello!")
        msgs = agent_session._session_manager.get_messages()
        # At minimum, user + assistant messages should be saved
        assert len(msgs) >= 1
        roles = [m.get("role") for m in msgs]
        assert "user" in roles or "assistant" in roles


class TestAutoRetryLogic:
    """2b: Auto-retry with exponential backoff."""

    def _make_error_msg(self, error_text: str):
        return AssistantMessage(
            role="assistant", content=[], api="anthropic", provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=Usage(), stop_reason="error",
            error_message=error_text,
            timestamp=_ts(),
        )

    def test_is_retryable_error_rate_limit(self, agent_session):
        msg = self._make_error_msg("rate limit exceeded, retry after 2 seconds")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_retryable_error_overloaded(self, agent_session):
        msg = self._make_error_msg("The API is currently overloaded")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_retryable_error_500(self, agent_session):
        msg = self._make_error_msg("500 internal server error")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_retryable_error_codex_opaque_backend_error(self, agent_session):
        msg = self._make_error_msg("Error Code None: None")
        assert agent_session._is_retryable_error(msg) is True

    def test_is_not_retryable_stop(self, agent_session):
        msg = AssistantMessage(
            role="assistant", content=[TextContent(type="text", text="ok")],
            api="anthropic", provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=Usage(), stop_reason="stop", timestamp=_ts(),
        )
        assert agent_session._is_retryable_error(msg) is False

    def test_is_not_retryable_overflow(self, agent_session):
        msg = self._make_error_msg("prompt is too long: 213462 tokens > 200000 maximum")
        # Overflow errors should not be retried (handled by compaction)
        assert agent_session._is_retryable_error(msg) is False

    @pytest.mark.asyncio
    async def test_retry_disabled_when_setting_off(self, agent_session, monkeypatch):
        """When retry is disabled, _handle_retryable_error returns False."""
        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": False, "maxRetries": 3, "baseDelayMs": 2000}
        )
        msg = self._make_error_msg("rate limit exceeded")
        agent_session._retry_attempt = 0
        result = await agent_session._handle_retryable_error(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_emits_auto_retry_start_event(self, agent_session, monkeypatch):
        """When retry fires, auto_retry_start event is emitted."""
        emitted = []
        agent_session.subscribe(lambda e: emitted.append(e))

        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": True, "maxRetries": 3, "baseDelayMs": 10}  # tiny delay
        )

        msg = self._make_error_msg("overloaded")
        agent_session._retry_attempt = 0

        # Patch asyncio.sleep to return immediately
        async def fast_sleep(_): pass
        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        result = await agent_session._handle_retryable_error(msg)
        assert result is True
        retry_events = [e for e in emitted if (isinstance(e, dict) and e.get("type") == "auto_retry_start")]
        assert len(retry_events) >= 1

    @pytest.mark.asyncio
    async def test_retry_stops_after_max_retries(self, agent_session, monkeypatch):
        """After maxRetries attempts, emits auto_retry_end with success=False."""
        emitted = []
        agent_session.subscribe(lambda e: emitted.append(e))

        monkeypatch.setattr(
            agent_session._settings_manager, "get_retry_settings",
            lambda: {"enabled": True, "maxRetries": 2, "baseDelayMs": 10}
        )
        msg = self._make_error_msg("overloaded")
        # Simulate already at maxRetries
        agent_session._retry_attempt = 2  # will be incremented to 3 > maxRetries=2
        agent_session._retry_event = asyncio.Event()

        result = await agent_session._handle_retryable_error(msg)
        assert result is False
        end_events = [e for e in emitted if isinstance(e, dict) and e.get("type") == "auto_retry_end"]
        assert any(not e.get("success", True) for e in end_events)


class TestToolManagement:
    """2d: set_active_tools_by_name and tool registry."""

    def test_get_all_tool_names(self, agent_session):
        names = agent_session.get_all_tool_names()
        assert "bash" in names
        assert "read" in names
        assert "write" in names
        assert "edit" in names

    def test_get_active_tool_names(self, agent_session):
        names = agent_session.get_active_tool_names()
        assert len(names) > 0

    def test_set_active_tools_by_name(self, agent_session):
        agent_session.set_active_tools_by_name(["bash", "read"])
        active = agent_session.get_active_tool_names()
        assert {"bash", "read"} <= set(active)
        # Goal tools are opt-in. Without an explicit `goal` alias or named
        # goal tool in the active set, they must not be force-included —
        # otherwise RPC/embedded sessions pay tool-list overhead even when
        # the developer never asked for goal machinery.
        assert not ({"get_goal", "set_goal", "update_goal", "clear_goal"} & set(active))
        assert "write" not in active
        assert "edit" not in active
        # System prompt should be rebuilt to reflect new tool set
        prompt = agent_session.system_prompt
        assert "bash" in prompt

    def test_set_active_tools_ignores_unknown(self, agent_session):
        agent_session.set_active_tools_by_name(["bash", "nonexistent_tool"])
        active = agent_session.get_active_tool_names()
        assert "bash" in active
        assert "nonexistent_tool" not in active


class TestContextUsageAndStats:
    """2e + 2f: get_context_usage and get_session_stats."""

    def test_get_context_usage_returns_none_with_no_model(self, session_dir):
        settings = Settings(auto_compact=False)
        sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
        sess = AgentSession(
            cwd=session_dir,
            model=get_model("anthropic", "claude-3-5-sonnet-20241022"),
            settings=settings,
            session_manager=sm,
        )
        # Force model to None
        sess._agent._state.model = None
        result = sess.get_context_usage()
        assert result is None

    def test_get_context_usage_returns_dict(self, agent_session):
        result = agent_session.get_context_usage()
        # May return None if no messages yet, or a dict with required keys
        if result is not None:
            assert "tokens" in result
            assert "contextWindow" in result
            assert "percent" in result

    def test_get_session_stats_structure(self, agent_session):
        stats = agent_session.get_session_stats()
        assert "sessionId" in stats
        assert "userMessages" in stats
        assert "assistantMessages" in stats
        assert "toolCalls" in stats
        assert "tokens" in stats
        assert isinstance(stats["tokens"], dict)
        assert "cost" in stats

    @pytest.mark.asyncio
    async def test_session_stats_message_counts(self, agent_session):
        await agent_session.prompt("Hello!")
        stats = agent_session.get_session_stats()
        assert stats["userMessages"] >= 1
        assert stats["assistantMessages"] >= 1


class TestModelCycling:
    """2g: cycle_model and set_model."""

    @pytest.mark.asyncio
    async def test_cycle_model_returns_none_if_single_model(self, agent_session, monkeypatch):
        """If only one model is available, cycle_model returns None."""
        current = agent_session.model

        async def single_available():
            return [current]

        monkeypatch.setattr(agent_session._model_registry, "get_available", single_available)
        result = await agent_session.cycle_model()
        assert result is None

    @pytest.mark.asyncio
    async def test_cycle_model_cycles_forward(self, agent_session, monkeypatch):
        """Cycling forward through multiple models."""
        from pi_ai import get_model as gm
        m1 = gm("anthropic", "claude-3-5-sonnet-20241022")
        m2 = gm("openai", "gpt-5.4-nano")

        async def fake_get_available():
            return [m1, m2]

        monkeypatch.setattr(agent_session._model_registry, "get_available", fake_get_available)
        # Set API key validation to always pass
        monkeypatch.setattr(
            agent_session._model_registry, "get_api_key",
            lambda p: "fake-key"
        )

        agent_session._agent.set_model(m1)
        result = await agent_session.cycle_model("forward")
        assert result is not None
        assert agent_session.model.id == m2.id

    @pytest.mark.asyncio
    async def test_set_model_raises_without_api_key(self, agent_session, monkeypatch):
        """set_model raises if no API key is configured for the provider."""
        monkeypatch.setattr(agent_session._model_registry, "get_api_key", lambda p: None)
        new_model = get_model("openai", "gpt-5.4-nano")
        with pytest.raises(RuntimeError, match="No API key"):
            await agent_session.set_model(new_model)


class TestThinkingLevelCycling:
    """2h: cycle_thinking_level."""

    def test_get_available_thinking_levels(self, agent_session):
        levels = agent_session.get_available_thinking_levels()
        assert "off" in levels
        assert isinstance(levels, list)

    def test_gpt_5_6_cycles_from_high_to_xhigh(self, agent_session):
        from pi_ai.types import Model, ModelCost

        agent_session._agent.set_model(Model(
            id="gpt-5.6-luna",
            name="GPT-5.6 Luna",
            api="openai-completions",
            provider="openrouter",
            reasoning=True,
            base_url="https://openrouter.ai/api/v1",
            cost=ModelCost(),
            context_window=1_050_000,
            max_tokens=128_000,
        ))

        agent_session._agent.set_thinking_level("high")

        assert agent_session.cycle_thinking_level() == "xhigh"
        assert agent_session.thinking_level == "xhigh"

    def test_claude_4_6_cycles_from_high_to_adaptive(self, agent_session):
        agent_session._agent.set_model(get_model("anthropic", "claude-sonnet-4-6"))
        agent_session._agent.set_thinking_level("high")

        assert agent_session.cycle_thinking_level() == "adaptive"
        assert agent_session.thinking_level == "adaptive"

    def test_cycle_thinking_level_advances(self, agent_session, monkeypatch):
        monkeypatch.setattr(
            agent_session, "get_available_thinking_levels",
            lambda: ["off", "minimal", "low", "medium", "high"]
        )
        agent_session._agent.set_thinking_level("off")
        new_level = agent_session.cycle_thinking_level()
        assert new_level == "minimal"

    def test_cycle_thinking_level_wraps_around(self, agent_session, monkeypatch):
        monkeypatch.setattr(
            agent_session, "get_available_thinking_levels",
            lambda: ["off", "minimal"]
        )
        agent_session._agent.set_thinking_level("minimal")
        new_level = agent_session.cycle_thinking_level()
        assert new_level == "off"


class TestQueueManagement:
    """2i + 2j: clear_queue, is_streaming, pending_message_count."""

    def test_is_streaming_false_when_idle(self, agent_session):
        assert agent_session.is_streaming is False

    def test_pending_message_count_zero_initially(self, agent_session):
        assert agent_session.pending_message_count == 0

    def test_clear_queue_returns_dict(self, agent_session):
        result = agent_session.clear_queue()
        assert "steering" in result
        assert "followUp" in result
        assert isinstance(result["steering"], list)
        assert isinstance(result["followUp"], list)

    def test_is_retrying_false_when_idle(self, agent_session):
        assert agent_session.is_retrying is False

    def test_is_compacting_false_when_idle(self, agent_session):
        assert agent_session.is_compacting is False

    def test_retry_attempt_starts_at_zero(self, agent_session):
        assert agent_session.retry_attempt == 0


class TestSessionProperties:
    """Extra session properties added in Group 2."""

    def test_thinking_level_property(self, agent_session):
        level = agent_session.thinking_level
        assert isinstance(level, str)

    def test_system_prompt_property(self, agent_session):
        prompt = agent_session.system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_last_assistant_text_none_when_no_messages(self, agent_session):
        result = agent_session.get_last_assistant_text()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_assistant_text_after_prompt(self, agent_session):
        await agent_session.prompt("Hello!")
        result = agent_session.get_last_assistant_text()
        # Should return the text of the last assistant message
        assert result is not None
        assert len(result) > 0


@pytest.mark.asyncio
async def test_prepare_next_turn_flushes_native_trace_when_durable_mode_enabled(
    monkeypatch,
):
    flushed: list[float] = []

    async def fake_flush(*, timeout_seconds: float) -> None:
        flushed.append(timeout_seconds)

    class NoExtensions:
        @staticmethod
        def has_handlers(_name: str) -> bool:
            return False

    session = object.__new__(AgentSession)
    session._extension_runner = NoExtensions()
    monkeypatch.setenv("TAU_INSTRUMENTATION_FLUSH_ON_TURN", "1")
    monkeypatch.setenv("TAU_INSTRUMENTATION_FLUSH_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setattr(
        "pi_coding_agent.core.agent_session._instr_flush",
        fake_flush,
    )

    await session._prepare_next_turn({})

    assert flushed == [7.5]


@pytest.mark.asyncio
async def test_prepare_next_turn_does_not_flush_native_trace_by_default(
    monkeypatch,
):
    flushed = False

    async def fake_flush(*, timeout_seconds: float) -> None:
        nonlocal flushed
        flushed = True

    class NoExtensions:
        @staticmethod
        def has_handlers(_name: str) -> bool:
            return False

    session = object.__new__(AgentSession)
    session._extension_runner = NoExtensions()
    monkeypatch.delenv("TAU_INSTRUMENTATION_FLUSH_ON_TURN", raising=False)
    monkeypatch.setattr(
        "pi_coding_agent.core.agent_session._instr_flush",
        fake_flush,
    )

    await session._prepare_next_turn({})

    assert flushed is False
