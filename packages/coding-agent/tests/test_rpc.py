"""
Tests for modes/rpc/ subpackage.

Covers: types.py (pydantic models), client.py API surface
"""
from __future__ import annotations

import asyncio

import pytest


# ============================================================================
# RPC Types
# ============================================================================

class TestRpcTypes:
    def test_rpc_command_prompt_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandPrompt
        cmd = RpcCommandPrompt(type="prompt", message="Hello world")
        data = cmd.model_dump()
        assert data["type"] == "prompt"
        assert data["message"] == "Hello world"

    def test_rpc_command_steer_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandSteer
        cmd = RpcCommandSteer(type="steer", message="Override")
        data = cmd.model_dump()
        assert data["type"] == "steer"

    def test_rpc_command_abort_serializes(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandAbort
        cmd = RpcCommandAbort(type="abort")
        data = cmd.model_dump()
        assert data["type"] == "abort"

    def test_rpc_response_success(self):
        from pi_coding_agent.modes.rpc.types import RpcResponseSuccess
        r = RpcResponseSuccess(command="get_state", data={"thinkingLevel": "off"})
        assert r.success is True
        assert r.command == "get_state"

    def test_rpc_response_error(self):
        from pi_coding_agent.modes.rpc.types import RpcResponseError
        r = RpcResponseError(command="prompt", error="Something went wrong")
        assert r.success is False
        assert "Something went wrong" in r.error

    def test_rpc_session_state_fields(self):
        from pi_coding_agent.modes.rpc.types import RpcSessionState
        state = RpcSessionState(
            thinkingLevel="off",
            isStreaming=False,
            isCompacting=False,
            steeringMode="all",
            followUpMode="all",
            sessionId="abc-123",
            autoCompactionEnabled=True,
            messageCount=5,
            pendingMessageCount=0,
        )
        assert state.sessionId == "abc-123"
        assert state.messageCount == 5

    def test_rpc_slash_command(self):
        from pi_coding_agent.modes.rpc.types import RpcSlashCommand
        cmd = RpcSlashCommand(
            name="test",
            source="skill",
            description="A skill cmd",
            sourceInfo={
                "path": "/tmp/skill/SKILL.md",
                "source": "local",
                "scope": "user",
                "origin": "top-level",
                "baseDir": "/tmp/skill",
            },
        )
        assert cmd.name == "test"
        assert cmd.source == "skill"
        assert cmd.sourceInfo["baseDir"] == "/tmp/skill"

    def test_rpc_extension_ui_request_notify(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIRequestNotify
        req = RpcExtensionUIRequestNotify(id="abc", method="notify", message="Hello!")
        data = req.model_dump()
        assert data["type"] == "extension_ui_request"
        assert data["method"] == "notify"

    def test_rpc_extension_ui_request_select(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIRequestSelect
        req = RpcExtensionUIRequestSelect(id="x", method="select", title="Choose", options=["A", "B"])
        assert req.options == ["A", "B"]

    def test_rpc_extension_ui_response_value(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIResponseValue
        resp = RpcExtensionUIResponseValue(id="x", value="chosen")
        assert resp.value == "chosen"

    def test_rpc_extension_ui_response_cancelled(self):
        from pi_coding_agent.modes.rpc.types import RpcExtensionUIResponseCancelled
        resp = RpcExtensionUIResponseCancelled(id="x", cancelled=True)
        assert resp.cancelled is True

    @pytest.mark.asyncio
    async def test_rpc_extension_ui_context_dialogs_emit_and_resolve(self):
        from pi_coding_agent.modes.rpc.mode import _create_extension_ui_context

        pending = {}
        output = []
        ctx = _create_extension_ui_context(pending, output.append)

        task = asyncio.create_task(ctx.select("Choose", ["A", "B"], {"timeout": 1}))
        await asyncio.sleep(0)

        assert len(output) == 1
        request = output[0]
        assert request["type"] == "extension_ui_request"
        assert request["method"] == "select"
        assert request["title"] == "Choose"
        assert request["options"] == ["A", "B"]
        assert request["timeout"] == 1

        pending[request["id"]].set_result({"id": request["id"], "value": "B"})

        assert await task == "B"

    @pytest.mark.asyncio
    async def test_rpc_extension_ui_context_aborted_dialog_does_not_emit(self):
        from pi_coding_agent.modes.rpc.mode import _create_extension_ui_context

        pending = {}
        output = []
        ctx = _create_extension_ui_context(pending, output.append)

        result = await ctx.confirm("Confirm", "Proceed?", {"signal": {"aborted": True}})

        assert result is False
        assert pending == {}
        assert output == []

    def test_rpc_extension_ui_context_node_style_fire_and_forget_methods(self):
        from pi_coding_agent.modes.rpc.mode import _create_extension_ui_context

        pending = {}
        output = []
        ctx = _create_extension_ui_context(pending, output.append)

        ctx.notify("Heads up", "warning")
        ctx.setStatus("sync", "running")
        ctx.setWidget("panel", ["line 1"], {"placement": "belowEditor"})
        ctx.setTitle("Pi Session")
        ctx.pasteToEditor("draft text")

        assert [item["method"] for item in output] == [
            "notify",
            "setStatus",
            "setWidget",
            "setTitle",
            "set_editor_text",
        ]
        assert output[0]["notifyType"] == "warning"
        assert output[1]["statusKey"] == "sync"
        assert output[1]["statusText"] == "running"
        assert output[2]["widgetKey"] == "panel"
        assert output[2]["widgetLines"] == ["line 1"]
        assert output[2]["widgetPlacement"] == "belowEditor"
        assert output[3]["title"] == "Pi Session"
        assert output[4]["text"] == "draft text"

    def test_rpc_command_set_model(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandSetModel
        cmd = RpcCommandSetModel(type="set_model", provider="anthropic", modelId="claude-3-5-sonnet")
        assert cmd.provider == "anthropic"
        assert cmd.modelId == "claude-3-5-sonnet"

    def test_rpc_command_compact(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandCompact
        cmd = RpcCommandCompact(type="compact", customInstructions="Focus on recent changes")
        assert cmd.customInstructions == "Focus on recent changes"

    def test_rpc_command_bash_accepts_exclude_from_context(self):
        from pi_coding_agent.modes.rpc.types import RpcCommandBash

        cmd = RpcCommandBash(type="bash", command="pwd", excludeFromContext=True)

        assert cmd.excludeFromContext is True


@pytest.mark.asyncio
async def test_rpc_bash_handler_passes_exclude_from_context() -> None:
    from pi_coding_agent.modes.rpc.mode import _handle_bash_command

    calls: list[dict[str, object]] = []

    class Session:
        async def execute_bash(self, command, on_chunk=None, exclude_from_context=False):
            calls.append(
                {
                    "command": command,
                    "on_chunk": on_chunk,
                    "exclude_from_context": exclude_from_context,
                }
            )
            return {"output": "ok", "exit_code": 0, "cancelled": False, "truncated": False}

    response = await _handle_bash_command(
        Session(),
        "cmd-1",
        {"type": "bash", "command": "pwd", "excludeFromContext": True},
    )

    assert response.command == "bash"
    assert response.success is True
    assert response.data["output"] == "ok"
    assert calls == [{"command": "pwd", "on_chunk": None, "exclude_from_context": True}]


@pytest.mark.asyncio
async def test_rpc_prompt_command_emits_success_after_preflight() -> None:
    from pi_coding_agent.modes.rpc.mode import _handle_prompt_command

    outputs = []
    calls = []

    class Session:
        async def prompt(self, message, **kwargs):
            calls.append((message, kwargs))
            kwargs["preflight_result"](True)

    _handle_prompt_command(
        Session(),
        "prompt-1",
        {
            "type": "prompt",
            "message": "hello",
            "images": [{"type": "image", "data": "x"}],
            "streamingBehavior": "followUp",
        },
        outputs.append,
    )
    await asyncio.sleep(0)

    assert len(outputs) == 1
    assert outputs[0].command == "prompt"
    assert outputs[0].success is True
    assert calls[0][0] == "hello"
    assert calls[0][1]["images"] == [{"type": "image", "data": "x"}]
    assert calls[0][1]["streaming_behavior"] == "followUp"
    assert calls[0][1]["source"] == "rpc"
    assert callable(calls[0][1]["preflight_result"])


@pytest.mark.asyncio
async def test_rpc_prompt_command_emits_error_when_preflight_fails() -> None:
    from pi_coding_agent.modes.rpc.mode import _handle_prompt_command

    outputs = []

    class Session:
        async def prompt(self, message, **kwargs):
            kwargs["preflight_result"](False)
            raise RuntimeError("No API key found for anthropic.")

    _handle_prompt_command(
        Session(),
        "prompt-2",
        {"type": "prompt", "message": "hello"},
        outputs.append,
    )
    await asyncio.sleep(0)

    assert len(outputs) == 1
    assert outputs[0].command == "prompt"
    assert outputs[0].success is False
    assert "No API key found" in outputs[0].error


@pytest.mark.asyncio
async def test_rpc_session_replacement_helpers_use_runtime_host() -> None:
    from pi_coding_agent.modes.rpc.mode import (
        _handle_clone_command,
        _handle_fork_command,
        _handle_new_session_command,
        _handle_switch_session_command,
    )

    calls: list[tuple[str, object]] = []

    class SessionManager:
        def get_leaf_id(self):
            return "leaf-1"

    class Session:
        session_manager = SessionManager()

    class RuntimeHost:
        session = Session()

        async def new_session(self, options=None):
            calls.append(("new_session", options))
            return {"cancelled": False}

        async def switch_session(self, session_path):
            calls.append(("switch_session", session_path))
            return {"cancelled": False}

        async def fork(self, entry_id, options=None):
            calls.append(("fork", {"entry_id": entry_id, "options": options}))
            return {"cancelled": False, "selectedText": "selected"}

    runtime = RuntimeHost()

    new_response = await _handle_new_session_command(
        runtime,
        "n1",
        {"type": "new_session", "parentSession": "parent.jsonl"},
    )
    switch_response = await _handle_switch_session_command(
        runtime,
        "s1",
        {"type": "switch_session", "sessionPath": "/tmp/session.jsonl"},
    )
    fork_response = await _handle_fork_command(
        runtime,
        "f1",
        {"type": "fork", "entryId": "entry-1"},
    )
    clone_response = await _handle_clone_command(runtime, "c1")

    assert new_response.data == {"cancelled": False}
    assert switch_response.data == {"cancelled": False}
    assert fork_response.data == {"text": "selected", "cancelled": False}
    assert clone_response.data == {"cancelled": False}
    assert calls == [
        ("new_session", {"parentSession": "parent.jsonl"}),
        ("switch_session", "/tmp/session.jsonl"),
        ("fork", {"entry_id": "entry-1", "options": None}),
        ("fork", {"entry_id": "leaf-1", "options": {"position": "at"}}),
    ]


@pytest.mark.asyncio
async def test_rpc_clone_runtime_host_requires_current_leaf() -> None:
    from pi_coding_agent.modes.rpc.mode import _handle_clone_command

    class SessionManager:
        def get_leaf_id(self):
            return None

    class Session:
        session_manager = SessionManager()

    class RuntimeHost:
        session = Session()

        async def new_session(self, options=None):
            return {"cancelled": False}

        async def switch_session(self, session_path):
            return {"cancelled": False}

        async def fork(self, entry_id, options=None):
            return {"cancelled": False}

    response = await _handle_clone_command(RuntimeHost(), "clone-1")

    assert response.success is False
    assert response.command == "clone"
    assert "no current entry selected" in response.error


def test_rpc_state_model_serialization_accepts_pydantic_model_object() -> None:
    from pi_ai.types import Model
    from pi_coding_agent.modes.rpc.mode import _rpc_model_dict
    from pi_coding_agent.modes.rpc.types import RpcSessionState

    model = Model(
        id="gpt-5.4-nano",
        name="GPT 5.4 Nano",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        context_window=400_000,
        max_tokens=128_000,
    )

    state = RpcSessionState(
        model=_rpc_model_dict(model),
        thinkingLevel="minimal",
        isStreaming=False,
        isCompacting=False,
        steeringMode="all",
        followUpMode="all",
        sessionId="session-1",
        autoCompactionEnabled=True,
        messageCount=0,
        pendingMessageCount=0,
    )

    data = state.model_dump()
    assert data["model"]["id"] == "gpt-5.4-nano"
    assert data["model"]["provider"] == "openai"
    assert data["model"]["baseUrl"] == "https://api.openai.com/v1"
    assert data["model"]["contextWindow"] == 400_000
    assert data["model"]["maxTokens"] == 128_000
    assert "base_url" not in data["model"]
    assert "context_window" not in data["model"]
    assert "max_tokens" not in data["model"]


def test_rpc_command_exception_response_preserves_command_identity() -> None:
    from pi_coding_agent.modes.rpc.mode import _command_exception_response

    response = _command_exception_response(
        {"id": "state-1", "type": "get_state"},
        RuntimeError("boom"),
    )

    assert response.id == "state-1"
    assert response.command == "get_state"
    assert response.success is False
    assert response.error == "boom"


# ============================================================================
# RpcClient instantiation and API surface
# ============================================================================

class TestRpcClientAPI:
    def test_rpc_client_instantiation(self):
        from pi_coding_agent.modes.rpc.client import RpcClient, RpcClientOptions
        opts = RpcClientOptions(cwd="/tmp", provider="anthropic")
        client = RpcClient(opts)
        assert client is not None

    def test_rpc_client_options_accept_node_camel_case_cli_path(self):
        from pi_coding_agent.modes.rpc.client import RpcClientOptions

        opts = RpcClientOptions(cliPath="dist/cli.js")

        assert opts.cli_path == "dist/cli.js"

    def test_rpc_client_default_options(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        assert client is not None

    def test_rpc_client_on_event_returns_unsubscribe(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        events = []
        unsub = client.on_event(lambda e: events.append(e))
        assert callable(unsub)
        # Manually trigger the listener
        client._handle_line({"type": "agent_start"})
        assert len(events) == 1
        unsub()
        client._handle_line({"type": "agent_end"})
        assert len(events) == 1  # No new events after unsubscribe

    def test_rpc_client_not_started_raises(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        import asyncio
        client = RpcClient()
        with pytest.raises((RuntimeError, Exception)):
            asyncio.get_event_loop().run_until_complete(client.prompt("hello"))

    def test_rpc_client_get_stderr_empty(self):
        from pi_coding_agent.modes.rpc.client import RpcClient
        client = RpcClient()
        assert client.get_stderr() == ""
        assert client.getStderr() == ""

    @pytest.mark.asyncio
    async def test_rpc_client_node_camel_case_methods_send_matching_commands(self):
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

        async def call_and_respond(client, call, data=None):
            line_count = len(client._process.stdin.lines)  # type: ignore[attr-defined]
            task = asyncio.create_task(call())
            while len(client._process.stdin.lines) == line_count:  # type: ignore[attr-defined]
                await asyncio.sleep(0)
            request_id = next(iter(client._pending_requests))  # type: ignore[attr-defined]
            sent = json.loads(client._process.stdin.lines[-1].decode("utf-8"))  # type: ignore[attr-defined]
            client._handle_line(  # type: ignore[attr-defined]
                {
                    "type": "response",
                    "id": request_id,
                    "command": sent["type"],
                    "success": True,
                    "data": data if data is not None else {},
                }
            )
            return sent, await task

        client = RpcClient()
        client._process = Process()  # type: ignore[attr-defined]

        command_specs = [
            (lambda: client.followUp("next", [{"kind": "image"}]), {"type": "follow_up", "message": "next", "images": [{"kind": "image"}]}, None, None),
            (lambda: client.newSession("parent.jsonl"), {"type": "new_session", "parentSession": "parent.jsonl"}, {"cancelled": False}, {"cancelled": False}),
            (lambda: client.getState(), {"type": "get_state"}, {"sessionId": "s1"}, {"sessionId": "s1"}),
            (lambda: client.setModel("openai", "gpt-5.4-nano"), {"type": "set_model", "provider": "openai", "modelId": "gpt-5.4-nano"}, {"provider": "openai", "id": "gpt-5.4-nano"}, {"provider": "openai", "id": "gpt-5.4-nano"}),
            (lambda: client.cycleModel(), {"type": "cycle_model"}, {"model": {"provider": "openai", "id": "gpt-5.4-nano"}}, {"model": {"provider": "openai", "id": "gpt-5.4-nano"}}),
            (lambda: client.getAvailableModels(), {"type": "get_available_models"}, {"models": [{"provider": "openai", "id": "gpt-5.4-nano"}]}, [{"provider": "openai", "id": "gpt-5.4-nano"}]),
            (lambda: client.setThinkingLevel("medium"), {"type": "set_thinking_level", "level": "medium"}, None, None),
            (lambda: client.cycleThinkingLevel(), {"type": "cycle_thinking_level"}, {"level": "high"}, {"level": "high"}),
            (lambda: client.setSteeringMode("all"), {"type": "set_steering_mode", "mode": "all"}, None, None),
            (lambda: client.setFollowUpMode("one-at-a-time"), {"type": "set_follow_up_mode", "mode": "one-at-a-time"}, None, None),
            (lambda: client.compact("focus"), {"type": "compact", "customInstructions": "focus"}, {"summary": "ok"}, {"summary": "ok"}),
            (lambda: client.setAutoCompaction(True), {"type": "set_auto_compaction", "enabled": True}, None, None),
            (lambda: client.setAutoRetry(False), {"type": "set_auto_retry", "enabled": False}, None, None),
            (lambda: client.abortRetry(), {"type": "abort_retry"}, None, None),
            (lambda: client.bash("pwd"), {"type": "bash", "command": "pwd"}, {"stdout": "/tmp"}, {"stdout": "/tmp"}),
            (lambda: client.abortBash(), {"type": "abort_bash"}, None, None),
            (lambda: client.getSessionStats(), {"type": "get_session_stats"}, {"messages": 2}, {"messages": 2}),
            (lambda: client.exportHtml("/tmp/out.html"), {"type": "export_html", "outputPath": "/tmp/out.html"}, {"path": "/tmp/out.html"}, {"path": "/tmp/out.html"}),
            (lambda: client.switchSession("other.jsonl"), {"type": "switch_session", "sessionPath": "other.jsonl"}, {"cancelled": False}, {"cancelled": False}),
            (lambda: client.fork("entry-1"), {"type": "fork", "entryId": "entry-1"}, {"text": "hello", "cancelled": False}, {"text": "hello", "cancelled": False}),
            (lambda: client.clone(), {"type": "clone"}, {"cancelled": False}, {"cancelled": False}),
            (lambda: client.getForkMessages(), {"type": "get_fork_messages"}, {"messages": [{"entryId": "e1", "text": "msg"}]}, [{"entryId": "e1", "text": "msg"}]),
            (lambda: client.getLastAssistantText(), {"type": "get_last_assistant_text"}, {"text": "last"}, "last"),
            (lambda: client.setSessionName("Work"), {"type": "set_session_name", "name": "Work"}, None, None),
            (lambda: client.getMessages(), {"type": "get_messages"}, {"messages": [{"role": "assistant"}]}, [{"role": "assistant"}]),
            (lambda: client.getCommands(), {"type": "get_commands"}, {"commands": [{"name": "cmd"}]}, [{"name": "cmd"}]),
        ]

        for call, expected_payload, response_data, expected_result in command_specs:
            sent, result = await call_and_respond(client, call, response_data)
            sent.pop("id")
            assert sent == expected_payload
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_rpc_client_prompt_and_wait_starts_collecting_before_prompt(self):
        from pi_coding_agent.modes.rpc.client import RpcClient

        client = RpcClient()
        events = []

        async def collect_events(timeout=60.0):
            events.append("collect-start")
            await asyncio.sleep(0)
            events.append("collect-end")
            return [{"type": "agent_end"}]

        async def prompt(message, images=None):
            events.append("prompt")

        client.collect_events = collect_events  # type: ignore[method-assign]
        client.prompt = prompt  # type: ignore[method-assign]

        result = await client.prompt_and_wait("hello")

        assert result == [{"type": "agent_end"}]
        assert events == ["collect-start", "prompt", "collect-end"]

    @pytest.mark.asyncio
    async def test_rpc_client_prompt_and_wait_cancels_collection_when_prompt_fails(self):
        from pi_coding_agent.modes.rpc.client import RpcClient

        client = RpcClient()
        cancelled = []

        async def collect_events(timeout=60.0):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def prompt(message, images=None):
            raise RuntimeError("prompt failed")

        client.collect_events = collect_events  # type: ignore[method-assign]
        client.prompt = prompt  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="prompt failed"):
            await client.prompt_and_wait("hello")
        await asyncio.sleep(0)

        assert cancelled == [True]

    @pytest.mark.asyncio
    async def test_rpc_client_rejects_in_flight_request_when_child_process_exits(self, tmp_path):
        from pi_coding_agent.modes.rpc.client import RpcClient, RpcClientOptions

        child = tmp_path / "child.py"
        child.write_text(
            "\n".join(
                [
                    "import sys",
                    "import time",
                    "sys.stdin.readline()",
                    "time.sleep(0.05)",
                    "sys.exit(43)",
                ]
            ),
            encoding="utf-8",
        )
        client = RpcClient(RpcClientOptions(cli_path=str(child)))

        await client.start()
        try:
            with pytest.raises(RuntimeError, match=r"Agent process exited \(code=43 signal=None\)"):
                await client.getCommands()
        finally:
            await client.stop()
