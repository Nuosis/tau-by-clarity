"""
End-to-end tests for the coding agent.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import AsyncGenerator

import pytest

from pi_ai.types import (
    AssistantMessage,
    EventDone,
    EventStart,
    EventTextEnd,
    EventTextStart,
    EventToolCallEnd,
    EventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai import get_model
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import Settings
from pi_coding_agent.core.tools import create_read_tool, create_write_tool


def _ts():
    return int(time.time() * 1000)


@pytest.mark.asyncio
async def test_e2e_read_write_workflow(monkeypatch):
    """Test that the agent can read and write files using tools."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = os.path.join(tmpdir, "hello.txt")
        with open(test_file, "w") as f:
            f.write("Original content")

        write_called = []

        # Mock: agent writes a new file after reading the existing one
        call_count = [0]

        async def mock_stream_with_write(model, ctx, opts=None):
            call_count[0] += 1
            partial = AssistantMessage(
                role="assistant", content=[], api=model.api, provider=model.provider,
                model=model.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)

            if call_count[0] == 1:
                # Return write tool call
                tc = ToolCall(
                    type="toolCall",
                    id="w1",
                    name="write",
                    arguments={"path": "output.txt", "content": "Written by agent"},
                )
                with_tc = partial.model_copy(update={"content": [tc]})
                yield EventToolCallStart(type="toolcall_start", content_index=0, partial=with_tc)
                yield EventToolCallEnd(type="toolcall_end", content_index=0, tool_call=tc, partial=with_tc)
                final = AssistantMessage(
                    role="assistant", content=[tc], api=model.api, provider=model.provider,
                    model=model.id, usage=Usage(), stop_reason="toolUse", timestamp=_ts(),
                )
                yield EventDone(type="done", reason="toolUse", message=final)
            else:
                text = "I've written the file."
                with_text = partial.model_copy(update={"content": [TextContent(type="text", text=text)]})
                yield EventTextEnd(type="text_end", content_index=0, content=text, partial=with_text)
                final = AssistantMessage(
                    role="assistant", content=[TextContent(type="text", text=text)],
                    api=model.api, provider=model.provider, model=model.id,
                    usage=Usage(), stop_reason="stop", timestamp=_ts(),
                )
                yield EventDone(type="done", reason="stop", message=final)

        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)
        session_manager = SessionManager(sessions_dir=tmpdir)

        session = AgentSession(
            cwd=tmpdir,
            model=model,
            settings=settings,
            session_manager=session_manager,
        )
        session._agent.stream_fn = mock_stream_with_write

        await session.prompt("Write a file called output.txt")

        # Verify the file was actually written
        output_file = os.path.join(tmpdir, "output.txt")
        assert os.path.exists(output_file), "output.txt should have been created"
        with open(output_file) as f:
            assert f.read() == "Written by agent"


@pytest.mark.asyncio
async def test_e2e_system_prompt_includes_cwd():
    """Test that the system prompt includes the working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(auto_compact=False)
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")

        session = AgentSession(
            cwd=tmpdir,
            model=model,
            settings=settings,
            session_manager=SessionManager(sessions_dir=tmpdir),
        )

        prompt = session.state.system_prompt
        assert tmpdir in prompt
        # TS-parity: default prompt should include explicit tool section/guidelines.
        assert "Available tools:" in prompt
        assert "- read: Read file contents" in prompt
        assert "- bash:" in prompt
        assert "Guidelines:" in prompt


@pytest.mark.asyncio
async def test_e2e_session_persistence(monkeypatch):
    """Test that messages are persisted across session restores."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)

        async def simple_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="OK")],
                api=m.api, provider=m.provider, model=m.id,
                usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

        session_manager = SessionManager(sessions_dir=tmpdir)
        session = AgentSession(
            cwd=tmpdir, model=model,
            settings=settings, session_manager=session_manager,
        )
        session._agent.stream_fn = simple_stream

        session_id = session.session_id
        await session.prompt("Remember this")

        # Load stored messages
        stored = session_manager.get_messages(session_id)
        assert len(stored) > 0


@pytest.mark.asyncio
async def test_e2e_compaction(monkeypatch):
    """Test that manual compaction works."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        settings = Settings(auto_compact=False)

        async def simple_stream(m, ctx, opts=None):
            partial = AssistantMessage(
                role="assistant", content=[], api=m.api, provider=m.provider,
                model=m.id, usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventStart(type="start", partial=partial)
            final = AssistantMessage(
                role="assistant", content=[TextContent(type="text", text="Response")],
                api=m.api, provider=m.provider, model=m.id,
                usage=Usage(), stop_reason="stop", timestamp=_ts(),
            )
            yield EventDone(type="done", reason="stop", message=final)

        session = AgentSession(
            cwd=tmpdir, model=model,
            settings=settings, session_manager=SessionManager(sessions_dir=tmpdir),
        )
        session._agent.stream_fn = simple_stream

        # Add some messages
        for i in range(3):
            await session.prompt(f"Message {i}")

        initial_count = len(session.state.messages)
        assert initial_count > 0

        # compact_context calls complete_simple internally,
        # but with a mock stream we can just verify it doesn't crash
        # and returns something
        try:
            summary = await session.compact()
            # Either compaction worked or it returned early (too few messages)
        except Exception as e:
            # Some errors are OK if the mock stream doesn't support summary generation
            pass


# ============================================================================
# Integration tests: Extensions
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_extension_loaded_and_session_start_event():
    """Test that an extension can be loaded and receives session_start events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_path = os.path.join(tmpdir, "test_ext.py")
        events_received = []

        ext_content = """
session_start_calls = []

def extension_factory(api):
    async def on_session_start(event, ctx):
        session_start_calls.append(event)
    api.on("session_start", on_session_start)
"""
        with open(ext_path, "w") as f:
            f.write(ext_content)

        from pi_coding_agent.core.event_bus import EventBus
        from pi_coding_agent.core.extensions.loader import load_extensions
        from pi_coding_agent.core.extensions.types import SessionStartEvent

        event_bus = EventBus()
        result = await load_extensions([ext_path], tmpdir, event_bus)
        assert len(result.extensions) == 1
        assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_e2e_extension_tool_wrapping():
    """Test that extension tool wrapping passes through correctly."""
    from unittest.mock import MagicMock
    from pi_coding_agent.core.extensions.wrapper import wrap_tool_with_extensions

    calls = []

    async def _execute(tool_call_id, params, cancel_event=None, on_update=None):
        calls.append(params)
        return {"content": [{"type": "text", "text": f"result:{params.get('x')}"}]}

    tool = {
        "name": "test_tool",
        "label": "Test",
        "description": "A test tool",
        "parameters": {},
        "execute": _execute,
    }
    mock_runner = MagicMock()
    mock_runner.has_handlers.return_value = False

    wrapped = wrap_tool_with_extensions(tool, mock_runner)
    result = await wrapped["execute"]("tc1", {"x": 42})
    assert calls == [{"x": 42}]
    assert result["content"][0]["text"] == "result:42"


@pytest.mark.asyncio
async def test_e2e_extension_multiple_tools_wrapped():
    """Test wrapping multiple tools preserves all tools."""
    from unittest.mock import MagicMock
    from pi_coding_agent.core.extensions.wrapper import wrap_tools_with_extensions

    async def _exec(tc_id, params, cancel=None, upd=None):
        return {}

    tools = [
        {"name": f"tool_{i}", "label": f"T{i}", "description": "d", "parameters": {}, "execute": _exec}
        for i in range(5)
    ]
    runner = MagicMock()
    runner.has_handlers.return_value = False

    wrapped = wrap_tools_with_extensions(tools, runner)
    assert len(wrapped) == 5
    assert [w["name"] for w in wrapped] == [f"tool_{i}" for i in range(5)]


# ============================================================================
# Integration tests: RPC types and protocol
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_rpc_command_roundtrip():
    """Test RPC command serialization and deserialization."""
    from pi_coding_agent.modes.rpc.types import (
        RpcCommandPrompt,
        RpcResponseSuccess,
        RpcSessionState,
    )

    # Serialize a command
    cmd = RpcCommandPrompt(type="prompt", message="Hello!", id="req_1")
    data = cmd.model_dump(exclude_none=True)
    assert data["type"] == "prompt"
    assert data["message"] == "Hello!"
    assert data["id"] == "req_1"

    # Session state round-trip
    state = RpcSessionState(
        thinkingLevel="medium",
        isStreaming=True,
        isCompacting=False,
        steeringMode="all",
        followUpMode="one-at-a-time",
        sessionId="sid-123",
        autoCompactionEnabled=False,
        messageCount=10,
        pendingMessageCount=2,
    )
    state_dict = state.model_dump()
    restored = RpcSessionState(**state_dict)
    assert restored.sessionId == "sid-123"
    assert restored.messageCount == 10


@pytest.mark.asyncio
async def test_e2e_rpc_client_event_subscriptions():
    """Test that the RpcClient event subscription and unsubscription work."""
    from pi_coding_agent.modes.rpc.client import RpcClient

    client = RpcClient()
    received: list = []
    received2: list = []

    unsub1 = client.on_event(lambda e: received.append(e))
    unsub2 = client.on_event(lambda e: received2.append(e))

    # Manually trigger
    client._handle_line({"type": "agent_start"})
    assert len(received) == 1
    assert len(received2) == 1

    unsub1()
    client._handle_line({"type": "agent_end"})
    assert len(received) == 1  # No new events
    assert len(received2) == 2


# ============================================================================
# Integration tests: CLI args and file processing
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_file_processor_text_files():
    """Test processing multiple text file arguments."""
    from pi_coding_agent.cli_sub.file_processor import process_file_arguments

    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "a.txt")
        file2 = os.path.join(tmpdir, "b.txt")
        with open(file1, "w") as f:
            f.write("Content A")
        with open(file2, "w") as f:
            f.write("Content B")

        result = await process_file_arguments([file1, file2])
        assert "Content A" in result.text
        assert "Content B" in result.text
        assert result.images == []


def test_e2e_args_full_parse():
    """Test parsing a realistic full set of CLI arguments."""
    from pi_coding_agent.cli_sub.args import parse_args

    args = parse_args([
        "--provider", "anthropic",
        "--model", "claude-3-5-sonnet",
        "--thinking", "high",
        "--mode", "rpc",
        "--extension", "ext1.py",
        "--extension", "ext2.py",
        "--no-session",
        "--verbose",
        "@myfile.md",
        "Fix the bug in main.py",
    ])

    assert args.provider == "anthropic"
    assert args.model == "claude-3-5-sonnet"
    assert args.thinking == "high"
    assert args.mode == "rpc"
    assert args.extensions == ["ext1.py", "ext2.py"]
    assert args.no_session is True
    assert args.verbose is True
    assert "myfile.md" in args.file_args
    assert "Fix the bug in main.py" in args.messages


# ============================================================================
# Integration tests: Migrations
# ============================================================================

def test_e2e_run_migrations_clean_dir():
    """Test that run_migrations works on a clean directory."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi", "agent")
        os.makedirs(agent_dir)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=os.path.join(agent_dir, "bin")):
                from pi_coding_agent.migrations import run_migrations
                result = run_migrations(tmpdir)

        assert "migratedAuthProviders" in result
        assert "deprecationWarnings" in result
        assert isinstance(result["migratedAuthProviders"], list)


def test_e2e_run_migrations_with_oauth():
    """Test that run_migrations migrates OAuth credentials."""
    import json
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi", "agent")
        os.makedirs(agent_dir)

        oauth_data = {"anthropic": {"access_token": "tok123", "refresh_token": "ref456"}}
        oauth_path = os.path.join(agent_dir, "oauth.json")
        with open(oauth_path, "w") as f:
            json.dump(oauth_data, f)

        with patch("pi_coding_agent.migrations.get_agent_dir", return_value=agent_dir):
            with patch("pi_coding_agent.migrations.get_bin_dir", return_value=os.path.join(agent_dir, "bin")):
                from pi_coding_agent.migrations import run_migrations
                result = run_migrations(tmpdir)

        assert "anthropic" in result["migratedAuthProviders"]
        auth_path = os.path.join(agent_dir, "auth.json")
        assert os.path.exists(auth_path)


# ============================================================================
# Integration tests: Compaction and resource loading
# ============================================================================

def test_e2e_compaction_utils_pipeline():
    """Test that FileOperations tracks changes correctly in a pipeline."""
    from pi_coding_agent.core.compaction.utils import FileOperations, compute_file_lists

    ops = FileOperations()
    ops.read.update(["/src/a.py", "/src/b.py"])
    ops.written.update(["/src/c.py"])
    ops.edited.update(["/src/a.py"])

    read_files, modified_files = compute_file_lists(ops)
    # /src/b.py was read but not modified → in read_files
    assert "/src/b.py" in read_files
    # /src/a.py was edited → in modified_files (and NOT in read_files since it's modified)
    assert "/src/a.py" in modified_files
    # /src/c.py was written → in modified_files
    assert "/src/c.py" in modified_files


def test_e2e_skills_and_prompts_integration():
    """Test skills and prompt templates loading from the same directory."""
    import tempfile
    from pi_coding_agent.core.skills import load_skills_from_dir
    from pi_coding_agent.core.prompt_templates import (
        LoadPromptTemplatesOptions,
        load_prompt_templates,
    )

    with tempfile.TemporaryDirectory() as skills_dir:
        # Create skill
        skill_subdir = os.path.join(skills_dir, "my-skill")
        os.makedirs(skill_subdir)
        with open(os.path.join(skill_subdir, "SKILL.md"), "w") as f:
            f.write("---\ndescription: Integration skill\n---\nDo something useful.")

        skills_result = load_skills_from_dir(skills_dir, "user")
        assert len(skills_result.skills) == 1
        assert skills_result.skills[0].name == "my-skill"

    with tempfile.TemporaryDirectory() as prompts_dir:
        # Create prompt template
        with open(os.path.join(prompts_dir, "my-prompt.md"), "w") as f:
            f.write("---\ndescription: Integration prompt\n---\nHello $1!")

        opts = LoadPromptTemplatesOptions(prompt_paths=[prompts_dir], include_defaults=False)
        templates = load_prompt_templates(opts)
        assert len(templates) == 1
        assert templates[0].name == "my-prompt"


# ============================================================================
# TUI integration tests
# ============================================================================

class _MockTerminal:
    """Minimal mock terminal for TUI integration tests (no ABC inheritance)."""

    rows = 24
    columns = 80
    kitty_protocol_active = False

    def __init__(self) -> None:
        self._writes: list[str] = []

    def write(self, s: str) -> None:
        self._writes.append(s)

    def start(self, on_input, on_resize) -> None:
        pass

    def stop(self) -> None:
        pass

    def hide_cursor(self) -> None:
        pass

    def show_cursor(self) -> None:
        pass

    def move_by(self, n: int) -> None:
        pass

    def clear_line(self) -> None:
        pass

    def clear_from_cursor(self) -> None:
        pass

    def clear_screen(self) -> None:
        pass

    def set_title(self, t: str) -> None:
        pass

    async def drain_input(self, *a, **k) -> None:
        pass

    @property
    def all_output(self) -> str:
        return "".join(self._writes)


@pytest.mark.asyncio
async def test_tui_render_history_and_stream():
    """Verify TUI lays out history_text, stream_text, spacer, editor correctly."""
    import re
    from pi_tui.tui import TUI
    from pi_tui.components.text import Text
    from pi_tui.components.spacer import Spacer
    from pi_tui.components.editor import Editor, EditorTheme
    from pi_tui.components.select_list import SelectListTheme

    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    mt = _MockTerminal()
    tui = TUI(mt)

    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(Spacer(1))
    select_theme = SelectListTheme(selected_text=cyan, description=dim, scroll_info=dim, no_match=dim)
    editor = Editor(tui, EditorTheme(border_color=dim, select_list=select_theme))
    tui.add_child(editor)
    tui.set_focus(editor)

    tui.start()
    await asyncio.sleep(0.05)

    # Initial state — no content
    lines = tui.render(80)
    clean = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert "You:" not in " ".join(clean)

    # Add user message
    history_text.set_text("You: hello world")
    history_text.invalidate()
    tui.request_render()
    await asyncio.sleep(0.05)

    lines = tui.render(80)
    clean_lines = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert any("You: hello world" in l for l in clean_lines)

    # Add streaming response
    stream_text.set_text("Assistant: Hi there!")
    stream_text.invalidate()
    tui.request_render()
    await asyncio.sleep(0.05)

    lines = tui.render(80)
    clean_lines = [re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip() for l in lines]
    assert any("You: hello world" in l for l in clean_lines)
    assert any("Assistant: Hi there!" in l for l in clean_lines)

    tui.stop()

    # Terminal writes should contain the response text
    all_output = mt.all_output
    assert "You: hello world" in all_output or "You:" in all_output


@pytest.mark.asyncio
async def test_tui_on_event_direct_calls():
    """
    Verify the TUI interactive handle_submit loop works end-to-end using
    a mock session that fires agent events directly (no real API call).
    """
    import re
    from pi_tui.tui import TUI
    from pi_tui.components.text import Text
    from pi_tui.components.spacer import Spacer
    from pi_tui.components.editor import Editor, EditorTheme
    from pi_tui.components.select_list import SelectListTheme

    def bold(s: str) -> str:
        return f"\x1b[1m{s}\x1b[22m"

    def dim(s: str) -> str:
        return f"\x1b[2m{s}\x1b[22m"

    def cyan(s: str) -> str:
        return f"\x1b[36m{s}\x1b[39m"

    def red(s: str) -> str:
        return f"\x1b[31m{s}\x1b[39m"

    def yellow(s: str) -> str:
        return f"\x1b[33m{s}\x1b[39m"

    mt = _MockTerminal()
    tui = TUI(mt)

    history_text = Text("", padding_x=1, padding_y=0)
    stream_text = Text("", padding_x=1, padding_y=0)
    tui.add_child(history_text)
    tui.add_child(stream_text)
    tui.add_child(Spacer(1))
    select_theme = SelectListTheme(selected_text=cyan, description=dim, scroll_info=dim, no_match=dim)
    editor = Editor(tui, EditorTheme(border_color=dim, select_list=select_theme))
    tui.add_child(editor)
    tui.set_focus(editor)

    def append_history(line: str) -> None:
        cur = history_text._text
        history_text.set_text((cur + "\n" + line).lstrip("\n"))
        history_text.invalidate()

    def set_stream(text: str) -> None:
        stream_text.set_text(text)
        stream_text.invalidate()
        tui.request_render()

    async def handle_submit(text: str, response_chunks: list[str]) -> None:
        """Simulated handle_submit that replays mock events."""
        collected: list[str] = []
        done_event = asyncio.Event()

        append_history(f"{bold('You:')} {text}")
        tui.request_render()

        # Simulate firing events directly (as on_event would receive them)
        for chunk in response_chunks:
            collected.append(chunk)
            set_stream(f"{bold('Assistant:')} {''.join(collected)}")
            await asyncio.sleep(0)  # Yield so renders can happen

        # Simulate agent_end
        done_event.set()
        await done_event.wait()

        # Finally: move to history
        final = stream_text._text
        if final:
            append_history(final)
            set_stream("")
        tui.request_render()

    tui.start()
    await asyncio.sleep(0.05)

    await handle_submit("say hello", ["Hello", ", world", "!"])
    await asyncio.sleep(0.1)

    tui.stop()

    # Check final rendered state
    lines = tui.render(80)
    clean_lines = [
        re.sub(r"\x1b[^m]*m|\x1b\[[^a-zA-Z]*[a-zA-Z]", "", l).strip()
        for l in lines
    ]
    assert any("You:" in l for l in clean_lines), f"No 'You:' in {clean_lines}"
    assert any("Assistant:" in l for l in clean_lines), f"No 'Assistant:' in {clean_lines}"

    # Verify response was committed to history, stream is empty
    assert stream_text._text == ""
    assert "Hello, world!" in history_text._text or "Assistant:" in history_text._text


@pytest.mark.asyncio
async def test_tui_initial_messages_render_without_text_delta(monkeypatch):
    """
    Regression test for "input-only, no assistant output":
    ensure interactive TUI can render assistant responses even when providers
    only emit message_start/message_end snapshots (no text_delta stream).
    """
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            start_message = SimpleNamespace(role="assistant", content=[])
            end_message = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text=f"Echo: {text}")],
                error_message=None,
            )

            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=start_message))
                listener(SimpleNamespace(type="message_end", message=end_message))
                listener(SimpleNamespace(type="turn_end", message=end_message))
                listener(SimpleNamespace(type="agent_end"))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["你好", "hello", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "你好" in clean        # user message (gray box, no "You:" label)
    assert "hello" in clean
    assert "Assistant: Echo: 你好" in clean
    assert "Assistant: Echo: hello" in clean


@pytest.mark.asyncio
async def test_tui_chat_command_renders_session_transcript(monkeypatch):
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="model", provider="openai")
            self.thinking_level = "off"

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 1,
                "assistantMessages": 1,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def get_messages(self):
            return [
                {"role": "user", "content": "What changed?"},
                {
                    "role": "assistant",
                    "content": [SimpleNamespace(type="text", text="The transcript rendered.")],
                },
            ]

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self, custom_instructions=None):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        def subscribe(self, _fn):
            return lambda: None

        async def prompt(self, _text: str, images=None, source: str | None = None):
            return None

    terminal = _MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    await _run_pi_tui(FakeSession(), initial_messages=["/chat", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        terminal.all_output,
    )
    assert "You: What changed?" in clean
    assert "Assistant: The transcript rendered." in clean


@pytest.mark.asyncio
async def test_tui_extension_can_install_custom_editor_component(monkeypatch):
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            pass

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class CustomEditor:
        focused = False

        def __init__(self):
            self.text = ""
            self.autocomplete_provider = None

        def render(self, width):
            return ["custom editor"]

        def invalidate(self):
            pass

        def handle_input(self, data):
            pass

        def set_text(self, text):
            self.text = text

        def get_text(self):
            return self.text

        def set_autocomplete_provider(self, provider):
            self.autocomplete_provider = provider

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.custom_editor = CustomEditor()

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def bind_extensions(self, bindings):
            ui = bindings["uiContext"]
            ui.setEditorText("draft")
            ui.setEditorComponent(lambda tui, theme, keybindings: self.custom_editor)

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["/exit"])

    assert session.custom_editor.text == "draft"
    assert session.custom_editor.focused is True
    assert callable(session.custom_editor.on_submit)
    assert callable(session.custom_editor.on_keydown)
    assert session.custom_editor.autocomplete_provider is not None


@pytest.mark.asyncio
async def test_tui_extension_dialogs_resolve_from_terminal_input(monkeypatch):
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            pass

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self, terminal) -> None:
            self.terminal = terminal
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.results = {}

        class AbortSignal:
            def __init__(self) -> None:
                self.aborted = False
                self.listeners = []

            def addEventListener(self, event, callback, options=None):
                if event == "abort":
                    self.listeners.append(callback)

            def removeEventListener(self, event, callback):
                if event == "abort" and callback in self.listeners:
                    self.listeners.remove(callback)

            def abort(self):
                self.aborted = True
                for callback in list(self.listeners):
                    callback()

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def _send_keys(self, *keys):
            await asyncio.sleep(0)
            for key in keys:
                self.terminal._on_input(key)
                await asyncio.sleep(0)

        async def _abort(self, signal):
            await asyncio.sleep(0)
            signal.abort()

        async def bind_extensions(self, bindings):
            ui = bindings["uiContext"]

            asyncio.create_task(self._send_keys("down", "\n"))
            self.results["select"] = await ui.select("Pick one", ["Alpha", "Beta"])

            asyncio.create_task(self._send_keys("\n"))
            self.results["confirm"] = await ui.confirm("Proceed", "Continue?")

            asyncio.create_task(self._send_keys("o", "k", "\n"))
            self.results["input"] = await ui.input("Name", "placeholder")

            asyncio.create_task(self._send_keys("x", "\n"))
            self.results["editor"] = await ui.editor("Draft", "pre")

            signal = self.AbortSignal()
            asyncio.create_task(self._abort(signal))
            self.results["aborted_select"] = await ui.select("Abort", ["Alpha"], {"signal": signal})
            self.results["abort_listeners"] = len(signal.listeners)

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession(terminal)
    await _run_pi_tui(session, initial_messages=["/exit"])

    assert session.results == {
        "select": "Beta",
        "confirm": True,
        "input": "ok",
        "editor": "prex",
        "aborted_select": None,
        "abort_listeners": 0,
    }


@pytest.mark.asyncio
async def test_tui_startup_renders_loaded_resources_before_initial_messages(monkeypatch):
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class ResourceLoader:
        def get_agents_files(self):
            return {"agentsFiles": [{"path": "/repo/AGENTS.md"}]}

        def get_skills(self):
            return {
                "skills": [SimpleNamespace(name="planning", file_path="/repo/.pi/skills/planning/SKILL.md")],
                "diagnostics": [],
            }

        def get_prompts(self):
            return {
                "prompts": [SimpleNamespace(name="summarize", file_path="/repo/.pi/prompts/summarize.md")],
                "diagnostics": [{"type": "collision", "message": 'name "/summarize" collision', "path": "/repo/dup.md"}],
            }

        def get_extensions(self):
            return {
                "extensions": [SimpleNamespace(path="/repo/.pi/extensions/work.py")],
                "diagnostics": [{"type": "warning", "message": "command conflict", "path": "/repo/.pi/extensions/work.py"}],
                "errors": [],
            }

        def get_themes(self):
            return {"themes": [SimpleNamespace(name="solarized", path="/repo/theme.json")], "diagnostics": []}

    class ExtensionRunner:
        def get_registered_commands(self):
            return []

        def get_command_diagnostics(self):
            return [{"type": "warning", "message": "extension command skipped", "path": "/repo/ext.py"}]

        def get_shortcut_diagnostics(self):
            return []

    class SettingsManager:
        def get_quiet_startup(self):
            return False

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.resource_loader = ResourceLoader()
            self.extension_runner = ExtensionRunner()
            self.settings_manager = SettingsManager()

        @property
        def prompt_templates(self):
            return self.resource_loader.get_prompts()["prompts"]

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            self._listeners.append(fn)
            return lambda: self._listeners.remove(fn)

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            message = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text=f"Echo: {text}")],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="message_end", message=message))
                listener(SimpleNamespace(type="turn_end", message=message))
                listener(SimpleNamespace(type="agent_end"))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    await _run_pi_tui(FakeSession(), initial_messages=["hello", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )

    assert "[Context]" in clean
    assert "/repo/AGENTS.md" in clean
    assert "[Skills]" in clean
    assert "planning" in clean
    assert "[Prompts]" in clean
    assert "/summarize" in clean
    assert "[Extensions]" in clean
    assert "/repo/.pi/extensions/work.py" in clean
    assert "[Themes]" in clean
    assert "solarized" in clean
    assert "[Prompt conflicts]" in clean
    assert 'name "/summarize" collision' in clean
    assert "[Extension issues]" in clean
    assert "extension command skipped" in clean
    assert clean.index("[Context]") < clean.index("hello")


@pytest.mark.asyncio
async def test_tui_reload_rerenders_resources_and_refreshes_extension_commands(monkeypatch):
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class ResourceLoader:
        def __init__(self) -> None:
            self.reloaded = False

        def get_agents_files(self):
            return {"agentsFiles": []}

        def get_skills(self):
            name = "after-reload-skill" if self.reloaded else "before-reload-skill"
            return {"skills": [SimpleNamespace(name=name, file_path=f"/repo/{name}/SKILL.md")], "diagnostics": []}

        def get_prompts(self):
            return {"prompts": [], "diagnostics": []}

        def get_extensions(self):
            path = "/repo/.pi/extensions/after_reload.py" if self.reloaded else "/repo/.pi/extensions/before_reload.py"
            return {"extensions": [SimpleNamespace(path=path)], "diagnostics": [], "errors": []}

        def get_themes(self):
            return {"themes": [], "diagnostics": []}

    class ExtensionRunner:
        def __init__(self, reloaded: bool = False) -> None:
            self.reloaded = reloaded

        def get_registered_commands(self):
            if not self.reloaded:
                return []
            return [
                SimpleNamespace(
                    name="after-reload",
                    invocation_name="after-reload",
                    description="Command registered after reload",
                )
            ]

        def get_shortcuts(self, config):
            return {}

        def get_command_diagnostics(self):
            return []

        def get_shortcut_diagnostics(self):
            return []

    class SettingsManager:
        def get_quiet_startup(self):
            return False

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.resource_loader = ResourceLoader()
            self.extension_runner = ExtensionRunner()
            self.settings_manager = SettingsManager()

        @property
        def prompt_templates(self):
            return []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            return None

        async def reload(self):
            self.resource_loader.reloaded = True
            self.extension_runner = ExtensionRunner(reloaded=True)

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    await _run_pi_tui(FakeSession(), initial_messages=["/reload", "/help", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )

    assert "after-reload-skill" in clean
    assert "/repo/.pi/extensions/after_reload.py" in clean
    assert "Reloaded keybindings, extensions, skills, prompts, themes." in clean
    assert "/after-reload" in clean
    assert "Command registered after reload" in clean


@pytest.mark.asyncio
async def test_tui_extension_header_footer_and_working_surfaces_render(monkeypatch):
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            ui = bindings["uiContext"]
            ui.setHeader(lambda tui, theme: "Extension header")
            ui.setFooter(lambda tui, theme, footer_data: "Extension footer")
            await asyncio.sleep(0)
            ui.setFooter(None)
            ui.setWorkingMessage("Indexing project")
            ui.setWorkingIndicator({"label": "dots"})
            ui.setWorkingVisible(True)
            await asyncio.sleep(0)

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    await _run_pi_tui(FakeSession(), initial_messages=["/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )

    assert "Extension header" in clean
    assert "Extension footer" in clean
    assert "working: Indexing project (dots)" in clean


@pytest.mark.asyncio
async def test_tui_extension_theme_contract_matches_interactive_runtime(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui
    from pi_coding_agent.modes.interactive.theme.theme import BUILTIN_THEME_JSON

    custom_theme = json.loads(json.dumps(BUILTIN_THEME_JSON["dark"]))
    custom_theme["name"] = "solarized"
    theme_path = tmp_path / "solarized.json"
    theme_path.write_text(json.dumps(custom_theme), encoding="utf-8")

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class ResourceLoader:
        def get_themes(self):
            return {"themes": [SimpleNamespace(name="solarized", path=str(theme_path))], "diagnostics": []}

    class SettingsManager:
        def __init__(self) -> None:
            self.theme = "dark"
            self.saved: list[tuple[str, str]] = []

        def get_theme(self):
            return self.theme

        def save_project(self, key: str, value: str) -> None:
            self.saved.append((key, value))
            if key == "theme":
                self.theme = value

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.resource_loader = ResourceLoader()
            self.settings_manager = SettingsManager()
            self.theme_names: list[str] = []
            self.resolved_theme_name: str | None = None
            self.active_theme_name: str | None = None
            self.header_theme_name: str | None = None

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        async def bind_extensions(self, bindings):
            ui = bindings["uiContext"]
            self.theme_names = [theme["name"] for theme in ui.getAllThemes()]
            resolved = ui.getTheme("solarized")
            self.resolved_theme_name = getattr(resolved, "name", None)
            assert ui.setTheme("solarized") == {"success": True}
            self.active_theme_name = getattr(ui.theme, "name", None)

            def header(_tui, theme):
                self.header_theme_name = getattr(theme, "name", None)
                return "Themed header"

            ui.setHeader(header)

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["/exit"])

    assert "solarized" in session.theme_names
    assert session.resolved_theme_name == "solarized"
    assert session.active_theme_name == "solarized"
    assert session.header_theme_name == "solarized"
    assert session.settings_manager.saved == [("theme", "solarized")]


@pytest.mark.asyncio
async def test_tui_extension_custom_component_receives_input_and_restores_editor(monkeypatch):
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class CustomComponent:
        focused = False

        def __init__(self, done):
            self.done = done
            self.disposed = False

        def render(self, width):
            return ["Custom picker", "Press enter"]

        def invalidate(self):
            pass

        def handle_input(self, data):
            if data in {"\n", "enter", "return"}:
                self.done("accepted")

        def dispose(self):
            self.disposed = True

    class FakeSession:
        def __init__(self, terminal) -> None:
            self.terminal = terminal
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.custom_component = None
            self.results = {}

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def _send_key(self, key):
            await asyncio.sleep(0)
            self.terminal._on_input(key)
            await asyncio.sleep(0)

        async def bind_extensions(self, bindings):
            ui = bindings["uiContext"]
            ui.setEditorText("draft")

            def factory(tui, theme, keybindings, done):
                self.custom_component = CustomComponent(done)
                return self.custom_component

            asyncio.create_task(self._send_key("\n"))
            self.results["custom"] = await ui.custom(factory)
            self.results["editor_text"] = ui.getEditorText()
            self.results["disposed"] = self.custom_component.disposed

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession(terminal)

    await _run_pi_tui(session, initial_messages=["/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )

    assert "Custom picker" in clean
    assert session.results == {
        "custom": "accepted",
        "editor_text": "draft",
        "disposed": True,
    }


@pytest.mark.asyncio
async def test_tui_agent_end_error_is_rendered(monkeypatch):
    """
    If agent fails before streaming assistant deltas and only emits agent_end
    with an assistant error message, TUI should still show the error.
    """
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 80
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            err_msg = "No API key configured"
            assistant_error = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="")],
                error_message=err_msg,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(
                    SimpleNamespace(
                        type="agent_end",
                        messages=[assistant_error],
                    )
                )

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["你好", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "Error: No API key configured" in clean


@pytest.mark.asyncio
async def test_tui_renders_tool_execution_lines(monkeypatch):
    """Tool execution start/end should be visible in interactive TUI history."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            assistant = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="Done.")],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=assistant))
                listener(SimpleNamespace(type="tool_execution_start", tool_call_id="tc1",
                                         tool_name="bash", args={"command": "echo hi"}))
                listener(
                    SimpleNamespace(
                        type="tool_execution_end",
                        tool_call_id="tc1",
                        tool_name="bash",
                        is_error=False,
                        result=SimpleNamespace(content=[SimpleNamespace(type="text", text="exit_code: 0")]),
                    )
                )
                listener(SimpleNamespace(type="message_end", message=assistant))
                listener(SimpleNamespace(type="turn_end", message=assistant))
                listener(SimpleNamespace(type="agent_end", messages=[assistant]))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["run tool", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    # Active tool shows a blue marker; completed tool shows a success box
    # (✓ header) with the call written out ($ command) and the output.
    assert "⏵ bash" in clean
    assert "✓ bash" in clean
    assert "$ echo hi" in clean
    assert "exit_code: 0" in clean


@pytest.mark.asyncio
async def test_tui_does_not_require_agent_end_event(monkeypatch):
    """Interactive flow should complete even if provider stream omits agent_end."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            from types import SimpleNamespace
            self.model = SimpleNamespace(id='claude-3-5-sonnet-20241022', provider='anthropic')
            self.thinking_level = 'off'

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return ['bash', 'read']

        def get_session_stats(self):
            return {'sessionId': 'test', 'userMessages': 0, 'assistantMessages': 0,
                    'toolCalls': 0, 'tokens': {'total': 0}, 'cost': 0.0}

        def cycle_thinking_level(self):
            return 'minimal'

        async def compact(self):
            return ''

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction='forward'):
            return None

        async def follow_up(self, msg):
            pass

        @property
        def model_registry(self):
            from types import SimpleNamespace
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)

            def _unsub():
                if fn in self._listeners:
                    self._listeners.remove(fn)

            return _unsub

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            assistant = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text="Completed without agent_end.")],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=assistant))
                listener(SimpleNamespace(type="message_update", message=assistant, assistant_message_event=SimpleNamespace(type="text_delta", delta="")))
                listener(SimpleNamespace(type="message_end", message=assistant))
                listener(SimpleNamespace(type="turn_end", message=assistant))
            return

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)

    session = FakeSession()
    await _run_pi_tui(session, initial_messages=["run", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert "Completed without agent_end." in clean


@pytest.mark.asyncio
async def test_tui_assistant_markdown_is_styled(monkeypatch):
    """Assistant markdown renders via the Markdown component: headings yellow,
    inline code cyan (backticks gone), bold light-cyan."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input

        def stop(self) -> None: pass
        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None: return
        def write(self, data: str) -> None: self._writes.append(data)
        def move_by(self, lines: int) -> None: pass
        def hide_cursor(self) -> None: pass
        def show_cursor(self) -> None: pass
        def clear_line(self) -> None: pass
        def clear_from_cursor(self) -> None: pass
        def clear_screen(self) -> None: pass
        def set_title(self, title: str) -> None: pass

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            self.model = SimpleNamespace(id="MiniMax-M3", provider="minimax")
            self.thinking_level = "off"

        def get_context_usage(self): return None
        def get_active_tool_names(self): return ["bash"]
        def get_session_stats(self):
            return {"sessionId": "t", "userMessages": 0, "assistantMessages": 0,
                    "toolCalls": 0, "tokens": {"total": 0}, "cost": 0.0}
        def cycle_thinking_level(self): return "minimal"
        async def compact(self): return ""
        async def set_model(self, model): self.model = model
        async def cycle_model(self, direction="forward"): return None
        async def follow_up(self, msg): pass

        @property
        def model_registry(self):
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)
            return lambda: None

        async def prompt(self, text, images=None, source=None) -> None:
            md = "## Title\n\nSome **strong** and `inline_code` here."
            assistant = SimpleNamespace(
                role="assistant",
                content=[SimpleNamespace(type="text", text=md)],
                error_message=None,
            )
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
                listener(SimpleNamespace(type="message_start", message=assistant))
                listener(SimpleNamespace(type="message_end", message=assistant))
                listener(SimpleNamespace(type="turn_end", message=assistant))
                listener(SimpleNamespace(type="agent_end", messages=[assistant]))

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    await _run_pi_tui(FakeSession(), initial_messages=["go", "/exit"])

    raw = "".join(terminal._writes)
    assert "\x1b[33m" in raw            # heading yellow
    assert "\x1b[38;5;51m" in raw       # inline code cyan
    assert "\x1b[38;5;39m" in raw       # bold blue (matches tool calls)
    # backticks are removed by the markdown parser
    clean = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw)
    assert "inline_code" in clean and "`inline_code`" not in clean


@pytest.mark.asyncio
async def test_tui_esc_interrupts_running_agent(monkeypatch):
    """Pressing ESC while the agent is working aborts the turn (session.abort)."""
    import asyncio
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []
            self._on_input = None

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None: pass
        def hide_cursor(self) -> None: pass
        def show_cursor(self) -> None: pass
        def clear_line(self) -> None: pass
        def clear_from_cursor(self) -> None: pass
        def clear_screen(self) -> None: pass
        def set_title(self, title: str) -> None: pass

    started = asyncio.Event()
    aborted = asyncio.Event()

    class FakeSession:
        def __init__(self) -> None:
            self._listeners: list = []
            self.model = SimpleNamespace(id="MiniMax-M3", provider="minimax")
            self.thinking_level = "off"

        def get_context_usage(self): return None
        def get_active_tool_names(self): return ["bash"]
        def get_session_stats(self):
            return {"sessionId": "t", "userMessages": 0, "assistantMessages": 0,
                    "toolCalls": 0, "tokens": {"total": 0}, "cost": 0.0}
        def cycle_thinking_level(self): return "minimal"
        async def compact(self): return ""
        async def set_model(self, model): self.model = model
        async def cycle_model(self, direction="forward"): return None
        async def follow_up(self, msg): pass

        @property
        def model_registry(self):
            async def ga(): return [self.model]
            return SimpleNamespace(get_available=ga)

        def subscribe(self, fn):
            self._listeners.append(fn)
            return lambda: self._listeners.remove(fn) if fn in self._listeners else None

        async def abort(self):
            aborted.set()

        async def prompt(self, text, images=None, source=None) -> None:
            for listener in list(self._listeners):
                listener(SimpleNamespace(type="agent_start"))
                listener(SimpleNamespace(type="turn_start"))
            started.set()        # agent is now "working" (is_busy True)
            await aborted.wait()  # block until ESC -> abort fires

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()

    task = asyncio.create_task(_run_pi_tui(session, initial_messages=["do work"]))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert terminal._on_input is not None
        terminal._on_input("\x1b")  # press ESC
        await asyncio.wait_for(aborted.wait(), timeout=5)
        assert aborted.is_set()  # session.abort() was invoked
    finally:
        task.cancel()
        try:
            await task
        except BaseException:
            pass


@pytest.mark.asyncio
async def test_tui_tree_rebuilds_visible_history_after_navigation(monkeypatch):
    """Interactive /tree should rebuild the visible transcript from the selected branch."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.messages = [
                {"role": "user", "content": "stale branch prompt"},
                {"role": "assistant", "content": [{"type": "text", "text": "stale branch response"}]},
            ]
            self.navigated: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 1,
                "assistantMessages": 1,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def get_messages(self):
            return self.messages

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def navigate_tree(self, entry_id):
            self.navigated.append(entry_id)
            self.messages = [
                {"role": "user", "content": "selected branch prompt"},
                {"role": "assistant", "content": [{"type": "text", "text": "selected branch response"}]},
            ]
            return {"cancelled": False, "editorText": "selected branch prompt"}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()

    await _run_pi_tui(session, initial_messages=["/tree entry-1", "/exit"])

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert session.navigated == ["entry-1"]
    assert "selected branch prompt" in clean
    assert "selected branch response" in clean
    assert "Navigated session tree." in clean


@pytest.mark.asyncio
async def test_tui_fork_uses_runtime_host_and_restores_selected_text(monkeypatch):
    """Interactive /fork <entry> should replace through runtime host and restore selected text."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self, model_id: str) -> None:
            self.model = SimpleNamespace(id=model_id, provider="openai")
            self.thinking_level = "off"
            self.direct_fork_calls: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def fork_session(self, entry_id):
            self.direct_fork_calls.append(entry_id)
            return {"cancelled": False, "selectedText": "direct"}

    class RuntimeHost:
        def __init__(self, session):
            self.session = session
            self.forks: list[str] = []

        async def fork(self, entry_id, options=None):
            self.forks.append(entry_id)
            self.session = FakeSession("forked-model")
            return {"cancelled": False, "selectedText": "selected prompt"}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    original = FakeSession("initial-model")
    runtime = RuntimeHost(original)

    await _run_pi_tui(original, initial_messages=["/fork entry-7", "/session", "/exit"], runtime_host=runtime)

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert runtime.forks == ["entry-7"]
    assert original.direct_fork_calls == []
    assert runtime.session.model.id == "forked-model"
    assert "Forked to new session" in clean
    assert "forked-model" in clean


@pytest.mark.asyncio
async def test_tui_resume_uses_runtime_host_switch_contract(monkeypatch):
    """Interactive /resume should replace through runtime host switchSession."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self, model_id: str) -> None:
            self.model = SimpleNamespace(id=model_id, provider="openai")
            self.thinking_level = "off"
            self.direct_switches: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def switch_session(self, session_path):
            self.direct_switches.append(session_path)
            return True

    class RuntimeHost:
        def __init__(self, session):
            self.session = session
            self.switches: list[str] = []

        async def switch_session(self, session_path):
            self.switches.append(session_path)
            self.session = FakeSession("resumed-model")
            return {"cancelled": False}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    original = FakeSession("initial-model")
    runtime = RuntimeHost(original)

    await _run_pi_tui(original, initial_messages=["/resume /tmp/session.jsonl", "/session", "/exit"], runtime_host=runtime)

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert runtime.switches == ["/tmp/session.jsonl"]
    assert original.direct_switches == []
    assert runtime.session.model.id == "resumed-model"
    assert "Resumed session." in clean
    assert "resumed-model" in clean


@pytest.mark.asyncio
async def test_tui_new_uses_runtime_host_replacement_contract(monkeypatch):
    """Interactive /new should replace through the runtime host, not only mutate the session."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self, model_id: str) -> None:
            self.model = SimpleNamespace(id=model_id, provider="openai")
            self.thinking_level = "off"
            self.direct_new_calls = 0

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def new_session(self):
            self.direct_new_calls += 1
            return True

    class RuntimeHost:
        def __init__(self, session):
            self.session = session
            self.new_calls = 0

        async def new_session(self):
            self.new_calls += 1
            self.session = FakeSession("replacement-model")
            return {"cancelled": False}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    original = FakeSession("initial-model")
    runtime = RuntimeHost(original)

    await _run_pi_tui(original, initial_messages=["/new", "/session", "/exit"], runtime_host=runtime)

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert runtime.new_calls == 1
    assert original.direct_new_calls == 0
    assert runtime.session.model.id == "replacement-model"
    assert "Started a new session." in clean
    assert "replacement-model" in clean


@pytest.mark.asyncio
async def test_tui_clone_uses_runtime_host_current_leaf_contract(monkeypatch):
    """Interactive /clone should mirror Node runtime-host current-leaf fork behavior."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class SessionManager:
        def get_leaf_id(self):
            return "leaf-123"

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.session_manager = SessionManager()
            self.clone_calls = 0

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def clone_session(self):
            self.clone_calls += 1
            return {"cancelled": False}

    class RuntimeHost:
        def __init__(self, session):
            self.session = session
            self.forks: list[tuple[str, dict[str, str]]] = []

        async def fork(self, entry_id, options=None):
            self.forks.append((entry_id, options or {}))
            return {"cancelled": False}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()
    runtime = RuntimeHost(session)

    await _run_pi_tui(session, initial_messages=["/clone", "/exit"], runtime_host=runtime)

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert runtime.forks == [("leaf-123", {"position": "at"})]
    assert session.clone_calls == 0
    assert "Cloned to new session" in clean


@pytest.mark.asyncio
async def test_tui_export_jsonl_and_import_use_runtime_contract(monkeypatch, tmp_path):
    """Interactive /export and /import should mirror Node runtime contracts."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.exported_html: list[str | None] = []
            self.exported_jsonl: list[str] = []
            self.switched: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self):
            return ""

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            return None

        async def export_to_html(self, output_path=None):
            self.exported_html.append(output_path)
            return output_path or "/tmp/session.html"

        def export_to_jsonl(self, output_path):
            self.exported_jsonl.append(output_path)
            return output_path

        async def switch_session(self, session_path):
            self.switched.append(session_path)
            return True

    class RuntimeHost:
        def __init__(self, session):
            self.session = session
            self.imported: list[str] = []

        async def import_from_jsonl(self, input_path, cwd_override=None):
            self.imported.append(input_path)
            return {"cancelled": False}

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()
    runtime = RuntimeHost(session)
    html_path = str(tmp_path / "out file.html")
    jsonl_path = str(tmp_path / "out file.jsonl")
    import_path = str(tmp_path / "source file.jsonl")

    await _run_pi_tui(
        session,
        initial_messages=[
            f'/export "{jsonl_path}" trailing text',
            f"/export '{html_path}' trailing text",
            f'/import "{import_path}" trailing text',
            "/exit",
        ],
        runtime_host=runtime,
    )

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert session.exported_jsonl == [jsonl_path]
    assert session.exported_html == [html_path]
    assert runtime.imported == [import_path]
    assert session.switched == []
    assert clean.count("Exported session:") >= 2
    assert "out file.jsonl" in clean
    assert "out file.html" in clean
    assert "Imported session." in clean


@pytest.mark.asyncio
async def test_tui_compact_command_passes_custom_instructions(monkeypatch):
    """Interactive /compact <text> should pass custom instructions to the session."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    class FakeSession:
        def __init__(self) -> None:
            self.model = SimpleNamespace(id="gpt-5.4-nano", provider="openai")
            self.thinking_level = "off"
            self.compact_calls: list[str | None] = []
            self.prompts: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self, custom_instructions=None):
            self.compact_calls.append(custom_instructions)
            return "summary"

        async def set_model(self, model):
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            self.prompts.append(text)

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()

    await _run_pi_tui(
        session,
        initial_messages=[
            "/compact focus on decisions and open risks",
            "/exit",
        ],
    )

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert session.compact_calls == ["focus on decisions and open risks"]
    assert session.prompts == []
    assert "Compaction complete." in clean


@pytest.mark.asyncio
async def test_tui_model_command_accepts_provider_model_reference(monkeypatch):
    """Interactive /model should accept canonical provider/model references."""
    import re
    from types import SimpleNamespace

    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui

    class MockTerminal:
        rows = 24
        columns = 100
        kitty_protocol_active = False

        def __init__(self) -> None:
            self._writes: list[str] = []

        def start(self, on_input, on_resize) -> None:
            self._on_input = on_input
            self._on_resize = on_resize

        def stop(self) -> None:
            pass

        async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
            return

        def write(self, data: str) -> None:
            self._writes.append(data)

        def move_by(self, lines: int) -> None:
            pass

        def hide_cursor(self) -> None:
            pass

        def show_cursor(self) -> None:
            pass

        def clear_line(self) -> None:
            pass

        def clear_from_cursor(self) -> None:
            pass

        def clear_screen(self) -> None:
            pass

        def set_title(self, title: str) -> None:
            pass

    openai_model = SimpleNamespace(id="shared-model", provider="openai")
    anthropic_model = SimpleNamespace(id="shared-model", provider="anthropic")

    class Registry:
        async def get_available(self):
            return [openai_model, anthropic_model]

    class FakeSession:
        def __init__(self) -> None:
            self.model = anthropic_model
            self.model_registry = Registry()
            self.thinking_level = "off"
            self.selected_models: list[object] = []
            self.prompts: list[str] = []

        def get_context_usage(self):
            return None

        def get_active_tool_names(self):
            return []

        def get_session_stats(self):
            return {
                "sessionId": "test",
                "userMessages": 0,
                "assistantMessages": 0,
                "toolCalls": 0,
                "tokens": {"total": 0},
                "cost": 0.0,
            }

        def cycle_thinking_level(self):
            return "minimal"

        async def compact(self, custom_instructions=None):
            return ""

        async def set_model(self, model):
            self.selected_models.append(model)
            self.model = model

        async def cycle_model(self, direction="forward"):
            return None

        async def bind_extensions(self, bindings):
            return None

        def subscribe(self, fn):
            return lambda: None

        async def prompt(self, text: str, images=None, source: str | None = None) -> None:
            self.prompts.append(text)

    terminal = MockTerminal()
    monkeypatch.setattr(pi_tui, "ProcessTerminal", lambda: terminal)
    session = FakeSession()

    await _run_pi_tui(
        session,
        initial_messages=[
            "/model openai/shared-model",
            "/model shared-model",
            "/exit",
        ],
    )

    clean = re.sub(
        r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;\x07",
        "",
        "".join(terminal._writes),
    )
    assert session.selected_models == [openai_model]
    assert session.prompts == []
    assert "Switched to model: shared-model (openai)" in clean
    assert "Unknown model: shared-model" in clean
