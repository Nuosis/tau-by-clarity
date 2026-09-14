"""Drive the production TUI/session with a captured terminal transport.

Provider responses are controlled. The rendered placeholder and submitted
messages are inspected separately; this does not measure LLM judgment.
"""
import asyncio
import re

import pytest
from pi_ai.types import ToolCall
from .test_model_router import make_session, model_command, envelope, intention_fixture
from .test_turn_review import done


class CaptureTerminal:
    rows = 32
    columns = 100
    kitty_protocol_active = False
    def __init__(self): self.writes = []
    def start(self, on_input, on_resize): self.on_input = on_input
    def stop(self): pass
    async def drain_input(self, **kwargs): pass
    def write(self, data): self.writes.append(data)
    def move_by(self, lines): pass
    def hide_cursor(self): pass
    def show_cursor(self): pass
    def clear_line(self): pass
    def clear_from_cursor(self): pass
    def clear_screen(self): pass
    def set_title(self, title): pass


@pytest.mark.asyncio
async def test_production_ui_receives_intention_without_submitting_it(tmp_path, monkeypatch):
    import pi_tui
    from pi_coding_agent.modes.interactive.tui import _run_pi_tui
    session, _ = make_session(tmp_path, monkeypatch, 'http://unused.invalid')
    terminal = CaptureTerminal()
    monkeypatch.setattr(pi_tui, 'ProcessTerminal', lambda: terminal)
    requests = []
    async def provider(model, context, options):
        names = {t.name for t in context.tools}
        if 'submit_intention' in names:
            args = {**intention_fixture(), 'outcome': 'PLACEHOLDER_ONLY_SENTINEL'}
            name = 'submit_intention'
        elif 'submit_review' in names:
            name = 'submit_review'
            args = dict(decision='accept', intention_met=True, answer_sound=True,
                        rationale='Complete.', evidence_refs=['message:0'], follow_up_requirements=[])
        else:
            requests.append(str(context.messages))
            assert 'PLACEHOLDER_ONLY_SENTINEL' not in requests[-1]
            await asyncio.sleep(0.05)  # Allow the terminal render callback to run.
            name, args = 'submit_response', envelope([], None)
        yield done(model, [ToolCall(id=name, name=name, arguments=args)])
    session._provider_stream = provider
    await model_command(session, '/model router')
    await _run_pi_tui(session, initial_messages=['Answer this fixture request.', '/exit'])
    rendered = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', ''.join(terminal.writes))
    assert 'PLACEHOLDER_ONLY_SENTINEL' in rendered
    assert len(requests) == 1
    assert session.agent.state.error is None
    assert all('PLACEHOLDER_ONLY_SENTINEL' not in str(m) for m in session.agent.state.messages)
