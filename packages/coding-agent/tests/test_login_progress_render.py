"""Login instructions must reach terminal output before device auth completes."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from pi_coding_agent.modes.interactive.tui import _handle_login_command
from pi_tui.tui import TUI
from pi_tui.components.text import Text


def test_device_login_renders_while_pending():
    async def scenario():
        output = []
        terminal = SimpleNamespace(columns=120, rows=30, write=output.append,
                                   hide_cursor=lambda: None, show_cursor=lambda: None)
        tui = TUI(terminal)
        history = Text("")
        tui.add_child(history)
        pending = asyncio.Event()
        release = asyncio.Event()

        def append_history(message):
            history.set_text((history._text + "\n" + message).lstrip("\n"))
            history.invalidate()

        async def login(callbacks):
            callbacks.on_auth(SimpleNamespace(url="https://auth.openai.com/codex/device"))
            callbacks.on_progress("Visit https://auth.openai.com/codex/device and enter code: TEST-CODE")
            pending.set()
            await release.wait()
            raise RuntimeError("simulated login timeout")

        async def select(*args):
            return "subscription"

        async def user_input(*args):
            raise AssertionError("Device login should not prompt in the terminal")

        identity = lambda text: text
        with patch("webbrowser.open", return_value=True), patch(
            "pi_ai.utils.oauth.openai_codex.openai_codex_oauth_provider.login", new=login
        ):
            task = asyncio.create_task(_handle_login_command(
                "/login openai", SimpleNamespace(), append_history, lambda: None,
                tui, select, user_input, identity, identity, identity, identity,
            ))
            try:
                await asyncio.wait_for(pending.wait(), 2)
                await asyncio.sleep(0)  # Let the scheduled renderer run, not auth finish.
                assert not task.done()
                rendered = "".join(output)
                assert "Opened browser for subscription login." in rendered
                assert "TEST-CODE" in rendered
                assert "Login failed" not in rendered
                release.set()
                await task
                await asyncio.sleep(0)
                assert "simulated login timeout" in "".join(output)
            finally:
                release.set()
                await task
    asyncio.run(scenario())
