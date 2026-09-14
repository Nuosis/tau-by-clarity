import asyncio
import os

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "interactive"])
async def test_cancel_reaps_child_and_drains_pipes(tmp_path, kind):
    from pi_coding_agent.core.tools.bash import create_bash_tool
    from pi_coding_agent.core.bash_executor import execute_bash
    marker = tmp_path / "pid"
    command = f"echo $$ > {marker}; trap '' TERM; exec sleep 30"
    if kind == "tool":
        task = asyncio.create_task(create_bash_tool(str(tmp_path)).execute("test", {"command": command}))
    else:
        task = asyncio.create_task(execute_bash(command, cwd=str(tmp_path)))
    try:
        async with asyncio.timeout(3):
            while not marker.exists() or not marker.read_text().strip():
                await asyncio.sleep(.01)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_bash_executor_does_not_wait_for_terminal_input(tmp_path):
    from pi_coding_agent.core.bash_executor import execute_bash

    result = await execute_bash(
        "read -r value; "
        "if [ -z \"$value\" ]; then echo stdin-eof; "
        "else echo \"stdin=$value\"; fi",
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0
    assert result.cancelled is False
    assert "stdin-eof" in result.output
