"""
Bash execution tool — mirrors packages/coding-agent/src/core/tools/bash.ts
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_tail


@dataclass
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _inside_project_root(path: str, root: str) -> bool:
    resolved = os.path.realpath(path)
    root_real = os.path.realpath(root)
    return resolved == root_real or resolved.startswith(root_real + os.sep)


def _pathish_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    out: list[str] = []
    for token in tokens:
        if not token or token.startswith("-") or _URL_RE.match(token):
            continue
        if (
            token.startswith("/")
            or token.startswith("~")
            or token in {".", ".."}
            or token.startswith("./")
            or token.startswith("../")
            or "/" in token
        ):
            out.append(token)
    return out


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _is_shell_separator(token: str) -> bool:
    return token in {";", "&&", "||", "|", "&"}


def _is_option(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _find_search_path_tokens(tokens: list[str], find_index: int) -> list[str]:
    paths: list[str] = []
    for token in tokens[find_index + 1:]:
        if _is_shell_separator(token):
            break
        if token in {"(", ")", "!", "not"} or _is_option(token):
            if paths:
                break
            continue
        if paths and not (token.startswith("/") or token.startswith(".") or token.startswith("~")):
            break
        paths.append(token)
    return paths


def _assert_no_broad_filesystem_scan(command: str, cwd: str) -> None:
    if os.environ.get("PI_ALLOW_BROAD_SHELL_SCAN") == "1":
        return
    tokens = _command_tokens(command)
    for index, token in enumerate(tokens):
        if os.path.basename(token) != "find":
            continue
        for raw_path in _find_search_path_tokens(tokens, index):
            expanded = os.path.expanduser(raw_path)
            path = expanded if os.path.isabs(expanded) else os.path.join(cwd, expanded)
            if raw_path == "/" or (os.path.isabs(expanded) and not _inside_project_root(path, cwd)):
                raise RuntimeError(
                    "Command blocked: broad filesystem scan via shell find. "
                    "Use the scoped find tool, or run find inside the current "
                    "working directory. Set PI_ALLOW_BROAD_SHELL_SCAN=1 to override."
                )


def _assert_command_within_project_root(command: str, cwd: str) -> None:
    _assert_no_broad_filesystem_scan(command, cwd)
    root = os.environ.get("PI_PROJECT_ROOT")
    if not root:
        return
    if not _inside_project_root(cwd, root):
        raise RuntimeError(
            f"Command blocked: cwd is outside the allowed project root "
            f"({cwd}); project root is {root}"
        )
    for token in _pathish_tokens(command):
        expanded = os.path.expanduser(token)
        path = expanded if os.path.isabs(expanded) else os.path.join(cwd, expanded)
        if not _inside_project_root(path, root):
            raise RuntimeError(
                f"Command blocked: path escapes project root ({token}); "
                f"project root is {root}"
            )


def _sandbox_profile(root: str) -> str:
    root_real = os.path.realpath(root).replace("\\", "\\\\").replace('"', '\\"')
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*)\n"
        f'(allow file-write* (subpath "{root_real}") '
        '(literal "/dev/null") (literal "/dev/tty"))\n'
    )


def _sandbox_exec_args(shell: str, args: list[str], command: str) -> tuple[str, list[str], str | None]:
    root = os.environ.get("PI_PROJECT_ROOT")
    sandbox_exec = shutil.which("sandbox-exec")
    if sys.platform != "darwin" or not root or not sandbox_exec:
        return shell, [*args, command], None
    fd, profile_path = tempfile.mkstemp(prefix="pi-project-boundary-", suffix=".sb")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_sandbox_profile(root))
    return sandbox_exec, ["-f", profile_path, shell, *args, command], profile_path


def _get_shell() -> tuple[str, list[str]]:
    """Get the shell command and args for the current platform."""
    if sys.platform == "win32":
        return "cmd.exe", ["/c"]
    return os.environ.get("SHELL", "/bin/bash"), ["-c"]


def _output_limit(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1024, value)


def _kill_process_tree(pid: int) -> None:
    """Kill a process and its entire child tree — mirrors TS killProcessTree."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def create_bash_tool(cwd: str, command_prefix: str | None = None) -> AgentTool:
    """
    Create a bash execution tool.
    Mirrors createBashTool() in TypeScript.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
        on_update: Callable | None = None,
    ) -> AgentToolResult:
        command: str = params["command"]
        timeout: float | None = params.get("timeout")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation aborted")

        if not os.path.exists(cwd):
            raise RuntimeError(f"Working directory does not exist: {cwd}")
        _assert_command_within_project_root(command, cwd)

        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command

        shell, args = _get_shell()
        executable, exec_args, sandbox_profile_path = _sandbox_exec_args(
            shell, args, resolved_command
        )

        # Track output
        chunks: list[bytes] = []
        chunks_bytes = 0
        max_output_bytes = _output_limit("PI_BASH_MAX_OUTPUT_BYTES", DEFAULT_MAX_BYTES)
        max_output_lines = _output_limit("PI_BASH_MAX_OUTPUT_LINES", DEFAULT_MAX_LINES)
        max_chunks_bytes = max_output_bytes * 2
        total_bytes = 0
        temp_file_path: str | None = None
        temp_file = None

        async def run() -> int | None:
            nonlocal total_bytes, temp_file_path, temp_file, chunks_bytes

            # start_new_session=True creates a new process group so we can
            # kill the whole tree with killpg (mirrors TS detached: true)
            kwargs: dict[str, Any] = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True

            process = await asyncio.create_subprocess_exec(
                executable, *exec_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env={**os.environ},
                **kwargs,
            )

            timed_out = False

            async def read_output():
                nonlocal total_bytes, temp_file_path, temp_file, chunks_bytes
                if process.stdout is None:
                    return
                async for chunk in process.stdout:
                    total_bytes += len(chunk)

                    if total_bytes > max_output_bytes and temp_file_path is None:
                        fd, temp_file_path = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
                        temp_file = os.fdopen(fd, "wb")
                        for existing in chunks:
                            temp_file.write(existing)

                    if temp_file is not None:
                        temp_file.write(chunk)

                    chunks.append(chunk)
                    chunks_bytes += len(chunk)
                    while chunks_bytes > max_chunks_bytes and len(chunks) > 1:
                        removed = chunks.pop(0)
                        chunks_bytes -= len(removed)

                    if on_update:
                        full_buf = b"".join(chunks)
                        full_text = full_buf.decode("utf-8", errors="replace")
                        trunc = truncate_tail(
                            full_text,
                            max_lines=max_output_lines,
                            max_bytes=max_output_bytes,
                        )
                        on_update(AgentToolResult(
                            content=[TextContent(type="text", text=trunc.content or "")],
                            details=BashToolDetails(
                                truncation=trunc if trunc.truncated else None,
                                full_output_path=temp_file_path,
                            ),
                        ))

            read_task = asyncio.create_task(read_output())

            def _kill_proc():
                if process.pid is not None:
                    _kill_process_tree(process.pid)

            # Set up cancellation
            cancel_task = None
            if cancel_event:
                async def watch_cancel():
                    await cancel_event.wait()
                    _kill_proc()
                cancel_task = asyncio.create_task(watch_cancel())

            timeout_task = None
            if timeout is not None and timeout > 0:
                async def do_timeout():
                    nonlocal timed_out
                    await asyncio.sleep(timeout)
                    timed_out = True
                    _kill_proc()
                timeout_task = asyncio.create_task(do_timeout())

            await read_task
            exit_code = await process.wait()

            if cancel_task:
                cancel_task.cancel()
            if timeout_task:
                timeout_task.cancel()

            if temp_file is not None:
                temp_file.close()
                temp_file = None

            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Command aborted")

            if timed_out:
                raise RuntimeError(f"Command timed out after {timeout} seconds")

            return exit_code

        try:
            exit_code = await run()
        finally:
            if sandbox_profile_path:
                try:
                    os.unlink(sandbox_profile_path)
                except OSError:
                    pass

        # Combine all chunks for final output
        full_buf = b"".join(chunks)
        full_output = full_buf.decode("utf-8", errors="replace")

        truncation = truncate_tail(
            full_output,
            max_lines=max_output_lines,
            max_bytes=max_output_bytes,
        )
        output_text = truncation.content or "(no output)"
        details: BashToolDetails | None = None

        if truncation.truncated:
            details = BashToolDetails(truncation=truncation, full_output_path=temp_file_path)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(len((full_output.split("\n") or [""])[-1].encode("utf-8")))
                output_text += f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. Full output: {temp_file_path}]"
            elif truncation.truncated_by == "lines":
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {temp_file_path}]"
            else:
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} ({format_size(max_output_bytes)} limit). Full output: {temp_file_path}]"

        if exit_code is not None and exit_code != 0:
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise RuntimeError(output_text)

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details=details,
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. "
            f"Returns stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
            f"Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
            },
            "required": ["command"],
        },
        execute=execute,
    )


bash_tool = create_bash_tool(os.getcwd())
