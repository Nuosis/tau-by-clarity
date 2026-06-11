"""
RPC mode: Headless operation with JSON stdin/stdout protocol.

Used for embedding the agent in other applications.
Receives commands as JSON on stdin, outputs events and responses as JSON on stdout.

Mirrors packages/coding-agent/src/modes/rpc/rpc-mode.ts
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any
from pi_coding_agent.core.source_info import create_synthetic_source_info, source_info_to_dict
from .jsonl import JsonlLineReader, serialize_json_line

from .types import (
    RpcCommand,
    RpcExtensionUIRequest,
    RpcExtensionUIRequestConfirm,
    RpcExtensionUIRequestEditor,
    RpcExtensionUIRequestInput,
    RpcExtensionUIRequestNotify,
    RpcExtensionUIRequestSelect,
    RpcExtensionUIRequestSetEditorText,
    RpcExtensionUIRequestSetStatus,
    RpcExtensionUIRequestSetTitle,
    RpcExtensionUIRequestSetWidget,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcResponseError,
    RpcResponseSuccess,
    RpcSessionState,
    RpcSlashCommand,
)

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.extensions.types import ExtensionUIContext


def _output(obj: dict[str, Any]) -> None:
    sys.stdout.write(serialize_json_line(obj))
    sys.stdout.flush()


def _success(cmd_id: str | None, command: str, data: Any = None) -> RpcResponseSuccess:
    return RpcResponseSuccess(id=cmd_id, command=command, data=data)


def _error(cmd_id: str | None, command: str, message: str) -> RpcResponseError:
    return RpcResponseError(id=cmd_id, command=command, error=message)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(exclude_none=True, mode="json"))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _rpc_model_dict(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    data = _jsonable(model)
    if not isinstance(data, dict):
        return None
    node_key_map = {
        "base_url": "baseUrl",
        "context_window": "contextWindow",
        "max_tokens": "maxTokens",
    }
    for python_key, node_key in node_key_map.items():
        if python_key in data and node_key not in data:
            data[node_key] = data.pop(python_key)
    return data


def _command_exception_response(command: Any, exc: Exception) -> RpcResponseError:
    cmd_id = command.get("id") if isinstance(command, dict) else None
    cmd_type = command.get("type") if isinstance(command, dict) else None
    return _error(cmd_id, cmd_type or "error", str(exc))


def _source_info_dict(resource: Any, fallback_path: str, fallback_source: str = "local") -> dict[str, Any]:
    source_info = getattr(resource, "source_info", None)
    if isinstance(resource, dict):
        source_info = resource.get("sourceInfo") or resource.get("source_info") or source_info
    if source_info is None:
        source_info = create_synthetic_source_info(
            fallback_path,
            source=fallback_source,
        )
    if isinstance(source_info, dict):
        data = dict(source_info)
        if "base_dir" in data and "baseDir" not in data:
            data["baseDir"] = data.pop("base_dir")
        return data
    return source_info_to_dict(source_info)


async def _handle_bash_command(session: Any, cmd_id: str | None, command: dict[str, Any]) -> RpcResponseSuccess:
    result = await session.execute_bash(
        command["command"],
        exclude_from_context=bool(command.get("excludeFromContext", False)),
    )
    return _success(cmd_id, "bash", result)


def _handle_prompt_command(
    session: Any,
    cmd_id: str | None,
    command: dict[str, Any],
    output: Any,
) -> None:
    """Start prompt handling and emit RPC response after prompt preflight resolves."""
    preflight_succeeded = False

    def preflight_result(success: bool) -> None:
        nonlocal preflight_succeeded
        if success and not preflight_succeeded:
            preflight_succeeded = True
            output(_success(cmd_id, "prompt"))

    async def run_prompt() -> None:
        try:
            await session.prompt(
                command["message"],
                images=command.get("images"),
                streaming_behavior=command.get("streamingBehavior"),
                source="rpc",
                preflight_result=preflight_result,
            )
        except Exception as exc:
            if not preflight_succeeded:
                output(_error(cmd_id, "prompt", str(exc)))

    asyncio.ensure_future(run_prompt())


def _looks_like_runtime_host(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("session", "new_session", "switch_session", "fork"))


def _runtime_session(runtime_or_session: Any) -> Any:
    return runtime_or_session.session if _looks_like_runtime_host(runtime_or_session) else runtime_or_session


async def _handle_new_session_command(runtime_or_session: Any, cmd_id: str | None, command: dict[str, Any]) -> RpcResponseSuccess:
    if _looks_like_runtime_host(runtime_or_session):
        options = {"parentSession": command["parentSession"]} if command.get("parentSession") else None
        result = await runtime_or_session.new_session(options)
        return _success(cmd_id, "new_session", result)

    session = runtime_or_session
    opts = {"parentSession": command["parentSession"]} if command.get("parentSession") else None
    cancelled = not await session.new_session(opts)
    return _success(cmd_id, "new_session", {"cancelled": cancelled})


async def _handle_switch_session_command(runtime_or_session: Any, cmd_id: str | None, command: dict[str, Any]) -> RpcResponseSuccess:
    if _looks_like_runtime_host(runtime_or_session):
        result = await runtime_or_session.switch_session(command["sessionPath"])
        return _success(cmd_id, "switch_session", result)

    cancelled = not await runtime_or_session.switch_session(command["sessionPath"])
    return _success(cmd_id, "switch_session", {"cancelled": cancelled})


async def _handle_fork_command(runtime_or_session: Any, cmd_id: str | None, command: dict[str, Any]) -> RpcResponseSuccess:
    if _looks_like_runtime_host(runtime_or_session):
        result = await runtime_or_session.fork(command["entryId"])
    else:
        fork_method = getattr(runtime_or_session, "fork_session", None)
        result = (
            await fork_method(command["entryId"])
            if callable(fork_method)
            else await runtime_or_session.fork(command["entryId"])
        )
    if not isinstance(result, dict):
        result = {"selectedText": "", "cancelled": False}
    return _success(
        cmd_id,
        "fork",
        {"text": result.get("selectedText", ""), "cancelled": result.get("cancelled", False)},
    )


async def _handle_clone_command(runtime_or_session: Any, cmd_id: str | None) -> RpcResponse:
    if _looks_like_runtime_host(runtime_or_session):
        session = runtime_or_session.session
        leaf_id = session.session_manager.get_leaf_id()
        if not leaf_id:
            return _error(cmd_id, "clone", "Cannot clone session: no current entry selected")
        result = await runtime_or_session.fork(leaf_id, {"position": "at"})
    else:
        clone_method = getattr(runtime_or_session, "clone_session", None)
        result = await clone_method() if callable(clone_method) else {"cancelled": True}
    if not isinstance(result, dict):
        result = {"cancelled": False}
    return _success(cmd_id, "clone", {"cancelled": bool(result.get("cancelled", False))})


async def _iter_jsonl_stdin_lines(reader: asyncio.StreamReader):
    """Yield strict LF-framed JSONL records from an asyncio reader."""
    lines: list[str] = []
    jsonl_reader = JsonlLineReader(lines.append)

    while True:
        chunk = await reader.read(4096)
        if not chunk:
            break
        jsonl_reader.feed(chunk)
        while lines:
            yield lines.pop(0)

    jsonl_reader.end()
    while lines:
        yield lines.pop(0)


def _create_extension_ui_context(
    pending_requests: dict[str, asyncio.Future[Any]],
    output_fn: Any,
) -> "ExtensionUIContext":
    """Create an ExtensionUIContext that uses the RPC protocol."""
    from pi_coding_agent.core.extensions.types import ExtensionUIContext

    def _opt(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _aborted(opts: Any) -> bool:
        signal = _opt(opts, "signal")
        if isinstance(signal, dict):
            return bool(signal.get("aborted", False))
        return bool(getattr(signal, "aborted", False))

    class RpcExtensionUIContextImpl(ExtensionUIContext):
        async def select(self, title: str, options: list[str], opts: Any = None) -> str | None:
            if opts and _aborted(opts):
                return None
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "select", "title": title, "options": options,
                "timeout": _opt(opts, "timeout"),
            })
            try:
                response = await asyncio.wait_for(future, timeout=_opt(opts, "timeout"))
                if isinstance(response, dict) and response.get("cancelled"):
                    return None
                return response.get("value")
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return None

        async def confirm(self, title: str, message: str, opts: Any = None) -> bool:
            if opts and _aborted(opts):
                return False
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "confirm", "title": title, "message": message,
                "timeout": _opt(opts, "timeout"),
            })
            try:
                response = await asyncio.wait_for(future, timeout=_opt(opts, "timeout"))
                if isinstance(response, dict) and response.get("cancelled"):
                    return False
                return bool(response.get("confirmed", False))
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return False

        async def input(self, title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
            if opts and _aborted(opts):
                return None
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({
                "type": "extension_ui_request", "id": req_id,
                "method": "input", "title": title, "placeholder": placeholder,
                "timeout": _opt(opts, "timeout"),
            })
            try:
                response = await asyncio.wait_for(future, timeout=_opt(opts, "timeout"))
                if isinstance(response, dict) and response.get("cancelled"):
                    return None
                return response.get("value")
            except asyncio.TimeoutError:
                pending_requests.pop(req_id, None)
                return None

        def notify(self, message: str, notify_type: str | None = None) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "notify", "message": message, "notifyType": notify_type,
            })

        def on_terminal_input(self, handler: Any) -> Any:
            return lambda: None

        def set_status(self, key: str, text: str | None) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setStatus", "statusKey": key, "statusText": text,
            })

        def set_working_message(self, message: str | None = None) -> None:
            pass

        def set_widget(self, key: str, content: Any, options: Any = None) -> None:
            if content is None or isinstance(content, list):
                output_fn({
                    "type": "extension_ui_request",
                    "id": str(uuid.uuid4()),
                    "method": "setWidget", "widgetKey": key,
                    "widgetLines": content,
                    "widgetPlacement": _opt(options, "placement"),
                })

        def set_footer(self, factory: Any) -> None:
            pass

        def set_header(self, factory: Any) -> None:
            pass

        def set_title(self, title: str) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setTitle", "title": title,
            })

        async def custom(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def paste_to_editor(self, text: str) -> None:
            self.set_editor_text(text)

        def set_editor_text(self, text: str) -> None:
            output_fn({
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "set_editor_text", "text": text,
            })

        def get_editor_text(self) -> str:
            return ""

        async def editor(self, title: str, prefill: str | None = None) -> str | None:
            req_id = str(uuid.uuid4())
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            pending_requests[req_id] = future
            output_fn({"type": "extension_ui_request", "id": req_id, "method": "editor", "title": title, "prefill": prefill})
            response = await future
            if isinstance(response, dict) and response.get("cancelled"):
                return None
            return response.get("value")

        def set_editor_component(self, *args: Any, **kwargs: Any) -> None:
            pass

        @property
        def theme(self) -> Any:
            return {}

        def get_all_themes(self) -> list[Any]:
            return []

        def get_theme(self, name: str) -> Any:
            return None

        def set_theme(self, theme_val: Any) -> dict[str, Any]:
            return {"success": False, "error": "Theme switching not supported in RPC mode"}

        def get_tools_expanded(self) -> bool:
            return False

        def set_tools_expanded(self, expanded: bool) -> None:
            pass

    return RpcExtensionUIContextImpl()


async def run_rpc_mode(session: "AgentSession") -> None:
    """
    Run in RPC mode.
    Listens for JSON commands on stdin, outputs events and responses on stdout.
    """
    pending_extension_requests: dict[str, asyncio.Future[Any]] = {}
    shutdown_requested = False
    runtime_host = session if _looks_like_runtime_host(session) else None
    active_session = _runtime_session(session)

    def output(obj: Any) -> None:
        if hasattr(obj, "model_dump"):
            _output(obj.model_dump(exclude_none=True))
        elif isinstance(obj, dict):
            _output(obj)
        else:
            _output(obj)

    ui_ctx = _create_extension_ui_context(pending_extension_requests, _output)

    await active_session.bind_extensions({
        "uiContext": ui_ctx,
        "commandContextActions": {
            "waitForIdle": lambda: active_session.agent.wait_for_idle(),
            "newSession": lambda opts=None: (
                runtime_host.new_session(opts) if runtime_host else active_session.new_session(opts)
            ),
            "fork": lambda entry_id: runtime_host.fork(entry_id) if runtime_host else active_session.fork(entry_id),
            "navigateTree": lambda target_id, opts=None: active_session.navigate_tree(target_id, opts),
            "switchSession": lambda path: (
                runtime_host.switch_session(path) if runtime_host else active_session.switch_session(path)
            ),
            "reload": lambda: active_session.reload(),
        },
        "shutdownHandler": lambda: None,  # shutdown_requested set below
        "onError": lambda err: output({"type": "extension_error", **err}),
    })

    # Forward all agent events as JSON
    active_session.subscribe(lambda event: output(event))

    async def handle_command(command: dict[str, Any]) -> RpcResponse | None:
        nonlocal active_session
        cmd_id = command.get("id")
        cmd_type = command.get("type", "")

        if cmd_type == "prompt":
            _handle_prompt_command(active_session, cmd_id, command, output)
            return None

        elif cmd_type == "steer":
            await active_session.steer(command["message"], command.get("images"))
            return _success(cmd_id, "steer")

        elif cmd_type == "follow_up":
            await active_session.follow_up(command["message"], command.get("images"))
            return _success(cmd_id, "follow_up")

        elif cmd_type == "abort":
            await active_session.abort()
            return _success(cmd_id, "abort")

        elif cmd_type == "new_session":
            response = await _handle_new_session_command(runtime_host or active_session, cmd_id, command)
            active_session = _runtime_session(runtime_host or active_session)
            return response

        elif cmd_type == "get_state":
            state = RpcSessionState(
                model=_rpc_model_dict(active_session.model),
                thinkingLevel=active_session.thinking_level,
                isStreaming=active_session.is_streaming,
                isCompacting=active_session.is_compacting,
                steeringMode=active_session.steering_mode,
                followUpMode=active_session.follow_up_mode,
                sessionFile=active_session.session_file,
                sessionId=active_session.session_id,
                sessionName=active_session.session_name,
                autoCompactionEnabled=active_session.auto_compaction_enabled,
                messageCount=len(active_session.messages),
                pendingMessageCount=active_session.pending_message_count,
            )
            return _success(cmd_id, "get_state", state.model_dump())

        elif cmd_type == "set_model":
            models = await active_session.model_registry.get_available()
            model = next(
                (m for m in models if m.get("provider") == command["provider"] and m.get("id") == command["modelId"]),
                None,
            )
            if not model:
                return _error(cmd_id, "set_model", f"Model not found: {command['provider']}/{command['modelId']}")
            await active_session.set_model(model)
            return _success(cmd_id, "set_model", model)

        elif cmd_type == "cycle_model":
            result = await active_session.cycle_model()
            return _success(cmd_id, "cycle_model", result)

        elif cmd_type == "get_available_models":
            models = await active_session.model_registry.get_available()
            return _success(cmd_id, "get_available_models", {"models": models})

        elif cmd_type == "set_thinking_level":
            active_session.set_thinking_level(command["level"])
            return _success(cmd_id, "set_thinking_level")

        elif cmd_type == "cycle_thinking_level":
            level = active_session.cycle_thinking_level()
            return _success(cmd_id, "cycle_thinking_level", {"level": level} if level else None)

        elif cmd_type == "set_steering_mode":
            active_session.set_steering_mode(command["mode"])
            return _success(cmd_id, "set_steering_mode")

        elif cmd_type == "set_follow_up_mode":
            active_session.set_follow_up_mode(command["mode"])
            return _success(cmd_id, "set_follow_up_mode")

        elif cmd_type == "compact":
            result = await active_session.compact(command.get("customInstructions"))
            return _success(cmd_id, "compact", result)

        elif cmd_type == "set_auto_compaction":
            active_session.set_auto_compaction_enabled(command["enabled"])
            return _success(cmd_id, "set_auto_compaction")

        elif cmd_type == "set_auto_retry":
            active_session.set_auto_retry_enabled(command["enabled"])
            return _success(cmd_id, "set_auto_retry")

        elif cmd_type == "abort_retry":
            active_session.abort_retry()
            return _success(cmd_id, "abort_retry")

        elif cmd_type == "bash":
            return await _handle_bash_command(active_session, cmd_id, command)

        elif cmd_type == "abort_bash":
            active_session.abort_bash()
            return _success(cmd_id, "abort_bash")

        elif cmd_type == "get_session_stats":
            stats = active_session.get_session_stats()
            return _success(cmd_id, "get_session_stats", stats)

        elif cmd_type == "export_html":
            path = await active_session.export_to_html(command.get("outputPath"))
            return _success(cmd_id, "export_html", {"path": path})

        elif cmd_type == "switch_session":
            response = await _handle_switch_session_command(runtime_host or active_session, cmd_id, command)
            active_session = _runtime_session(runtime_host or active_session)
            return response

        elif cmd_type == "fork":
            response = await _handle_fork_command(runtime_host or active_session, cmd_id, command)
            active_session = _runtime_session(runtime_host or active_session)
            return response

        elif cmd_type == "clone":
            response = await _handle_clone_command(runtime_host or active_session, cmd_id)
            active_session = _runtime_session(runtime_host or active_session)
            return response

        elif cmd_type == "get_fork_messages":
            messages = active_session.get_user_messages_for_forking()
            return _success(cmd_id, "get_fork_messages", {"messages": messages})

        elif cmd_type == "get_last_assistant_text":
            text = active_session.get_last_assistant_text()
            return _success(cmd_id, "get_last_assistant_text", {"text": text})

        elif cmd_type == "set_session_name":
            name = command.get("name", "").strip()
            if not name:
                return _error(cmd_id, "set_session_name", "Session name cannot be empty")
            active_session.set_session_name(name)
            return _success(cmd_id, "set_session_name")

        elif cmd_type == "get_messages":
            return _success(cmd_id, "get_messages", {"messages": active_session.messages})

        elif cmd_type == "get_commands":
            commands: list[RpcSlashCommand] = []
            runner = getattr(active_session, "extension_runner", None)
            if runner:
                for cmd_info in runner.get_registered_commands_with_paths():
                    path = cmd_info.get("extensionPath") or cmd_info.get("path") or "<extension>"
                    commands.append(RpcSlashCommand(
                        name=cmd_info["command"]["name"],
                        description=cmd_info["command"].get("description"),
                        source="extension",
                        sourceInfo=_source_info_dict(cmd_info, path),
                    ))
            for template in getattr(active_session, "prompt_templates", []):
                path = getattr(template, "file_path", None) or f"<prompt:{template.name}>"
                commands.append(RpcSlashCommand(
                    name=template.name,
                    description=getattr(template, "description", None),
                    source="prompt",
                    sourceInfo=_source_info_dict(template, path),
                ))
            resource_loader = getattr(active_session, "resource_loader", None)
            if resource_loader:
                skills_result = resource_loader.get_skills()
                if isinstance(skills_result, dict):
                    skills = skills_result.get("skills") or []
                else:
                        skills = getattr(skills_result, "skills", []) or []
                for skill in skills:
                    path = getattr(skill, "file_path", None) or f"<skill:{skill.name}>"
                    commands.append(RpcSlashCommand(
                        name=f"skill:{skill.name}",
                        description=getattr(skill, "description", None),
                        source="skill",
                        sourceInfo=_source_info_dict(skill, path),
                    ))
            return _success(cmd_id, "get_commands", {"commands": [c.model_dump(exclude_none=True) for c in commands]})

        else:
            return _error(None, cmd_type, f"Unknown command: {cmd_type}")

    # Read strict LF-framed JSONL records from stdin asynchronously
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async for line in _iter_jsonl_stdin_lines(reader):
        parsed: Any = None
        try:
            if not line:
                continue

            parsed = json.loads(line)

            # Handle extension UI responses
            if parsed.get("type") == "extension_ui_response":
                req_id = parsed.get("id")
                if req_id and req_id in pending_extension_requests:
                    fut = pending_extension_requests.pop(req_id)
                    if not fut.done():
                        fut.set_result(parsed)
                continue

            response = await handle_command(parsed)
            if response is not None:
                output(response)

        except json.JSONDecodeError as e:
            output(_error(None, "parse", f"Failed to parse command: {e}"))
        except Exception as e:  # noqa: BLE001
            output(_command_exception_response(parsed, e))
