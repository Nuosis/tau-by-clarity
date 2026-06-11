"""
Extension runner — mirrors packages/coding-agent/src/core/extensions/runner.ts

Manages loaded extensions and dispatches events to their handlers.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
from typing import Any

from pi_coding_agent.core.source_info import create_synthetic_source_info, source_info_to_dict

from .types import (
    Extension,
    ExtensionContext,
    RegisteredCommand,
    ExtensionShortcut,
    RegisteredTool,
)

RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS = {
    "app.interrupt",
    "app.clear",
    "app.exit",
    "app.suspend",
    "app.thinking.cycle",
    "app.model.cycleForward",
    "app.model.cycleBackward",
    "app.model.select",
    "app.tools.expand",
    "app.thinking.toggle",
    "app.editor.external",
    "app.message.followUp",
    "tui.input.submit",
    "tui.select.confirm",
    "tui.select.cancel",
    "tui.input.copy",
    "tui.editor.deleteToLineEnd",
}


class ExtensionRunner:
    """
    Runs extensions and dispatches events.
    Mirrors ExtensionRunner in TypeScript.
    """

    def __init__(
        self,
        extensions: list[Extension],
        runtime: dict[str, Any] | None = None,
        cwd: str = "",
        session_id: str = "",
    ) -> None:
        self._extensions = extensions
        self._runtime = runtime if runtime is not None else {"flagValues": {}}
        self._runtime.setdefault("flagValues", {})
        self._context_actions: dict[str, Any] = {}
        self._context_values: dict[str, Any] = {}
        self._command_context_actions: dict[str, Any] = {}
        self._shortcut_diagnostics: list[dict[str, Any]] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._error_listeners: list[Any] = []
        self._ui_context: Any = None
        self._mode = "print"
        self._stale_message: str | None = None
        self._cwd = cwd
        self._session_id = session_id

    @property
    def extensions(self) -> list[Extension]:
        return self._extensions

    def bind_core(
        self,
        actions: dict[str, Any] | None = None,
        context_actions: dict[str, Any] | None = None,
        provider_actions: dict[str, Any] | None = None,
    ) -> None:
        del provider_actions
        runtime = self._runtime
        for key, value in dict(actions or {}).items():
            runtime[key] = value
        self.bind_context_actions(context_actions or {})

    def bind_command_context(self, actions: dict[str, Any] | None = None) -> None:
        self.bind_command_context_actions(actions)

    def set_ui_context(self, ui_context: Any = None, mode: str = "print") -> None:
        self._ui_context = ui_context
        self._mode = mode

    def get_ui_context(self) -> Any:
        return self._ui_context

    def has_ui(self) -> bool:
        return self._ui_context is not None

    def get_extension_paths(self) -> list[str]:
        return [getattr(ext, "path", "") for ext in self._extensions]

    def add_diagnostic(self, diagnostic: dict[str, Any]) -> None:
        self._diagnostics.append(dict(diagnostic))

    def invalidate(self, message: str | None = None) -> None:
        self._stale_message = message or (
            "This extension ctx is stale after session replacement or reload. "
            "Do not use a captured pi or command ctx after ctx.newSession(), "
            "ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, "
            "fork, and switchSession, move post-replacement work into withSession "
            "and use the ctx passed to withSession. For reload, do not use the old "
            "ctx after await ctx.reload()."
        )
        invalidate = getattr(self._runtime, "invalidate", None)
        if callable(invalidate):
            invalidate(self._stale_message)

    def on_error(self, listener: Any) -> Any:
        self._error_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._error_listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def emit_error(self, error: dict[str, Any]) -> None:
        for listener in list(self._error_listeners):
            listener(error)

    def has_handlers(self, event_type: str) -> bool:
        """Check if any extension has handlers for this event type."""
        return any(
            event_type in ext.handlers and len(ext.handlers[event_type]) > 0
            for ext in self._extensions
        )

    def set_flag_value(self, name: str, value: bool | str) -> None:
        self._runtime.setdefault("flagValues", {})[name] = value

    def get_flag_values(self) -> dict[str, bool | str]:
        return dict(self._runtime.get("flagValues", {}))

    def get_shortcuts(
        self,
        resolved_keybindings: dict[str, Any] | None = None,
    ) -> dict[str, ExtensionShortcut]:
        """Resolve executable extension shortcuts with Node-style conflict rules."""
        self._shortcut_diagnostics = []
        builtin_keybindings = self._build_builtin_keybindings(resolved_keybindings or {})
        extension_shortcuts: dict[str, ExtensionShortcut] = {}

        for ext in self._extensions:
            for key, shortcut in ext.shortcuts.items():
                normalized_key = key.lower()
                builtin = builtin_keybindings.get(normalized_key)
                if builtin and builtin["restrictOverride"]:
                    self._add_shortcut_diagnostic(
                        (
                            f"Extension shortcut '{key}' from {shortcut.extension_path} "
                            "conflicts with built-in shortcut. Skipping."
                        ),
                        shortcut.extension_path,
                    )
                    continue
                if builtin and not builtin["restrictOverride"]:
                    self._add_shortcut_diagnostic(
                        (
                            f"Extension shortcut conflict: '{key}' is built-in shortcut "
                            f"for {builtin['keybinding']} and {shortcut.extension_path}. "
                            f"Using {shortcut.extension_path}."
                        ),
                        shortcut.extension_path,
                    )
                existing = extension_shortcuts.get(normalized_key)
                if existing is not None:
                    self._add_shortcut_diagnostic(
                        (
                            f"Extension shortcut conflict: '{key}' registered by both "
                            f"{existing.extension_path} and {shortcut.extension_path}. "
                            f"Using {shortcut.extension_path}."
                        ),
                        shortcut.extension_path,
                    )
                extension_shortcuts[normalized_key] = shortcut
        return extension_shortcuts

    def _build_builtin_keybindings(self, resolved_keybindings: dict[str, Any]) -> dict[str, dict[str, Any]]:
        builtin: dict[str, dict[str, Any]] = {}
        for keybinding, keys in resolved_keybindings.items():
            if keys is None:
                continue
            key_list = keys if isinstance(keys, list) else [keys]
            restrict_override = keybinding in RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS
            for key in key_list:
                if not isinstance(key, str):
                    continue
                normalized = key.lower()
                existing = builtin.get(normalized)
                if existing and existing.get("restrictOverride") and not restrict_override:
                    continue
                builtin[normalized] = {
                    "keybinding": keybinding,
                    "restrictOverride": restrict_override,
                }
        return builtin

    def _add_shortcut_diagnostic(self, message: str, extension_path: str) -> None:
        self._shortcut_diagnostics.append({
            "type": "warning",
            "message": message,
            "path": extension_path,
        })

    def get_shortcut_diagnostics(self) -> list[dict[str, Any]]:
        return list(self._shortcut_diagnostics)

    async def execute_shortcut(
        self,
        key: str,
        resolved_keybindings: dict[str, Any] | None = None,
    ) -> bool:
        """Execute an extension shortcut handler for a key. Returns true when handled."""
        shortcut = self.get_shortcuts(resolved_keybindings).get(key.lower())
        if shortcut is None or shortcut.handler is None:
            return False
        result = shortcut.handler(self.create_context())
        if inspect.isawaitable(result):
            await result
        return True

    def bind_command_context_actions(self, actions: dict[str, Any] | None) -> None:
        self._command_context_actions = dict(actions or {})

    def bind_context_actions(
        self,
        actions: dict[str, Any] | None = None,
        values: dict[str, Any] | None = None,
    ) -> None:
        self._context_actions = dict(actions or {})
        self._context_values = dict(values or {})

    def create_context(self) -> ExtensionContext:
        ctx = ExtensionContext(
            cwd=self._cwd,
            session_id=self._session_id,
            model=self._resolve_context_value("model"),
            stale_message_getter=lambda: self._stale_message,
        )
        setattr(ctx, "ui", self._context_values.get("ui", self._ui_context))
        setattr(ctx, "uiContext", self._context_values.get("uiContext", self._ui_context))
        setattr(ctx, "mode", self._context_values.get("mode", self._mode))
        setattr(ctx, "hasUI", self.has_ui())
        for name, value in self._context_values.items():
            if name == "model":
                continue
            setattr(ctx, name, self._resolve_context_value(name))
        for name, action in self._context_actions.items():
            setattr(ctx, name, action)
        return ctx

    def _resolve_context_value(self, name: str) -> Any:
        value = self._context_values.get(name)
        return value() if callable(value) else value

    def create_command_context(self) -> ExtensionContext:
        ctx = self.create_context()
        for name, action in {
            **self._context_actions,
            **self._command_context_actions,
        }.items():
            async def _wrapped_action(*args: Any, _action: Any = action, **kwargs: Any) -> Any:
                result = _action(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            setattr(ctx, name, _wrapped_action)
        return ctx

    async def _invoke_handler(
        self,
        handler: Any,
        ctx: ExtensionContext,
        event: dict[str, Any],
    ) -> Any:
        """Call extension handlers with Node order by default, preserving ctx-first Python handlers."""
        args: tuple[Any, ...] = (event, ctx)
        try:
            positional = [
                param
                for param in inspect.signature(handler).parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if len(positional) == 0:
                args = ()
            elif len(positional) == 1:
                args = (event,)
            else:
                first_name = positional[0].name.lower()
                if first_name in {"ctx", "context", "extension_context"}:
                    args = (ctx, event)
        except (TypeError, ValueError):
            pass

        result = handler(*args)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def emit(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit an event to all extensions."""
        event_type = event.get("type", "")
        combined_result: dict[str, Any] | None = None

        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    result = await self._invoke_handler(handler, ctx, event)
                    if isinstance(result, dict):
                        if combined_result is None:
                            combined_result = {}
                        combined_result.update(result)
                except Exception:
                    pass

        return combined_result

    async def emit_input(
        self,
        text: str,
        images: list[Any] | None = None,
        source: str = "interactive",
    ) -> dict[str, Any]:
        """Emit input event. Returns {action, text, images}."""
        result = await self.emit({
            "type": "input",
            "text": text,
            "images": images,
            "source": source,
        })
        if result and result.get("action") in ("handled", "transform"):
            return result
        return {"action": "pass", "text": text, "images": images}

    async def emit_before_agent_start(
        self,
        prompt: str,
        images: list[Any] | None,
        system_prompt: str,
    ) -> dict[str, Any] | None:
        """Emit before_agent_start event."""
        return await self.emit({
            "type": "before_agent_start",
            "prompt": prompt,
            "images": images,
            "systemPrompt": system_prompt,
        })

    async def emit_before_provider_request(self, payload: Any) -> Any:
        """Emit before_provider_request and allow handlers to replace the payload."""
        current_payload = payload
        for ext in self._extensions:
            handlers = ext.handlers.get("before_provider_request", [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    result = await self._invoke_handler(
                        handler,
                        ctx,
                        {"type": "before_provider_request", "payload": current_payload},
                    )
                    if result is not None:
                        current_payload = result
                except Exception:
                    pass
        return current_payload

    async def emit_after_provider_response(self, response: Any) -> None:
        """Emit after_provider_response with provider response metadata."""
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "status_code", None)
        if status is None and isinstance(response, dict):
            status = response.get("status") or response.get("status_code")
        headers = getattr(response, "headers", None)
        if headers is None and isinstance(response, dict):
            headers = response.get("headers")
        event = {
            "type": "after_provider_response",
            "status": status,
            "headers": dict(headers or {}),
        }
        await self.emit(event)

    async def emit_context(self, messages: list[Any]) -> list[Any]:
        """
        Emit context event for message modification.
        Uses chain passing - each handler receives the output of the previous handler.
        Mirrors ExtensionRunner.emitContext() in TypeScript.
        """
        current_messages = messages
        
        for ext in self._extensions:
            handlers = ext.handlers.get("context", [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    result = await self._invoke_handler(
                        handler,
                        ctx,
                        {"type": "context", "messages": current_messages},
                    )
                    
                    # Update current_messages with handler result (chain passing)
                    if isinstance(result, dict) and "messages" in result:
                        current_messages = result["messages"]
                except Exception:
                    # On error, skip this handler but continue chain
                    pass
        
        return current_messages

    async def emit_tool_call(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit tool_call event (before tool execution)."""
        current_event = self._normalize_event_dict(event)
        current_event["type"] = "tool_call"
        result: dict[str, Any] | None = None

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call", [])
            for handler in handlers:
                ctx = self.create_context()
                handler_result = await self._invoke_handler(handler, ctx, current_event)
                if isinstance(handler_result, dict):
                    result = dict(handler_result)
                    if result.get("block"):
                        return result
        return result

    async def emit_tool_result(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Emit tool_result event (after tool execution)."""
        current_event = self._normalize_event_dict(event)
        current_event["type"] = "tool_result"
        modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result", [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    handler_result = await self._invoke_handler(handler, ctx, current_event)
                    if not isinstance(handler_result, dict):
                        continue
                    if "content" in handler_result:
                        current_event["content"] = handler_result["content"]
                        modified = True
                    if "details" in handler_result:
                        current_event["details"] = handler_result["details"]
                        modified = True
                    if "isError" in handler_result or "is_error" in handler_result:
                        current_event["isError"] = handler_result.get("isError", handler_result.get("is_error"))
                        current_event["is_error"] = current_event["isError"]
                        modified = True
                except Exception:
                    continue

        if not modified:
            return None
        return {
            "content": current_event.get("content"),
            "details": current_event.get("details"),
            "isError": current_event.get("isError", current_event.get("is_error")),
            "is_error": current_event.get("is_error", current_event.get("isError")),
        }

    async def emit_message_end(self, event: dict[str, Any]) -> Any:
        current_event = self._normalize_event_dict(event)
        current_event["type"] = "message_end"
        current_message = current_event.get("message")
        modified = False
        for ext in self._extensions:
            handlers = ext.handlers.get("message_end", [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    handler_result = await self._invoke_handler(handler, ctx, current_event)
                    if isinstance(handler_result, dict) and "message" in handler_result:
                        next_message = handler_result["message"]
                        current_role = getattr(current_message, "role", None)
                        next_role = getattr(next_message, "role", None)
                        if isinstance(current_message, dict):
                            current_role = current_message.get("role")
                        if isinstance(next_message, dict):
                            next_role = next_message.get("role")
                        if current_role and next_role and current_role != next_role:
                            self.emit_error({
                                "extensionPath": getattr(ext, "path", ""),
                                "event": "message_end",
                                "error": "message_end handlers must return a message with the same role",
                            })
                            continue
                        current_message = next_message
                        current_event["message"] = current_message
                        modified = True
                except Exception as exc:
                    self.emit_error({
                        "extensionPath": getattr(ext, "path", ""),
                        "event": "message_end",
                        "error": str(exc),
                    })
        return current_message if modified else None

    async def emit_user_bash(self, event: dict[str, Any]) -> dict[str, Any] | None:
        result = await self.emit({**self._normalize_event_dict(event), "type": "user_bash"})
        return result if isinstance(result, dict) else None

    def _normalize_event_dict(self, event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            data = dict(event)
        elif dataclasses.is_dataclass(event):
            data = dataclasses.asdict(event)
        else:
            data = dict(getattr(event, "__dict__", {}) or {})
        if "tool_call_id" in data and "toolCallId" not in data:
            data["toolCallId"] = data["tool_call_id"]
        if "toolCallId" in data and "tool_call_id" not in data:
            data["tool_call_id"] = data["toolCallId"]
        if "tool_name" in data and "toolName" not in data:
            data["toolName"] = data["tool_name"]
        if "toolName" in data and "tool_name" not in data:
            data["tool_name"] = data["toolName"]
        if "is_error" in data and "isError" not in data:
            data["isError"] = data["is_error"]
        if "isError" in data and "is_error" not in data:
            data["is_error"] = data["isError"]
        return data

    async def emit_resources_discover(
        self,
        cwd: str,
        reason: str = "init",
    ) -> dict[str, list[str]]:
        """Emit resources_discover event."""
        skill_paths: list[Any] = []
        prompt_paths: list[Any] = []
        theme_paths: list[Any] = []
        event = {
            "type": "resources_discover",
            "cwd": cwd,
            "reason": reason,
        }

        def normalize_entries(entries: list[Any], extension_path: str) -> list[Any]:
            normalized: list[Any] = []
            for entry in entries:
                if isinstance(entry, str):
                    normalized.append({"path": entry, "extensionPath": extension_path})
                elif isinstance(entry, dict):
                    copy = dict(entry)
                    copy.setdefault("extensionPath", copy.get("extension_path") or extension_path)
                    normalized.append(copy)
            return normalized

        for ext in self._extensions:
            handlers = ext.handlers.get("resources_discover", [])
            for handler in handlers:
                try:
                    ctx = self.create_context()
                    result = await self._invoke_handler(handler, ctx, event)
                    if not isinstance(result, dict):
                        continue
                    extension_path = getattr(ext, "path", "") or getattr(ext, "resolved_path", "")
                    skill_paths.extend(
                        normalize_entries(
                            result.get("skillPaths") or result.get("skill_paths") or [],
                            extension_path,
                        )
                    )
                    prompt_paths.extend(
                        normalize_entries(
                            result.get("promptPaths") or result.get("prompt_paths") or [],
                            extension_path,
                        )
                    )
                    theme_paths.extend(
                        normalize_entries(
                            result.get("themePaths") or result.get("theme_paths") or [],
                            extension_path,
                        )
                    )
                except Exception:
                    pass

        return {
            "skillPaths": skill_paths,
            "promptPaths": prompt_paths,
            "themePaths": theme_paths,
        }

    def get_all_registered_tools(self) -> list[Any]:
        """Get all tools registered by extensions."""
        tools_by_name: dict[str, Any] = {}
        for ext in self._extensions:
            for tool in ext.tools.values():
                name = getattr(tool, "name", None)
                if isinstance(name, str) and name not in tools_by_name:
                    tools_by_name[name] = tool
        return list(tools_by_name.values())

    def get_tool_definition(self, tool_name: str) -> RegisteredTool | None:
        """Get a specific tool's definition."""
        for ext in self._extensions:
            if tool_name in ext.tools:
                return ext.tools[tool_name]
        return None

    def get_all_commands(self) -> list[RegisteredCommand]:
        """Get all commands registered by extensions."""
        commands: list[Any] = []
        for ext in self._extensions:
            commands.extend(ext.commands.values())
        return self._resolve_registered_commands(commands)

    def get_registered_commands(self) -> list[RegisteredCommand]:
        """Return commands with Node-style invocation names resolved."""
        return self.get_all_commands()

    def _resolve_registered_commands(self, commands: list[Any]) -> list[Any]:
        counts: dict[str, int] = {}
        for command in commands:
            counts[command.name] = counts.get(command.name, 0) + 1
        seen: dict[str, int] = {}
        taken: set[str] = set()
        resolved: list[Any] = []
        for command in commands:
            occurrence = seen.get(command.name, 0) + 1
            seen[command.name] = occurrence
            invocation_name = f"{command.name}:{occurrence}" if counts[command.name] > 1 else command.name
            if invocation_name in taken:
                suffix = occurrence
                while invocation_name in taken:
                    suffix += 1
                    invocation_name = f"{command.name}:{suffix}"
            taken.add(invocation_name)
            setattr(command, "invocation_name", invocation_name)
            setattr(command, "invocationName", invocation_name)
            resolved.append(command)
        return resolved

    def get_registered_commands_with_paths(self) -> list[dict[str, Any]]:
        """Return registered commands with their extension paths for RPC/UI callers."""
        commands: list[dict[str, Any]] = []
        for command in self.get_all_commands():
            invocation_name = getattr(command, "invocation_name", command.name)
            source_info = (
                command.source_info
                or create_synthetic_source_info(command.extension_path, source="local")
            )
            commands.append({
                "command": {
                    "name": invocation_name,
                    "description": command.description,
                },
                "extensionPath": command.extension_path,
                "sourceInfo": source_info_to_dict(source_info),
            })
        return commands

    def get_command(self, name: str) -> RegisteredCommand | None:
        """Get a specific command."""
        for command in self.get_all_commands():
            if getattr(command, "invocation_name", command.name) == name:
                return command
        return None

    async def get_argument_completions(self, name: str, argument_prefix: str) -> list[Any] | None:
        """Resolve argument completions for a registered command invocation."""
        command = self.get_command(name)
        if not command or command.get_argument_completions is None:
            return None
        result = command.get_argument_completions(argument_prefix)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def execute_command(self, name: str, args: str = "") -> Any:
        """Execute a registered command."""
        cmd = self.get_command(name)
        if not cmd:
            raise ValueError(f"Command not found: {name}")
        if cmd.handler is None:
            raise ValueError(f"Command has no handler: {name}")
        context = self.create_command_context()
        try:
            positional = [
                param
                for param in inspect.signature(cmd.handler).parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        except (TypeError, ValueError):
            positional = []
        result = cmd.handler(args) if len(positional) == 1 else cmd.handler(args, context)
        if inspect.isawaitable(result):
            result = await result
        return result

    getRegisteredCommands = get_registered_commands
    getCommandDiagnostics = lambda self: list(self._diagnostics)
    getCommand = get_command
    executeCommand = execute_command
    getArgumentCompletions = get_argument_completions
    getShortcuts = get_shortcuts
    getShortcutDiagnostics = get_shortcut_diagnostics
    createContext = create_context
    createCommandContext = create_command_context
    bindCore = bind_core
    bindCommandContext = bind_command_context
    bindCommandContextActions = bind_command_context_actions
    bindContextActions = bind_context_actions
    setUIContext = set_ui_context
    getUIContext = get_ui_context
    hasUI = has_ui
    getExtensionPaths = get_extension_paths
    getAllRegisteredTools = get_all_registered_tools
    getToolDefinition = get_tool_definition
    getFlags = lambda self: {name: flag for ext in self._extensions for name, flag in ext.flags.items()}
    getFlagValues = get_flag_values
    setFlagValue = set_flag_value
    getMessageRenderer = lambda self, custom_type: next(
        (
            renderer
            for ext in self._extensions
            for name, renderer in ext.message_renderers.items()
            if name == custom_type
        ),
        None,
    )
    addDiagnostic = add_diagnostic
    onError = on_error
    emitError = emit_error
    hasHandlers = has_handlers
    emitMessageEnd = emit_message_end
    emitToolResult = emit_tool_result
    emitToolCall = emit_tool_call
    emitUserBash = emit_user_bash
    emitContext = emit_context
    emitBeforeProviderRequest = emit_before_provider_request
    emitBeforeAgentStart = emit_before_agent_start
    emitResourcesDiscover = emit_resources_discover
    emitInput = emit_input
    executeShortcut = execute_shortcut

    async def shutdown(self, reason: str | None = None) -> None:
        """Emit session_shutdown to all extensions."""
        event = {"type": "session_shutdown"}
        if reason:
            event["reason"] = reason
        await self.emit(event)
