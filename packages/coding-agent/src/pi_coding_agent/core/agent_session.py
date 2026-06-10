"""
AgentSession — mirrors packages/coding-agent/src/core/agent-session.ts

Central class managing agent lifecycle, session persistence, tools, and events.
Full parity with TypeScript: auto-retry, overflow compaction, tool management,
model/thinking cycling, context usage, session stats, and queue management.
"""
from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

from pi_agent import Agent, AgentOptions
from pi_agent.types import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    ThinkingLevel,
)
from pi_ai import get_model, is_context_overflow
from pi_ai.types import AssistantMessage, ImageContent, Model, TextContent, UserMessage

from pi_coding_agent.config import get_share_viewer_url
from .auth_storage import AuthStorage
from .compaction import compact_context, should_compact
from .extensions.runner import ExtensionRunner
from .messages import CustomMessage, wrap_convert_to_llm
from .model_registry import ModelRegistry
from .session_manager import SessionManager
from .settings_manager import Settings, SettingsManager
from .system_prompt import build_system_prompt
from .trust_manager import ProjectTrustDecision, ProjectTrustStore
from .tools import (
    create_tool_definition_from_agent_tool,
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
    wrap_tool_definition,
)

# ── Thinking levels (mirrors TS constants) ────────────────────────────────────
_THINKING_LEVELS: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]
_THINKING_LEVELS_WITH_XHIGH: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]

# ── Retry error pattern (mirrors TS _isRetryableError regex) ─────────────────
_RETRY_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|429|500|502|503|504|"
    r"service.?unavailable|server error|internal error|connection.?error|"
    r"connection.?refused|other side closed|fetch failed|upstream.?connect|"
    r"reset before headers|terminated|retry delay",
    re.IGNORECASE,
)


class AgentSession:
    """
    Manages an agent session with persistence, tools, and events.
    Mirrors AgentSession in TypeScript.

    Key features vs. old version:
    - Per-message session persistence (message_end, not agent_end)
    - Auto-retry with exponential backoff
    - Overflow-aware auto-compaction (two paths: overflow vs threshold)
    - Tool registry with set_active_tools_by_name()
    - Model cycling (cycle_model), thinking cycling (cycle_thinking_level)
    - Context usage and session statistics
    - Queue management (clear_queue, pending_message_count)
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: Model | None = None,
        settings: Settings | None = None,
        session_id: str | None = None,
        session_manager: SessionManager | None = None,
        auth_storage: AuthStorage | None = None,
        model_registry: ModelRegistry | None = None,
        settings_manager: SettingsManager | None = None,
        resource_loader: Any | None = None,
        custom_tools: list[Any] | None = None,
        initial_active_tool_names: list[str] | None = None,
        session_start_event: dict[str, Any] | None = None,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self._settings = settings or Settings()
        self._auth_storage = auth_storage or AuthStorage()
        self._model_registry = model_registry or ModelRegistry()
        self._settings_manager = settings_manager or SettingsManager.create(cwd=self.cwd)
        self._resource_loader = resource_loader
        self._custom_tools = list(custom_tools or [])
        self._session_start_event = session_start_event or {
            "type": "session_start",
            "reason": "startup",
        }
        self._extension_bindings: dict[str, Any] = {}
        self._steering_mode_override: str | None = None
        self._follow_up_mode_override: str | None = None

        if session_manager is not None:
            self._session_manager = session_manager
        else:
            self._session_manager = SessionManager.create(cwd=self.cwd)

        self.session_id = self._session_manager.get_session_id()
        self._extension_runner = self._create_extension_runner()

        # Build all tools; keep registry for set_active_tools_by_name
        self._all_tools: list[AgentTool] = self._build_tools()
        active_names = (
            list(initial_active_tool_names)
            if initial_active_tool_names is not None
            else ["read", "bash", "edit", "write"]
        )
        active_tools = self._tools_for_names(active_names)

        # Resolve model
        resolved_model = model or self._resolve_default_model()

        # Build system prompt (stored as _base_system_prompt so it can be rebuilt)
        self._base_system_prompt = self._build_system_prompt([t.name for t in active_tools])

        # Create convertToLlm wrapper with blockImages support
        convert_to_llm_fn = wrap_convert_to_llm(self._settings_manager.get_block_images())

        # Create Agent with convert_to_llm and transform_context
        opts = AgentOptions(
            get_api_key=self._resolve_api_key,
            convert_to_llm=convert_to_llm_fn,
            transform_context=self._transform_context,
            on_payload=self._on_provider_payload,
            on_response=self._on_provider_response,
            beforeToolCall=self._before_tool_call,
            afterToolCall=self._after_tool_call,
        )
        self._agent = Agent(opts)
        self._agent.set_model(resolved_model)
        self._agent.set_system_prompt(self._base_system_prompt)
        self._agent.set_tools(active_tools)
        self._agent.set_thinking_level(self._settings.thinking_level)

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._agent.subscribe(self._on_agent_event)

        # ── Auto-retry state ──────────────────────────────────────────────────
        self._retry_attempt: int = 0
        self._retry_event: asyncio.Event | None = None      # set when retry resolves/fails
        self._retry_success: bool = False
        self._bash_cancel_event: asyncio.Event | None = None

        # ── Auto-compaction abort ─────────────────────────────────────────────
        self._compaction_abort: asyncio.Event = asyncio.Event()
        self._compaction_running: bool = False
        self._overflow_recovery_attempted: bool = False  # one-shot guard against infinite overflow loops

        # ── Last assistant message tracker (for auto-compaction/retry check) ──
        self._last_assistant_msg: AssistantMessage | None = None

        # ── Bash execution state ──────────────────────────────────────────────
        self._pending_bash_messages: list[AgentMessage] = []
        self._pending_next_turn_messages: list[AgentMessage] = []

        # ── Scoped models (for cycling) ───────────────────────────────────────
        self._scoped_models: list[dict[str, Model | ThinkingLevel | None]] | None = None
        self._bind_extension_context({})

        # ── Project-local memory (P5) — flag-gated, default off ───────────────
        # When PI_MEMORY_ENABLED=1, attach the project-local store so the recall hook
        # in _transform_context fires. Live auto-curation + compaction replacement are
        # validated end-to-end by P6; default-off keeps existing behaviour unchanged.
        self._memory = None
        self._memory_store = None
        self._memory_scope = None
        try:
            from .memory.integration import MemoryIntegration, memory_enabled
            if memory_enabled():
                self._memory = MemoryIntegration(os.getcwd())
                self._memory_store = self._memory.store
                self._memory_scope = self._memory.scope
        except Exception:
            self._memory = None  # never let memory wiring break session construction

    # ── Tool construction ─────────────────────────────────────────────────────

    def _build_tools(self) -> list[AgentTool]:
        """Create all default coding tools."""
        tools = [
            create_read_tool(self.cwd),
            create_write_tool(self.cwd),
            create_edit_tool(self.cwd),
            create_bash_tool(self.cwd),
            create_grep_tool(self.cwd),
            create_find_tool(self.cwd),
            create_ls_tool(self.cwd),
        ]
        for extension_tool in self._extension_runner.get_all_registered_tools():
            try:
                tools.append(
                    wrap_tool_definition(
                        extension_tool,
                        self._extension_runner.create_context,
                    )
                )
            except Exception:
                pass
        for custom_tool in self._custom_tools:
            adapted = self._adapt_custom_tool(custom_tool)
            if adapted is not None:
                tools.append(adapted)
        return tools

    def _adapt_custom_tool(self, custom_tool: Any) -> AgentTool | None:
        if isinstance(custom_tool, AgentTool):
            return custom_tool

        name = getattr(custom_tool, "name", None)
        execute = getattr(custom_tool, "execute", None)
        if not isinstance(name, str) or not callable(execute):
            return None
        return wrap_tool_definition(custom_tool)

    def _tools_for_names(self, tool_names: list[str]) -> list[AgentTool]:
        tools_by_name = {tool.name: tool for tool in self._all_tools}
        return [tools_by_name[name] for name in tool_names if name in tools_by_name]

    def _build_system_prompt(self, selected_tools: list[str]) -> str:
        loader = self._resource_loader
        custom_prompt = None
        append_parts: list[str] = []
        context_files: list[dict[str, str]] = []
        skills: list[dict[str, str]] = []

        if loader is not None:
            get_system_prompt = getattr(loader, "get_system_prompt", None)
            if callable(get_system_prompt):
                custom_prompt = get_system_prompt()

            get_append_system_prompt = getattr(loader, "get_append_system_prompt", None)
            if callable(get_append_system_prompt):
                append_parts = [part for part in get_append_system_prompt() if part]

            get_agents_files = getattr(loader, "get_agents_files", None)
            if callable(get_agents_files):
                agents_result = get_agents_files() or {}
                context_files = list(
                    agents_result.get("agentsFiles")
                    or agents_result.get("agents_files")
                    or []
                )

            get_skills = getattr(loader, "get_skills", None)
            if callable(get_skills):
                skills_result = get_skills() or {}
                skills = self._skills_for_prompt(skills_result.get("skills") or [])

        return build_system_prompt(
            self.cwd,
            custom_prompt=custom_prompt,
            selected_tools=selected_tools,
            append_system_prompt="\n\n".join(append_parts) if append_parts else None,
            context_files=context_files,
            skills=skills,
        )

    def _skills_for_prompt(self, loaded_skills: list[Any]) -> list[dict[str, str]]:
        skills: list[dict[str, str]] = []
        for skill in loaded_skills:
            name = getattr(skill, "name", None)
            if not isinstance(name, str):
                continue
            content = getattr(skill, "content", None)
            if not isinstance(content, str):
                file_path = getattr(skill, "file_path", None)
                if isinstance(file_path, str) and os.path.exists(file_path):
                    try:
                        with open(file_path, encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except OSError:
                        content = ""
                else:
                    content = ""
            skills.append({"name": name, "content": content})
        return skills

    def _create_extension_runner(
        self,
        flag_values: dict[str, bool | str] | None = None,
    ) -> ExtensionRunner:
        extensions: list[Any] = []
        runtime: dict[str, Any] = {"flagValues": {}}
        loader = self._resource_loader
        if loader is not None:
            get_extensions = getattr(loader, "get_extensions", None)
            if callable(get_extensions):
                result = get_extensions()
                if isinstance(result, dict):
                    extensions = list(result.get("extensions") or [])
                    runtime_result = result.get("runtime")
                    if isinstance(runtime_result, dict):
                        runtime = runtime_result
                else:
                    extensions = list(getattr(result, "extensions", []) or [])
                    runtime_result = getattr(result, "runtime", None)
                    if isinstance(runtime_result, dict):
                        runtime = runtime_result
        runtime.setdefault("flagValues", {})
        runtime.setdefault("pendingProviderRegistrations", [])
        self._flush_pending_provider_registrations(runtime)
        if flag_values:
            runtime["flagValues"].update(flag_values)
        return ExtensionRunner(
            extensions=extensions,
            runtime=runtime,
            cwd=self.cwd,
            session_id=self.session_id,
        )

    def _flush_pending_provider_registrations(self, runtime: dict[str, Any]) -> None:
        pending = list(runtime.get("pendingProviderRegistrations") or [])
        for registration in pending:
            if not isinstance(registration, dict):
                continue
            name = registration.get("name")
            config = registration.get("config")
            if isinstance(name, str) and isinstance(config, dict):
                try:
                    self._model_registry.register_provider(name, config)
                except Exception:
                    pass
        runtime["pendingProviderRegistrations"] = []

    # ── Model resolution ──────────────────────────────────────────────────────

    def _resolve_default_model(self) -> Model:
        """Resolve the default model from settings."""
        try:
            resolved = self._model_registry.resolve_model(
                model_id=self._settings.model_id,
                provider=self._settings.provider,
            )
            explicit_requested = bool(self._settings.model_id or self._settings.provider)
            has_auth = bool(self._model_registry.get_api_key(resolved.provider))
            if explicit_requested and not has_auth:
                for prov, mid in (
                    ("openai", "gpt-5.5"),
                    ("anthropic", "claude-3-5-sonnet-20241022"),
                    ("google", "gemini-2.5-pro"),
                ):
                    if self._model_registry.get_api_key(prov):
                        fallback = self._model_registry.find(prov, mid)
                        if fallback:
                            return fallback
            return resolved
        except Exception:
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                if not os.environ.get("OPENAI_API_KEY"):
                    return get_model("google", "gemini-2.5-pro")
            # Default to OpenAI gpt-5.5 in every other case.
            return get_model("openai", "gpt-5.5")

    async def _transform_context(
        self, 
        messages: list[AgentMessage], 
        signal: asyncio.Event | None = None
    ) -> list[AgentMessage]:
        """
        Transform context before convert_to_llm.
        
        Currently returns messages unchanged. 
        Will be connected to ExtensionRunner.emit_context when extension support is added.
        Mirrors the transform_context callback in TypeScript SDK.
        """
        if self._extension_runner.has_handlers("context"):
            messages = await self._extension_runner.emit_context(messages)
        # P1: memory recall — inject a tail recall block when a store is attached.
        # Off by default (store is None) so existing behaviour is unchanged.
        store = getattr(self, "_memory_store", None)
        if store is not None:
            from .memory.recall import build_recall_block, latest_user_query
            from .messages import CustomMessage
            query = latest_user_query(messages)
            block = build_recall_block(store, query, getattr(self, "_memory_scope", None))
            if block:
                messages = list(messages) + [
                    CustomMessage(custom_type="memory_recall", content=block, display=False)
                ]
        return messages

    async def _resolve_api_key(self, provider: str) -> str | None:
        return self._auth_storage.resolve_api_key(provider)

    async def _on_provider_payload(self, payload: Any, model: Model | None = None) -> Any:
        if not self._extension_runner.has_handlers("before_provider_request"):
            return payload
        return await self._extension_runner.emit_before_provider_request(payload)

    async def _on_provider_response(self, response: Any, model: Model | None = None) -> None:
        if not self._extension_runner.has_handlers("after_provider_response"):
            return None
        await self._extension_runner.emit_after_provider_response(response)
        return None

    async def _before_tool_call(
        self,
        context: dict[str, Any],
        signal: asyncio.Event | None = None,
    ) -> dict[str, Any] | None:
        if not self._extension_runner.has_handlers("tool_call"):
            return None
        tool_call = context.get("toolCall") or context.get("tool_call")
        tool_name = getattr(tool_call, "name", "")
        tool_call_id = getattr(tool_call, "id", "")
        args = context.get("args") or {}
        try:
            return await self._extension_runner.emit_tool_call({
                "type": "tool_call",
                "toolName": tool_name,
                "toolCallId": tool_call_id,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "input": args,
            })
        except Exception as err:
            if isinstance(err, Exception):
                raise
            raise RuntimeError(f"Extension failed, blocking execution: {err}")

    async def _after_tool_call(
        self,
        context: dict[str, Any],
        signal: asyncio.Event | None = None,
    ) -> dict[str, Any] | None:
        if not self._extension_runner.has_handlers("tool_result"):
            return None
        tool_call = context.get("toolCall") or context.get("tool_call")
        tool_name = getattr(tool_call, "name", "")
        tool_call_id = getattr(tool_call, "id", "")
        args = context.get("args") or {}
        result = context.get("result")
        is_error = bool(context.get("isError", context.get("is_error", False)))
        content = result.get("content", []) if isinstance(result, dict) else getattr(result, "content", [])
        details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
        hook_result = await self._extension_runner.emit_tool_result({
            "type": "tool_result",
            "toolName": tool_name,
            "toolCallId": tool_call_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "input": args,
            "content": content,
            "details": details,
            "isError": is_error,
            "is_error": is_error,
        })
        if not hook_result:
            return None
        return {
            "content": hook_result.get("content"),
            "details": hook_result.get("details"),
            "isError": hook_result.get("isError", hook_result.get("is_error")),
            "is_error": hook_result.get("is_error", hook_result.get("isError")),
        }

    # ── Event handling ────────────────────────────────────────────────────────

    def _on_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events — persist messages and notify listeners."""
        # ── 2a: Persist messages on message_end (not agent_end) ──────────────
        if event.type == "message_end":
            msg = getattr(event, "message", None)
            if msg is not None:
                role = getattr(msg, "role", "")
                if role in ("user", "assistant", "toolResult"):
                    self._session_manager.append_message(_message_to_dict(msg))
                # Track last assistant message for retry/compaction
                if role == "assistant":
                    self._last_assistant_msg = msg
                    # Reset retry on successful non-error response
                    stop_reason = getattr(msg, "stop_reason", "")
                    if stop_reason != "error" and self._retry_attempt > 0:
                        self._emit({"type": "auto_retry_end", "success": True,
                                    "attempt": self._retry_attempt})
                        self._retry_attempt = 0
                        self._resolve_retry(success=True)

        # ── agent_end: check retry and compaction ─────────────────────────────
        if event.type == "agent_end":
            if self._last_assistant_msg is not None:
                msg = self._last_assistant_msg
                self._last_assistant_msg = None
                # Schedule retry / compaction check asynchronously
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._post_turn_checks(msg)
                    ))
                except RuntimeError:
                    pass  # no running loop in sync context

        # Notify external listeners
        for listener in list(self._listeners):
            listener(event)

    async def _post_turn_checks(self, msg: AssistantMessage) -> None:
        """Check retry and compaction after a turn completes (mirrors TS _handleAgentEvent)."""
        # Reset overflow recovery on successful turns
        if getattr(msg, "stop_reason", "") not in ("error", "aborted"):
            self._overflow_recovery_attempted = False
        # Retry takes priority over compaction
        if self._is_retryable_error(msg):
            did_retry = await self._handle_retryable_error(msg)
            if did_retry:
                return
        await self._check_compaction(msg)

    def _emit(self, event: dict | Any) -> None:
        """Emit a synthetic session event to all listeners."""
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to session events. Returns unsubscribe function."""
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    # ── Agent control ─────────────────────────────────────────────────────────

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
        source: str | None = None,
        expand_prompt_templates: bool = True,
        streaming_behavior: str | None = None,
        preflight_result: Callable[[bool], None] | None = None,
    ) -> None:
        """
        Send a prompt to the agent and wait for completion (including retries).
        Mirrors prompt() in TypeScript with:
        - Model/API key validation
        - Pre-prompt compaction check
        - Bash flush
        - pendingNextTurnMessages injection
        """
        current_text: str | None = None
        current_images = images

        # Normalize input
        if isinstance(message, str):
            current_text = message
        elif isinstance(message, list):
            pass  # already message list
        else:
            pass  # single message

        try:
            # If streaming, queue via steer/followUp
            if self._agent.state.is_streaming:
                if not streaming_behavior:
                    raise RuntimeError(
                        "Agent is already processing. Specify streaming_behavior ('steer' or 'followUp') to queue."
                    )
                if current_text is not None:
                    user_msg = UserMessage(
                        role="user",
                        content=[TextContent(type="text", text=current_text)],
                        timestamp=int(time.time() * 1000),
                    )
                    if streaming_behavior == "followUp":
                        self._agent.follow_up(user_msg)
                    else:
                        self._agent.steer(user_msg)
                if preflight_result is not None:
                    preflight_result(True)
                return

            if current_text is not None and await self._try_execute_extension_command(current_text):
                if preflight_result is not None:
                    preflight_result(True)
                return

            # Reset overflow recovery flag for each new user-initiated turn
            self._overflow_recovery_attempted = False

            # Flush pending bash messages
            self._flush_pending_bash_messages()

            # Validate model
            if not self._agent.state.model:
                raise RuntimeError("No model selected. Use /login or set an API key environment variable.")

            # Validate API key
            model = self._agent.state.model
            api_key = await self._resolve_api_key(model.provider)
            if not api_key:
                raise RuntimeError(
                    f"No API key found for {model.provider}. "
                    "Use /login or set an API key environment variable."
                )

            # Pre-compaction check on last assistant message
            last_assistant = self._find_last_assistant_message()
            if last_assistant:
                await self._check_compaction(last_assistant, skip_aborted=False)

            if current_text is not None and self._extension_runner.has_handlers("input"):
                input_result = await self._extension_runner.emit_input(
                    current_text,
                    current_images,
                    source or "interactive",
                )
                action = input_result.get("action")
                if action == "handled":
                    if preflight_result is not None:
                        preflight_result(True)
                    return
                if action == "transform":
                    current_text = input_result.get("text", current_text)
                    current_images = input_result.get("images", current_images)

            # Build messages array
            msgs: list[AgentMessage]
            if isinstance(message, list):
                msgs = message
            elif current_text is not None:
                content_parts: list[TextContent | ImageContent] = [TextContent(type="text", text=current_text)]
                if current_images:
                    content_parts.extend(current_images)
                msgs = [UserMessage(
                    role="user",
                    content=content_parts,
                    timestamp=int(time.time() * 1000),
                )]
            else:
                msgs = [message]

            # Inject pending next-turn messages
            if self._pending_next_turn_messages:
                msgs.extend(self._pending_next_turn_messages)
                self._pending_next_turn_messages = []

            # Reset system prompt to base
            self._agent.set_system_prompt(self._base_system_prompt)

            if current_text is not None and self._extension_runner.has_handlers("before_agent_start"):
                before_result = await self._extension_runner.emit_before_agent_start(
                    current_text,
                    current_images,
                    self._base_system_prompt,
                )
                if isinstance(before_result, dict) and before_result.get("system_prompt"):
                    self._agent.set_system_prompt(before_result["system_prompt"])
        except Exception:
            if preflight_result is not None:
                preflight_result(False)
            raise

        # Reset retry state
        self._retry_event = asyncio.Event()
        self._retry_success = False
        self._retry_attempt = 0

        if preflight_result is not None:
            preflight_result(True)
        await self._agent.prompt(msgs)
        await self._wait_for_retry()

    def _flush_pending_bash_messages(self) -> None:
        """Flush pending bash messages into agent state and session."""
        if not self._pending_bash_messages:
            return
        for bash_msg in self._pending_bash_messages:
            self._agent.append_message(bash_msg)
            self._session_manager.append_message(_message_to_dict(bash_msg))
        self._pending_bash_messages = []

    def _find_last_assistant_message(self) -> AssistantMessage | None:
        """Find the last assistant message in the current context."""
        for m in reversed(self._agent.state.messages):
            if getattr(m, "role", "") == "assistant":
                return m
        return None

    async def _try_execute_extension_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        args = "" if space_index == -1 else text[space_index + 1:]
        if not self._extension_runner.get_command(command_name):
            return False
        await self._extension_runner.execute_command(command_name, args)
        return True

    def record_bash_result(
        self,
        command: str,
        result: dict[str, Any],
        exclude_from_context: bool | dict[str, Any] = False,
    ) -> None:
        """
        Record a bash execution result.
        If streaming, queues for later flush; otherwise adds immediately.
        """
        from .messages import BashExecutionMessage

        if isinstance(exclude_from_context, dict):
            exclude_from_context = bool(
                exclude_from_context.get("excludeFromContext")
                or exclude_from_context.get("exclude_from_context")
            )

        bash_msg = BashExecutionMessage(
            role="bashExecution",
            command=command,
            output=result.get("output", ""),
            exit_code=result.get("exit_code", result.get("exitCode")),
            cancelled=bool(result.get("cancelled", False)),
            truncated=bool(result.get("truncated", False)),
            full_output_path=result.get("full_output_path", result.get("fullOutputPath")),
            timestamp=int(time.time() * 1000),
            exclude_from_context=exclude_from_context,
        )
        if self._agent.state.is_streaming:
            self._pending_bash_messages.append(bash_msg)
        else:
            self._agent.append_message(bash_msg)
            self._session_manager.append_message(_message_to_dict(bash_msg))

    # ── Session switching / tree navigation ────────────────────────────────────

    async def switch_session(self, session_path: str) -> bool:
        """
        Switch to a different session file.
        Mirrors switchSession() in TypeScript.
        Returns True if switch succeeded.
        """
        new_sm = SessionManager.open(session_path)
        await self._switch_to_session_manager(new_sm)
        return True

    async def _switch_to_session_manager(self, session_manager: SessionManager) -> None:
        """Switch this AgentSession to an already-created SessionManager."""
        self._agent.abort()
        await self._agent.wait_for_idle()

        self._agent.clear_all_queues()
        self._pending_bash_messages = []
        self._pending_next_turn_messages = []

        self._session_manager = session_manager
        self.session_id = session_manager.get_session_id()

        # Restore context from session
        context = session_manager.build_context()
        self._agent.replace_messages(context.messages)

        # Restore model/thinking from session
        if context.model:
            try:
                model = get_model(context.model["provider"], context.model["model_id"])
                self._agent.set_model(model)
            except Exception:
                pass
        if context.thinking_level:
            self._agent.set_thinking_level(context.thinking_level)

        self._extension_runner = self._create_extension_runner()
        self._bind_extension_context(self._extension_bindings)

    async def new_session(self, options: dict[str, Any] | None = None) -> bool:
        """
        Start a new session in-place.
        Mirrors the Node runtime /new behavior at the session-management layer.
        """
        parent_session = None
        if options:
            parent_session = options.get("parentSession") or options.get("parent_session")
        new_sm = SessionManager.create(
            self.cwd,
            session_dir=self._session_manager.get_session_dir(),
            parent_session=parent_session,
        )
        if self.model:
            new_sm.append_model_change(self.model.provider, self.model.id)
        new_sm.append_thinking_level_change(self.thinking_level)
        await self._switch_to_session_manager(new_sm)
        return True

    async def clone_session(self) -> dict[str, Any]:
        """
        Duplicate the current session at its current file and switch to the clone.
        """
        current_file = self._session_manager.get_session_file()
        if not current_file:
            await self.new_session()
            return {"cancelled": False}
        cloned = SessionManager.fork_from(
            current_file,
            self.cwd,
            self._session_manager.get_session_dir(),
        )
        await self._switch_to_session_manager(cloned)
        return {"cancelled": False}

    async def fork_session(self, entry_id: str) -> dict[str, Any]:
        """
        Fork from a user message entry and switch to the fork in-place.
        Returns the selected user text so callers can place it back in the editor.
        """
        entry = self._session_manager.get_entry(entry_id)
        if not entry:
            return {"cancelled": True, "selectedText": ""}

        selected_text = ""
        msg_data = entry.data.get("message", {})
        if isinstance(msg_data, dict) and msg_data.get("role") == "user":
            selected_text = self._extract_user_message_text(msg_data.get("content", ""))

        branch_point = entry.parent_id if entry.parent_id else None
        forked_sm = self._session_manager.branch(branch_point, self.cwd)
        await self._switch_to_session_manager(forked_sm)
        return {"cancelled": False, "selectedText": selected_text}

    def get_session_tree_entries(self) -> list[dict[str, Any]]:
        """Return a flat session-tree listing for RPC/TUI fallback commands."""
        rows: list[dict[str, Any]] = []
        for entry in self._session_manager.get_entries():
            text = ""
            msg_data = entry.data.get("message", {})
            if isinstance(msg_data, dict):
                text = self._extract_user_message_text(msg_data.get("content", ""))
            rows.append({
                "entry_id": entry.id,
                "parent_id": entry.parent_id,
                "type": entry.type,
                "label": self._session_manager.get_label(entry.id),
                "text": text,
            })
        return rows

    async def navigate_tree(
        self,
        target_id: str,
        summarize: bool = False,
    ) -> dict[str, Any]:
        """
        Navigate to a different point in the session tree.
        Mirrors navigateTree() in TypeScript.
        """
        old_leaf_id = self._session_manager.get_leaf_id()
        if target_id == old_leaf_id:
            return {"cancelled": False}

        self._agent.abort()
        await self._agent.wait_for_idle()

        # Optionally summarize the branch being left
        if summarize and old_leaf_id:
            from .compaction.branch_summarization import summarize_branch
            try:
                summary_result = await summarize_branch(
                    self._session_manager,
                    old_leaf_id,
                    target_id,
                    self._agent.state.model,
                )
                if summary_result and summary_result.summary:
                    self._session_manager.append_branch_summary(
                        summary_result.summary,
                        from_id=old_leaf_id,
                    )
            except Exception:
                pass

        self._session_manager.set_leaf_id(target_id)

        # Rebuild context from new position
        context = self._session_manager.build_context(target_id)
        self._agent.replace_messages(context.messages)

        if context.model:
            try:
                model = get_model(context.model["provider"], context.model["model_id"])
                self._agent.set_model(model)
            except Exception:
                pass
        if context.thinking_level:
            self._agent.set_thinking_level(context.thinking_level)

        # Check if target is a user message (return its text for editor)
        target_entry = self._session_manager.get_entry(target_id)
        editor_text: str | None = None
        if target_entry and target_entry.type == "message":
            msg_data = target_entry.data.get("message", {})
            if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                content = msg_data.get("content", [])
                if isinstance(content, str):
                    editor_text = content
                elif isinstance(content, list):
                    editor_text = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )

        return {"cancelled": False, "editorText": editor_text}

    async def create_branched_session(self, branch_point_id: str) -> "AgentSession":
        """
        Create a new session branching from a specific entry.
        Mirrors createBranchedSession() via SessionManager.branch().
        """
        new_sm = self._session_manager.branch(branch_point_id, self.cwd)
        branched = AgentSession(
            cwd=self.cwd,
            model=self._agent.state.model,
            settings=self._settings,
            session_manager=new_sm,
            auth_storage=self._auth_storage,
            model_registry=self._model_registry,
        )
        context = new_sm.build_context()
        branched._agent.replace_messages(context.messages)
        return branched

    def _user_message_from_text(
        self,
        message: str,
        images: list[ImageContent] | None = None,
    ) -> UserMessage:
        content_parts: list[TextContent | ImageContent] = [TextContent(type="text", text=message)]
        if images:
            content_parts.extend(images)
        return UserMessage(
            role="user",
            content=content_parts,
            timestamp=int(time.time() * 1000),
        )

    async def steer(
        self,
        message: str | AgentMessage,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Queue a steering message."""
        if isinstance(message, str):
            message = self._user_message_from_text(message, images)
        self._agent.steer(message)

    async def follow_up(
        self,
        message: str | AgentMessage,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Queue a follow-up message."""
        if isinstance(message, str):
            message = self._user_message_from_text(message, images)
        self._agent.follow_up(message)

    async def abort(self) -> None:
        """Abort current operation and wait for agent to become idle."""
        self._abort_retry()
        self._agent.abort()
        await self._agent.wait_for_idle()

    async def wait_for_idle(self) -> None:
        await self._agent.wait_for_idle()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self):
        return self._agent.state

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def model(self) -> Model | None:
        return self._agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self._agent.state.is_streaming

    @property
    def is_compacting(self) -> bool:
        return self._compaction_running

    @property
    def is_retrying(self) -> bool:
        return self._retry_attempt > 0

    @property
    def is_bash_running(self) -> bool:
        return self._bash_cancel_event is not None

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def auth_storage(self) -> AuthStorage:
        return self._auth_storage

    @property
    def resource_loader(self) -> Any | None:
        return self._resource_loader

    @property
    def prompt_templates(self) -> list[Any]:
        loader = self._resource_loader
        if loader is None:
            return []
        get_prompts = getattr(loader, "get_prompts", None)
        if not callable(get_prompts):
            return []
        result = get_prompts() or {}
        if isinstance(result, dict):
            return list(result.get("prompts") or [])
        return list(getattr(result, "prompts", []) or [])

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def settings_manager(self) -> SettingsManager:
        return self._settings_manager

    @property
    def session_file(self) -> str | None:
        return self._session_manager.get_session_file()

    @property
    def session_name(self) -> str | None:
        return self._session_manager.get_session_name()

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._session_manager.get_messages()

    @property
    def steering_mode(self) -> str:
        if self._steering_mode_override is not None:
            return self._steering_mode_override
        getter = getattr(self._settings_manager, "get_steering_mode", None)
        return getter() if callable(getter) else "steer"

    @property
    def follow_up_mode(self) -> str:
        if self._follow_up_mode_override is not None:
            return self._follow_up_mode_override
        getter = getattr(self._settings_manager, "get_follow_up_mode", None)
        return getter() if callable(getter) else "followUp"

    @property
    def system_prompt(self) -> str:
        return self._agent.state.system_prompt

    # ── Queue management ──────────────────────────────────────────────────────

    @property
    def pending_message_count(self) -> int:
        return (len(self._agent._steering_queue)
                + len(self._agent._follow_up_queue))

    def get_steering_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent._steering_queue]

    def get_follow_up_messages(self) -> list[str]:
        return [getattr(m, "content", str(m)) for m in self._agent._follow_up_queue]

    def clear_queue(self) -> dict[str, list]:
        steering = list(self._agent._steering_queue)
        follow_up = list(self._agent._follow_up_queue)
        self._agent.clear_all_queues()
        return {"steering": steering, "followUp": follow_up}

    # ── Tool management (2d) ──────────────────────────────────────────────────

    def get_active_tool_names(self) -> list[str]:
        """Get names of currently active tools."""
        return [t.name for t in self._agent.state.tools]

    def get_all_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return [t.name for t in self._all_tools]

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all configured tools with definition metadata."""
        tools: list[dict[str, Any]] = []
        for tool in self._all_tools:
            definition = create_tool_definition_from_agent_tool(tool)
            tools.append({
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
                "promptGuidelines": definition.prompt_guidelines,
                "sourceInfo": None,
            })
        return tools

    def get_tool_definition(self, name: str) -> Any | None:
        """Get a registered tool definition by name."""
        extension_definition = self._extension_runner.get_tool_definition(name)
        if extension_definition is not None:
            return extension_definition
        for tool in self._all_tools:
            if tool.name == name:
                return create_tool_definition_from_agent_tool(tool)
        return None

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """
        Set active tools by name. Rebuilds system prompt to reflect new tool set.
        Mirrors setActiveToolsByName() in TypeScript.
        """
        active = self._tools_for_names(tool_names)
        self._agent.set_tools(active)
        valid_names = [t.name for t in active]
        self._base_system_prompt = self._build_system_prompt(valid_names)
        self._agent.set_system_prompt(self._base_system_prompt)

    # ── Model management (2g) ─────────────────────────────────────────────────

    async def set_model(self, model: Model) -> None:
        """
        Switch the active model with API key validation.
        Mirrors setModel() in TypeScript.
        """
        api_key = self._model_registry.get_api_key(model.provider)
        if not api_key:
            raise RuntimeError(f"No API key for {model.provider}/{model.id}")
        self._agent.set_model(model)
        self._session_manager.append_model_change(model.provider, model.id)
        # Re-clamp thinking level for new model
        self.set_thinking_level(self.thinking_level)

    async def cycle_model(self, direction: str = "forward") -> dict | None:
        """
        Cycle to next/previous available model.
        Mirrors cycleModel() in TypeScript.
        Returns new model info or None if only one model available.
        """
        available = await self._model_registry.get_available()
        if len(available) <= 1:
            return None
        current = self._agent.state.model
        current_idx = next(
            (i for i, m in enumerate(available)
             if m.provider == getattr(current, "provider", "") and m.id == getattr(current, "id", "")),
            0,
        )
        n = len(available)
        next_idx = (current_idx + 1) % n if direction == "forward" else (current_idx - 1 + n) % n
        next_model = available[next_idx]
        await self.set_model(next_model)
        return {"model": next_model}

    # ── Thinking level management (2h) ────────────────────────────────────────

    def get_available_thinking_levels(self) -> list[ThinkingLevel]:
        """Get thinking levels available for current model."""
        from pi_ai import supports_xhigh
        model = self._agent.state.model
        if model and supports_xhigh(model):
            return list(_THINKING_LEVELS_WITH_XHIGH)
        return list(_THINKING_LEVELS)

    def supports_thinking(self) -> bool:
        """Return whether the current model supports reasoning/thinking."""
        return bool(getattr(self.model, "reasoning", False))

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set thinking level, clamped to model capabilities. Persists to session."""
        available = self.get_available_thinking_levels()
        effective = level if level in available else _clamp_thinking_level(level, available)
        is_changing = effective != self._agent.state.thinking_level
        self._agent.set_thinking_level(effective)
        if is_changing:
            self._session_manager.append_thinking_level_change(effective)

    def cycle_thinking_level(self) -> ThinkingLevel | None:
        """
        Cycle to next thinking level.
        Mirrors cycleThinkingLevel() in TypeScript.
        Returns new level or None if model doesn't support thinking.
        """
        available = self.get_available_thinking_levels()
        if available == ["off"]:
            return None
        current = self._agent.state.thinking_level
        idx = available.index(current) if current in available else 0
        next_level = available[(idx + 1) % len(available)]
        self.set_thinking_level(next_level)
        return next_level

    # ── Session statistics (2f) ───────────────────────────────────────────────

    def get_session_stats(self) -> dict[str, Any]:
        """
        Get session statistics (message counts, token totals, cost).
        Mirrors getSessionStats() in TypeScript.
        """
        msgs = self._agent.state.messages
        user_messages = sum(1 for m in msgs if getattr(m, "role", "") == "user")
        assistant_messages = sum(1 for m in msgs if getattr(m, "role", "") == "assistant")
        tool_results = sum(1 for m in msgs if getattr(m, "role", "") == "toolResult")
        tool_calls = 0
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0

        for m in msgs:
            if getattr(m, "role", "") == "assistant":
                content = getattr(m, "content", [])
                tool_calls += sum(
                    1 for c in content if getattr(c, "type", "") == "toolCall"
                )
                usage = getattr(m, "usage", None)
                if usage:
                    total_input += getattr(usage, "input", 0)
                    total_output += getattr(usage, "output", 0)
                    total_cache_read += getattr(usage, "cache_read", 0)
                    total_cache_write += getattr(usage, "cache_write", 0)
                    cost = getattr(usage, "cost", None)
                    if cost:
                        total_cost += getattr(cost, "total", 0) or 0.0

        return {
            "sessionId": self.session_id,
            "sessionFile": self._session_manager.get_session_file(),
            "userMessages": user_messages,
            "assistantMessages": assistant_messages,
            "toolCalls": tool_calls,
            "toolResults": tool_results,
            "totalMessages": len(msgs),
            "tokens": {
                "input": total_input,
                "output": total_output,
                "cacheRead": total_cache_read,
                "cacheWrite": total_cache_write,
                "total": total_input + total_output + total_cache_read + total_cache_write,
            },
            "cost": total_cost,
        }

    # ── Context usage (2e) ────────────────────────────────────────────────────

    def get_context_usage(self) -> dict | None:
        """
        Get current context window usage.
        Mirrors getContextUsage() in TypeScript.
        """
        model = self._agent.state.model
        if not model:
            return None
        context_window = getattr(model, "context_window", 0) or 0
        if context_window <= 0:
            return None
        tokens = self._estimate_context_tokens()
        percent = (tokens / context_window * 100) if context_window else 0
        return {"tokens": tokens, "contextWindow": context_window, "percent": percent}

    def _estimate_context_tokens(self) -> int:
        """Estimate current context size from last assistant usage or message lengths."""
        msgs = self._agent.state.messages
        # Walk backwards to find last assistant message with usage
        for m in reversed(msgs):
            if getattr(m, "role", "") == "assistant":
                usage = getattr(m, "usage", None)
                if usage:
                    inp = getattr(usage, "input", 0) or 0
                    out = getattr(usage, "output", 0) or 0
                    cr = getattr(usage, "cache_read", 0) or 0
                    if inp + out + cr > 0:
                        return inp + cr  # context tokens = input + cache_read
        # Fallback: estimate from character count
        total_chars = sum(
            len(str(getattr(m, "content", ""))) for m in msgs
        )
        return total_chars // 4

    # ── Session management ────────────────────────────────────────────────────

    async def fork(self, entry_id: str | None = None) -> "AgentSession":
        """
        Fork the session from a specific entry (or current leaf).
        Mirrors fork() in TypeScript.
        """
        branch_point = entry_id
        if not branch_point:
            leaf = self._session_manager.get_leaf_entry()
            branch_point = leaf.id if leaf else None

        if branch_point:
            # Check if entry has a parent; if not, create new session
            entry = self._session_manager.get_entry(branch_point)
            if entry and entry.parent_id:
                return await self.create_branched_session(entry.parent_id)

        sessions_dir = self._session_manager.get_session_dir()
        src_path = self._session_manager.get_session_file()
        forked_sm = SessionManager.fork_from(src_path, self.cwd, sessions_dir)
        forked = AgentSession(
            cwd=self.cwd,
            model=self._agent.state.model,
            settings=self._settings,
            session_manager=forked_sm,
            auth_storage=self._auth_storage,
            model_registry=self._model_registry,
        )
        forked._agent.replace_messages(list(self._agent.state.messages))
        return forked

    def get_session_info(self) -> dict[str, Any]:
        """Get basic session information (backwards compat)."""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "model": self._agent.state.model.id if self._agent.state.model else None,
            "message_count": len(self._agent.state.messages),
            "is_streaming": self._agent.state.is_streaming,
        }

    async def send_custom_message(
        self,
        message: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send or persist a custom extension message."""
        opts = options or {}
        custom_type = message.get("customType") or message.get("custom_type") or "custom"
        content = message.get("content", "")
        display = bool(message.get("display", True))
        details = message.get("details")
        app_message = CustomMessage(
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
            timestamp=int(time.time() * 1000),
        )

        deliver_as = opts.get("deliverAs") or opts.get("deliver_as")
        if deliver_as == "nextTurn":
            self._pending_next_turn_messages.append(app_message)
        elif self.is_streaming:
            if deliver_as == "followUp":
                self._agent.follow_up(app_message)
            else:
                self._agent.steer(app_message)
        elif opts.get("triggerTurn") or opts.get("trigger_turn"):
            await self.prompt(app_message)
        else:
            self._agent.append_message(app_message)
            self._session_manager.append_custom_message_entry(
                custom_type,
                content,
                display,
                details,
            )
            self._emit({"type": "message_start", "message": app_message})
            self._emit({"type": "message_end", "message": app_message})

    async def send_user_message(
        self,
        content: str | list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send a user message through the normal prompt path."""
        opts = options or {}
        if isinstance(content, str):
            text = content
            images = None
        else:
            text_parts: list[str] = []
            images: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    images.append(part)
            text = "\n".join(text_parts)
            if not images:
                images = None
        await self.prompt(
            text,
            images=images,
            streaming_behavior=opts.get("deliverAs") or opts.get("deliver_as"),
            source="extension",
        )

    # ── Compaction ────────────────────────────────────────────────────────────

    async def compact(self, custom_instructions: str | None = None) -> str:
        """Manually compact the context. Returns the summary."""
        messages = self._agent.state.messages
        model = self._agent.state.model
        if not model:
            return ""

        from pi_ai import stream_simple
        new_messages, summary = await compact_context(
            messages,
            self._agent.state.system_prompt,
            stream_simple,
            model,
        )
        self._agent.replace_messages(new_messages)

        if summary:
            first_kept_id = str(len(messages)) if messages else "0"
            self._session_manager.append_compaction(summary, first_kept_id)

        return summary

    async def _check_compaction(self, msg: AssistantMessage, skip_aborted: bool = True) -> None:
        """
        Check if compaction is needed and run it.
        Mirrors _checkCompaction() in TypeScript with two cases:
        1. Overflow — LLM returned context overflow error → compact + retry
        2. Threshold — context over threshold → compact (no auto-retry)
        """
        settings = self._settings_manager.get_compaction_settings()
        if not settings.get("enabled", True):
            return
        if skip_aborted and getattr(msg, "stop_reason", "") == "aborted":
            return

        model = self._agent.state.model
        context_window = getattr(model, "context_window", 0) if model else 0

        # Case 1: Overflow
        same_model = (
            model and
            getattr(msg, "provider", None) == model.provider and
            getattr(msg, "model", None) == model.id
        )
        if same_model and is_context_overflow(msg, context_window):
            if self._overflow_recovery_attempted:
                # Already tried once — don't loop infinitely
                self._emit({
                    "type": "auto_compaction_end",
                    "result": None,
                    "aborted": False,
                    "willRetry": False,
                    "errorMessage": (
                        "Context overflow recovery failed after one compact-and-retry attempt. "
                        "Try reducing context or switching to a larger-context model."
                    ),
                })
                return
            self._overflow_recovery_attempted = True
            # Remove the error message from agent state (keep in session history)
            messages = self._agent.state.messages
            if messages and getattr(messages[-1], "role", "") == "assistant":
                self._agent.replace_messages(messages[:-1])
            await self._run_auto_compaction("overflow", will_retry=True)
            return

        # Case 2: Threshold
        if getattr(msg, "stop_reason", "") == "error":
            return  # non-overflow errors have no usage data
        tokens = self._estimate_context_tokens()
        reserve = settings.get("reserveTokens", 16384)
        if context_window > 0 and should_compact(
            self._agent.state.messages,
            context_window,
            (context_window - reserve) / context_window,
        ):
            await self._run_auto_compaction("threshold", will_retry=False)

    async def _run_auto_compaction(self, reason: str, will_retry: bool) -> None:
        """Run auto-compaction with events (mirrors _runAutoCompaction in TS)."""
        self._compaction_running = True
        self._emit({"type": "auto_compaction_start", "reason": reason})
        try:
            model = self._agent.state.model
            if not model:
                self._emit({"type": "auto_compaction_end", "result": None,
                            "aborted": False, "willRetry": False})
                return

            from pi_ai import stream_simple
            messages = self._agent.state.messages
            new_messages, summary = await compact_context(
                messages,
                self._agent.state.system_prompt,
                stream_simple,
                model,
            )

            self._agent.replace_messages(new_messages)

            result = {"summary": summary, "tokensBefore": self._estimate_context_tokens()}
            if summary:
                first_kept_id = str(len(messages)) if messages else "0"
                self._session_manager.append_compaction(summary, first_kept_id)

            self._emit({"type": "auto_compaction_end", "result": result,
                        "aborted": False, "willRetry": will_retry})

            if will_retry:
                # Schedule agent.continue() via call_soon to break out of event chain
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._agent.continue_from_context()
                    ))
                except RuntimeError:
                    pass
            elif self._agent.has_queued_messages():
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon(lambda: asyncio.ensure_future(
                        self._agent.continue_from_context()
                    ))
                except RuntimeError:
                    pass

        except Exception as e:
            err_msg = str(e)
            prefix = "Context overflow recovery failed" if reason == "overflow" else "Auto-compaction failed"
            self._emit({"type": "auto_compaction_end", "result": None, "aborted": False,
                        "willRetry": False, "errorMessage": f"{prefix}: {err_msg}"})
        finally:
            self._compaction_running = False

    def abort_compaction(self) -> None:
        self._compaction_abort.set()

    def abort_branch_summary(self) -> None:
        """Cancel in-progress branch summarization when a cancel signal exists."""
        branch_abort = getattr(self, "_branch_summary_abort", None)
        if branch_abort is not None and hasattr(branch_abort, "set"):
            branch_abort.set()

    # ── Auto-retry (2b) ───────────────────────────────────────────────────────

    def _is_retryable_error(self, msg: AssistantMessage) -> bool:
        """
        Check if error is retryable (rate limit, overloaded, server errors).
        Context overflow is NOT retryable — handled by compaction.
        Mirrors _isRetryableError() in TypeScript.
        """
        if getattr(msg, "stop_reason", "") != "error":
            return False
        model = self._agent.state.model
        context_window = getattr(model, "context_window", 0) if model else 0
        if is_context_overflow(msg, context_window):
            return False
        err = getattr(msg, "error_message", "") or ""
        return bool(_RETRY_PATTERN.search(err))

    async def _handle_retryable_error(self, msg: AssistantMessage) -> bool:
        """
        Handle retryable errors with exponential backoff.
        Mirrors _handleRetryableError() in TypeScript.
        Returns True if retry was initiated.
        """
        settings = self._settings_manager.get_retry_settings()
        if not settings.get("enabled", True):
            return False

        self._retry_attempt += 1
        max_retries = settings.get("maxRetries", 3)
        base_delay_ms = settings.get("baseDelayMs", 2000)

        if self._retry_attempt > max_retries:
            self._emit({
                "type": "auto_retry_end",
                "success": False,
                "attempt": self._retry_attempt - 1,
                "finalError": getattr(msg, "error_message", "Unknown error"),
            })
            self._retry_attempt = 0
            self._resolve_retry(success=False)
            return False

        delay_ms = base_delay_ms * (2 ** (self._retry_attempt - 1))

        self._emit({
            "type": "auto_retry_start",
            "attempt": self._retry_attempt,
            "maxAttempts": max_retries,
            "delayMs": delay_ms,
            "errorMessage": getattr(msg, "error_message", "Unknown error"),
        })

        # Remove error message from state (keep in session for history)
        messages = self._agent.state.messages
        if messages and getattr(messages[-1], "role", "") == "assistant":
            self._agent.replace_messages(messages[:-1])

        # Wait with exponential backoff
        try:
            await asyncio.sleep(delay_ms / 1000.0)
        except asyncio.CancelledError:
            self._emit({"type": "auto_retry_end", "success": False,
                        "attempt": self._retry_attempt, "finalError": "Retry cancelled"})
            self._retry_attempt = 0
            self._resolve_retry(success=False)
            return False

        # Schedule retry via call_soon
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: asyncio.ensure_future(
                self._agent.continue_from_context()
            ))
        except RuntimeError:
            pass

        return True

    def _resolve_retry(self, success: bool) -> None:
        """Resolve the pending retry event."""
        self._retry_success = success
        if self._retry_event:
            self._retry_event.set()

    def _abort_retry(self) -> None:
        """Cancel in-progress retry."""
        if self._retry_event:
            self._retry_event.set()

    def abort_retry(self) -> None:
        """Cancel in-progress retry."""
        self._abort_retry()

    async def _wait_for_retry(self) -> None:
        """Wait for any in-progress retry to complete (mirrors waitForRetry in TS)."""
        if self._retry_attempt == 0:
            return
        if self._retry_event and not self._retry_event.is_set():
            await self._retry_event.wait()

    # ── HTML export ───────────────────────────────────────────────────────────

    async def export_to_html(self, output_path: str | None = None) -> str:
        """Export current session messages to a basic HTML transcript."""
        messages = self._session_manager.get_messages()
        if not output_path:
            output_path = os.path.join(self.cwd, f"{self.session_id}.html")
        rows: list[str] = []
        for msg in messages:
            role = html.escape(str(msg.get("role", "unknown")))
            content = html.escape(str(msg.get("content", "")))
            rows.append(f"<div><strong>{role}</strong>: <pre>{content}</pre></div>")
        body = "\n".join(rows) or "<div><em>No messages</em></div>"
        html_doc = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Session Export</title></head><body>"
            f"{body}</body></html>"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return output_path

    def export_to_jsonl(self, output_path: str | None = None) -> str:
        """Export the current session branch to JSONL."""
        if not output_path:
            output_path = os.path.join(self.cwd, f"session-{int(time.time() * 1000)}.jsonl")
        output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        header = dict(self._session_manager.get_header() or {})
        header.update({
            "type": "session",
            "version": header.get("version", 3),
            "id": self._session_manager.get_session_id(),
            "timestamp": header.get("timestamp") or int(time.time() * 1000),
            "cwd": self._session_manager.get_cwd(),
        })

        lines = [json.dumps(header, ensure_ascii=False)]
        previous_id: str | None = None
        for entry in self._session_manager.get_branch():
            raw = dict(entry.data)
            raw["parentId"] = previous_id
            raw.pop("parent_id", None)
            lines.append(json.dumps(raw, ensure_ascii=False))
            previous_id = entry.id

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return output_path

    def get_last_assistant_text(self) -> str | None:
        """Get text of last assistant message (for /copy command)."""
        for m in reversed(self._agent.state.messages):
            if getattr(m, "role", "") != "assistant":
                continue
            stop = getattr(m, "stop_reason", "")
            content = getattr(m, "content", [])
            if stop == "aborted" and not content:
                continue
            text = "".join(
                getattr(c, "text", "")
                for c in content
                if getattr(c, "type", "") == "text"
            )
            return text.strip() or None
        return None

    # ── Additional session methods (mirrors TS) ───────────────────────────────

    def dispose(self) -> None:
        """Clean up session resources and event listeners."""
        self._extension_runner.invalidate()
        self._listeners = []

    def set_scoped_models(self, scoped_models: list[dict[str, Model | ThinkingLevel | None]]) -> None:
        """
        Set models available for cycling (Ctrl+P).
        Mirrors setScopedModels() in TypeScript.
        """
        self._scoped_models = scoped_models

    @property
    def scoped_models(self) -> list[dict[str, Model | ThinkingLevel | None]]:
        """Scoped models for cycling. Mirrors scopedModels in TypeScript."""
        return list(self._scoped_models or [])

    @property
    def has_pending_bash_messages(self) -> bool:
        """Whether bash messages are queued for the next prompt."""
        return bool(self._pending_bash_messages)

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-compaction."""
        self._settings_manager.set_compaction_enabled(enabled)

    @property
    def auto_compaction_enabled(self) -> bool:
        """Whether auto-compaction is currently enabled."""
        return self._settings_manager.get_compaction_settings().get("enabled", True)

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-retry."""
        self._settings_manager.set_retry_enabled(enabled)

    def set_steering_mode(self, mode: str) -> None:
        """Set runtime steering mode for RPC/interactive callers."""
        self._steering_mode_override = mode

    def set_follow_up_mode(self, mode: str) -> None:
        """Set runtime follow-up mode for RPC/interactive callers."""
        self._follow_up_mode_override = mode

    def login_api_key(self, provider: str, api_key: str) -> None:
        """Persist an API key for a provider."""
        cleaned_provider = provider.strip()
        cleaned_key = api_key.strip()
        if not cleaned_provider:
            raise ValueError("Provider is required")
        if not cleaned_key:
            raise ValueError("API key is required")
        self._auth_storage.set_api_key(cleaned_provider, cleaned_key)

    def logout_provider(self, provider: str) -> None:
        """Remove persisted credentials for a provider."""
        cleaned_provider = provider.strip()
        if not cleaned_provider:
            raise ValueError("Provider is required")
        self._auth_storage.logout(cleaned_provider)

    def get_project_trust(self) -> ProjectTrustDecision:
        """Return the persisted trust decision for this session cwd."""
        return ProjectTrustStore().get(self.cwd)

    def set_project_trust(self, trusted: ProjectTrustDecision) -> None:
        """Persist the project trust decision for this session cwd."""
        ProjectTrustStore().set(self.cwd, trusted)

    async def share_session(self) -> dict[str, str]:
        """
        Share the current session as a secret GitHub gist using the gh CLI.

        Returns a dict with gist_url, gist_id, and share_url.
        """
        session_file = self._session_manager.get_session_file()
        if not session_file or not os.path.exists(session_file):
            raise RuntimeError("No session file is available to share")
        gh_path = shutil.which("gh")
        if not gh_path:
            raise RuntimeError("GitHub CLI 'gh' is required for /share")

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            prefix="pi-session-",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            with open(session_file, encoding="utf-8", errors="replace") as src:
                tmp.write(src.read())

        try:
            process = await asyncio.create_subprocess_exec(
                gh_path,
                "gist",
                "create",
                "--public=false",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                message = stderr.decode("utf-8", errors="replace").strip() or "Unknown error"
                raise RuntimeError(f"Failed to create gist: {message}")
            gist_url = stdout.decode("utf-8", errors="replace").strip()
            gist_id = gist_url.rstrip("/").split("/")[-1] if gist_url else ""
            if not gist_id:
                raise RuntimeError("Failed to parse gist ID from gh output")
            return {
                "gist_url": gist_url,
                "gist_id": gist_id,
                "share_url": get_share_viewer_url(gist_id),
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @property
    def auto_retry_enabled(self) -> bool:
        """Whether auto-retry is currently enabled."""
        return self._settings_manager.get_retry_settings().get("enabled", True)

    async def bind_extensions(self, bindings: dict[str, Any]) -> None:
        """
        Bind extension UI context and handlers.
        Mirrors bindExtensions() in TypeScript.
        """
        self._extension_bindings = dict(bindings)
        self._bind_extension_context(bindings)
        command_context_actions = self._default_extension_command_context_actions()
        command_context_actions.update(
            bindings.get("commandContextActions")
            or bindings.get("command_context_actions")
            or {}
        )
        self._extension_runner.bind_command_context_actions(command_context_actions)
        await self._extension_runner.emit(self._session_start_event)
        reason = "reload" if self._session_start_event.get("reason") == "reload" else "startup"
        await self._extend_resources_from_extensions(reason)

    def _bind_extension_context(self, bindings: dict[str, Any]) -> None:
        shutdown_handler = (
            bindings.get("shutdownHandler")
            or bindings.get("shutdown_handler")
            or (lambda: None)
        )

        def _compact(options: Any = None) -> Any:
            custom_instructions = None
            if isinstance(options, dict):
                custom_instructions = (
                    options.get("customInstructions")
                    or options.get("custom_instructions")
                )
            elif isinstance(options, str):
                custom_instructions = options
            return self.compact(custom_instructions)

        actions = {
            "isIdle": lambda: not self._agent.state.is_streaming,
            "abort": lambda: self.abort(),
            "hasPendingMessages": self._has_pending_extension_messages,
            "shutdown": shutdown_handler,
            "getContextUsage": self.get_context_usage,
            "compact": _compact,
            "getSystemPrompt": lambda: self.system_prompt,
            "getRegisteredCommands": self._extension_runner.get_registered_commands,
            "getCommandDiagnostics": lambda: [],
        }
        values = {
            "ui": bindings.get("uiContext") or bindings.get("ui_context"),
            "uiContext": bindings.get("uiContext") or bindings.get("ui_context"),
            "mode": bindings.get("mode"),
            "hasUI": lambda: bool(bindings.get("uiContext") or bindings.get("ui_context")),
            "sessionManager": lambda: self._session_manager,
            "modelRegistry": lambda: self._model_registry,
            "model": lambda: self.model,
            "signal": bindings.get("signal"),
        }
        # Bind core actions into the shared extension runtime FIRST. bind_core()
        # resets the context bindings, so the bind_context_actions() call below
        # must come after it to restore ui/sessionManager/etc on ctx.
        core_actions = {
            "getActiveTools": self.get_active_tool_names,
            "getAllTools": self.get_all_tools,
            "getAllToolNames": self.get_all_tool_names,
            "setActiveTools": self.set_active_tools_by_name,
            "getCommands": self._get_extension_commands,
            "appendEntry": (
                lambda custom_type, data=None: self._session_manager.append_custom_entry(
                    custom_type, data
                )
            ),
        }
        self._extension_runner.bind_core(actions=core_actions)
        self._extension_runner.bind_context_actions(actions=actions, values=values)

    def _get_extension_commands(self) -> list[dict[str, Any]]:
        """
        Aggregate all slash commands (extension + prompt + skill) with their
        source. Backs pi.get_commands(). Mirrors getCommands() in TypeScript.
        """
        from pi_coding_agent.core.source_info import source_info_to_dict

        def _attr(obj: Any, name: str, default: Any = "") -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        def _source_dict(obj: Any) -> dict[str, Any] | None:
            si = _attr(obj, "source_info", None) or _attr(obj, "sourceInfo", None)
            try:
                return source_info_to_dict(si) if si else None
            except Exception:
                return None

        commands: list[dict[str, Any]] = []
        for cmd in self._extension_runner.get_registered_commands():
            commands.append({
                "name": getattr(cmd, "invocation_name", cmd.name),
                "description": cmd.description,
                "source": "extension",
                "sourceInfo": _source_dict(cmd),
            })

        loader = self._resource_loader
        if loader is not None:
            try:
                for prompt in (loader.get_prompts().get("prompts") or []):
                    commands.append({
                        "name": _attr(prompt, "name"),
                        "description": _attr(prompt, "description"),
                        "source": "prompt",
                        "sourceInfo": _source_dict(prompt),
                    })
            except Exception:
                pass
            try:
                for skill in (loader.get_skills().get("skills") or []):
                    commands.append({
                        "name": f"skill:{_attr(skill, 'name')}",
                        "description": _attr(skill, "description"),
                        "source": "skill",
                        "sourceInfo": _source_dict(skill),
                    })
            except Exception:
                pass
        return commands

    def _default_extension_command_context_actions(self) -> dict[str, Any]:
        return {
            "getSystemPromptOptions": lambda: {},
            "waitForIdle": lambda: self._agent.wait_for_idle(),
            "newSession": lambda opts=None: self.new_session(opts),
            "fork": lambda entry_id, opts=None: self.fork_session(entry_id),
            "navigateTree": lambda target_id, opts=None: self.navigate_tree(target_id, opts),
            "switchSession": lambda path, opts=None: self.switch_session(path),
            "reload": lambda: self.reload(),
        }

    def _has_pending_extension_messages(self) -> bool:
        return bool(
            self._pending_bash_messages
            or self._pending_next_turn_messages
            or self.pending_message_count
        )

    async def _extend_resources_from_extensions(self, reason: str) -> None:
        if not self._extension_runner.has_handlers("resources_discover"):
            return

        discovered = await self._extension_runner.emit_resources_discover(self.cwd, reason)
        skill_paths = discovered.get("skillPaths") or []
        prompt_paths = discovered.get("promptPaths") or []
        theme_paths = discovered.get("themePaths") or []
        if not skill_paths and not prompt_paths and not theme_paths:
            return

        if self._resource_loader is None:
            return
        extend_resources = getattr(self._resource_loader, "extend_resources", None)
        if not callable(extend_resources):
            return

        from .resource_loader import ResourceExtensionPaths

        paths = ResourceExtensionPaths(
            skill_paths=self._build_extension_resource_paths(skill_paths),
            prompt_paths=self._build_extension_resource_paths(prompt_paths),
            theme_paths=self._build_extension_resource_paths(theme_paths),
        )
        extend_resources(paths)
        self._base_system_prompt = self._build_system_prompt(self.get_active_tool_names())
        self._agent.set_system_prompt(self._base_system_prompt)

    def _build_extension_resource_paths(self, entries: list[Any]) -> list[dict[str, Any]]:
        paths: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, str):
                path = entry
                extension_path = None
            elif isinstance(entry, dict):
                path = entry.get("path")
                extension_path = entry.get("extensionPath") or entry.get("extension_path")
            else:
                path = getattr(entry, "path", None)
                extension_path = getattr(entry, "extensionPath", None) or getattr(entry, "extension_path", None)
            if not isinstance(path, str) or not path:
                continue
            metadata = {
                "source": "extension",
                "scope": "temporary",
                "origin": "top-level",
            }
            if isinstance(extension_path, str) and extension_path:
                metadata["baseDir"] = os.path.dirname(extension_path)
            paths.append({"path": path, "metadata": metadata})
        return paths

    async def reload(self) -> None:
        """
        Reload configuration and resources.
        Mirrors reload() in TypeScript.
        """
        previous_flag_values = self._extension_runner.get_flag_values()
        await self._extension_runner.shutdown("reload")
        self._extension_runner.invalidate()
        settings_reload = self._settings_manager.reload()
        if asyncio.iscoroutine(settings_reload):
            await settings_reload
        self._model_registry.reset_registered_providers()
        if self._resource_loader is not None:
            reload_fn = getattr(self._resource_loader, "reload", None)
            if callable(reload_fn):
                result = reload_fn()
                if asyncio.iscoroutine(result):
                    await result
            self._extension_runner = self._create_extension_runner(previous_flag_values)
            self._bind_extension_context(self._extension_bindings)
            if self._extension_bindings:
                command_context_actions = self._default_extension_command_context_actions()
                command_context_actions.update(
                    self._extension_bindings.get("commandContextActions")
                    or self._extension_bindings.get("command_context_actions")
                    or {}
                )
                self._extension_runner.bind_command_context_actions(command_context_actions)
                await self._extension_runner.emit({"type": "session_start", "reason": "reload"})
                await self._extend_resources_from_extensions("reload")

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        exclude_from_context: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a bash command and return the result.
        Mirrors executeBash() in TypeScript.
        
        Returns dict with:
            - output: str
            - exit_code: int
            - cancelled: bool
            - truncated: bool
            - full_output_path: str | None
        """
        from .bash_executor import execute_bash as execute_bash_command
        
        # Apply command prefix if configured
        prefix = self._settings_manager.get_shell_command_prefix()
        resolved_command = f"{prefix}\n{command}" if prefix else command

        cancel_event = asyncio.Event()
        self._bash_cancel_event = cancel_event
        try:
            bash_result = await execute_bash_command(
                resolved_command,
                on_chunk=on_chunk,
                cancel_event=cancel_event,
                cwd=self.cwd,
            )
            result = {
                "output": bash_result.output,
                "exit_code": bash_result.exit_code,
                "cancelled": bash_result.cancelled,
                "truncated": bash_result.truncated,
                "full_output_path": bash_result.full_output_path,
            }
        finally:
            self._bash_cancel_event = None

        self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
        
        return result

    def abort_bash(self) -> None:
        """
        Abort current bash execution.
        Mirrors abortBash() in TypeScript.
        """
        if self._bash_cancel_event is not None:
            self._bash_cancel_event.set()

    def set_session_name(self, name: str) -> None:
        """
        Set a human-readable name for the session.
        Mirrors setSessionName() in TypeScript.
        """
        self._session_manager.append_session_info(name)

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """
        Get all user messages suitable for forking.
        Mirrors getUserMessagesForForking() in TypeScript.
        
        Returns list of dicts with:
            - entry_id: str
            - text: str
        """
        entries = self._session_manager.get_entries()
        result: list[dict[str, str]] = []
        
        for entry in entries:
            if entry.type != "message":
                continue
            msg_data = entry.data.get("message", {})
            if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                text = self._extract_user_message_text(msg_data.get("content", ""))
                if text:
                    result.append({"entry_id": entry.id, "text": text})
        
        return result

    def _extract_user_message_text(self, content: str | list[dict] | Any) -> str:
        """Extract text from user message content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        return ""

    def has_extension_handlers(self, event_type: str) -> bool:
        """
        Check if any extensions handle the given event type.
        Mirrors hasExtensionHandlers() in TypeScript.
        """
        return self._extension_runner.has_handlers(event_type)

    def create_replaced_session_context(self) -> Any:
        """Create the context shape passed after session replacement."""
        context = self._extension_runner.create_command_context()
        setattr(context, "sendMessage", self.send_custom_message)
        setattr(context, "sendUserMessage", self.send_user_message)
        return context

    @property
    def extension_runner(self) -> ExtensionRunner:
        return self._extension_runner

    # Node/TypeScript public API compatibility aliases.
    thinkingLevel = thinking_level
    isStreaming = is_streaming
    isCompacting = is_compacting
    isRetrying = is_retrying
    isBashRunning = is_bash_running
    retryAttempt = retry_attempt
    modelRegistry = model_registry
    authStorage = auth_storage
    resourceLoader = resource_loader
    promptTemplates = prompt_templates
    sessionManager = session_manager
    settingsManager = settings_manager
    sessionFile = session_file
    sessionName = session_name
    steeringMode = steering_mode
    followUpMode = follow_up_mode
    systemPrompt = system_prompt
    pendingMessageCount = pending_message_count
    autoCompactionEnabled = auto_compaction_enabled
    autoRetryEnabled = auto_retry_enabled
    extensionRunner = extension_runner
    sessionId = property(lambda self: self.session_id)
    scopedModels = scoped_models
    hasPendingBashMessages = has_pending_bash_messages

    followUp = follow_up
    waitForIdle = wait_for_idle
    switchSession = switch_session
    newSession = new_session
    cloneSession = clone_session
    forkSession = fork_session
    getSessionTreeEntries = get_session_tree_entries
    navigateTree = navigate_tree
    getSteeringMessages = get_steering_messages
    getFollowUpMessages = get_follow_up_messages
    clearQueue = clear_queue
    getActiveToolNames = get_active_tool_names
    getAllTools = get_all_tools
    getToolDefinition = get_tool_definition
    getAllToolNames = get_all_tool_names
    setActiveToolsByName = set_active_tools_by_name
    setModel = set_model
    cycleModel = cycle_model
    getAvailableThinkingLevels = get_available_thinking_levels
    supportsThinking = supports_thinking
    setThinkingLevel = set_thinking_level
    cycleThinkingLevel = cycle_thinking_level
    getSessionStats = get_session_stats
    getContextUsage = get_context_usage
    getSessionInfo = get_session_info
    sendCustomMessage = send_custom_message
    sendUserMessage = send_user_message
    abortCompaction = abort_compaction
    abortBranchSummary = abort_branch_summary
    abortRetry = abort_retry
    exportToHtml = export_to_html
    exportToJsonl = export_to_jsonl
    getLastAssistantText = get_last_assistant_text
    setScopedModels = set_scoped_models
    setAutoCompactionEnabled = set_auto_compaction_enabled
    setAutoRetryEnabled = set_auto_retry_enabled
    setSteeringMode = set_steering_mode
    setFollowUpMode = set_follow_up_mode
    loginApiKey = login_api_key
    logoutProvider = logout_provider
    getProjectTrust = get_project_trust
    setProjectTrust = set_project_trust
    shareSession = share_session
    bindExtensions = bind_extensions
    executeBash = execute_bash
    recordBashResult = record_bash_result
    abortBash = abort_bash
    setSessionName = set_session_name
    getUserMessagesForForking = get_user_messages_for_forking
    createReplacedSessionContext = create_replaced_session_context
    hasExtensionHandlers = has_extension_handlers


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp_thinking_level(level: ThinkingLevel, available: list[ThinkingLevel]) -> ThinkingLevel:
    """Clamp a thinking level to the closest available level."""
    ordered = _THINKING_LEVELS_WITH_XHIGH
    avail_set = set(available)
    idx = ordered.index(level) if level in ordered else 0
    for i in range(idx, len(ordered)):
        if ordered[i] in avail_set:
            return ordered[i]
    for i in range(idx - 1, -1, -1):
        if ordered[i] in avail_set:
            return ordered[i]
    return available[0] if available else "off"


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a message to a dict for persistence."""
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if dataclasses.is_dataclass(msg):
        return dataclasses.asdict(msg)
    return {"role": getattr(msg, "role", "unknown")}
