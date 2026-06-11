"""Runtime owner for an AgentSession and its cwd-bound services."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.agent_session_services import (
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
)
from pi_coding_agent.core.session_cwd import assert_session_cwd_exists
from pi_coding_agent.core.session_manager import SessionManager


class SessionImportFileNotFoundError(FileNotFoundError):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")


@dataclass
class CreateAgentSessionRuntimeResult:
    session: AgentSession
    services: AgentSessionServices
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)
    extensions_result: dict[str, Any] | None = None
    model_fallback_message: str | None = None


CreateAgentSessionRuntimeFactory = Callable[[dict[str, Any]], Awaitable[CreateAgentSessionRuntimeResult]]


def _extract_user_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class AgentSessionRuntime:
    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        create_runtime: CreateAgentSessionRuntimeFactory,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
        model_fallback_message: str | None = None,
    ) -> None:
        self._session = session
        self._services = services
        self._create_runtime = create_runtime
        self._diagnostics = list(diagnostics or [])
        self._model_fallback_message = model_fallback_message
        self._rebind_session: Callable[[AgentSession], Awaitable[None]] | None = None
        self._before_session_invalidate: Callable[[], None] | None = None

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def cwd(self) -> str:
        return self._services.cwd

    @property
    def diagnostics(self) -> list[AgentSessionRuntimeDiagnostic]:
        return self._diagnostics

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    def set_rebind_session(self, rebind_session: Callable[[AgentSession], Awaitable[None]] | None = None) -> None:
        self._rebind_session = rebind_session

    def set_before_session_invalidate(self, before_session_invalidate: Callable[[], None] | None = None) -> None:
        self._before_session_invalidate = before_session_invalidate

    async def _emit_before_switch(
        self,
        reason: str,
        target_session_file: str | None = None,
    ) -> dict[str, bool]:
        runner = self._session.extension_runner
        if not runner.has_handlers("session_before_switch"):
            return {"cancelled": False}
        result = await runner.emit({
            "type": "session_before_switch",
            "reason": reason,
            "targetSessionFile": target_session_file,
        })
        return {"cancelled": bool(result and result.get("cancel") is True)}

    async def _emit_before_fork(self, entry_id: str, position: str) -> dict[str, bool]:
        runner = self._session.extension_runner
        if not runner.has_handlers("session_before_fork"):
            return {"cancelled": False}
        result = await runner.emit({
            "type": "session_before_fork",
            "entryId": entry_id,
            "position": position,
        })
        return {"cancelled": bool(result and result.get("cancel") is True)}

    async def _teardown_current(self, reason: str, target_session_file: str | None = None) -> None:
        runner = self._session.extension_runner
        if runner.has_handlers("session_shutdown"):
            await runner.emit({
                "type": "session_shutdown",
                "reason": reason,
                "targetSessionFile": target_session_file,
            })
        if self._before_session_invalidate:
            self._before_session_invalidate()
        self._session.dispose()

    def _apply(self, result: CreateAgentSessionRuntimeResult) -> None:
        self._session = result.session
        self._services = result.services
        self._diagnostics = list(result.diagnostics)
        self._model_fallback_message = result.model_fallback_message

    async def _finish_replacement(self, with_session: Callable[[AgentSession], Awaitable[None]] | None = None) -> None:
        if self._rebind_session:
            await self._rebind_session(self._session)
        if with_session:
            await with_session(self._session)

    async def switch_session(
        self,
        session_path: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        opts = options or {}
        before_result = await self._emit_before_switch("resume", session_path)
        if before_result["cancelled"]:
            return before_result
        session_manager = SessionManager.open(session_path)
        assert_session_cwd_exists(session_manager, self.cwd)
        previous_session_file = self._session.session_file
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(await self._create_runtime({
            "cwd": session_manager.get_cwd(),
            "agent_dir": self._services.agent_dir,
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "resume",
                "previousSessionFile": previous_session_file,
            },
        }))
        await self._finish_replacement(opts.get("withSession") or opts.get("with_session"))
        return {"cancelled": False}

    async def new_session(self, options: dict[str, Any] | None = None) -> dict[str, bool]:
        opts = options or {}
        before_result = await self._emit_before_switch("new")
        if before_result["cancelled"]:
            return before_result
        previous_session_file = self._session.session_file
        session_manager = SessionManager.create(
            self.cwd,
            session_dir=self._session.session_manager.get_session_dir(),
            parent_session=opts.get("parentSession") or opts.get("parent_session"),
        )
        setup = opts.get("setup")
        if callable(setup):
            maybe = setup(session_manager)
            if hasattr(maybe, "__await__"):
                await maybe
        await self._teardown_current("new", session_manager.get_session_file())
        self._apply(await self._create_runtime({
            "cwd": self.cwd,
            "agent_dir": self._services.agent_dir,
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "new",
                "previousSessionFile": previous_session_file,
            },
        }))
        await self._finish_replacement(opts.get("withSession") or opts.get("with_session"))
        return {"cancelled": False}

    async def fork(self, entry_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = options or {}
        position = opts.get("position") or "before"
        before_result = await self._emit_before_fork(entry_id, position)
        if before_result["cancelled"]:
            return {"cancelled": True}
        entry = self._session.session_manager.get_entry(entry_id)
        if not entry:
            raise ValueError("Invalid entry ID for forking")
        msg_data = entry.data.get("message", {}) if hasattr(entry, "data") else {}
        selected_text = ""
        if isinstance(msg_data, dict) and msg_data.get("role") == "user":
            selected_text = _extract_user_message_text(msg_data.get("content"))
        branch_point = entry.id if position == "at" else entry.parent_id
        previous_session_file = self._session.session_file
        session_manager = self._session.session_manager.branch(branch_point, self.cwd)
        await self._teardown_current("fork", session_manager.get_session_file())
        self._apply(await self._create_runtime({
            "cwd": self.cwd,
            "agent_dir": self._services.agent_dir,
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "fork",
                "previousSessionFile": previous_session_file,
            },
        }))
        await self._finish_replacement(opts.get("withSession") or opts.get("with_session"))
        return {"cancelled": False, "selectedText": selected_text}

    async def clone(self) -> dict[str, bool]:
        current_file = self._session.session_file
        if not current_file:
            return await self.new_session()
        previous_session_file = current_file
        session_manager = SessionManager.fork_from(
            current_file,
            self.cwd,
            self._session.session_manager.get_session_dir(),
        )
        await self._teardown_current("fork", session_manager.get_session_file())
        self._apply(await self._create_runtime({
            "cwd": self.cwd,
            "agent_dir": self._services.agent_dir,
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "fork",
                "previousSessionFile": previous_session_file,
            },
        }))
        await self._finish_replacement()
        return {"cancelled": False}

    async def navigate_tree(self, target_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = options or {}
        return await self._session.navigate_tree(
            target_id,
            summarize=bool(opts.get("summarize", False)),
        )

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, bool]:
        resolved_path = os.path.abspath(os.path.expanduser(input_path))
        if not os.path.exists(resolved_path):
            raise SessionImportFileNotFoundError(resolved_path)
        session_dir = self._session.session_manager.get_session_dir()
        os.makedirs(session_dir, exist_ok=True)
        destination = os.path.join(session_dir, os.path.basename(resolved_path))
        if os.path.abspath(destination) != resolved_path:
            shutil.copyfile(resolved_path, destination)
        session_manager = SessionManager.open(destination)
        if cwd_override:
            session_manager._cwd = cwd_override
            if session_manager._header is not None:
                session_manager._header["cwd"] = cwd_override
        assert_session_cwd_exists(session_manager, self.cwd)
        previous_session_file = self._session.session_file
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(await self._create_runtime({
            "cwd": session_manager.get_cwd(),
            "agent_dir": self._services.agent_dir,
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "resume",
                "previousSessionFile": previous_session_file,
            },
        }))
        await self._finish_replacement()
        return {"cancelled": False}

    async def dispose(self) -> None:
        await self._teardown_current("quit")

    switchSession = switch_session
    newSession = new_session
    navigateTree = navigate_tree
    importFromJsonl = import_from_jsonl


async def create_agent_session_runtime(
    create_runtime: CreateAgentSessionRuntimeFactory,
    options: dict[str, Any],
) -> AgentSessionRuntime:
    session_manager = options["session_manager"]
    assert_session_cwd_exists(session_manager, options["cwd"])
    result = await create_runtime(options)
    return AgentSessionRuntime(
        result.session,
        result.services,
        create_runtime,
        result.diagnostics,
        result.model_fallback_message,
    )


__all__ = [
    "AgentSessionRuntime",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "SessionImportFileNotFoundError",
    "create_agent_session_runtime",
]
