"""Session cwd validation helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionCwdIssue:
    session_cwd: str
    fallback_cwd: str
    session_file: str | None = None


def _call_or_attr(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    for name in (snake, camel):
        value = getattr(obj, name, None)
        if callable(value):
            return value()
        if value is not None:
            return value
    if isinstance(obj, dict):
        return obj.get(snake, obj.get(camel, default))
    return default


def get_missing_session_cwd_issue(session_manager: Any, fallback_cwd: str) -> SessionCwdIssue | None:
    session_file = _call_or_attr(session_manager, "get_session_file", "getSessionFile")
    if not session_file:
        return None
    session_cwd = _call_or_attr(session_manager, "get_cwd", "getCwd", "")
    if not session_cwd or os.path.exists(str(session_cwd)):
        return None
    return SessionCwdIssue(
        session_file=str(session_file),
        session_cwd=str(session_cwd),
        fallback_cwd=fallback_cwd,
    )


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    session_file = f"\nSession file: {issue.session_file}" if issue.session_file else ""
    return (
        f"Stored session working directory does not exist: {issue.session_cwd}"
        f"{session_file}\nCurrent working directory: {issue.fallback_cwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    return f"cwd from session file does not exist\n{issue.session_cwd}\n\ncontinue in current cwd\n{issue.fallback_cwd}"


class MissingSessionCwdError(RuntimeError):
    def __init__(self, issue: SessionCwdIssue) -> None:
        self.issue = issue
        super().__init__(format_missing_session_cwd_error(issue))


def assert_session_cwd_exists(session_manager: Any, fallback_cwd: str) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue is not None:
        raise MissingSessionCwdError(issue)


__all__ = [
    "MissingSessionCwdError",
    "SessionCwdIssue",
    "assert_session_cwd_exists",
    "format_missing_session_cwd_error",
    "format_missing_session_cwd_prompt",
    "get_missing_session_cwd_issue",
]
