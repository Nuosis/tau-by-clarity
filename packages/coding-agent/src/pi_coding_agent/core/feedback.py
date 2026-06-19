"""Native feedback issue submission for Tau."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal


FeedbackType = Literal["bug", "feature"]
GITHUB_OWNER = "Nuosis"
GITHUB_REPO = "tau-by-clarity"
GITHUB_ISSUES_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
GITHUB_ISSUES_WEB = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
MAX_SESSION_CHARS = 12000


@dataclass(frozen=True)
class FeedbackRequest:
    feedback_type: FeedbackType
    include_session: bool
    issue: str
    expected: str | None = None
    session_snapshot: str | None = None


@dataclass(frozen=True)
class SubmittedIssue:
    number: int | None
    url: str


class FeedbackError(RuntimeError):
    pass


def normalize_feedback_type(value: str) -> FeedbackType:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"bug", "bug-report", "defect"}:
        return "bug"
    if normalized in {"feature", "feature-request", "request"}:
        return "feature"
    raise FeedbackError("Feedback type must be bug or feature request.")


def parse_yes_no(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes", "true", "1", "include"}:
        return True
    if normalized in {"n", "no", "false", "0", "skip"}:
        return False
    raise FeedbackError("Include session must be yes or no.")


def parse_feedback_args(args: str) -> FeedbackRequest | None:
    args = (args or "").strip()
    if not args:
        return None
    parts = shlex.split(args)
    if len(parts) < 3:
        raise FeedbackError('Usage: /feedback <bug|feature> <yes|no> "<issue/request>" ["expected behavior"]')
    feedback_type = normalize_feedback_type(parts[0])
    include_session = parse_yes_no(parts[1])
    issue = parts[2].strip()
    if not issue:
        raise FeedbackError("Issue/request cannot be empty.")
    expected = " ".join(parts[3:]).strip() if len(parts) > 3 else None
    if feedback_type == "bug" and not expected:
        raise FeedbackError('Bug feedback requires "what would you like to have happened instead".')
    return FeedbackRequest(
        feedback_type=feedback_type,
        include_session=include_session,
        issue=issue,
        expected=expected or None,
    )


def github_token_from_env() -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        output = subprocess.check_output(
            ["gh", "auth", "token"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    return output or None


def collect_session_snapshot(session: Any, *, max_chars: int = MAX_SESSION_CHARS) -> str:
    session_manager = getattr(session, "session_manager", None) or getattr(session, "sessionManager", None)
    session_id = getattr(session, "session_id", None) or getattr(session, "sessionId", None)
    cwd = getattr(session, "cwd", None)
    header: Any = None
    leaf_id: str | None = None
    messages: Any = None

    if session_manager is not None:
        get_session_id = getattr(session_manager, "get_session_id", None) or getattr(session_manager, "getSessionId", None)
        get_cwd = getattr(session_manager, "get_cwd", None) or getattr(session_manager, "getCwd", None)
        get_header = getattr(session_manager, "get_header", None) or getattr(session_manager, "getHeader", None)
        get_leaf_id = getattr(session_manager, "get_leaf_id", None) or getattr(session_manager, "getLeafId", None)
        get_messages = getattr(session_manager, "get_messages", None) or getattr(session_manager, "getMessages", None)
        if callable(get_session_id):
            session_id = get_session_id()
        if callable(get_cwd):
            cwd = get_cwd()
        if callable(get_header):
            header = get_header()
        if callable(get_leaf_id):
            leaf_id = get_leaf_id()
        if callable(get_messages):
            messages = get_messages()

    if messages is None:
        get_messages = getattr(session, "get_messages", None) or getattr(session, "getMessages", None)
        if callable(get_messages):
            messages = get_messages()
        else:
            messages = getattr(session, "messages", [])

    payload = {
        "session_id": session_id,
        "cwd": cwd,
        "leaf_id": leaf_id,
        "header": header,
        "messages": messages or [],
    }
    text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... truncated {omitted} chars ..."


def _issue_preview(text: str, *, max_chars: int = 72) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def build_github_issue_payload(request: FeedbackRequest) -> dict[str, Any]:
    label = "Bug" if request.feedback_type == "bug" else "Feature request"
    title = f"[{label}] {_issue_preview(request.issue)}"
    lines = [
        f"## Type\n{label}",
        f"## Issue/request\n{request.issue.strip()}",
    ]
    if request.feedback_type == "bug":
        lines.append(f"## Expected behavior\n{(request.expected or '').strip()}")
    if request.include_session:
        snapshot = request.session_snapshot or "Session was requested but no session snapshot was available."
        lines.append(f"## Session\n```json\n{snapshot}\n```")
    else:
        lines.append("## Session\nNot included.")
    lines.append(f"## Environment\nTau feedback command on {platform.platform()}")
    return {
        "title": title,
        "body": "\n\n".join(lines),
        "labels": ["bug" if request.feedback_type == "bug" else "enhancement"],
    }


def submit_github_issue(
    request: FeedbackRequest,
    *,
    token: str | None = None,
    opener: Any | None = None,
) -> SubmittedIssue:
    token = token or github_token_from_env()
    if not token:
        raise FeedbackError(
            "Missing GitHub token. Set GITHUB_TOKEN/GH_TOKEN or run `gh auth login` "
            "with permission to create issues."
        )

    payload = json.dumps(build_github_issue_payload(request)).encode("utf-8")
    http_request = urllib.request.Request(
        GITHUB_ISSUES_API,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "tau-feedback",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(http_request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise FeedbackError(f"GitHub issue submission failed ({exc.code}): {body}") from exc
    except Exception as exc:
        raise FeedbackError(f"GitHub issue submission failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    url = data.get("html_url") or GITHUB_ISSUES_WEB
    number = data.get("number") if isinstance(data.get("number"), int) else None
    return SubmittedIssue(number=number, url=str(url))
