import json

import pytest


def test_parse_feedback_args_requires_expected_for_bug():
    from pi_coding_agent.core.feedback import FeedbackError, parse_feedback_args

    with pytest.raises(FeedbackError, match="Bug feedback requires"):
        parse_feedback_args('bug yes "It crashed"')


def test_parse_feedback_args_accepts_feature_request():
    from pi_coding_agent.core.feedback import parse_feedback_args

    request = parse_feedback_args('feature no "Add keyboard macros"')

    assert request is not None
    assert request.feedback_type == "feature"
    assert request.include_session is False
    assert request.issue == "Add keyboard macros"
    assert request.expected is None


def test_build_github_issue_payload_includes_expected_and_session():
    from pi_coding_agent.core.feedback import FeedbackRequest, build_github_issue_payload

    payload = build_github_issue_payload(
        FeedbackRequest(
            feedback_type="bug",
            include_session=True,
            issue="The editor submitted a blank message",
            expected="It should have kept focus in the editor",
            session_snapshot='{"session_id":"abc"}',
        )
    )

    assert payload["title"] == "[Bug] The editor submitted a blank message"
    assert payload["labels"] == ["bug"]
    assert "## Expected behavior\nIt should have kept focus in the editor" in payload["body"]
    assert '```json\n{"session_id":"abc"}\n```' in payload["body"]


def test_collect_session_snapshot_uses_session_manager_messages():
    from pi_coding_agent.core.feedback import collect_session_snapshot

    class Manager:
        def get_session_id(self):
            return "sess-1"

        def get_cwd(self):
            return "/tmp/project"

        def get_header(self):
            return {"id": "sess-1", "cwd": "/tmp/project"}

        def get_leaf_id(self):
            return "leaf-1"

        def get_messages(self):
            return [{"role": "user", "content": "hello"}]

    class Session:
        session_manager = Manager()

    snapshot = json.loads(collect_session_snapshot(Session()))

    assert snapshot["session_id"] == "sess-1"
    assert snapshot["cwd"] == "/tmp/project"
    assert snapshot["leaf_id"] == "leaf-1"
    assert snapshot["messages"] == [{"role": "user", "content": "hello"}]


def test_github_token_falls_back_to_gh_cli(monkeypatch):
    from pi_coding_agent.core import feedback

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(feedback.subprocess, "check_output", lambda *args, **kwargs: "ghs_from_cli\n")

    assert feedback.github_token_from_env() == "ghs_from_cli"


def test_submit_github_issue_posts_to_tau_repo_without_live_network():
    from pi_coding_agent.core.feedback import FeedbackRequest, submit_github_issue

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"number": 42, "html_url": "https://github.com/Nuosis/tau-by-clarity/issues/42"}'

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    issue = submit_github_issue(
        FeedbackRequest(
            feedback_type="feature",
            include_session=False,
            issue="Add a command palette",
        ),
        token="ghp_test",
        opener=opener,
    )

    assert issue.number == 42
    assert issue.url.endswith("/issues/42")
    assert captured["url"] == "https://api.github.com/repos/Nuosis/tau-by-clarity/issues"
    assert captured["method"] == "POST"
    assert captured["headers"]["Authorization"] == "Bearer ghp_test"
    assert captured["body"]["labels"] == ["enhancement"]
    assert captured["body"]["title"] == "[Feature request] Add a command palette"
    assert captured["timeout"] == 20
