"""
Tests for cli_sub/ subpackage.

Covers: args.py, file_processor.py, list_models.py, session_picker.py
"""
from __future__ import annotations

import os
import tempfile

import pytest


# ============================================================================
# Args parsing
# ============================================================================

class TestArgParsing:
    def test_parse_empty(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args([])
        assert args.messages == []
        assert args.file_args == []
        assert args.provider is None

    def test_parse_messages(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["Hello", "world"])
        assert args.messages == ["Hello", "world"]

    def test_parse_session_vars_from_generic_channels(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--agent-context", "devflow", "--var", "ACTIVE_PATH=/repo/app"])
        assert args.session_vars == {
            "AGENT_CONTEXT": "devflow",
            "ACTIVE_PATH": "/repo/app",
        }
        assert args.messages == []

    def test_parse_session_vars_from_key_value_positional(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["ACTIVE_PATH=/repo/app", "Run the checks"])
        assert args.session_vars == {"ACTIVE_PATH": "/repo/app"}
        assert args.messages == ["Run the checks"]

    def test_parse_provider(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--provider", "anthropic"])
        assert args.provider == "anthropic"

    def test_parse_model(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--model", "claude-3-5-sonnet"])
        assert args.model == "claude-3-5-sonnet"

    def test_parse_mode(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--mode", "rpc"])
        assert args.mode == "rpc"

    def test_parse_invalid_mode(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--mode", "invalid"])
        assert args.mode is None

    def test_parse_help(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--help"])
        assert args.help is True

    def test_parse_version(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-v"])
        assert args.version is True

    def test_parse_continue(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-c"])
        assert args.continue_ is True

    def test_parse_resume(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-r"])
        assert args.resume is True

    def test_parse_print(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-p"])
        assert args.print_mode is True

    def test_parse_print_consumes_following_prompt_like_node(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--print", "Summarize this"])
        assert args.print_mode is True
        assert args.messages == ["Summarize this"]

        dash_prompt = parse_args(["-p", "--- prompt starts with dashes"])
        assert dash_prompt.messages == ["--- prompt starts with dashes"]

    def test_parse_file_args(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["@file.txt", "@image.png"])
        assert "file.txt" in args.file_args
        assert "image.png" in args.file_args

    def test_parse_no_tools(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--no-tools"])
        assert args.no_tools is True

    def test_parse_tools(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--tools", "read,bash"])
        assert args.tools == ["read", "bash"]

    def test_parse_node_tool_flags(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-nbt", "-t", "read,bash", "-xt", "bash,edit"])
        assert args.no_builtin_tools is True
        assert args.tools == ["read", "bash"]
        assert args.exclude_tools == ["bash", "edit"]

    def test_parse_invalid_tool_skipped(self, capsys):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--tools", "read,nonexistent"])
        assert "nonexistent" not in (args.tools or [])
        assert "read" in (args.tools or [])

    def test_parse_thinking_level(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--thinking", "high"])
        assert args.thinking == "high"

    def test_parse_invalid_thinking_level(self, capsys):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--thinking", "invalid"])
        assert args.thinking is None

    def test_parse_no_session(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--no-session"])
        assert args.no_session is True

    def test_parse_session_path(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--session", "/path/to/session.jsonl"])
        assert args.session == "/path/to/session.jsonl"

    def test_parse_node_session_and_trust_flags(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args([
            "--name", "Planning",
            "--session-id", "session-123",
            "--fork", "abc",
            "--no-context-files",
            "--approve",
            "--offline",
        ])
        assert args.name == "Planning"
        assert args.session_id == "session-123"
        assert args.fork == "abc"
        assert args.no_context_files is True
        assert args.project_trust_override is True
        assert args.offline is True

        denied = parse_args(["--no-approve"])
        assert denied.project_trust_override is False

    def test_parse_repeated_append_prompt_and_unknown_flags(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args([
            "--append-system-prompt", "one",
            "--append-system-prompt", "two",
            "--foo=bar",
            "--baz", "qux",
            "-z",
        ])
        assert args.append_system_prompt == ["one", "two"]
        assert args.unknown_flags == {"foo": "bar", "baz": "qux"}
        assert args.diagnostics == [{"type": "error", "message": "Unknown option: -z"}]

    def test_parse_multiple_extensions(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["-e", "ext1.py", "-e", "ext2.py"])
        assert args.extensions == ["ext1.py", "ext2.py"]

    def test_parse_list_models_flag_only(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--list-models"])
        assert args.list_models is True

    def test_parse_list_models_with_search(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--list-models", "sonnet"])
        assert args.list_models == "sonnet"

    def test_parse_verbose(self):
        from pi_coding_agent.cli_sub.args import parse_args
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_is_valid_thinking_level(self):
        from pi_coding_agent.cli_sub.args import is_valid_thinking_level
        assert is_valid_thinking_level("high") is True
        assert is_valid_thinking_level("invalid") is False
        assert is_valid_thinking_level("off") is True

    def test_print_help_runs(self, capsys):
        from pi_coding_agent.cli_sub.args import print_help
        print_help()
        captured = capsys.readouterr()
        assert "pi" in captured.out.lower() or "usage" in captured.out.lower()
        assert "uninstall <source> [-l]" in captured.out
        assert "update [source|self|pi]" in captured.out
        assert "config [--no-approve]" in captured.out
        assert "Show help for install/remove/uninstall/update/list" in captured.out
        assert "--session <path|id>" in captured.out
        assert "--offline" in captured.out
        assert "same as PI_OFFLINE=1" in captured.out
        assert "default: ~/.pi-py/agent" in captured.out
        assert "~/..pi/agent" not in captured.out

    def test_root_exports_parse_args_and_config_paths(self):
        import pi_coding_agent

        assert pi_coding_agent.parseArgs is pi_coding_agent.parse_args
        parsed = pi_coding_agent.parseArgs(["--mode", "rpc"])
        assert parsed.mode == "rpc"
        assert pi_coding_agent.getAgentDir is pi_coding_agent.get_agent_dir
        assert pi_coding_agent.getDocsPath is pi_coding_agent.get_docs_path
        assert pi_coding_agent.getExamplesPath is pi_coding_agent.get_examples_path
        assert pi_coding_agent.getPackageDir is pi_coding_agent.get_package_dir
        assert pi_coding_agent.getReadmePath is pi_coding_agent.get_readme_path
        assert pi_coding_agent.ENV_SESSION_DIR == "PI_CODING_AGENT_SESSION_DIR"

    @pytest.mark.asyncio
    async def test_resource_loader_can_disable_context_files(self, tmp_path):
        from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions

        agent_dir = tmp_path / "agent"
        project = tmp_path / "project"
        agent_dir.mkdir()
        project.mkdir()
        (agent_dir / "AGENTS.md").write_text("global context", encoding="utf-8")
        (project / "AGENTS.md").write_text("project context", encoding="utf-8")

        loader = DefaultResourceLoader(
            DefaultResourceLoaderOptions(
                cwd=str(project),
                agent_dir=str(agent_dir),
                no_context_files=True,
                no_extensions=True,
                no_skills=True,
                no_prompt_templates=True,
                no_themes=True,
            )
        )
        await loader.reload()

        assert loader.get_agents_files()["agentsFiles"] == []

    def test_session_manager_can_create_exact_session_id(self, tmp_path):
        from pi_coding_agent.core.session_manager import SessionManager

        manager = SessionManager.create(str(tmp_path), session_id="fixed-session")

        assert manager.get_session_id() == "fixed-session"
        assert manager.get_session_file().endswith("fixed-session.jsonl")


# ============================================================================
# File processor
# ============================================================================

class TestFileProcessor:
    @pytest.mark.asyncio
    async def test_process_empty_file_args(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        result = await process_file_arguments([])
        assert result.text == ""
        assert result.images == []

    @pytest.mark.asyncio
    async def test_process_text_file(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello, world!")
            path = f.name
        try:
            result = await process_file_arguments([path])
            assert "Hello, world!" in result.text
            assert path in result.text
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_process_empty_file_skipped(self):
        from pi_coding_agent.cli_sub.file_processor import process_file_arguments
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            path = f.name
        try:
            result = await process_file_arguments([path])
            assert result.text == ""
        finally:
            os.unlink(path)


# ============================================================================
# List models
# ============================================================================

class TestListModels:
    @pytest.mark.asyncio
    async def test_list_models_empty_registry(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockRegistry:
            async def get_available(self):
                return []

        await list_models(MockRegistry())
        captured = capsys.readouterr()
        assert "No models" in captured.out

    @pytest.mark.asyncio
    async def test_list_models_with_data(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockModel:
            provider = "anthropic"
            id = "claude-3-5-sonnet"
            contextWindow = 200000
            maxTokens = 8192
            reasoning = False
            input = ["text", "image"]

        class MockRegistry:
            async def get_available(self):
                return [MockModel()]

        await list_models(MockRegistry())
        captured = capsys.readouterr()
        assert "anthropic" in captured.out
        assert "claude-3-5-sonnet" in captured.out

    @pytest.mark.asyncio
    async def test_list_models_with_search(self, capsys):
        from pi_coding_agent.cli_sub.list_models import list_models

        class MockModel:
            provider = "anthropic"
            id = "claude-3-5-haiku"
            contextWindow = 200000
            maxTokens = 4096
            reasoning = False
            input = ["text"]

        class MockModel2:
            provider = "openai"
            id = "gpt-5.4-nano"
            contextWindow = 128000
            maxTokens = 4096
            reasoning = False
            input = ["text", "image"]

        class MockRegistry:
            async def get_available(self):
                return [MockModel(), MockModel2()]

        await list_models(MockRegistry(), search_pattern="haiku")
        captured = capsys.readouterr()
        assert "haiku" in captured.out
        assert "gpt-5.4-nano" not in captured.out

    def test_format_token_count(self):
        from pi_coding_agent.cli_sub.list_models import _format_token_count
        assert _format_token_count(200000) == "200K"
        assert _format_token_count(1000000) == "1M"
        assert _format_token_count(500) == "500"
        assert _format_token_count(1500000) == "1.5M"
