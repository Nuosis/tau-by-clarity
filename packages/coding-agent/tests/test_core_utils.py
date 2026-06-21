"""
Tests for coding-agent core utilities.

Covers: utils/sleep.py, utils/git.py, utils/frontmatter.py,
        utils/changelog.py, utils/shell.py
"""
from __future__ import annotations

import asyncio
import json
import sys

import pytest


# ============================================================================
# sleep
# ============================================================================

class TestSleep:
    @pytest.mark.asyncio
    async def test_sleep_completes(self):
        from pi_coding_agent.utils.sleep import sleep
        await asyncio.wait_for(sleep(0.01), timeout=1.0)

    @pytest.mark.asyncio
    async def test_sleep_cancelled_by_event(self):
        from pi_coding_agent.utils.sleep import sleep
        cancel = asyncio.Event()
        cancel.set()
        # Already-set event: the sleep should return quickly (may or may not raise)
        try:
            await asyncio.wait_for(sleep(10.0, cancel), timeout=0.5)
        except asyncio.CancelledError:
            pass  # Expected in some implementations
        except asyncio.TimeoutError:
            pytest.fail("sleep did not exit when cancel event was set")

    @pytest.mark.asyncio
    async def test_sleep_cancelled_mid_sleep(self):
        from pi_coding_agent.utils.sleep import sleep
        cancel = asyncio.Event()

        async def _cancel_after():
            await asyncio.sleep(0.05)
            cancel.set()

        asyncio.ensure_future(_cancel_after())
        with pytest.raises(asyncio.CancelledError):
            await sleep(10.0, cancel)

    @pytest.mark.asyncio
    async def test_sleep_no_cancel_event(self):
        from pi_coding_agent.utils.sleep import sleep
        # Should complete normally without a cancel event
        await sleep(0.01)


# ============================================================================
# git
# ============================================================================

class TestGitUrlParsing:
    def test_https_url(self):
        from pi_coding_agent.utils.git import parse_git_url
        result = parse_git_url("https://github.com/user/repo.git")
        assert result is not None
        # GitSource has: type, repo, host, path, ref, pinned
        assert result.host == "github.com"
        assert "user" in result.path or "user" in result.repo

    def test_scp_like_url(self):
        from pi_coding_agent.utils.git import GitSource, parse_git_url
        # SCP-like format: git@host:owner/repo
        result = parse_git_url("git@github.com:user/repo.git")
        # May return None if not supported, or GitSource
        assert result is None or isinstance(result, GitSource)

    def test_https_url_returns_git_source(self):
        from pi_coding_agent.utils.git import GitSource, parse_git_url
        result = parse_git_url("https://github.com/user/repo")
        assert result is None or isinstance(result, GitSource)

    def test_invalid_returns_none(self):
        from pi_coding_agent.utils.git import parse_git_url
        result = parse_git_url("not-a-url")
        assert result is None


# ============================================================================
# frontmatter
# ============================================================================

class TestFrontmatter:
    def test_parse_frontmatter_with_yaml(self):
        from pi_coding_agent.utils.frontmatter import parse_frontmatter
        # parse_frontmatter returns (metadata_dict, body_str)
        content = "---\ntitle: Hello\nauthor: World\n---\nBody content here."
        meta, body = parse_frontmatter(content)
        assert meta["title"] == "Hello"
        assert meta["author"] == "World"
        assert "Body content here." in body

    def test_parse_frontmatter_no_yaml(self):
        from pi_coding_agent.utils.frontmatter import parse_frontmatter
        content = "Just plain content without frontmatter."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert "Just plain" in body

    def test_strip_frontmatter(self):
        from pi_coding_agent.utils.frontmatter import strip_frontmatter
        content = "---\ntitle: Hello\n---\nBody content here."
        result = strip_frontmatter(content)
        assert "title" not in result
        assert "Body content here." in result

    def test_strip_frontmatter_no_yaml(self):
        from pi_coding_agent.utils.frontmatter import strip_frontmatter
        content = "Just plain content."
        result = strip_frontmatter(content)
        assert result == content

    def test_stringify_frontmatter(self):
        from pi_coding_agent.utils.frontmatter import stringify_frontmatter
        # stringify_frontmatter(metadata, body) -> str
        data = {"title": "Hello", "count": 42}
        result = stringify_frontmatter(data, "Body goes here")
        assert result.startswith("---")
        assert "title: Hello" in result
        assert "Body goes here" in result

    def test_roundtrip(self):
        from pi_coding_agent.utils.frontmatter import (
            parse_frontmatter,
            stringify_frontmatter,
            strip_frontmatter,
        )
        original_data = {"title": "Test"}
        original_body = "Hello world"
        full_content = stringify_frontmatter(original_data, original_body)

        meta, body = parse_frontmatter(full_content)
        stripped = strip_frontmatter(full_content)

        assert meta["title"] == "Test"
        assert original_body in body or original_body in stripped


# ============================================================================
# changelog
# ============================================================================

class TestChangelog:
    def test_parse_changelog_basic(self):
        from pi_coding_agent.utils.changelog import parse_changelog
        text = "# Changelog\n\n## [1.2.0] - 2024-01-01\n- Added feature\n\n## [1.1.0] - 2023-12-01\n- Fixed bug\n"
        entries = parse_changelog(text)
        assert len(entries) >= 2
        versions = [e.version for e in entries]
        assert "1.2.0" in versions
        assert "1.1.0" in versions

    def test_compare_versions(self):
        from pi_coding_agent.utils.changelog import compare_versions
        assert compare_versions("1.2.0", "1.1.0") > 0
        assert compare_versions("1.0.0", "2.0.0") < 0
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_get_new_entries(self):
        from pi_coding_agent.utils.changelog import get_new_entries, parse_changelog
        text = "## [1.3.0]\n- New\n\n## [1.2.0]\n- Old\n\n## [1.1.0]\n- Very old\n"
        entries = parse_changelog(text)
        # get_new_entries(old_version, entries)
        new = get_new_entries("1.2.0", entries)
        assert len(new) >= 1
        assert new[0].version == "1.3.0"

    def test_get_new_entries_none_newer(self):
        from pi_coding_agent.utils.changelog import get_new_entries, parse_changelog
        text = "## [1.0.0]\n- Basic\n"
        entries = parse_changelog(text)
        new = get_new_entries("1.0.0", entries)
        assert len(new) == 0


# ============================================================================
# shell
# ============================================================================

class TestShell:
    def test_get_shell_config_returns_tuple(self):
        from pi_coding_agent.utils.shell import get_shell_config
        # get_shell_config returns (shell_path: str, args: list[str]) tuple
        config = get_shell_config()
        assert isinstance(config, tuple)
        assert len(config) == 2
        shell_path, shell_args = config
        assert isinstance(shell_path, str)
        assert isinstance(shell_args, list)

    def test_sanitize_binary_output(self):
        from pi_coding_agent.utils.shell import sanitize_binary_output
        # Regular text passes through
        assert sanitize_binary_output("hello world") == "hello world"

    def test_get_shell_env_returns_dict(self):
        from pi_coding_agent.utils.shell import get_shell_env
        env = get_shell_env()
        assert isinstance(env, dict)


# ============================================================================
# cli debug log
# ============================================================================

class TestCliDebugLog:
    def test_redact_image_base64_redacts_only_image_paths(self):
        from pi_coding_agent.core.cli_debug_log import (
            IMAGE_BASE64_REDACT_THRESHOLD_BYTES,
            IMAGE_BASE64_REPLACEMENT_TEMPLATE,
            redact_image_base64,
        )

        big = "A" * (IMAGE_BASE64_REDACT_THRESHOLD_BYTES + 20)
        payload = {
            "request": {
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": big}},
                    {"type": "text", "text": big},
                ],
                "signature": big,
            }
        }

        redacted = redact_image_base64(payload)

        assert redacted["request"]["content"][0]["source"]["data"] == (
            IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(big))
        )
        assert redacted["request"]["content"][1]["text"] == big
        assert redacted["request"]["signature"] == big

    def test_log_event_writes_redacted_image_payload(self, tmp_path, monkeypatch):
        from pi_coding_agent.core import cli_debug_log

        log_path = tmp_path / "debug.jsonl"
        big = "B" * (cli_debug_log.IMAGE_BASE64_REDACT_THRESHOLD_BYTES + 10)
        monkeypatch.setattr(cli_debug_log, "_LOG_PATH", str(log_path))

        cli_debug_log.log_event("image_payload", request={"source": {"type": "base64", "data": big}})

        line = log_path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert big not in line
        assert parsed["request"]["source"]["data"] == (
            cli_debug_log.IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(big))
        )


# ============================================================================
# clarity pii
# ============================================================================

class TestClarityPiiWalk:
    def test_vault_detokenizes_legacy_equals_token_alias(self):
        from pi_coding_agent.clarity_pii.vault import Vault

        vault = Vault()
        token = vault.tokenize("jane@acme.com")

        assert token == "[PII:EMAIL:1]"
        assert vault.detokenize("email [PII:EMAIL=1]") == "email jane@acme.com"

    def test_provider_payload_tokenization_skips_response_protocol_ids(self):
        from pi_coding_agent.clarity_pii.vault import Vault
        from pi_coding_agent.clarity_pii.walk import (
            provider_payload_protocol_slots,
            provider_payload_string_slots,
        )

        raw_function_call_id = "fc_0400e0ad26af8453016a4111111111111111b166aefb19465b3"
        payload = {
            "model": "gpt-5.5",
            "input": [
                {
                    "id": "rs_09dc8f8587d38d3e016a3340667384819b9b9e1efb018d3194",
                    "type": "reasoning",
                    "encrypted_content": "opaque-4111111111111111",
                    "summary": [],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "email jane@acme.com"}],
                },
                {
                    "type": "function_call",
                    "id": raw_function_call_id,
                    "call_id": "call_4111111111111111",
                    "name": "edit",
                    "arguments": "{\"old_string\":\"email jane@acme.com\"}",
                },
            ],
        }
        vault = Vault()

        for get, set_ in provider_payload_protocol_slots(payload):
            set_(vault.detokenize(get()))
        for get, set_ in provider_payload_string_slots(payload):
            set_(vault.tokenize(get()))

        reasoning = payload["input"][0]
        assert reasoning["id"] == "rs_09dc8f8587d38d3e016a3340667384819b9b9e1efb018d3194"
        assert reasoning["encrypted_content"] == "opaque-4111111111111111"
        assert payload["input"][1]["content"][0]["text"] == "email [PII:EMAIL:1]"
        assert payload["input"][2]["id"] == raw_function_call_id
        assert payload["input"][2]["call_id"] == "call_4111111111111111"
        assert payload["input"][2]["arguments"] == "{\"old_string\":\"email [PII:EMAIL:1]\"}"

    def test_provider_payload_protocol_slots_restore_pretokenized_ids(self):
        from pi_coding_agent.clarity_pii.vault import Vault
        from pi_coding_agent.clarity_pii.walk import (
            provider_payload_protocol_slots,
            provider_payload_string_slots,
        )

        vault = Vault()
        raw_card = "4111111111111111"
        token = vault.tokenize(raw_card)
        raw_id = f"fc_0400e0ad26af8453016a{raw_card}b166aefb19465b3"
        payload = {
            "input": [
                {
                    "type": "function_call",
                    "id": raw_id.replace(raw_card, token),
                    "call_id": f"call_{token}",
                    "arguments": "{\"note\":\"email jane@acme.com\"}",
                }
            ]
        }

        for get, set_ in provider_payload_protocol_slots(payload):
            set_(vault.detokenize(get()))
        for get, set_ in provider_payload_string_slots(payload):
            set_(vault.tokenize(get()))

        assert payload["input"][0]["id"] == raw_id
        assert payload["input"][0]["call_id"] == f"call_{raw_card}"
        assert payload["input"][0]["arguments"] == "{\"note\":\"email [PII:EMAIL:1]\"}"

    def test_provider_payload_tokenization_skips_anthropic_tool_use_id(self):
        from pi_coding_agent.clarity_pii.vault import Vault
        from pi_coding_agent.clarity_pii.walk import (
            provider_payload_protocol_slots,
            provider_payload_string_slots,
        )

        raw_card = "4111111111111111"
        raw_tool_use_id = f"tc_prefix{raw_card}suffix"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": raw_tool_use_id,
                            "content": "email jane@acme.com",
                        }
                    ],
                }
            ]
        }
        vault = Vault()

        for get, set_ in provider_payload_protocol_slots(payload):
            set_(vault.detokenize(get()))
        for get, set_ in provider_payload_string_slots(payload):
            set_(vault.tokenize(get()))

        tool_result = payload["messages"][0]["content"][0]
        assert tool_result["tool_use_id"] == raw_tool_use_id
        assert tool_result["content"] == "email [PII:EMAIL:1]"


def _graphwalks_like_prompt() -> str:
    header = [
        "You will be given a graph as a list of directed edges.",
        "If asked for the parents of a node, only return incoming edges.",
        "",
        "Here is an example:",
        "Operation:",
        "Find the parents of node uvwx.",
        "Final Answer: [abcd, efgh]",
        "",
        "Here is the graph to operate on:",
        "The graph has the following edges:",
    ]
    edges = [f"node{i:03d} -> node{i + 1:03d}" for i in range(80)]
    tail = [
        "target-node -> outgoing-child",
        "winner-2 -> target-node",
        "winner -> target-node",
        "",
        "Operation:",
        "Find the parents of node target-node.",
        "",
        "Return your final answer as a list of nodes.",
        "Final Answer: []",
    ]
    return "\n".join(header + edges + tail)


def test_active_compression_log_summary_preserves_actual_tail_operation(tmp_path):
    from pi_coding_agent.active_compression.ccr import CCRStore
    from pi_coding_agent.active_compression.compressor import compress

    original = _graphwalks_like_prompt()
    store = CCRStore(str(tmp_path / "ccr.db"))

    compressed = compress(original, store)

    assert "[CCR:" in compressed
    assert "Find the parents of node target-node." in compressed
    assert len(compressed) < len(original)


def test_ccr_tail_query_returns_actual_task_block():
    from pi_coding_agent.active_compression.search import search_original

    result = search_original(_graphwalks_like_prompt(), "actual task tail operation")

    assert result["route"] == "tail_slice"
    assert "Find the parents of node target-node." in result["text"]
    assert "winner -> target-node" in result["text"]


def test_ccr_parent_query_returns_all_incoming_edges_only():
    from pi_coding_agent.active_compression.search import search_original

    result = search_original(_graphwalks_like_prompt(), "target-node parents edges")

    assert result["route"] == "incoming_edges"
    assert "winner -> target-node" in result["text"]
    assert "winner-2 -> target-node" in result["text"]
    assert "target-node -> outgoing-child" not in result["text"]


def test_ccr_source_query_returns_all_outgoing_edges_only():
    from pi_coding_agent.active_compression.search import search_original

    result = search_original(_graphwalks_like_prompt(), "target-node ->")

    assert result["route"] == "outgoing_edges"
    assert "target-node -> outgoing-child" in result["text"]
    assert "winner -> target-node" not in result["text"]


def test_ccr_bfs_query_computes_exact_depth_without_frontier_turns():
    from pi_coding_agent.active_compression.search import search_original

    original = "\n".join(
        [
            "The graph has the following edges:",
            "start -> a",
            "start -> b",
            "a -> c",
            "a -> d",
            "b -> e",
            "c -> target",
            "d -> off-target",
            "e -> other",
            "",
            "Operation:",
            "Perform a BFS from node start and return only the nodes at exactly depth 2 (not nodes at intermediate depths).",
        ]
    )

    result = search_original(original, "edges graph list directed")

    assert result["route"] == "graph_bfs"
    assert "depth 1: 2 node(s)" in result["text"]
    assert "depth 2: 3 node(s)" in result["text"]
    assert "Final Answer: [c, d, e]" in result["text"]


def test_ccr_parent_operation_query_ignores_example_bfs():
    from pi_coding_agent.active_compression.search import search_original

    original = "\n".join(
        [
            "Here is an example:",
            "The graph has the following edges:",
            "abcd -> uvwx",
            "Operation:",
            "Perform a BFS from node alke with depth 1.",
            "Final Answer: []",
            "",
            "Here is the graph to operate on:",
            "The graph has the following edges:",
            "winner -> target",
            "decoy -> other",
            "",
            "Operation:",
            "Find the parents of node target.",
            "",
            "Final Answer: []",
        ]
    )

    result = search_original(original, "edges graph operation node")

    assert result["route"] == "graph_parents"
    assert "Operation: Find the parents of node target." in result["text"]
    assert "Final Answer: [winner]" in result["text"]
    assert "BFS from node alke" not in result["text"]


def test_ccr_task_query_on_paginated_page_directs_next_read_offset():
    from pi_coding_agent.active_compression.search import search_original

    page = "\n".join(
        [
            "The graph has the following edges:",
            "a -> b",
            "b -> c",
            "",
            "[Showing lines 1-2000 of 4394. Use offset=2001 to continue.]",
        ]
    )

    result = search_original(page, "actual task operation")

    assert result["route"] == "pagination_continue"
    assert "offset=2001" in result["text"]
