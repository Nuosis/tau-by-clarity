from __future__ import annotations

import math

import pi_ai
import pytest
from pi_ai.compression_policy import AuthMode, CompressionPolicy, policy_default_payg, policy_for_mode, resolve_policy
from pi_ai.types import Context, TextContent, ToolResultMessage


def test_compression_policy_matches_headroom_modes():
    payg = policy_for_mode(AuthMode.PAYG)
    oauth = policy_for_mode(AuthMode.OAUTH)
    subscription = policy_for_mode(AuthMode.SUBSCRIPTION)

    assert payg == CompressionPolicy(
        live_zone_only=False,
        cache_aligner_enabled=True,
        volatile_token_threshold=128,
        max_lossy_ratio=0.45,
        toin_read_only=False,
    )
    assert oauth == payg
    assert subscription == CompressionPolicy(
        live_zone_only=True,
        cache_aligner_enabled=False,
        volatile_token_threshold=32,
        max_lossy_ratio=0.25,
        toin_read_only=True,
    )
    assert policy_default_payg() == payg


def test_compression_policy_net_cost_formula_matches_headroom_anchors():
    policy = policy_for_mode("payg")

    assert abs(policy.net_mutation_gain(2_000, 50_000, 10.0, 1.0) - (-55_500.0)) < 1.0
    assert not policy.should_mutate_deep(2_000, 50_000, 10.0, 1.0)
    assert abs(policy.net_mutation_gain(50_000, 10_000, 3.0, 1.0) - 3_500.0) < 1.0
    assert policy.should_mutate_deep(50_000, 10_000, 3.0, 1.0)
    assert math.isfinite(policy.net_mutation_gain(2_000, 50_000, float("nan"), float("nan")))
    assert abs(policy.break_even_reads(2_000, 50_000) - 287.5) < 0.5
    assert abs(policy.break_even_reads(50_000, 10_000) - 2.3) < 0.05
    assert policy.break_even_reads(0, 10_000) == 0.0


def test_compression_policy_resolver_honors_enforcement_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT", "off")

    assert resolve_policy("subscription") == policy_for_mode("payg")


def test_subscription_policy_preserves_prompt_bytes_but_compresses_live_tool_results():
    seen_policies = []

    def compressor(text: str) -> str:
        seen_policies.append(pi_ai.get_current_compression_policy())
        return f"COMPRESSED:{text[:16]}"

    pi_ai.register_compressor(compressor)
    try:
        ctx = Context(
            compression_auth_mode="subscription",
            system_prompt="system instructions " * 120,
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(text="tool output payload " * 120)],
                    timestamp=0,
                )
            ],
        )

        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    assert out.system_prompt == ctx.system_prompt
    assert out.messages[0].content[0].text.startswith("COMPRESSED:tool output")
    assert seen_policies == [policy_for_mode("subscription")]


def test_payg_and_oauth_policy_record_compression_learning_events():
    def compressor(text: str) -> str:
        return f"COMPRESSED:{text[:16]}"

    pi_ai.register_compressor(compressor)
    pi_ai.reset_compression_learning_stats()
    try:
        for auth_mode in ("payg", "oauth"):
            ctx = Context(
                compression_auth_mode=auth_mode,
                messages=[
                    ToolResultMessage(
                        tool_call_id=f"call-{auth_mode}",
                        tool_name="bash",
                        content=[TextContent(text=f"{auth_mode} tool output payload " * 120)],
                        timestamp=0,
                    )
                ],
            )
            out = pi_ai.compress_context(ctx)
            assert out.messages[0].content[0].text.startswith(f"COMPRESSED:{auth_mode}")
    finally:
        pi_ai.unregister_compressor()

    stats = pi_ai.get_compression_learning_stats()
    assert stats.total_events == 2
    assert stats.total_skipped_read_only == 0
    assert stats.events_by_strategy == {"text": 2}
    assert stats.total_tokens_saved > 0


def test_subscription_policy_skips_compression_learning_writes():
    def compressor(text: str) -> str:
        return f"COMPRESSED:{text[:16]}"

    pi_ai.register_compressor(compressor)
    pi_ai.reset_compression_learning_stats()
    try:
        ctx = Context(
            compression_auth_mode="subscription",
            messages=[
                ToolResultMessage(
                    tool_call_id="call-subscription",
                    tool_name="bash",
                    content=[TextContent(text="subscription tool output payload " * 120)],
                    timestamp=0,
                )
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    stats = pi_ai.get_compression_learning_stats()
    assert out.messages[0].content[0].text.startswith("COMPRESSED:subscription")
    assert stats.total_events == 0
    assert stats.total_skipped_read_only == 1
    assert stats.events_by_strategy == {}
    assert stats.skipped_by_strategy == {"text": 1}


def test_cache_alignment_detector_matches_headroom_volatile_labels():
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    findings = pi_ai.detect_volatile_content(
        "Session: 550e8400-e29b-41d4-a716-446655440000 "
        "Now: 2024-01-15T10:30:00 "
        f"Token: {jwt} "
        "Hash: d41d8cd98f00b204e9800998ecf8427e"
    )

    assert [finding.label for finding in findings] == ["uuid", "iso8601", "jwt", "hex_hash"]
    assert "..." in findings[0].sample
    assert pi_ai.detect_volatile_content("You are a helpful assistant. Be concise.") == []


def test_payg_cache_alignment_detector_records_findings_without_mutating_prompt():
    def compressor(text: str) -> str:
        return text

    pi_ai.register_compressor(compressor)
    pi_ai.reset_cache_alignment_stats()
    try:
        ctx = Context(
            compression_auth_mode="payg",
            system_prompt=(
                "Session: 550e8400-e29b-41d4-a716-446655440000\n"
                "Hash: d41d8cd98f00b204e9800998ecf8427e"
            ),
            messages=[],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    stats = pi_ai.get_cache_alignment_stats()
    assert out.system_prompt == ctx.system_prompt
    assert stats.total_scans == 1
    assert stats.total_findings == 2
    assert stats.skipped_by_policy == 0
    assert stats.findings_by_label == {"hex_hash": 1, "uuid": 1}
    assert pi_ai.cache_alignment_score([{"role": "system", "content": ctx.system_prompt}]) == 80.0


def test_subscription_cache_alignment_detector_is_policy_skipped():
    def compressor(text: str) -> str:
        return text

    pi_ai.register_compressor(compressor)
    pi_ai.reset_cache_alignment_stats()
    try:
        ctx = Context(
            compression_auth_mode="subscription",
            system_prompt="Session: 550e8400-e29b-41d4-a716-446655440000",
            messages=[],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    stats = pi_ai.get_cache_alignment_stats()
    assert out.system_prompt == ctx.system_prompt
    assert stats.total_scans == 0
    assert stats.total_findings == 0
    assert stats.skipped_by_policy == 1
    assert stats.findings_by_label == {}
