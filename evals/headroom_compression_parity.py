#!/usr/bin/env python3
"""Compression-only parity checks against Headroom's recorded fixtures.

This is intentionally not an SDK/MCP/proxy comparison. It reads Headroom's
recorded transform fixtures and checks Tau's public active-compression entry
point against behavior-level invariants: whether sizeable content is compressed,
whether compressed content is CCR-reversible, and whether high-signal content is
still visible after compression.

Usage:
    python evals/headroom_compression_parity.py [/tmp/headroom-src]
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages", "ai", "src"))
sys.path.insert(0, os.path.join(ROOT, "packages", "coding-agent", "src"))

import pi_ai  # noqa: E402
from pi_ai.compression_policy import AuthMode, policy_for_mode, resolve_policy  # noqa: E402
from pi_ai.tokenization import count_text_tokens  # noqa: E402
from pi_coding_agent import active_compression as active_compression_runtime  # noqa: E402
from pi_coding_agent.active_compression.ccr import CCRStore  # noqa: E402
from pi_coding_agent.active_compression.compressor import (  # noqa: E402
    CompressionConfig,
    _detect_content_type,
    _protect_custom_tags,
    _restore_custom_tags,
    compress,
)
from pi_coding_agent.active_compression.extension import extension_factory  # noqa: E402
from pi_coding_agent.core.cli_debug_log import (  # noqa: E402
    IMAGE_BASE64_REDACT_THRESHOLD_BYTES,
    IMAGE_BASE64_REPLACEMENT_TEMPLATE,
    redact_image_base64,
)

FIXTURE_GROUPS = ("smart_crusher", "log_compressor", "diff_compressor")
SYNTHETIC_GROUP = "search_code_html"
CONTENT_DETECTOR_GROUP = "content_detector"
TOKENIZER_GROUP = "tokenizer"
IMAGE_COMPRESSION_GROUP = "image_compression"
IMAGE_LOG_REDACTION_GROUP = "image_log_redaction"
COMPRESSION_SUMMARY_GROUP = "compression_summary"
DETERMINISM_GROUP = "determinism"
COMPRESSION_POLICY_GROUP = "compression_policy"
SYNTHETIC_EXPECTED_PASSTHROUGH = frozenset(
    {
        "search_above_ccr_ratio",
        "log_below_min_lines",
        "log_above_ccr_ratio",
        "diff_below_min_lines",
    }
)
PROVIDER_SHAPE_GROUP = "anthropic_tool_result"
ERROR_PROTECTION_GROUP = "error_output_protection"
CODE_PROTECTION_GROUP = "code_output_protection"
CACHE_CONTROL_GROUP = "cache_control_protection"
CACHE_ALIGNER_GROUP = "cache_aligner"
MARKER_PINNING_GROUP = "marker_pinning"
MARKER_PRESERVING_GROUP = "marker_preserving"
CUSTOM_TAG_GROUP = "custom_tag_protection"
TOOL_EXCLUSION_GROUP = "tool_exclusion"
NON_SHRINKING_GROUP = "non_shrinking_rejection"
COMPRESSION_FAILURE_GROUP = "compression_failure"
COMPRESSION_CIRCUIT_BREAKER_GROUP = "compression_circuit_breaker"
INFLATION_GUARD_GROUP = "inflation_guard"
TOKEN_FLOOR_GROUP = "token_floor"
CONFIG_CONTROL_GROUP = "config_controls"
TARGET_RATIO_GROUP = "target_ratio"
READ_LIFECYCLE_GROUP = "read_lifecycle"
CCR_RECOVERY_GROUP = "ccr_recovery"
COMPRESSION_COMMAND_GROUP = "compression_command"
CCR_STORE_GROUP = "ccr_store"
UNIT_CACHE_GROUP = "unit_cache"
CCR_HANDLE_RE = re.compile(r"\[CCR:([0-9a-f]{12})\]")
READ_MARKER_HANDLE_RE = re.compile(r"hash=([0-9a-f]{12})")
ROW_OFFLOAD_HANDLE_RE = re.compile(r"<<ccr:([0-9a-f]{12})\s+\d+_rows_offloaded>>")
ERROR_RE = re.compile(r"error|fail|exception|traceback|fatal|critical", re.IGNORECASE)
SUMMARY_RE = re.compile(r"\b\d+\s+(failed|passed|skipped|errors?|warnings?)\b", re.IGNORECASE)


@dataclass
class CaseResult:
    group: str
    name: str
    ok: bool
    before_chars: int
    after_chars: int
    checks: list[str]
    failures: list[str]


class _LooseContext:
    def __init__(
        self,
        messages: list[Any],
        *,
        system_prompt: str | None = None,
        compression_compress_user_messages: bool = False,
        compression_compress_system_messages: bool = False,
        compression_protect_recent: int = 4,
        compression_target_ratio: float | None = None,
        compression_min_tokens: int | None = None,
        compression_max_items_after_crush: int | None = None,
        compression_lossless_min_savings_ratio: float | None = None,
        compression_enable_ccr_marker: bool | None = None,
        compression_image_optimize: bool = True,
        compression_auth_mode: str | None = None,
        compression_compress_stale_reads: bool = True,
        compression_compress_superseded_reads: bool = False,
        compression_read_lifecycle_min_bytes: int = 512,
    ):
        self.messages = messages
        self.system_prompt = system_prompt
        self.tools = None
        self.compression_frozen_message_count = 0
        self.compression_compress_user_messages = compression_compress_user_messages
        self.compression_compress_system_messages = compression_compress_system_messages
        self.compression_protect_recent = compression_protect_recent
        self.compression_target_ratio = compression_target_ratio
        self.compression_min_tokens = compression_min_tokens
        self.compression_max_items_after_crush = compression_max_items_after_crush
        self.compression_lossless_min_savings_ratio = compression_lossless_min_savings_ratio
        self.compression_enable_ccr_marker = compression_enable_ccr_marker
        self.compression_image_optimize = compression_image_optimize
        self.compression_auth_mode = compression_auth_mode
        self.compression_compress_stale_reads = compression_compress_stale_reads
        self.compression_compress_superseded_reads = compression_compress_superseded_reads
        self.compression_read_lifecycle_min_bytes = compression_read_lifecycle_min_bytes

    def model_copy(self, update: dict[str, Any] | None = None) -> _LooseContext:
        update = update or {}
        return _LooseContext(
            messages=update.get("messages", self.messages),
            system_prompt=update.get("system_prompt", self.system_prompt),
            compression_compress_user_messages=update.get(
                "compression_compress_user_messages",
                self.compression_compress_user_messages,
            ),
            compression_compress_system_messages=update.get(
                "compression_compress_system_messages",
                self.compression_compress_system_messages,
            ),
            compression_protect_recent=update.get(
                "compression_protect_recent",
                self.compression_protect_recent,
            ),
            compression_target_ratio=update.get(
                "compression_target_ratio",
                self.compression_target_ratio,
            ),
            compression_min_tokens=update.get(
                "compression_min_tokens",
                self.compression_min_tokens,
            ),
            compression_max_items_after_crush=update.get(
                "compression_max_items_after_crush",
                self.compression_max_items_after_crush,
            ),
            compression_lossless_min_savings_ratio=update.get(
                "compression_lossless_min_savings_ratio",
                self.compression_lossless_min_savings_ratio,
            ),
            compression_enable_ccr_marker=update.get(
                "compression_enable_ccr_marker",
                self.compression_enable_ccr_marker,
            ),
            compression_image_optimize=update.get(
                "compression_image_optimize",
                self.compression_image_optimize,
            ),
            compression_auth_mode=update.get(
                "compression_auth_mode",
                self.compression_auth_mode,
            ),
            compression_compress_stale_reads=update.get(
                "compression_compress_stale_reads",
                self.compression_compress_stale_reads,
            ),
            compression_compress_superseded_reads=update.get(
                "compression_compress_superseded_reads",
                self.compression_compress_superseded_reads,
            ),
            compression_read_lifecycle_min_bytes=update.get(
                "compression_read_lifecycle_min_bytes",
                self.compression_read_lifecycle_min_bytes,
            ),
        )


class _FakePi:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}

    def register_tool(self, **kwargs: Any) -> None:
        self.tools[kwargs["name"]] = kwargs

    def register_command(self, name: str, command: Any) -> None:
        self.commands[name] = command


def _fixture_input(data: dict[str, Any]) -> str:
    raw = data.get("input", "")
    if isinstance(raw, dict):
        return str(raw.get("content", ""))
    return str(raw)


def _headroom_modified(data: dict[str, Any]) -> bool:
    output = data.get("output", {})
    if isinstance(output, dict) and "was_modified" in output:
        return bool(output["was_modified"])
    compressed = output.get("compressed") if isinstance(output, dict) else None
    original = output.get("original") if isinstance(output, dict) else None
    baseline = original if isinstance(original, str) else _fixture_input(data)
    return isinstance(compressed, str) and compressed != baseline


def _headroom_material_savings(data: dict[str, Any], content: str) -> bool:
    output = data.get("output", {})
    compressed = output.get("compressed") if isinstance(output, dict) else None
    if not isinstance(compressed, str) or compressed == content:
        return False
    saved = len(content) - len(compressed)
    return saved >= 200 or len(compressed) <= len(content) * 0.9


def _headroom_compressed_lines(data: dict[str, Any]) -> list[str]:
    output = data.get("output", {})
    compressed = output.get("compressed") if isinstance(output, dict) else None
    return compressed.splitlines() if isinstance(compressed, str) else []


def _approx_tokens(text: str) -> int:
    return count_text_tokens(text)


def _is_reversible(output: str, original: str, store: CCRStore) -> bool:
    marker = CCR_HANDLE_RE.search(output)
    if marker is not None:
        return store.get(marker.group(1)) == original

    marker = ROW_OFFLOAD_HANDLE_RE.search(output)
    if marker is not None:
        stored = store.get(marker.group(1))
        try:
            return json.loads(stored or "") == json.loads(original)
        except (TypeError, ValueError):
            return stored == original

    try:
        return json.loads(output) == json.loads(original)
    except (TypeError, ValueError):
        pass
    return output == original


def _has_retrieval_marker(output: str) -> bool:
    return bool(CCR_HANDLE_RE.search(output) or READ_MARKER_HANDLE_RE.search(output) or "<<ccr:" in output)


def _handle_from_marker(output: str) -> str:
    marker = CCR_HANDLE_RE.search(output)
    if marker is not None:
        return marker.group(1)
    raise AssertionError(f"no CCR marker found in output: {output[:120]!r}")


def _json_error_needles(content: str) -> list[str]:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    needles: list[str] = []
    for item in parsed:
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if ERROR_RE.search(encoded):
            needles.extend(str(value) for value in item.values() if isinstance(value, str) and ERROR_RE.search(value))
    return needles


def _log_needles(content: str) -> list[str]:
    needles: list[str] = []
    for line in content.splitlines():
        if ERROR_RE.search(line) or SUMMARY_RE.search(line):
            needles.append(line.strip())
    return needles[:20]


def _diff_needles(data: dict[str, Any]) -> list[str]:
    needles: list[str] = []
    for line in _headroom_compressed_lines(data):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            needles.append(line)
    return needles[:40]


def _run_case(group: str, path: Path, tempdir: str) -> CaseResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    content = _fixture_input(data)
    store = CCRStore(os.path.join(tempdir, f"{group}-{path.stem}.db"))
    output = compress(content, store)
    modified = output != content
    headroom_modified = _headroom_modified(data)
    headroom_material_savings = _headroom_material_savings(data, content)
    checks: list[str] = []
    failures: list[str] = []

    if headroom_modified and headroom_material_savings and _approx_tokens(content) >= 200:
        if modified:
            checks.append("modified")
        else:
            failures.append("Headroom materially compressed sizeable content but Tau passed it through")
    elif not headroom_modified:
        if not modified:
            checks.append("passthrough")
        else:
            failures.append("Headroom passed content through but Tau modified it")

    if group == "smart_crusher" and modified and not _has_retrieval_marker(output):
        checks.append("lossless-direct")
    elif _is_reversible(output, content, store):
        checks.append("reversible")
    else:
        failures.append("compressed output was not CCR-reversible")

    if group == "smart_crusher":
        for needle in _json_error_needles(content):
            if needle not in output:
                failures.append(f"missing JSON error needle: {needle[:80]}")
    elif group == "log_compressor":
        for needle in _log_needles(content):
            if needle and needle not in output:
                failures.append(f"missing log signal: {needle[:80]}")
    elif group == "diff_compressor":
        for needle in _diff_needles(data):
            if needle not in output:
                failures.append(f"missing diff changed line: {needle[:80]}")

    if not failures:
        checks.append("signals")

    return CaseResult(
        group=group,
        name=path.name,
        ok=not failures,
        before_chars=len(content),
        after_chars=len(output),
        checks=checks,
        failures=failures,
    )


def _run_synthetic_case(name: str, content: str, needles: list[str], forbidden: list[str], tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{SYNTHETIC_GROUP}-{name}.db"))
    output = compress(content, store)
    failures: list[str] = []

    expected_passthrough = name in SYNTHETIC_EXPECTED_PASSTHROUGH
    if output == content and _approx_tokens(content) >= 200 and not expected_passthrough:
        failures.append("Tau passed through a sizeable Headroom-derived compression case")
    if expected_passthrough and output != content:
        failures.append("Tau compressed a Headroom expected pass-through case")
    if not _is_reversible(output, content, store):
        failures.append("compressed output was not CCR-reversible")
    for needle in needles:
        if needle not in output:
            failures.append(f"missing signal: {needle[:80]}")
    for value in forbidden:
        if value in output:
            failures.append(f"retained noise: {value[:80]}")

    return CaseResult(
        group=SYNTHETIC_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(content),
        after_chars=len(output),
        checks=["synthetic"] if not failures else [],
        failures=failures,
    )


def _synthetic_search_case() -> tuple[str, str, list[str], list[str]]:
    lines = [f"src/auth.py:{i}:auth event {i}" for i in range(1, 51)]
    lines += [f"src/db.py:{i}:db query {i}" for i in range(1, 31)]
    lines.append("src/auth.py:77:ERROR auth failed for TARGET_SEARCH_77")
    content = "\n".join(lines)
    return (
        "search_grep_context",
        content,
        ["compressed search results", "src/auth.py", "ERROR auth failed for TARGET_SEARCH_77"],
        ["src/db.py:14:db query 14"],
    )


def _synthetic_min_search_case() -> tuple[str, str, list[str], list[str]]:
    lines = [
        (
            "src/services/authentication/session_recovery.py:"
            f"{i}:ordinary session recovery result with repeated explanatory context {i}"
        )
        for i in range(1, 13)
    ]
    content = "\n".join(lines)
    return (
        "search_min_match_threshold",
        content,
        ["compressed search results", "src/services/authentication/session_recovery.py", "12:"],
        ["compressed text"],
    )


def _synthetic_above_ccr_ratio_search_case() -> tuple[str, str, list[str], list[str]]:
    lines = []
    for match_index in range(25):
        file_index = match_index % 20
        path = f"src/{file_index:02d}/" + ("longname_" * (file_index % 5 + 1)) + "file.py"
        lines.append(f"{path}:{match_index + 1}:x{match_index}")
    content = "\n".join(lines)
    return (
        "search_above_ccr_ratio",
        content,
        ["src/00/longname_file.py:1:x0", "src/19/longname_longname_longname_longname_longname_file.py:20:x19"],
        ["compressed search results", "compressed text"],
    )


def _synthetic_below_threshold_log_case() -> tuple[str, str, list[str], list[str]]:
    lines = ["npm WARN deprecated x"] * 44 + ["npm ERR! something broke"] * 5
    content = "\n".join(lines)
    return (
        "log_below_min_lines",
        content,
        ["npm WARN deprecated x", "npm ERR! something broke"],
        ["compressed text", "compressed log"],
    )


def _synthetic_above_ccr_ratio_log_case() -> tuple[str, str, list[str], list[str]]:
    lines = [
        f"INFO setup line {i:03d} with moderately long shared context for ratio"
        for i in range(20)
    ]
    lines.extend(
        f"warning: distinct warning message {i:03d} with moderately long shared context"
        for i in range(10)
    )
    lines.extend(
        f"INFO cleanup line {i:03d} with moderately long shared context for ratio"
        for i in range(20)
    )
    content = "\n".join(lines)
    return (
        "log_above_ccr_ratio",
        content,
        ["warning: distinct warning message 009", "INFO cleanup line 019"],
        ["compressed log", "compacted log templates"],
    )


def _synthetic_heterogeneous_json_no_signal_case() -> tuple[str, str, list[str], list[str]]:
    rows = [
        {
            f"unique_{i}": "x" * 40,
            "id": i,
        }
        for i in range(40)
    ]
    content = json.dumps(rows, indent=2)
    return (
        "heterogeneous_json_no_signal",
        content,
        ["unique_39", '"id":39'],
        ["compressed JSON array", "[CCR:"],
    )


def _synthetic_code_case() -> tuple[str, str, list[str], list[str]]:
    blocks: list[str] = ["from typing import Any", ""]
    for i in range(20):
        blocks.extend(
            [
                f"def process_{i}(arg: Any) -> str:",
                f'    """Process argument {i}.',
                "",
                "    Args:",
                "        arg: The argument to process.",
                '    """',
                "    result = str(arg)",
                "    for j in range(10):",
                "        result += str(j)",
                "    return result",
                "",
            ]
        )
    content = "\n".join(blocks)
    return (
        "code_first_line_docstrings",
        content,
        ["compressed code", "20 bodies compressed: process_0()", "def process_0", '"""Process argument 19."""'],
        ["Args:", "for j in range(10)"],
    )


def _synthetic_html_case() -> tuple[str, str, list[str], list[str]]:
    article = "\n".join(
        f"<p>Main article paragraph {i} with TARGET_HTML_{i} content.</p>" for i in range(30)
    )
    nav = "\n".join(f"<a href='/nav-{i}'>Navigation {i}</a>" for i in range(60))
    content = f"""<!doctype html>
<html>
<head>
  <title>Headroom HTML Compression</title>
  <script>{'console.log("noise");' * 120}</script>
  <style>{'.hidden{display:none;}' * 120}</style>
</head>
<body>
  <nav>{nav}</nav>
  <main><article><h1>Main Article</h1>{article}</article></main>
  <footer>{'Footer noise ' * 120}</footer>
</body>
</html>"""
    return (
        "html_main_content",
        content,
        ["extracted HTML content", "Headroom HTML Compression", "TARGET_HTML_29"],
        ["console.log", "Navigation 59", "Footer noise"],
    )


def _synthetic_long_preamble_diff_case() -> tuple[str, str, list[str], list[str]]:
    prefix = [
        "commit abc1234567890abcdef",
        "Author: Tester <tester@example.com>",
        "Date:   Mon Apr 25 12:00:00 2026",
        "",
    ]
    prefix.extend(f"    detailed commit message line {i}" for i in range(120))
    diff_lines = [
        "diff --git a/src/app.py b/src/app.py",
        "index abc..def 100644",
        "--- a/src/app.py",
        "+++ b/src/app.py",
        "@@ -1,5 +1,5 @@",
    ]
    diff_lines.extend(f" context before {i}" for i in range(40))
    diff_lines.extend(["-old_auth_token = read_old()", "+new_auth_token = read_new()"])
    diff_lines.extend(f" context after {i}" for i in range(80))
    content = "\n".join(prefix + [""] + diff_lines)
    return (
        "diff_after_long_commit_message",
        content,
        ["compressed diff", "commit abc1234567890abcdef", "diff --git a/src/app.py b/src/app.py"],
        ["compressed text"],
    )


def _synthetic_below_threshold_diff_case() -> tuple[str, str, list[str], list[str]]:
    diff_lines = [
        "diff --git a/app.py b/app.py",
        "--- a/app.py",
        "+++ b/app.py",
        "@@ -1,44 +1,44 @@",
    ]
    diff_lines.extend(f" context before {i}" for i in range(20))
    diff_lines.extend(["-old_auth_token = read_old()", "+new_auth_token = read_new()"])
    diff_lines.extend(f" context after {i}" for i in range(23))
    content = "\n".join(diff_lines)
    return (
        "diff_below_min_lines",
        content,
        ["diff --git a/app.py b/app.py", "-old_auth_token = read_old()"],
        ["compressed text", "compressed diff"],
    )


def _run_synthetic_cases(tempdir: str) -> list[CaseResult]:
    return [
        _run_synthetic_case(name, content, needles, forbidden, tempdir)
        for name, content, needles, forbidden in (
            _synthetic_search_case(),
            _synthetic_min_search_case(),
            _synthetic_above_ccr_ratio_search_case(),
            _synthetic_below_threshold_log_case(),
            _synthetic_above_ccr_ratio_log_case(),
            _synthetic_heterogeneous_json_no_signal_case(),
            _synthetic_below_threshold_diff_case(),
            _synthetic_code_case(),
            _synthetic_html_case(),
            _synthetic_long_preamble_diff_case(),
        )
    ]


def _run_content_detector_cases(root: Path) -> list[CaseResult]:
    fixture_dir = root / "tests" / "parity" / "fixtures" / "content_detector"
    if not fixture_dir.exists():
        return [
            CaseResult(
                group=CONTENT_DETECTOR_GROUP,
                name="missing_fixtures",
                ok=False,
                before_chars=0,
                after_chars=0,
                checks=[],
                failures=[f"missing Headroom content detector fixtures at {fixture_dir}"],
            )
        ]

    results: list[CaseResult] = []
    for path in sorted(fixture_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        content = _fixture_input(data)
        output = data.get("output", {})
        expected = output.get("content_type") if isinstance(output, dict) else None
        actual = _detect_content_type(content)
        failures = [] if actual == expected else [f"expected {expected!r}, got {actual!r}"]
        results.append(
            CaseResult(
                group=CONTENT_DETECTOR_GROUP,
                name=path.name,
                ok=not failures,
                before_chars=len(content),
                after_chars=len(actual),
                checks=["content_type"] if not failures else [],
                failures=failures,
            )
        )
    return results


def _run_tokenizer_cases(root: Path) -> list[CaseResult]:
    fixture_dir = root / "tests" / "parity" / "fixtures" / "tokenizer"
    if not fixture_dir.exists():
        return [
            CaseResult(
                group=TOKENIZER_GROUP,
                name="missing_fixtures",
                ok=False,
                before_chars=0,
                after_chars=0,
                checks=[],
                failures=[f"missing Headroom tokenizer fixtures at {fixture_dir}"],
            )
        ]

    results: list[CaseResult] = []
    for path in sorted(fixture_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        content = _fixture_input(data)
        expected = data.get("output")
        actual = count_text_tokens(content)
        failures = [] if actual == expected else [f"expected {expected!r}, got {actual!r}"]
        results.append(
            CaseResult(
                group=TOKENIZER_GROUP,
                name=path.name,
                ok=not failures,
                before_chars=len(content),
                after_chars=actual,
                checks=["token_count"] if not failures else [],
                failures=failures,
            )
        )
    return results


def _run_provider_shape_case(
    name: str,
    tool_content: Any,
    expected_payload: str,
    tempdir: str,
    *,
    message_role: str = "user",
    tool_name: str = "Bash",
) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{PROVIDER_SHAPE_GROUP}-{name}.db"))
    if message_role == "user":
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": tool_name,
                        "input": {"file_path": "src/app.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": tool_content,
                    }
                ],
            },
        ]
    else:
        messages = [{"role": message_role, "tool_call_id": "toolu_1", "tool_name": tool_name, "content": tool_content}]
    before = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()
    after = json.dumps(out.messages, ensure_ascii=False, separators=(",", ":"))
    failures: list[str] = []

    if expected_payload in after:
        failures.append("provider tool_result payload was not compressed")
    lossless_structured = name == "dict_content" and ("]{" in after or "__buckets:" in after)
    if not CCR_HANDLE_RE.search(after) and not lossless_structured:
        failures.append("compressed provider tool_result did not include a CCR handle")
    marker = CCR_HANDLE_RE.search(after)
    if marker is not None and store.get(marker.group(1)) != expected_payload:
        failures.append("CCR handle did not recover the original provider tool_result payload")
    if pi_ai.get_compression_stats().compressions_by_strategy.get("text") != 1:
        failures.append("provider tool_result compression was not recorded as text compression")

    return CaseResult(
        group=PROVIDER_SHAPE_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(before),
        after_chars=len(after),
        checks=["provider_shape"] if not failures else [],
        failures=failures,
    )


def _run_provider_shape_cases(tempdir: str) -> list[CaseResult]:
    payload = "anthropic provider payload line\n" * 160
    rows = {"rows": [{"id": i, "padding": "x" * 80, "status": "active"} for i in range(240)]}
    rows_payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return [
        _run_provider_shape_case("string_content", payload, payload, tempdir),
        _run_provider_shape_case(
            "nested_text_list",
            [{"type": "text", "text": payload}],
            payload,
            tempdir,
        ),
        _run_provider_shape_case("dict_content", rows, rows_payload, tempdir),
        _run_provider_shape_case("openai_tool_string", payload, payload, tempdir, message_role="tool"),
        _run_provider_shape_case(
            "openai_tool_text_block",
            [{"type": "text", "text": payload}],
            payload,
            tempdir,
            message_role="tool",
        ),
    ]


def _run_error_protection_case(name: str, content: str, should_compress: bool, tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{ERROR_PROTECTION_GROUP}-{name}.db"))
    messages = [{"role": "tool", "tool_call_id": "toolu_1", "content": content}]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[0]["content"]
    failures: list[str] = []
    if should_compress:
        marker = CCR_HANDLE_RE.search(output)
        if marker is None:
            failures.append("large tool error output did not compress")
        elif store.get(marker.group(1)) != content:
            failures.append("large tool error CCR handle did not recover original")
    else:
        if output != content:
            failures.append("small tool error output was compressed")
        if pi_ai.get_compression_stats().compressions_by_strategy:
            failures.append("small tool error emitted compression stats")

    return CaseResult(
        group=ERROR_PROTECTION_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(content),
        after_chars=len(output),
        checks=["error_protection"] if not failures else [],
        failures=failures,
    )


def _run_error_protection_cases(tempdir: str) -> list[CaseResult]:
    small = "Traceback (most recent call last):\nRuntimeError: crashed\n" + "exact failure context\n" * 80
    large = "\n".join(
        ["Traceback (most recent call last):", "RuntimeError: crashed"]
        + [f"debug context line {i}" for i in range(700)]
    )
    return [
        _run_error_protection_case("small_tool_error", small, False, tempdir),
        _run_error_protection_case("large_tool_error", large, True, tempdir),
    ]


def _large_python_code() -> str:
    lines = ["import os", "from typing import Any", ""]
    for i in range(80):
        lines.extend(
            [
                f"def process_{i}(value: Any) -> str:",
                f'    """Process value {i}."""',
                "    result = str(value)",
                "    for j in range(5):",
                "        result += str(j)",
                "    return result",
                "",
            ]
        )
    return "\n".join(lines)


def _run_code_protection_case(
    name: str,
    messages: list[Any],
    expected: str,
    should_compress: bool,
    tempdir: str,
    **context_kwargs: Any,
) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{CODE_PROTECTION_GROUP}-{name}.db"))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages, **context_kwargs))
    finally:
        pi_ai.unregister_compressor()

    content = out.messages[0]["content"]
    if isinstance(content, list):
        output = content[0]["text"]
    else:
        output = content
    failures: list[str] = []
    if should_compress:
        marker = CCR_HANDLE_RE.search(output)
        if marker is None:
            failures.append("compressible non-code tool output did not compress")
        elif store.get(marker.group(1)) != expected:
            failures.append("compressed non-code CCR handle did not recover original")
    else:
        if output != expected:
            failures.append("protected code tool output was compressed")
        if pi_ai.get_compression_stats().compressions_by_strategy:
            failures.append("protected code emitted compression stats")

    return CaseResult(
        group=CODE_PROTECTION_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["code_protection"] if not failures else [],
        failures=failures,
    )


def _run_code_protection_cases(tempdir: str) -> list[CaseResult]:
    code = _large_python_code()
    plain = "plain data line\n" * 120
    return [
        _run_code_protection_case(
            "recent_code",
            [{"role": "tool", "tool_call_id": "toolu_1", "content": code}],
            code,
            False,
            tempdir,
        ),
        _run_code_protection_case(
            "recent_code_protect_recent_zero",
            [{"role": "tool", "tool_call_id": "toolu_1", "content": code}],
            code,
            True,
            tempdir,
            compression_protect_recent=0,
        ),
        _run_code_protection_case(
            "analysis_intent_code",
            [
                {"role": "tool", "tool_call_id": "toolu_1", "content": code},
                {"role": "user", "content": "ack"},
                {"role": "user", "content": "ack"},
                {"role": "user", "content": "ack"},
                {"role": "user", "content": "ack"},
                {"role": "user", "content": "please review this code for bugs"},
            ],
            code,
            False,
            tempdir,
        ),
        _run_code_protection_case(
            "recent_non_code",
            [{"role": "tool", "tool_call_id": "toolu_1", "content": plain}],
            plain,
            True,
            tempdir,
        ),
    ]


def _run_cache_control_case(name: str, messages: list[Any], expected: str, tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{CACHE_CONTROL_GROUP}-{name}.db"))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    first_content = out.messages[0]["content"][0]
    output = first_content["content"] if "content" in first_content else first_content["text"]
    failures: list[str] = []
    if output != expected:
        failures.append("cache_control content block was compressed")
    if first_content.get("cache_control") != {"type": "ephemeral"}:
        failures.append("cache_control metadata was not preserved")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("cache_control block emitted compression stats")

    return CaseResult(
        group=CACHE_CONTROL_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["cache_control"] if not failures else [],
        failures=failures,
    )


def _run_cache_control_cases(tempdir: str) -> list[CaseResult]:
    anthropic_payload = "cache controlled anthropic payload line\n" * 120
    tool_payload = "cache controlled tool text block payload line\n" * 120
    return [
        _run_cache_control_case(
            "anthropic_tool_result",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "cache_control": {"type": "ephemeral"},
                            "content": anthropic_payload,
                        }
                    ],
                }
            ],
            anthropic_payload,
            tempdir,
        ),
        _run_cache_control_case(
            "tool_text_block",
            [
                {
                    "role": "tool",
                    "tool_call_id": "toolu_1",
                    "content": [
                        {
                            "type": "text",
                            "cache_control": {"type": "ephemeral"},
                            "text": tool_payload,
                        }
                    ],
                }
            ],
            tool_payload,
            tempdir,
        ),
    ]


def _run_cache_aligner_case() -> CaseResult:
    live_payload = "live alpha beta gamma delta epsilon zeta eta theta iota kappa " * 40
    messages = [
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "name": "bash",
            "content": [
                {"type": "text", "text": live_payload},
                {"type": "text", "text": "prefix-" + "b" * 1000, "cache_zone": "prefix"},
                {"type": "text", "text": "cachezone-" + "c" * 1000, "cacheZone": "prefix"},
                {"type": "text", "text": "fixed-" + "d" * 1000, "mutable": False},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_2",
            "name": "bash",
            "cache_zone": "prefix",
            "content": "message-prefix-" + "e" * 1000,
        },
    ]
    pi_ai.reset_compression_stats()
    pi_ai.reset_cache_alignment_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    blocks = out.messages[0]["content"]
    failures: list[str] = []
    if blocks[0]["text"] != "COMPRESSED:live alp":
        failures.append("live dict block was not compressed")
    if blocks[1]["text"] != "prefix-" + "b" * 1000:
        failures.append("dict cache_zone prefix block was compressed")
    if blocks[2]["text"] != "cachezone-" + "c" * 1000:
        failures.append("dict cacheZone prefix block was compressed")
    if blocks[3]["text"] != "fixed-" + "d" * 1000:
        failures.append("dict immutable block was compressed")
    if out.messages[1]["content"] != "message-prefix-" + "e" * 1000:
        failures.append("dict prefix message was compressed")
    if pi_ai.get_compression_stats().compressions_by_strategy != {"text": 1}:
        failures.append("cache-aligner case did not emit exactly one live text compression")

    detector_prompt = (
        "Session: 550e8400-e29b-41d4-a716-446655440000\n"
        "Hash: d41d8cd98f00b204e9800998ecf8427e"
    )
    pi_ai.reset_cache_alignment_stats()
    pi_ai.register_compressor(lambda text: text)
    try:
        detector_out = pi_ai.compress_context(_LooseContext([], system_prompt=detector_prompt))
        subscription_out = pi_ai.compress_context(
            _LooseContext(
                [],
                system_prompt="Session: 550e8400-e29b-41d4-a716-446655440000",
                compression_auth_mode="subscription",
            )
        )
    finally:
        pi_ai.unregister_compressor()
    detector_stats = pi_ai.get_cache_alignment_stats()
    if detector_out.system_prompt != detector_prompt:
        failures.append("cache aligner detector mutated PAYG system prompt")
    if subscription_out.system_prompt != "Session: 550e8400-e29b-41d4-a716-446655440000":
        failures.append("cache aligner detector mutated subscription system prompt")
    if detector_stats.total_scans != 1:
        failures.append("cache aligner detector did not scan PAYG system prompt")
    if detector_stats.total_findings != 2:
        failures.append("cache aligner detector did not count volatile PAYG findings")
    if detector_stats.findings_by_label != {"hex_hash": 1, "uuid": 1}:
        failures.append("cache aligner detector labels did not match Headroom volatile classes")
    if detector_stats.skipped_by_policy != 1:
        failures.append("cache aligner detector did not honor subscription policy skip")

    return CaseResult(
        group=CACHE_ALIGNER_GROUP,
        name="dict_cache_zone_metadata",
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["cache_zone_stability", "detector_only"] if not failures else [],
        failures=failures,
    )


def _large_png_base64() -> str:
    from PIL import Image

    img = Image.new("RGB", (900, 700))
    rng = random.Random(1234)
    pixels = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(900 * 700)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_size(data: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data))) as img:
        return img.size


def _image_payload_bytes_from_data_url(url: str) -> bytes | None:
    if not url.startswith("data:image/jpeg;base64,"):
        return None
    return base64.b64decode(url.split(",", 1)[1])


def _run_image_compression_cases() -> list[CaseResult]:
    original_data = _large_png_base64()
    original_bytes = len(base64.b64decode(original_data))
    cases = [
        (
            "openai_dict_image_url",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{original_data}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            lambda out: _image_payload_bytes_from_data_url(out.messages[0]["content"][1]["image_url"]["url"]),
        ),
        (
            "anthropic_dict_image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": original_data},
                        },
                    ],
                }
            ],
            lambda out: (
                base64.b64decode(out.messages[0]["content"][1]["source"]["data"])
                if out.messages[0]["content"][1]["source"].get("media_type") == "image/jpeg"
                else None
            ),
        ),
        (
            "google_dict_inline_data",
            [
                {
                    "role": "user",
                    "content": [
                        {"text": "What do you see?"},
                        {"inlineData": {"mimeType": "image/png", "data": original_data}},
                    ],
                }
            ],
            lambda out: (
                base64.b64decode(out.messages[0]["content"][1]["inlineData"]["data"])
                if out.messages[0]["content"][1]["inlineData"].get("mimeType") == "image/jpeg"
                else None
            ),
        ),
    ]

    results: list[CaseResult] = []
    for name, messages, image_getter in cases:
        pi_ai.reset_compression_stats()
        pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
        try:
            out = pi_ai.compress_context(_LooseContext(messages))
        finally:
            pi_ai.unregister_compressor()
        failures: list[str] = []
        compressed = image_getter(out)
        if compressed is None:
            failures.append("image block was not converted to image/jpeg")
            after_bytes = original_bytes
        else:
            after_bytes = len(compressed)
            if after_bytes >= original_bytes:
                failures.append("compressed image bytes were not smaller than original")
            encoded = base64.b64encode(compressed).decode("ascii")
            if max(_image_size(encoded)) > pi_ai.compression.IMAGE_MAX_DIMENSION:
                failures.append("compressed image exceeded max dimension")
        if pi_ai.get_compression_stats().compressions_by_strategy != {"image_resize": 1}:
            failures.append("image compression did not emit exactly one image_resize stat")
        results.append(
            CaseResult(
                group=IMAGE_COMPRESSION_GROUP,
                name=name,
                ok=not failures,
                before_chars=original_bytes,
                after_chars=after_bytes,
                checks=["image_resize"] if not failures else [],
                failures=failures,
            )
        )

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        out = pi_ai.compress_context(
            _LooseContext(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "cache_zone": "prefix",
                                "image_url": {"url": f"data:image/png;base64,{original_data}"},
                            }
                        ],
                    }
                ]
            )
        )
    finally:
        pi_ai.unregister_compressor()
    failures = []
    if out.messages[0]["content"][0]["image_url"]["url"] != f"data:image/png;base64,{original_data}":
        failures.append("prefix cache-zone image was rewritten")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("prefix cache-zone image emitted compression stats")
    results.append(
        CaseResult(
            group=IMAGE_COMPRESSION_GROUP,
            name="cached_dict_image_skipped",
            ok=not failures,
            before_chars=original_bytes,
            after_chars=original_bytes,
            checks=["cache_zone_skip"] if not failures else [],
            failures=failures,
        )
    )

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        out = pi_ai.compress_context(
            _LooseContext(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{original_data}"},
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "bash",
                        "content": "tool output\n" * 240,
                    },
                ],
                compression_image_optimize=False,
            )
        )
    finally:
        pi_ai.unregister_compressor()
    failures = []
    if out.messages[0]["content"][1]["image_url"]["url"] != f"data:image/png;base64,{original_data}":
        failures.append("image optimize disabled still rewrote image bytes")
    if out.messages[1]["content"] != "COMPRESSED:tool out":
        failures.append("image optimize disabled also disabled text compression")
    stats = pi_ai.get_compression_stats()
    if stats.compressions_by_strategy != {"text": 1}:
        failures.append("image optimize disabled emitted unexpected compression stats")
    unit_stats = pi_ai.get_unit_outcome_stats()
    if unit_stats.outcomes_by_reason.get("image_optimize_disabled") != 1:
        failures.append("image optimize disabled did not record a unit outcome")
    results.append(
        CaseResult(
            group=IMAGE_COMPRESSION_GROUP,
            name="image_optimize_disabled_preserves_images_only",
            ok=not failures,
            before_chars=original_bytes,
            after_chars=original_bytes,
            checks=["image_optimize_disabled"] if not failures else [],
            failures=failures,
        )
    )

    original_ocr = pi_ai.compression._ocr_extract_image_text
    pi_ai.compression._ocr_extract_image_text = lambda _image_bytes: "Traceback line 42\nOperationalError"
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        out = pi_ai.compress_context(
            _LooseContext(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read the text in this screenshot"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{original_data}"},
                            },
                        ],
                    }
                ]
            )
        )
    finally:
        pi_ai.unregister_compressor()
        pi_ai.compression._ocr_extract_image_text = original_ocr
    failures = []
    content = out.messages[0]["content"]
    if len(content) < 2 or content[1].get("type") != "text":
        failures.append("text-intent image was not replaced by an OCR text block")
        after_chars = original_bytes
    else:
        text = content[1].get("text", "")
        after_chars = len(text)
        if "[OCR from image]" not in text or "OperationalError" not in text:
            failures.append("OCR text block did not include extracted image text")
    if pi_ai.get_compression_stats().compressions_by_strategy != {"image_ocr": 1}:
        failures.append("image OCR did not emit exactly one image_ocr stat")
    results.append(
        CaseResult(
            group=IMAGE_COMPRESSION_GROUP,
            name="openai_dict_image_ocr_transcode",
            ok=not failures,
            before_chars=original_bytes,
            after_chars=after_chars,
            checks=["image_ocr"] if not failures else [],
            failures=failures,
        )
    )
    return results


def _run_image_log_redaction_cases() -> list[CaseResult]:
    big = "A" * (IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)
    image_payload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image", "source": {"type": "base64", "data": big}},
        ],
    }
    redacted = redact_image_base64(image_payload)
    failures: list[str] = []
    expected = IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(big))
    if redacted["content"][1]["source"]["data"] != expected:
        failures.append("image source.data payload was not redacted with byte-count placeholder")
    if redacted["content"][0] != {"type": "text", "text": "describe this image"}:
        failures.append("non-image text block was modified during image redaction")

    data_url = f"data:image/png;base64,{big}"
    data_url_redacted = redact_image_base64({"response_content": data_url})
    data_url_failures: list[str] = []
    if data_url_redacted["response_content"] != IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(data_url)):
        data_url_failures.append("data:image URL outside image path was not redacted")

    non_image = {"signature": big, "arguments": big}
    non_image_redacted = redact_image_base64(non_image)
    non_image_failures: list[str] = []
    if non_image_redacted != non_image:
        non_image_failures.append("non-image base64-shaped fields were redacted")

    return [
        CaseResult(
            group=IMAGE_LOG_REDACTION_GROUP,
            name="image_path_redaction",
            ok=not failures,
            before_chars=len(big),
            after_chars=len(expected),
            checks=["image_log_redaction"] if not failures else [],
            failures=failures,
        ),
        CaseResult(
            group=IMAGE_LOG_REDACTION_GROUP,
            name="data_url_redaction",
            ok=not data_url_failures,
            before_chars=len(data_url),
            after_chars=len(IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(data_url))),
            checks=["data_url_redaction"] if not data_url_failures else [],
            failures=data_url_failures,
        ),
        CaseResult(
            group=IMAGE_LOG_REDACTION_GROUP,
            name="non_image_base64_passthrough",
            ok=not non_image_failures,
            before_chars=len(json.dumps(non_image)),
            after_chars=len(json.dumps(non_image_redacted)),
            checks=["non_image_passthrough"] if not non_image_failures else [],
            failures=non_image_failures,
        ),
    ]


def _run_compression_summary_case(tempdir: str) -> CaseResult:
    rows = [
        {
            "id": i,
            "status": "active",
            "name": f"item-{i}",
            "payload": {"common": i % 5, f"unique_{i}": "y" * 80},
        }
        for i in range(60)
    ]
    rows.extend(
        {
            "id": 100 + i,
            "status": "archived",
            "name": f"old-{i}",
            "payload": {"common": i % 5, f"archived_unique_{i}": "z" * 80},
        }
        for i in range(20)
    )
    original = json.dumps(rows, indent=2)
    store = CCRStore(os.path.join(tempdir, f"{COMPRESSION_SUMMARY_GROUP}-dropped.db"))
    output = compress(original, store)
    failures: list[str] = []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        parsed = []
        failures.append(f"compressed output was not JSON: {exc}")
    sentinel = parsed[-1] if parsed and isinstance(parsed[-1], dict) else {}
    summary = sentinel.get("_ccr_summary", "")
    if "rows omitted" not in summary:
        failures.append("dropped-row summary did not report omitted rows")
    if "status:" not in summary:
        failures.append("dropped-row summary did not include category field")
    if "active=" not in summary and "archived=" not in summary:
        failures.append("dropped-row summary did not include status counts")
    if "https://" in summary or len(summary) >= 500:
        failures.append("dropped-row summary included noisy/oversized content")
    if not _is_reversible(output, original, store):
        failures.append("summarized row-offload output was not CCR-reversible")

    return CaseResult(
        group=COMPRESSION_SUMMARY_GROUP,
        name="dropped_row_categories",
        ok=not failures,
        before_chars=len(original),
        after_chars=len(output),
        checks=["dropped_row_summary"] if not failures else [],
        failures=failures,
    )


def _determinism_fixtures() -> dict[str, str]:
    search_lines = []
    for i in range(1, 180):
        body = f"ordinary search result {i}"
        if i == 57:
            body = "ERROR TARGET_DETERMINISM preserve this search signal"
        search_lines.append(f"src/service.py:{i}:{body}")

    diff_lines = []
    for file_idx in range(30):
        diff_lines.extend(
            [
                f"diff --git a/file_{file_idx}.py b/file_{file_idx}.py",
                f"--- a/file_{file_idx}.py",
                f"+++ b/file_{file_idx}.py",
                "@@ -1,8 +1,8 @@",
            ]
        )
        for line_idx in range(8):
            diff_lines.append(f" context line {file_idx}-{line_idx}")
            diff_lines.append(f"-old value {file_idx}-{line_idx}")
            diff_lines.append(f"+new value {file_idx}-{line_idx}")

    log_lines = [f"INFO worker completed item {i}" for i in range(160)]
    log_lines.insert(83, "ERROR TARGET_DETERMINISM failed payment import")

    rows = [
        {
            "id": i,
            "status": "active" if i % 3 else "archived",
            "score": i,
            "payload": {"common": i % 5, f"unique_{i}": "x" * 100},
        }
        for i in range(90)
    ]
    rows[44]["status"] = "failed"
    rows[44]["payload"]["message"] = "TARGET_DETERMINISM row signal"

    return {
        "plain_text": "\n".join(
            [f"plain section {i} with repeated operational detail" for i in range(260)]
        ),
        "search_results": "\n".join(search_lines),
        "diff": "\n".join(diff_lines),
        "log": "\n".join(log_lines),
        "json_row_offload": json.dumps(rows, indent=2),
    }


def _run_determinism_cases(tempdir: str) -> list[CaseResult]:
    results: list[CaseResult] = []
    for name, original in _determinism_fixtures().items():
        first_store = CCRStore(os.path.join(tempdir, f"{DETERMINISM_GROUP}-{name}-a.db"))
        second_store = CCRStore(os.path.join(tempdir, f"{DETERMINISM_GROUP}-{name}-b.db"))
        first = compress(original, first_store)
        second = compress(original, second_store)
        failures: list[str] = []

        if first == original:
            failures.append("determinism fixture passed through instead of exercising compression")
        if first != second:
            failures.append("fresh CCR stores produced different compressed bytes")
        if not _is_reversible(first, original, first_store):
            failures.append("first compressed output was not recoverable")
        if not _is_reversible(second, original, second_store):
            failures.append("second compressed output was not recoverable")

        results.append(
            CaseResult(
                group=DETERMINISM_GROUP,
                name=name,
                ok=not failures,
                before_chars=len(original),
                after_chars=len(first),
                checks=["byte_stable", "reversible"] if not failures else [],
                failures=failures,
            )
        )
    return results


def _run_compression_policy_cases() -> list[CaseResult]:
    results: list[CaseResult] = []
    policy_expectations = {
        "payg": (AuthMode.PAYG, False, True, 128, 0.45, False),
        "oauth": (AuthMode.OAUTH, False, True, 128, 0.45, False),
        "subscription": (AuthMode.SUBSCRIPTION, True, False, 32, 0.25, True),
    }
    for name, (mode, live_only, aligner, volatile_threshold, lossy_ratio, toin_read_only) in policy_expectations.items():
        policy = policy_for_mode(mode)
        failures: list[str] = []
        if policy.live_zone_only is not live_only:
            failures.append("live_zone_only mismatch")
        if policy.cache_aligner_enabled is not aligner:
            failures.append("cache_aligner_enabled mismatch")
        if policy.volatile_token_threshold != volatile_threshold:
            failures.append("volatile_token_threshold mismatch")
        if abs(policy.max_lossy_ratio - lossy_ratio) > 1e-9:
            failures.append("max_lossy_ratio mismatch")
        if policy.toin_read_only is not toin_read_only:
            failures.append("toin_read_only mismatch")
        results.append(
            CaseResult(
                group=COMPRESSION_POLICY_GROUP,
                name=f"{name}_mode",
                ok=not failures,
                before_chars=1,
                after_chars=1,
                checks=["mode_values"] if not failures else [],
                failures=failures,
            )
        )

    policy = policy_for_mode(AuthMode.PAYG)
    formula_failures: list[str] = []
    if abs(policy.net_mutation_gain(2_000, 50_000, 10.0, 1.0) - (-55_500.0)) >= 1.0:
        formula_failures.append("small-shave/deep-suffix gain drifted")
    if policy.should_mutate_deep(2_000, 50_000, 10.0, 1.0):
        formula_failures.append("small-shave/deep-suffix decision drifted")
    if abs(policy.net_mutation_gain(50_000, 10_000, 3.0, 1.0) - 3_500.0) >= 1.0:
        formula_failures.append("big-shave/shallow-suffix gain drifted")
    if not policy.should_mutate_deep(50_000, 10_000, 3.0, 1.0):
        formula_failures.append("big-shave/shallow-suffix decision drifted")
    if abs(policy.break_even_reads(2_000, 50_000) - 287.5) >= 0.5:
        formula_failures.append("2K/50K break-even drifted")
    if abs(policy.break_even_reads(50_000, 10_000) - 2.3) >= 0.05:
        formula_failures.append("50K/10K break-even drifted")
    results.append(
        CaseResult(
            group=COMPRESSION_POLICY_GROUP,
            name="net_cost_formula",
            ok=not formula_failures,
            before_chars=1,
            after_chars=1,
            checks=["net_cost"] if not formula_failures else [],
            failures=formula_failures,
        )
    )

    old_env = os.environ.get("HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT")
    os.environ["HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"] = "off"
    try:
        fallback_ok = resolve_policy(AuthMode.SUBSCRIPTION) == policy_for_mode(AuthMode.PAYG)
    finally:
        if old_env is None:
            os.environ.pop("HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT", None)
        else:
            os.environ["HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"] = old_env
    results.append(
        CaseResult(
            group=COMPRESSION_POLICY_GROUP,
            name="enforcement_fallback",
            ok=fallback_ok,
            before_chars=1,
            after_chars=1,
            checks=["fallback_payg"] if fallback_ok else [],
            failures=[] if fallback_ok else ["disabled enforcement did not fall back to PAYG"],
        )
    )

    def learning_compressor(text: str) -> str:
        return f"COMPRESSED:{text[:24]}"

    learning_failures: list[str] = []
    pi_ai.reset_compression_learning_stats()
    pi_ai.register_compressor(learning_compressor)
    try:
        for auth_mode in ("payg", "oauth", "subscription"):
            out = pi_ai.compress_context(
                _LooseContext(
                    [{"role": "tool", "tool_call_id": f"toolu_{auth_mode}", "content": f"{auth_mode} output " * 260}],
                    compression_auth_mode=auth_mode,
                )
            )
            if not str(out.messages[0]["content"]).startswith("COMPRESSED:"):
                learning_failures.append(f"{auth_mode} tool output did not compress")
    finally:
        pi_ai.unregister_compressor()
    learning_stats = pi_ai.get_compression_learning_stats()
    if learning_stats.total_events != 2:
        learning_failures.append(f"expected two writable learning events, got {learning_stats.total_events}")
    if learning_stats.total_skipped_read_only != 1:
        learning_failures.append(
            f"expected one read-only learning skip, got {learning_stats.total_skipped_read_only}"
        )
    if learning_stats.events_by_strategy.get("text") != 2:
        learning_failures.append("PAYG/OAuth text learning strategy counts drifted")
    if learning_stats.skipped_by_strategy.get("text") != 1:
        learning_failures.append("subscription text learning skip count drifted")
    results.append(
        CaseResult(
            group=COMPRESSION_POLICY_GROUP,
            name="toin_learning_gate",
            ok=not learning_failures,
            before_chars=1,
            after_chars=1,
            checks=["toin_gate"] if not learning_failures else [],
            failures=learning_failures,
        )
    )
    return results


def _run_marker_pinning_case(name: str, messages: list[Any], expected: str, tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{MARKER_PINNING_GROUP}-{name}.db"))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    content = out.messages[0]["content"]
    if isinstance(content, list):
        first = content[0]
        output = first["content"] if "content" in first else first["text"]
    else:
        output = content
    failures: list[str] = []
    if output != expected:
        failures.append("already-compressed tool output was rewritten")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("already-compressed tool output emitted compression stats")

    return CaseResult(
        group=MARKER_PINNING_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["marker_pinning"] if not failures else [],
        failures=failures,
    )


def _run_marker_pinning_cases(tempdir: str) -> list[CaseResult]:
    own = ("prefix\n" * 200) + "[CCR:abcdef123456] compressed previous payload\n" + ("noise\n" * 200)
    original = ("prefix\n" * 200) + "[CCR:abcdef123456] compressed previous payload\n" + ("noise\n" * 200)
    return [
        _run_marker_pinning_case(
            "anthropic_tool_result",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": original,
                        }
                    ],
                }
            ],
            original,
            tempdir,
        ),
        _run_marker_pinning_case(
            "openai_tool_role",
            [{"role": "tool", "tool_call_id": "toolu_1", "content": own}],
            own,
            tempdir,
        ),
    ]


def _run_marker_preserving_case() -> CaseResult:
    marker = "Retrieve more: hash=abcdef123456"
    original = ("prefix\n" * 200) + marker + "\n" + ("noise\n" * 200)
    messages = [{"role": "tool", "tool_call_id": "toolu_1", "content": original}]
    calls: list[str] = []

    def span_compressor(text: str) -> str:
        calls.append(text)
        return f"<compressed:{len(text)}>"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(span_compressor)
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[0]["content"]
    failures: list[str] = []
    if output == original:
        failures.append("live retrieval-marker text was pinned instead of surrounding spans being compressed")
    if marker not in output:
        failures.append("retrieval marker was not preserved")
    if calls != ["prefix\n" * 200, "\n" + "noise\n" * 200]:
        failures.append("compressor did not receive exactly the marker-free surrounding spans")
    if pi_ai.get_compression_stats().compressions_by_strategy.get("text") != 1:
        failures.append("marker-preserving compression did not emit exactly one text stat")

    return CaseResult(
        group=MARKER_PRESERVING_GROUP,
        name="live_tool_result_retrieval_marker",
        ok=not failures,
        before_chars=len(original),
        after_chars=len(output),
        checks=["marker_preserving"] if not failures else [],
        failures=failures,
    )


def _run_custom_tag_protection_cases(tempdir: str) -> list[CaseResult]:
    protected = (
        '<system-reminder priority="critical">'
        "Rule 1: validate input. Rule 2: never skip auth."
        "</system-reminder>"
    )
    original = (
        ("verbose setup line\n" * 140)
        + protected
        + "\n"
        + "<tool_call>{\"name\":\"search\",\"args\":{\"query\":\"test\"}}</tool_call>"
        + "\n"
        + ("verbose tail line\n" * 140)
    )
    store = CCRStore(os.path.join(tempdir, "custom-tags.db"))
    output = compress(original, store)
    failures: list[str] = []
    if protected not in output:
        failures.append("protected system-reminder tag block did not survive compression")
    if "<tool_call>" not in output or '"name":"search"' not in output:
        failures.append("protected tool_call tag block did not survive compression")
    if "Preserved custom tags" in output:
        failures.append("legacy missing-placeholder appendix leaked into compressed output")
    if len(output) >= len(original):
        failures.append("tag-protected payload did not compress")
    if "[CCR:" in output and store.get(_handle_from_marker(output)) != original:
        failures.append("tag-protected compressed output was not CCR-reversible")

    lost_placeholder_output = _restore_custom_tags(
        "head {{TAU_TAG_1}} tail",
        [
            ("{{TAU_TAG_0}}", "<system-reminder>lost</system-reminder>"),
            ("{{TAU_TAG_1}}", "<context>kept</context>"),
        ],
    )
    restore_failures: list[str] = []
    if lost_placeholder_output != "head <context>kept</context> tail":
        restore_failures.append("lost placeholder was not discarded byte-for-byte")
    if "lost" in lost_placeholder_output or "Preserved custom tags" in lost_placeholder_output:
        restore_failures.append("lost placeholder tag content was re-injected")

    invariant_failures: list[str] = []
    duplicate = (
        "<system-reminder>same</system-reminder> middle "
        "<system-reminder>same</system-reminder>"
    )
    duplicate_cleaned, duplicate_blocks = _protect_custom_tags(duplicate)
    if duplicate_cleaned != "{{TAU_TAG_0}} middle {{TAU_TAG_1}}":
        invariant_failures.append("duplicate custom blocks were not replaced by distinct placeholders")
    if len(duplicate_blocks) != 2 or duplicate_blocks[0][0] == duplicate_blocks[1][0]:
        invariant_failures.append("duplicate custom blocks did not get distinct placeholder records")
    if _restore_custom_tags(duplicate_cleaned, duplicate_blocks) != duplicate:
        invariant_failures.append("duplicate custom block roundtrip failed")

    nested = "<lvl>" * 60 + "core" + "</lvl>" * 60
    nested_cleaned, nested_blocks = _protect_custom_tags(nested)
    if nested_cleaned != "{{TAU_TAG_0}}" or len(nested_blocks) != 1:
        invariant_failures.append("deep nested custom tags leaked instead of protecting the outer span")
    if _restore_custom_tags(nested_cleaned, nested_blocks) != nested:
        invariant_failures.append("deep nested custom tag roundtrip failed")

    self_closing = "<marker/> middle <marker/>"
    self_cleaned, self_blocks = _protect_custom_tags(self_closing)
    if self_cleaned != "{{TAU_TAG_0}} middle {{TAU_TAG_1}}":
        invariant_failures.append("duplicate self-closing custom tags were not distinct")
    if len(self_blocks) != 2 or self_blocks[0][0] == self_blocks[1][0]:
        invariant_failures.append("duplicate self-closing tags did not get distinct placeholder records")
    if _restore_custom_tags(self_cleaned, self_blocks) != self_closing:
        invariant_failures.append("duplicate self-closing custom tag roundtrip failed")

    collision = (
        "User wrote {{TAU_TAG_0}} on purpose. "
        "<system-reminder>real one</system-reminder>"
    )
    collision_cleaned, collision_blocks = _protect_custom_tags(collision)
    if not collision_blocks or collision_blocks[0][0] == "{{TAU_TAG_0}}":
        invariant_failures.append("literal placeholder collision was not avoided")
    if _restore_custom_tags(collision_cleaned, collision_blocks) != collision:
        invariant_failures.append("literal placeholder collision roundtrip failed")

    return [
        CaseResult(
            group=CUSTOM_TAG_GROUP,
            name="workflow_tags_survive",
            ok=not failures,
            before_chars=len(original),
            after_chars=len(output),
            checks=["custom_tag_survival"] if not failures else [],
            failures=failures,
        ),
        CaseResult(
            group=CUSTOM_TAG_GROUP,
            name="lost_placeholder_discard",
            ok=not restore_failures,
            before_chars=1,
            after_chars=1,
            checks=["discard_lost_placeholder"] if not restore_failures else [],
            failures=restore_failures,
        ),
        CaseResult(
            group=CUSTOM_TAG_GROUP,
            name="tag_protector_invariants",
            ok=not invariant_failures,
            before_chars=len(duplicate) + len(nested) + len(self_closing) + len(collision),
            after_chars=len(duplicate_cleaned)
            + len(nested_cleaned)
            + len(self_cleaned)
            + len(collision_cleaned),
            checks=["nested_duplicate_collision_invariants"] if not invariant_failures else [],
            failures=invariant_failures,
        ),
    ]


def _run_tool_exclusion_case(
    name: str,
    tool_name: str,
    payload: str,
    should_compress: bool,
    tempdir: str,
) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{TOOL_EXCLUSION_GROUP}-{name}.db"))
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": payload}],
        },
    ]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, store))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[1]["content"][0]["content"]
    failures: list[str] = []
    if should_compress:
        marker = CCR_HANDLE_RE.search(output)
        if marker is None:
            failures.append(f"{tool_name} output did not compress")
        elif store.get(marker.group(1)) != payload:
            failures.append(f"{tool_name} CCR handle did not recover original")
    else:
        if output != payload:
            failures.append(f"fresh excluded {tool_name} output was compressed")
        if pi_ai.get_compression_stats().compressions_by_strategy:
            failures.append(f"fresh excluded {tool_name} output emitted compression stats")

    return CaseResult(
        group=TOOL_EXCLUSION_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["tool_exclusion"] if not failures else [],
        failures=failures,
    )


def _run_tool_exclusion_cases(tempdir: str) -> list[CaseResult]:
    read_payload = "fresh exact file line\n" * 120
    grep_payload = "\n".join(f"src/app.py:{i}:target match {i}" for i in range(1, 140))
    bash_payload = "bash output line\n" * 120
    return [
        _run_tool_exclusion_case("fresh_read", "Read", read_payload, False, tempdir),
        _run_tool_exclusion_case("fresh_grep", "Grep", grep_payload, False, tempdir),
        _run_tool_exclusion_case("bash_compressible", "Bash", bash_payload, True, tempdir),
    ]


def _run_non_shrinking_rejection_case(tempdir: str) -> CaseResult:
    payload = "short words " * 80
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": payload}],
        }
    ]
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: text + (" expanded" * 80))
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[0]["content"][0]["content"]
    failures: list[str] = []
    if output != payload:
        failures.append("non-shrinking provider tool_result replacement was accepted")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("non-shrinking rejection emitted compression stats")

    return CaseResult(
        group=NON_SHRINKING_GROUP,
        name="provider_tool_result",
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["rejected_not_smaller"] if not failures else [],
        failures=failures,
    )


def _run_compression_failure_case(
    name: str,
    compressor: Callable[[str], Any],
    messages: list[Any],
    output_getter: Callable[[Any], str],
    expected: str,
) -> CaseResult:
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = output_getter(out)
    failures: list[str] = []
    if output != expected:
        failures.append("compression failure did not fail open to original content")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("compression failure emitted compression stats")

    return CaseResult(
        group=COMPRESSION_FAILURE_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["fail_open"] if not failures else [],
        failures=failures,
    )


def _run_compression_failure_cases() -> list[CaseResult]:
    tool_payload = "large direct compression payload\n" * 120
    read_payload = "old stale read payload\n" * 120

    def raising_compressor(text: str) -> str:
        raise RuntimeError(f"cannot compress {len(text)} bytes")

    return [
        _run_compression_failure_case(
            "tool_result_exception",
            raising_compressor,
            [{"role": "tool", "tool_call_id": "toolu_1", "content": tool_payload}],
            lambda out: out.messages[0]["content"],
            tool_payload,
        ),
        _run_compression_failure_case(
            "tool_result_non_string",
            lambda text: {"compressed": text[:8]},
            [{"role": "tool", "tool_call_id": "toolu_1", "content": tool_payload}],
            lambda out: out.messages[0]["content"],
            tool_payload,
        ),
        _run_compression_failure_case(
            "read_lifecycle_exception",
            raising_compressor,
            _read_messages(read_payload, edit_after_first=True),
            lambda out: out.messages[1]["content"][0]["content"],
            read_payload,
        ),
        _run_compression_failure_case(
            "read_lifecycle_without_ccr_handle",
            lambda text: f"COMPRESSED:{text[:8]}",
            _read_messages(read_payload, edit_after_first=True),
            lambda out: out.messages[1]["content"][0]["content"],
            read_payload,
        ),
    ]


def _run_compression_circuit_breaker_cases() -> list[CaseResult]:
    results: list[CaseResult] = []
    original = "breaker compression payload alpha beta gamma delta\n" * 80

    old_threshold = os.environ.get("HEADROOM_PIPELINE_BREAKER_THRESHOLD")
    old_cooldown = os.environ.get("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S")
    old_cache_entries = os.environ.get("TAU_COMPRESSION_CACHE_MAX_ENTRIES")

    def restore_env() -> None:
        if old_threshold is None:
            os.environ.pop("HEADROOM_PIPELINE_BREAKER_THRESHOLD", None)
        else:
            os.environ["HEADROOM_PIPELINE_BREAKER_THRESHOLD"] = old_threshold
        if old_cooldown is None:
            os.environ.pop("HEADROOM_PIPELINE_BREAKER_COOLDOWN_S", None)
        else:
            os.environ["HEADROOM_PIPELINE_BREAKER_COOLDOWN_S"] = old_cooldown
        if old_cache_entries is None:
            os.environ.pop("TAU_COMPRESSION_CACHE_MAX_ENTRIES", None)
        else:
            os.environ["TAU_COMPRESSION_CACHE_MAX_ENTRIES"] = old_cache_entries

    try:
        os.environ["TAU_COMPRESSION_CACHE_MAX_ENTRIES"] = "0"
        os.environ["HEADROOM_PIPELINE_BREAKER_THRESHOLD"] = "3"
        os.environ["HEADROOM_PIPELINE_BREAKER_COOLDOWN_S"] = "60"
        calls = 0

        def failing_compressor(_text: str) -> str:
            nonlocal calls
            calls += 1
            raise RuntimeError("compressor down")

        pi_ai.reset_compression_circuit_breaker()
        pi_ai.register_compressor(failing_compressor)
        try:
            for _ in range(4):
                pi_ai.compress_context(
                    _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
                )
            state = pi_ai.get_compression_circuit_breaker_state()
        finally:
            pi_ai.unregister_compressor()
        failures: list[str] = []
        if calls != 3:
            failures.append("breaker did not suppress calls after threshold")
        if not state["open"]:
            failures.append("breaker did not report open after threshold")
        results.append(
            CaseResult(
                group=COMPRESSION_CIRCUIT_BREAKER_GROUP,
                name="opens_after_threshold",
                ok=not failures,
                before_chars=len(original),
                after_chars=len(original),
                checks=["circuit_open"] if not failures else [],
                failures=failures,
            )
        )

        os.environ["HEADROOM_PIPELINE_BREAKER_THRESHOLD"] = "3"
        calls = 0

        def flaky_compressor(text: str) -> str:
            nonlocal calls
            calls += 1
            if calls in {1, 2, 4, 5}:
                raise RuntimeError("temporary compressor failure")
            return f"COMPRESSED:{text[:8]}"

        pi_ai.reset_compression_circuit_breaker()
        pi_ai.register_compressor(flaky_compressor)
        try:
            outputs = [
                pi_ai.compress_context(
                    _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
                ).messages[0]["content"]
                for _ in range(5)
            ]
            state = pi_ai.get_compression_circuit_breaker_state()
        finally:
            pi_ai.unregister_compressor()
        failures = []
        if outputs[2] != "COMPRESSED:breaker ":
            failures.append("successful compression did not run between failures")
        if state["open"]:
            failures.append("success did not reset consecutive failures")
        if calls != 5:
            failures.append("breaker opened despite success reset")
        results.append(
            CaseResult(
                group=COMPRESSION_CIRCUIT_BREAKER_GROUP,
                name="success_resets_failures",
                ok=not failures,
                before_chars=len(original),
                after_chars=sum(len(out) for out in outputs),
                checks=["success_reset"] if not failures else [],
                failures=failures,
            )
        )

        os.environ["HEADROOM_PIPELINE_BREAKER_THRESHOLD"] = "1"
        os.environ["HEADROOM_PIPELINE_BREAKER_COOLDOWN_S"] = "0.05"
        calls = 0

        def cooldown_compressor(text: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("initial failure")
            return f"COMPRESSED:{text[:8]}"

        pi_ai.reset_compression_circuit_breaker()
        pi_ai.register_compressor(cooldown_compressor)
        try:
            first = pi_ai.compress_context(
                _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
            ).messages[0]["content"]
            second = pi_ai.compress_context(
                _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
            ).messages[0]["content"]
            time.sleep(0.1)
            third = pi_ai.compress_context(
                _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
            ).messages[0]["content"]
        finally:
            pi_ai.unregister_compressor()
        failures = []
        if first != original or second != original:
            failures.append("open breaker did not pass through original content")
        if third != "COMPRESSED:breaker ":
            failures.append("breaker did not close after cooldown")
        if calls != 2:
            failures.append("breaker did not skip compressor while open")
        results.append(
            CaseResult(
                group=COMPRESSION_CIRCUIT_BREAKER_GROUP,
                name="cooldown_expires",
                ok=not failures,
                before_chars=len(original),
                after_chars=len(first) + len(second) + len(third),
                checks=["cooldown"] if not failures else [],
                failures=failures,
            )
        )

        os.environ["HEADROOM_PIPELINE_BREAKER_THRESHOLD"] = "0"
        calls = 0
        pi_ai.reset_compression_circuit_breaker()
        pi_ai.register_compressor(failing_compressor)
        try:
            for _ in range(4):
                pi_ai.compress_context(
                    _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
                )
        finally:
            pi_ai.unregister_compressor()
        failures = []
        if calls != 4:
            failures.append("disabled breaker suppressed compressor calls")
        results.append(
            CaseResult(
                group=COMPRESSION_CIRCUIT_BREAKER_GROUP,
                name="disabled_by_env",
                ok=not failures,
                before_chars=len(original),
                after_chars=len(original),
                checks=["disabled"] if not failures else [],
                failures=failures,
            )
        )
    finally:
        pi_ai.reset_compression_circuit_breaker()
        restore_env()
    return results


def _run_inflation_guard_case() -> CaseResult:
    original = "inflation guard payload alpha beta gamma delta\n" * 80
    ctx = _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": original}])
    original_inner = pi_ai.compression._compress_context_inner

    def bloating_inner(context: Any, _fn: Callable[[str], str], _policy: Any) -> Any:
        return context.model_copy(
            update={
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "toolu_1",
                        "content": "PADDING " * 600,
                    }
                ]
            }
        )

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    pi_ai.compression._compress_context_inner = bloating_inner
    try:
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.compression._compress_context_inner = original_inner
        pi_ai.unregister_compressor()

    failures: list[str] = []
    if out is not ctx:
        failures.append("inflated compression did not return original context")
    if out.messages[0]["content"] != original:
        failures.append("inflated compression did not preserve original content")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("inflation guard left successful compression stats behind")

    return CaseResult(
        group=INFLATION_GUARD_GROUP,
        name="reverts_bloated_context",
        ok=not failures,
        before_chars=len(original),
        after_chars=len(out.messages[0]["content"]),
        checks=["reverted"] if not failures else [],
        failures=failures,
    )


def _run_token_floor_case(tempdir: str) -> CaseResult:
    payload = "small output " * 40
    calls: list[str] = []
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": payload}],
        }
    ]

    def eager_compressor(text: str) -> str:
        calls.append(text)
        return "COMPRESSED"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(eager_compressor)
    try:
        out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[0]["content"][0]["content"]
    failures: list[str] = []
    if output != payload:
        failures.append("provider tool_result below token floor was compressed")
    if calls:
        failures.append("compressor was called for provider tool_result below token floor")
    if pi_ai.get_compression_stats().compressions_by_strategy:
        failures.append("below-floor provider tool_result emitted compression stats")

    return CaseResult(
        group=TOKEN_FLOOR_GROUP,
        name="provider_tool_result",
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["min_tokens_to_compress"] if not failures else [],
        failures=failures,
    )


def _run_config_control_case(
    name: str,
    ctx: _LooseContext,
    output_getter: Callable[[Any], str | None],
    expected: str,
    should_compress: bool,
) -> CaseResult:
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    try:
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    output = output_getter(out)
    failures: list[str] = []
    if output != expected:
        failures.append(f"expected {expected!r}, got {output!r}")
    stats = pi_ai.get_compression_stats().compressions_by_strategy
    if should_compress and stats != {"text": 1}:
        failures.append(f"expected one text compression, got {stats}")
    if not should_compress and stats:
        failures.append(f"expected no compression stats, got {stats}")

    return CaseResult(
        group=CONFIG_CONTROL_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(ctx.messages, ensure_ascii=False, default=str))
        + len(ctx.system_prompt or ""),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str))
        + len(getattr(out, "system_prompt", None) or ""),
        checks=["compress_config_controls"] if not failures else [],
        failures=failures,
    )


def _run_provider_config_control_case(tempdir: str) -> CaseResult:
    rows = [10 for _ in range(120)]
    rows[37] = 5000
    payload = json.dumps(rows, indent=2)
    messages = [{"role": "tool", "tool_call_id": "toolu_1", "name": "bash", "content": payload}]
    store = CCRStore(os.path.join(tempdir, f"{CONFIG_CONTROL_GROUP}-provider-structured.db"))

    def config_aware_compressor(text: str) -> str:
        defaults = CompressionConfig()
        return compress(
            text,
            store,
            config=CompressionConfig(
                min_tokens=pi_ai.get_current_compression_min_tokens() or defaults.min_tokens,
                max_items_after_crush=(
                    pi_ai.get_current_compression_max_items_after_crush()
                    or defaults.max_items_after_crush
                ),
                lossless_min_savings_ratio=(
                    pi_ai.get_current_compression_lossless_min_savings_ratio()
                    if pi_ai.get_current_compression_lossless_min_savings_ratio() is not None
                    else defaults.lossless_min_savings_ratio
                ),
                enable_ccr_marker=(
                    pi_ai.get_current_compression_enable_ccr_marker()
                    if pi_ai.get_current_compression_enable_ccr_marker() is not None
                    else defaults.enable_ccr_marker
                ),
            ),
        )

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(config_aware_compressor)
    try:
        out = pi_ai.compress_context(
            _LooseContext(
                messages,
                compression_min_tokens=1,
                compression_max_items_after_crush=8,
                compression_enable_ccr_marker=False,
            )
        )
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[0]["content"]
    failures: list[str] = []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        failures.append(f"provider config output was not JSON: {exc}")
        parsed = []
    if len(parsed) >= len(rows):
        failures.append("provider config controls did not lower the compression floor")
    if len(parsed) > 8:
        failures.append("provider config controls did not apply structured item budget")
    if 5000 not in parsed:
        failures.append("provider config controls lost numeric outlier")
    if any(isinstance(item, dict) and "_ccr_dropped" in item for item in parsed):
        failures.append("provider config controls did not disable CCR dropped-row marker")
    if pi_ai.get_compression_stats().compressions_by_strategy != {"text": 1}:
        failures.append("provider config controls did not emit exactly one text stat")

    return CaseResult(
        group=CONFIG_CONTROL_GROUP,
        name="provider_structured_knobs",
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["provider_config_controls"] if not failures else [],
        failures=failures,
    )


def _run_config_control_cases(tempdir: str) -> list[CaseResult]:
    user_payload = "user pasted context " * 120
    system_payload = "system instructions " * 120
    return [
        _run_config_control_case(
            "user_default_protected",
            _LooseContext([{"role": "user", "content": user_payload}]),
            lambda out: out.messages[0]["content"],
            user_payload,
            False,
        ),
        _run_config_control_case(
            "user_opt_in_compresses",
            _LooseContext(
                [{"role": "user", "content": user_payload}],
                compression_compress_user_messages=True,
            ),
            lambda out: out.messages[0]["content"],
            "COMPRESSED:user pas",
            True,
        ),
        _run_config_control_case(
            "system_prompt_default_protected",
            _LooseContext([], system_prompt=system_payload),
            lambda out: out.system_prompt,
            system_payload,
            False,
        ),
        _run_config_control_case(
            "system_prompt_opt_in_compresses",
            _LooseContext(
                [],
                system_prompt=system_payload,
                compression_compress_system_messages=True,
            ),
            lambda out: out.system_prompt,
            "COMPRESSED:system i",
            True,
        ),
        _run_provider_config_control_case(tempdir),
    ]


def _run_target_ratio_case(tempdir: str) -> CaseResult:
    lines = [f"intro paragraph filler {i}" for i in range(40)]
    lines.extend(f"ANCHOR-{i:04d} important detail line {i}" for i in range(120))
    lines.extend(f"tail paragraph filler {i}" for i in range(40))
    payload = "\n".join(lines)
    messages = [{"role": "tool", "tool_call_id": "toolu_1", "content": payload}]

    default_store = CCRStore(os.path.join(tempdir, f"{TARGET_RATIO_GROUP}-default.db"))
    targeted_store = CCRStore(os.path.join(tempdir, f"{TARGET_RATIO_GROUP}-targeted.db"))

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(lambda text: compress(text, default_store))
    try:
        default_out = pi_ai.compress_context(_LooseContext(messages))
    finally:
        pi_ai.unregister_compressor()

    pi_ai.reset_compression_stats()

    def target_aware_compressor(text: str) -> str:
        return compress(
            text,
            targeted_store,
            target_ratio=pi_ai.get_current_compression_target_ratio(),
        )

    pi_ai.register_compressor(target_aware_compressor)
    try:
        targeted_out = pi_ai.compress_context(
            _LooseContext(messages, compression_target_ratio=0.15)
        )
    finally:
        pi_ai.unregister_compressor()

    default_text = default_out.messages[0]["content"]
    targeted_text = targeted_out.messages[0]["content"]
    failures: list[str] = []
    marker = CCR_HANDLE_RE.search(targeted_text)
    if marker is None:
        failures.append("target-ratio compression did not emit a CCR handle")
    elif targeted_store.get(marker.group(1)) != payload:
        failures.append("target-ratio CCR handle did not recover original")
    if len(targeted_text) >= len(default_text):
        failures.append("target-ratio output was not smaller than default compression")
    if pi_ai.get_compression_stats().compressions_by_strategy != {"text": 1}:
        failures.append("target-ratio compression did not emit exactly one text stat")

    return CaseResult(
        group=TARGET_RATIO_GROUP,
        name="plain_text",
        ok=not failures,
        before_chars=len(json.dumps(messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(targeted_out.messages, ensure_ascii=False, default=str)),
        checks=["target_ratio"] if not failures else [],
        failures=failures,
    )


def _read_messages(first_read: str, second_read: str | None = None, *, edit_after_first: bool = False) -> list[Any]:
    messages: list[Any] = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "src/app.py"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": first_read}],
        },
    ]
    if edit_after_first:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "edit-1", "name": "Edit", "input": {"file_path": "src/app.py"}}],
            }
        )
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "edit-1", "content": "ok"}]})
    if second_read is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "read-2", "name": "Read", "input": {"file_path": "src/app.py"}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "read-2", "content": second_read}],
            }
        )
    return messages


def _run_read_lifecycle_case(name: str, ctx: _LooseContext, expected_prefix: str | None, tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{READ_LIFECYCLE_GROUP}-{name}.db"))
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(
        lambda text: compress(
            text,
            store,
            force=pi_ai.get_current_compression_force_compression(),
        )
    )
    try:
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    output = out.messages[1]["content"][0]["content"]
    failures: list[str] = []
    if expected_prefix is None:
        if output != ctx.messages[1]["content"][0]["content"]:
            failures.append("superseded Read was compressed by default")
        if pi_ai.get_compression_stats().compressions_by_strategy:
            failures.append("default superseded Read emitted compression stats")
    else:
        if not output.startswith(expected_prefix):
            failures.append(f"expected read lifecycle marker prefix {expected_prefix!r}")
        marker = READ_MARKER_HANDLE_RE.search(output)
        if marker is None:
            failures.append("read lifecycle marker did not include a CCR handle")
        elif store.get(marker.group(1)) != ctx.messages[1]["content"][0]["content"]:
            failures.append("read lifecycle CCR handle did not recover original")

    return CaseResult(
        group=READ_LIFECYCLE_GROUP,
        name=name,
        ok=not failures,
        before_chars=len(json.dumps(ctx.messages, ensure_ascii=False, default=str)),
        after_chars=len(json.dumps(out.messages, ensure_ascii=False, default=str)),
        checks=["read_lifecycle"] if not failures else [],
        failures=failures,
    )


def _run_read_lifecycle_cases(tempdir: str) -> list[CaseResult]:
    first_read = "first exact source snapshot line\n" * 120
    second_read = "second exact source snapshot line\n" * 120
    mid_read = "mid exact source snapshot\n" * 24
    return [
        _run_read_lifecycle_case(
            "superseded_default_passthrough",
            _LooseContext(_read_messages(first_read, second_read)),
            None,
            tempdir,
        ),
        _run_read_lifecycle_case(
            "superseded_opt_in",
            _LooseContext(_read_messages(first_read, second_read), compression_compress_superseded_reads=True),
            "[Read content superseded:",
            tempdir,
        ),
        _run_read_lifecycle_case(
            "stale_default",
            _LooseContext(_read_messages(first_read, edit_after_first=True)),
            "[Read content stale:",
            tempdir,
        ),
        _run_read_lifecycle_case(
            "stale_disabled",
            _LooseContext(_read_messages(first_read, edit_after_first=True), compression_compress_stale_reads=False),
            None,
            tempdir,
        ),
        _run_read_lifecycle_case(
            "stale_default_min_size",
            _LooseContext(_read_messages(mid_read, edit_after_first=True)),
            "[Read content stale:",
            tempdir,
        ),
        _run_read_lifecycle_case(
            "stale_raised_min_size",
            _LooseContext(
                _read_messages(mid_read, edit_after_first=True),
                compression_read_lifecycle_min_bytes=2000,
            ),
            None,
            tempdir,
        ),
    ]


def _run_ccr_recovery_case(tempdir: str) -> CaseResult:
    store = CCRStore(os.path.join(tempdir, f"{CCR_RECOVERY_GROUP}.db"))
    previous_store = active_compression_runtime._store
    handle = "abc123abc123"
    original = json.dumps([
        {"id": "a", "message": "ordinary startup"},
        {"id": "b", "message": "needle payment failure"},
        {"id": "c", "message": "ordinary shutdown"},
    ])
    store.put_with_handle(handle, original)
    active_compression_runtime._store = store
    try:
        pi = _FakePi()
        extension_factory(pi)
        failures: list[str] = []
        if "headroom_retrieve" not in pi.tools:
            failures.append("Headroom-compatible headroom_retrieve tool was not registered")
            full_result = {"content": [{"text": ""}], "details": {}}
            query_result = {"content": [{"text": ""}], "details": {}}
        else:
            schema = pi.tools["headroom_retrieve"]["parameters"]
            if schema.get("required") != ["hash"]:
                failures.append("headroom_retrieve does not accept Headroom's hash-only required schema")
            execute = pi.tools["headroom_retrieve"]["execute"]
            full_result = asyncio.run(execute("tool-1", {"hash": handle}, None, None, None))
            query_result = asyncio.run(
                execute("tool-2", {"hash": handle, "query": "needle payment"}, None, None, None)
            )
    finally:
        active_compression_runtime._store = previous_store

    try:
        payload = json.loads(full_result["content"][0]["text"])
    except Exception as exc:
        failures.append(f"full CCR retrieval did not return JSON payload: {exc}")
        payload = {}
    if payload.get("hash") != handle:
        failures.append("full CCR retrieval did not echo hash")
    if payload.get("original_content") != original:
        failures.append("full CCR retrieval did not recover original")
    if not store.is_expanded(handle):
        failures.append("full CCR retrieval did not mark handle expanded")

    query_text = query_result["content"][0]["text"]
    if "needle payment failure" not in query_text:
        failures.append("query CCR retrieval did not return matching item")
    if "ordinary startup" in query_text:
        failures.append("query CCR retrieval returned unrelated item")

    return CaseResult(
        group=CCR_RECOVERY_GROUP,
        name="headroom_retrieve_alias",
        ok=not failures,
        before_chars=len(original),
        after_chars=len(query_text),
        checks=["headroom_retrieve"] if not failures else [],
        failures=failures,
    )


def _run_compression_command_case() -> CaseResult:
    pi_ai.unregister_compressor()
    pi_ai.reset_compression_stats()
    pi_ai.reset_compression_learning_stats()
    pi_ai.reset_cache_alignment_stats()
    pi_ai.register_compressor(lambda text: f"COMPRESSED:{text[:8]}")
    failures: list[str] = []
    try:
        pi_ai.compress_context(
            _LooseContext([{"role": "tool", "tool_call_id": "toolu_1", "content": "payg payload " * 260}])
        )
        pi_ai.compress_context(
            _LooseContext(
                [{"role": "tool", "tool_call_id": "toolu_2", "content": "subscription payload " * 260}],
                compression_auth_mode="subscription",
            )
        )
        pi = _FakePi()
        extension_factory(pi)
        text = asyncio.run(pi.commands["compression"]["handler"]("stats"))
        if "total compressions: 2" not in text:
            failures.append("compression command did not report total compression count")
        if "learning events: 1 (read-only skipped 1)" not in text:
            failures.append("compression command did not report learning write/skip counts")
        if "learning by strategy: text: 1" not in text:
            failures.append("compression command did not report learning events by strategy")
        if "learning skipped by strategy: text: 1" not in text:
            failures.append("compression command did not report learning skips by strategy")
        if "cache alignment scans: 0 (findings 0, policy skipped 0)" not in text:
            failures.append("compression command did not report cache alignment stats")
        if "compression cache: hits 0, misses 2, entries 2" not in text:
            failures.append("compression command did not report compression cache stats")
        if "unit outcomes: 2" not in text:
            failures.append("compression command did not report unit outcome count")
        if "unit outcomes by category: applied: 2" not in text:
            failures.append("compression command did not report unit outcome categories")
        if "unit outcomes by reason: applied: 2" not in text:
            failures.append("compression command did not report unit outcome reasons")
        reset_text = asyncio.run(pi.commands["compression"]["handler"]("reset"))
        if reset_text != "Active compression stats reset.":
            failures.append("compression command reset returned unexpected text")
        if pi_ai.get_compression_stats().total_compressions != 0:
            failures.append("compression command reset did not clear compression stats")
        learning = pi_ai.get_compression_learning_stats()
        if learning.total_events != 0 or learning.total_skipped_read_only != 0:
            failures.append("compression command reset did not clear learning stats")
        cache_alignment = pi_ai.get_cache_alignment_stats()
        if cache_alignment.total_scans != 0 or cache_alignment.skipped_by_policy != 0:
            failures.append("compression command reset did not clear cache alignment stats")
        if pi_ai.get_compression_cache_stats().entries != 0:
            failures.append("compression command reset did not clear compression cache stats")
        if pi_ai.get_unit_outcome_stats().total_units != 0:
            failures.append("compression command reset did not clear unit outcome stats")
    finally:
        pi_ai.unregister_compressor()
        pi_ai.reset_compression_stats()
        pi_ai.reset_compression_learning_stats()
        pi_ai.reset_cache_alignment_stats()

    return CaseResult(
        group=COMPRESSION_COMMAND_GROUP,
        name="stats_learning_reset",
        ok=not failures,
        before_chars=1,
        after_chars=1,
        checks=["stats", "learning", "reset"] if not failures else [],
        failures=failures,
    )


def _run_ccr_store_cases(tempdir: str) -> list[CaseResult]:
    results: list[CaseResult] = []

    store = CCRStore(os.path.join(tempdir, f"{CCR_STORE_GROUP}-metadata.db"))
    original = json.dumps([
        {"id": "a", "message": "ordinary startup"},
        {"id": "b", "message": "needle payment failure"},
    ])
    handle = store.put_with_handle(
        "444444444444",
        original,
        compressed_content="[]",
        original_tokens=100,
        compressed_tokens=10,
        original_item_count=2,
        compressed_item_count=0,
        tool_name="search_tool",
        query_context="payment issue",
        compression_strategy="json_rows",
    )
    search_results = store.search(handle, "needle payment")
    entry = store.retrieve(handle, query="needle payment")
    metadata = store.get_metadata(handle)
    stats = store.get_stats()
    events = store.get_retrieval_events(tool_name="search_tool")
    failures: list[str] = []
    if search_results != [{"id": "b", "message": "needle payment failure"}]:
        failures.append("query-scoped CCR store search did not return only the matching item")
    if entry is None or entry.search_queries != ["needle payment"]:
        failures.append("CCR store did not track unique retrieval/search queries")
    if not metadata or metadata.get("compressed_content") != "[]" or metadata.get("query_context") != "payment issue":
        failures.append("CCR store did not preserve compression metadata")
    if stats.get("entry_count") != 1 or stats.get("total_original_tokens") != 100:
        failures.append("CCR store stats did not reflect stored compression entry")
    if {event.retrieval_type for event in events} != {"full", "search"}:
        failures.append("CCR store did not record full and query retrieval events")
    results.append(
        CaseResult(
            group=CCR_STORE_GROUP,
            name="metadata_search_events",
            ok=not failures,
            before_chars=len(original),
            after_chars=len(json.dumps(search_results, ensure_ascii=False)),
            checks=["metadata", "search", "events"] if not failures else [],
            failures=failures,
        )
    )

    ttl_store = CCRStore(os.path.join(tempdir, f"{CCR_STORE_GROUP}-ttl.db"), default_ttl=10)
    ttl_handle = ttl_store.put_with_handle("555555555555", "payload", ttl=1)
    with ttl_store._connect() as c:
        c.execute("UPDATE ccr SET created_at = ? WHERE handle = ?", (time.time() - 2, ttl_handle))
    status = ttl_store.get_entry_status(ttl_handle, clean_expired=True)
    failures = []
    if status.get("status") != "expired":
        failures.append("expired CCR entry did not report expired status")
    if ttl_store.get(ttl_handle) is not None:
        failures.append("expired CCR entry was still retrievable")
    results.append(
        CaseResult(
            group=CCR_STORE_GROUP,
            name="ttl_expiration",
            ok=not failures,
            before_chars=len("payload"),
            after_chars=0 if not failures else len("payload"),
            checks=["ttl"] if not failures else [],
            failures=failures,
        )
    )

    eviction_store = CCRStore(os.path.join(tempdir, f"{CCR_STORE_GROUP}-eviction.db"), max_entries=2)
    first = eviction_store.put_with_handle("111111111111", "first")
    time.sleep(0.001)
    second = eviction_store.put_with_handle("222222222222", "second")
    time.sleep(0.001)
    third = eviction_store.put_with_handle("333333333333", "third")
    failures = []
    if eviction_store.exists(first):
        failures.append("CCR store did not evict oldest entry at capacity")
    if eviction_store.get(second) != "second" or eviction_store.get(third) != "third":
        failures.append("CCR store evicted a newer entry instead of the oldest entry")
    results.append(
        CaseResult(
            group=CCR_STORE_GROUP,
            name="bounded_eviction",
            ok=not failures,
            before_chars=len("firstsecondthird"),
            after_chars=len("secondthird") if not failures else len("firstsecondthird"),
            checks=["eviction"] if not failures else [],
            failures=failures,
        )
    )

    return results


def _run_unit_cache_case() -> CaseResult:
    original = "identical provider unit payload line\n" * 120
    calls = {"count": 0}

    def compressor(text: str) -> str:
        calls["count"] += 1
        return f"COMPRESSED:{text[:16]}"

    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(compressor)
    try:
        ctx = _LooseContext(
            [
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": original,
                },
                {
                    "role": "tool",
                    "tool_call_id": "c2",
                    "content": original,
                },
            ]
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()

    failures: list[str] = []
    first = out.messages[0]["content"]
    second = out.messages[1]["content"]
    if calls["count"] != 1:
        failures.append(f"identical live compression unit was compressed {calls['count']} times")
    if first != "COMPRESSED:identical provid" or second != first:
        failures.append("identical live compression units did not reuse the same compressed output")
    if pi_ai.get_compression_stats().compressions_by_strategy != {"text": 2}:
        failures.append("unit cache reuse did not preserve per-unit compression stats")
    return CaseResult(
        group=UNIT_CACHE_GROUP,
        name="identical_live_text_unit_reuse",
        ok=not failures,
        before_chars=len(original) * 2,
        after_chars=len(str(first)) + len(str(second)),
        checks=["unit_cache"] if not failures else [],
        failures=failures,
    )


def _fixtures_root(headroom_root: Path) -> Path:
    return headroom_root / "tests" / "parity" / "fixtures"


def main(argv: list[str]) -> int:
    headroom_root = Path(argv[1]) if len(argv) > 1 else Path("/tmp/headroom-src")
    root = _fixtures_root(headroom_root)
    if not root.exists():
        print(f"Headroom parity fixtures not found: {root}", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory() as tempdir:
        for group in FIXTURE_GROUPS:
            for path in sorted((root / group).glob("*.json")):
                results.append(_run_case(group, path, tempdir))
        results.extend(_run_synthetic_cases(tempdir))
        results.extend(_run_provider_shape_cases(tempdir))
        results.extend(_run_error_protection_cases(tempdir))
        results.extend(_run_code_protection_cases(tempdir))
        results.extend(_run_cache_control_cases(tempdir))
        results.append(_run_cache_aligner_case())
        results.extend(_run_image_compression_cases())
        results.extend(_run_image_log_redaction_cases())
        results.append(_run_compression_summary_case(tempdir))
        results.extend(_run_determinism_cases(tempdir))
        results.extend(_run_compression_policy_cases())
        results.extend(_run_marker_pinning_cases(tempdir))
        results.append(_run_marker_preserving_case())
        results.extend(_run_custom_tag_protection_cases(tempdir))
        results.extend(_run_tool_exclusion_cases(tempdir))
        results.append(_run_non_shrinking_rejection_case(tempdir))
        results.extend(_run_compression_failure_cases())
        results.extend(_run_compression_circuit_breaker_cases())
        results.append(_run_inflation_guard_case())
        results.append(_run_token_floor_case(tempdir))
        results.extend(_run_config_control_cases(tempdir))
        results.append(_run_target_ratio_case(tempdir))
        results.extend(_run_read_lifecycle_cases(tempdir))
        results.append(_run_ccr_recovery_case(tempdir))
        results.append(_run_compression_command_case())
        results.extend(_run_ccr_store_cases(tempdir))
        results.append(_run_unit_cache_case())
        results.extend(_run_content_detector_cases(headroom_root))
        results.extend(_run_tokenizer_cases(headroom_root))

    failures = [result for result in results if not result.ok]
    print("Headroom compression parity fixtures")
    print("-" * 72)
    for group in (
        *FIXTURE_GROUPS,
        SYNTHETIC_GROUP,
        PROVIDER_SHAPE_GROUP,
        ERROR_PROTECTION_GROUP,
        CODE_PROTECTION_GROUP,
        CACHE_CONTROL_GROUP,
        CACHE_ALIGNER_GROUP,
        IMAGE_COMPRESSION_GROUP,
        IMAGE_LOG_REDACTION_GROUP,
        COMPRESSION_SUMMARY_GROUP,
        DETERMINISM_GROUP,
        COMPRESSION_POLICY_GROUP,
        MARKER_PINNING_GROUP,
        MARKER_PRESERVING_GROUP,
        CUSTOM_TAG_GROUP,
        TOOL_EXCLUSION_GROUP,
        NON_SHRINKING_GROUP,
        COMPRESSION_FAILURE_GROUP,
        COMPRESSION_CIRCUIT_BREAKER_GROUP,
        INFLATION_GUARD_GROUP,
        TOKEN_FLOOR_GROUP,
        CONFIG_CONTROL_GROUP,
        TARGET_RATIO_GROUP,
        READ_LIFECYCLE_GROUP,
        CCR_RECOVERY_GROUP,
        COMPRESSION_COMMAND_GROUP,
        CCR_STORE_GROUP,
        UNIT_CACHE_GROUP,
        CONTENT_DETECTOR_GROUP,
        TOKENIZER_GROUP,
    ):
        group_results = [result for result in results if result.group == group]
        ok_count = sum(1 for result in group_results if result.ok)
        print(f"{group:<18} {ok_count:>3}/{len(group_results):<3} passing")
    print("-" * 72)
    if failures:
        for result in failures[:20]:
            print(f"{result.group}/{result.name}:")
            for failure in result.failures:
                print(f"  - {failure}")
        if len(failures) > 20:
            print(f"... {len(failures) - 20} more failing fixture(s)")
        return 1

    total_before = sum(result.before_chars for result in results)
    total_after = sum(result.after_chars for result in results)
    saved = 100 * (1 - total_after / total_before) if total_before else 0.0
    print(f"all {len(results)} fixtures passed; aggregate char savings {saved:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
