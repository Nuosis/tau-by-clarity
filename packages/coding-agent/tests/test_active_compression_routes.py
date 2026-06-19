from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pi_ai
from pi_ai.types import Context, TextContent, ToolResultMessage
from pi_coding_agent import active_compression as active_compression_runtime
from pi_coding_agent.active_compression.ccr import CCRStore
from pi_coding_agent.active_compression.compressor import (
    CompressionConfig,
    _detect_content_type,
    _protect_custom_tags,
    _restore_custom_tags,
    compress,
)


def _handle(text: str) -> str:
    match = re.search(r"\[CCR:([0-9a-f]{12})\]", text)
    assert match is not None
    return match.group(1)


def test_content_detector_matches_headroom_route_fixtures():
    fixture_dir = Path("/tmp/headroom-src/tests/parity/fixtures/content_detector")
    assert fixture_dir.exists(), "Headroom content_detector fixtures are required for parity"
    mismatches = []

    for path in sorted(fixture_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("input", "")
        content = str(raw.get("content", "")) if isinstance(raw, dict) else str(raw)
        expected = data["output"]["content_type"]
        actual = _detect_content_type(content)
        if actual != expected:
            mismatches.append(f"{path.name}: expected {expected}, got {actual}")

    assert not mismatches


def test_search_output_compresses_by_file_and_preserves_errors(tmp_path):
    lines = []
    for i in range(1, 80):
        body = f"def ordinary_{i}(): pass"
        if i == 42:
            body = "ERROR_TARGET = 'needle preserved'"
        lines.append(f"src/service.py:{i}:{body}")
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" in out
    assert "src/service.py (79 matches" in out
    assert "1: def ordinary_1(): pass" in out
    assert "42: ERROR_TARGET = 'needle preserved'" in out
    assert "79: def ordinary_79(): pass" in out
    assert "[... and 74 more matches]" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_search_output_keeps_high_signal_importance_match(tmp_path):
    lines = []
    for i in range(1, 80):
        body = f"ordinary search result {i}"
        if i == 57:
            body = "TODO TARGET_KEEP reconcile account 7741"
        lines.append(f"src/service.py:{i}:{body}")
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" in out
    assert "1: ordinary search result 1" in out
    assert "57: TODO TARGET_KEEP reconcile account 7741" in out
    assert "79: ordinary search result 79" in out
    assert "56: ordinary search result 56" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_search_output_compresses_at_headroom_min_match_threshold(tmp_path):
    lines = [
        (
            "src/services/authentication/session_recovery.py:"
            f"{i}:ordinary session recovery result with repeated explanatory context {i}"
        )
        for i in range(1, 13)
    ]
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" in out
    assert "src/services/authentication/session_recovery.py" in out
    assert "1: ordinary session recovery result" in out
    assert "12: ordinary session recovery result" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_search_output_keeps_high_signal_file_beyond_file_cap(tmp_path):
    lines = []
    for file_index in range(18):
        path = f"src/file_{file_index:02d}.py"
        for line_no in range(1, 4):
            lines.append(f"{path}:{line_no}:ordinary search result {file_index}-{line_no}")
    for line_no in range(1, 4):
        body = (
            "ERROR TARGET_SEARCH_FILE auth token 7741"
            if line_no == 2
            else f"ordinary zeta result {line_no}"
        )
        lines.append(f"src/zz_target.py:{line_no}:{body}")
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" in out
    assert "src/zz_target.py" in out
    assert "ERROR TARGET_SEARCH_FILE auth token 7741" in out
    assert "src/file_00.py" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_search_output_rejects_negative_line_number_context_rows(tmp_path):
    original = "\n".join(
        f"src/file_{i:02d}.py--{i}-ERROR invalid negative line number {i} should not parse"
        for i in range(40)
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" not in out
    assert out == original


def test_search_output_caps_total_matches_but_keeps_high_signal_file(tmp_path):
    lines = []
    for file_index in range(20):
        path = f"src/search_{file_index:02d}.py"
        for line_no in range(1, 8):
            body = f"ordinary capped search result {file_index}-{line_no}"
            if file_index == 19 and line_no == 4:
                body = "ERROR TARGET_GLOBAL_CAP auth token 7741"
            lines.append(f"{path}:{line_no}:{body}")
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    visible_match_rows = [
        line
        for line in out.splitlines()
        if re.match(r"^(?:  )?\d+:", line) or re.match(r"^src/search_\\d+\\.py[:-]\\d+[:-]", line)
    ]
    assert "compressed search results" in out
    assert len(visible_match_rows) <= 30
    assert "src/search_19.py" in out
    assert "ERROR TARGET_GLOBAL_CAP auth token 7741" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_search_output_above_headroom_ccr_ratio_threshold_passes_through(tmp_path):
    lines = []
    for match_index in range(25):
        file_index = match_index % 20
        path = f"src/{file_index:02d}/" + ("longname_" * (file_index % 5 + 1)) + "file.py"
        lines.append(f"{path}:{match_index + 1}:x{match_index}")
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed search results" not in out
    assert out == original


def test_diff_output_preserves_change_lines_and_summary(tmp_path):
    hunks = []
    for i in range(1, 70):
        hunks.append(f" context line {i}")
    original = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "index 111..222 100644",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -1,70 +1,70 @@",
            *hunks[:30],
            "-old_value = 1",
            "+new_value = 2",
            *hunks[30:],
        ]
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "-old_value = 1" in out
    assert "+new_value = 2" in out
    assert "[1 files changed, +1 -1 lines]" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_diff_below_headroom_line_threshold_passes_through(tmp_path):
    diff_lines = [
        "diff --git a/app.py b/app.py",
        "--- a/app.py",
        "+++ b/app.py",
        "@@ -1,44 +1,44 @@",
    ]
    diff_lines.extend(f" context before {i}" for i in range(20))
    diff_lines.extend(["-old_auth_token = read_old()", "+new_auth_token = read_new()"])
    diff_lines.extend(f" context after {i}" for i in range(23))
    original = "\n".join(diff_lines)
    assert len(original.splitlines()) == 49
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" not in out
    assert "compressed text" not in out
    assert out == original


def test_large_diff_caps_hunks_but_keeps_first_last_and_high_signal(tmp_path):
    diff_lines = [
        "diff --git a/app.py b/app.py",
        "index 111..222 100644",
        "--- a/app.py",
        "+++ b/app.py",
    ]
    for i in range(20):
        diff_lines.extend(
            [
                f"@@ -{i * 10 + 1},5 +{i * 10 + 1},5 @@",
                f" context before {i}",
                f"-old ordinary value {i}",
                (
                    "+TODO SECURITY_TARGET rotate token 7741"
                    if i == 11
                    else f"+new ordinary value {i}"
                ),
                f" context after {i}",
            ]
        )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "@@ -1,5 +1,5 @@" in out
    assert "+TODO SECURITY_TARGET rotate token 7741" in out
    assert "@@ -191,5 +191,5 @@" in out
    assert "@@ -91,5 +91,5 @@" not in out
    assert "10 hunks omitted" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_multi_file_diff_preserves_headers_for_selected_later_file(tmp_path):
    diff_lines = [
        "diff --git a/alpha.py b/alpha.py",
        "index aaa..bbb 100644",
        "--- a/alpha.py",
        "+++ b/alpha.py",
    ]
    for i in range(25):
        diff_lines.extend(
            [
                f"@@ -{i * 10 + 1},5 +{i * 10 + 1},5 @@",
                f" alpha context {i}",
                f"-alpha old {i}",
                f"+alpha new {i}",
                f" alpha after {i}",
            ]
        )
    diff_lines.extend(
        [
            "diff --git a/beta.py b/beta.py",
            "index ccc..ddd 100644",
            "--- a/beta.py",
            "+++ b/beta.py",
            "@@ -1,5 +1,5 @@",
            " beta context",
            "-old beta token",
            "+TODO SECURITY_TARGET beta token 7741",
            " beta after",
        ]
    )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "diff --git a/beta.py b/beta.py" in out
    assert "--- a/beta.py" in out
    assert "+++ b/beta.py" in out
    assert "+TODO SECURITY_TARGET beta token 7741" in out
    assert "diff --git a/beta.py b/beta.py\nindex ccc..ddd 100644\n--- a/beta.py\n+++ b/beta.py" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_multi_file_diff_keeps_high_signal_file_beyond_file_cap(tmp_path):
    diff_lines = []
    for i in range(25):
        path = f"file_{i:02d}.py"
        diff_lines.extend(
            [
                f"diff --git a/{path} b/{path}",
                f"index {i:03x}..{i + 1:03x} 100644",
                f"--- a/{path}",
                f"+++ b/{path}",
                "@@ -1,25 +1,25 @@",
                *(f" context before {i}-{j}" for j in range(12)),
                f"-old value {i}",
                (
                    "+TODO SECURITY_TARGET rotate token 7741"
                    if i == 23
                    else f"+new value {i}"
                ),
                *(f" context after {i}-{j}" for j in range(12)),
            ]
        )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "diff --git a/file_23.py b/file_23.py" in out
    assert "+TODO SECURITY_TARGET rotate token 7741" in out
    assert "diff --git a/file_09.py b/file_09.py" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_multi_file_diff_keeps_binary_metadata_beyond_file_cap(tmp_path):
    diff_lines = []
    for i in range(25):
        path = f"file_{i:02d}.py"
        diff_lines.extend(
            [
                f"diff --git a/{path} b/{path}",
                f"index {i:03x}..{i + 1:03x} 100644",
                f"--- a/{path}",
                f"+++ b/{path}",
                "@@ -1,5 +1,5 @@",
                f" context {i}",
                f"-old value {i}",
                f"+new value {i}",
                f" after {i}",
            ]
        )
    diff_lines.extend(
        [
            "diff --git a/assets/logo.png b/assets/logo.png",
            "Binary files a/assets/logo.png and b/assets/logo.png differ",
        ]
    )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "diff --git a/assets/logo.png b/assets/logo.png" in out
    assert "Binary files a/assets/logo.png and b/assets/logo.png differ" in out
    assert "diff --git a/file_09.py b/file_09.py" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_multi_file_diff_keeps_rename_metadata_beyond_file_cap(tmp_path):
    diff_lines = []
    for i in range(25):
        path = f"file_{i:02d}.py"
        diff_lines.extend(
            [
                f"diff --git a/{path} b/{path}",
                f"index {i:03x}..{i + 1:03x} 100644",
                f"--- a/{path}",
                f"+++ b/{path}",
                "@@ -1,5 +1,5 @@",
                f" context {i}",
                f"-old value {i}",
                f"+new value {i}",
                f" after {i}",
            ]
        )
    diff_lines.extend(
        [
            "diff --git a/src/old_name.py b/src/new_name.py",
            "similarity index 92%",
            "rename from src/old_name.py",
            "rename to src/new_name.py",
        ]
    )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "diff --git a/src/old_name.py b/src/new_name.py" in out
    assert "similarity index 92%" in out
    assert "rename from src/old_name.py" in out
    assert "rename to src/new_name.py" in out
    assert "diff --git a/file_09.py b/file_09.py" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_headroom_fixture_style_multi_file_diff_compresses(tmp_path):
    diff_lines = []
    for file_index in range(8):
        diff_lines.extend(
            [
                f"diff --git a/file_{file_index}.py b/file_{file_index}.py",
                f"--- a/file_{file_index}.py",
                f"+++ b/file_{file_index}.py",
                "@@ -1,10 +1,12 @@",
                *(f" context_{i}_{file_index}" for i in range(5)),
                *(f"-removed_{i}_{file_index}" for i in range(3)),
                *(f"+added_{i}_{file_index}" for i in range(5)),
                *(f" tail_{i}_{file_index}" for i in range(5)),
            ]
        )
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "diff --git a/file_0.py b/file_0.py" in out
    assert "diff --git a/file_7.py b/file_7.py" in out
    assert "+added_4_7" in out
    assert "context_0_0" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_combined_diff_compresses_and_preserves_three_way_hunk(tmp_path):
    diff_lines = [
        "commit abc123",
        "Merge: aaa bbb",
        "diff --combined src/merge.py",
        "index 111,222..333",
        "--- a/src/merge.py",
        "+++ b/src/merge.py",
        "@@@ -1,80 -1,80 +1,82 @@@",
    ]
    diff_lines.extend(f"  unchanged before {i}" for i in range(40))
    diff_lines.extend(
        [
            " -old_branch_1",
            "- old_branch_2",
            "++TODO MERGE_TARGET reconcile token 7741",
            " +new_added",
        ]
    )
    diff_lines.extend(f"  unchanged after {i}" for i in range(80))
    original = "\n".join(diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "commit abc123" in out
    assert "diff --combined src/merge.py" in out
    assert "@@@ -1,80 -1,80 +1,82 @@@" in out
    assert " -old_branch_1" in out
    assert "- old_branch_2" in out
    assert "++TODO MERGE_TARGET reconcile token 7741" in out
    assert " +new_added" in out
    assert "unchanged before 20" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_diff_after_long_commit_message_routes_to_diff_compressor(tmp_path):
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
    original = "\n".join(prefix + [""] + diff_lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed diff" in out
    assert "commit abc1234567890abcdef" in out
    assert "diff --git a/src/app.py b/src/app.py" in out
    assert "-old_auth_token = read_old()" in out
    assert "+new_auth_token = read_new()" in out
    assert store.get(_handle(out)) == original


def test_code_output_preserves_structure_not_all_bodies(tmp_path):
    blocks = []
    for i in range(40):
        blocks.extend(
            [
                f"def function_{i}(value):",
                f"    intermediate_{i} = value + {i}",
                f"    detail_{i} = intermediate_{i} * 2",
                f"    return detail_{i}",
                "",
            ]
        )
    original = "\n".join(["import os", "import sys", "", *blocks])
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed code" in out
    assert "40 bodies compressed: function_0(), function_1()" in out
    assert "(+34 more)" in out
    assert "import os" in out
    assert "def function_0(value):" in out
    assert "def function_39(value):" in out
    assert "detail_20 = intermediate_20 * 2" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_code_output_preserves_first_line_docstrings(tmp_path):
    blocks = []
    for i in range(24):
        blocks.extend(
            [
                f"def function_{i}(arg, optional=None):",
                f'    """Process argument {i}.',
                "",
                "    Args:",
                "        arg: The argument to process.",
                "        optional: Optional detail.",
                '    """',
                "    result = str(arg)",
                "    for j in range(10):",
                "        result += str(j)",
                "    return result",
                "",
            ]
        )
    original = "\n".join(["from typing import Any", "", *blocks])
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed code" in out
    assert '"""Process argument 0."""' in out
    assert '"""Process argument 23."""' in out
    assert "Args:" not in out
    assert "for j in range(10)" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_log_output_preserves_complete_traceback_block(tmp_path):
    traceback_lines = [
        "Traceback (most recent call last):",
        *(
            f'  File "/srv/app/frame_{i}.py", line {i}, in function_{i}'
            for i in range(15)
        ),
        "ValueError: customer target TRACE-NEEDLE failed",
    ]
    original = "\n".join(
        [
            *(f"INFO setup line {i}" for i in range(80)),
            *traceback_lines,
            *(f"INFO cleanup line {i}" for i in range(80)),
        ]
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed log" in out
    assert "Traceback (most recent call last):" in out
    assert 'File "/srv/app/frame_0.py", line 0, in function_0' in out
    assert 'File "/srv/app/frame_7.py", line 7, in function_7' in out
    assert 'File "/srv/app/frame_14.py", line 14, in function_14' in out
    assert "ValueError: customer target TRACE-NEEDLE failed" in out
    assert "INFO setup line 40" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_short_log_output_passes_through_even_with_failure(tmp_path):
    original = "\n".join(
        [
            "============================= test session starts ==============================",
            "collected 42 items",
            *(f"tests/test_mod_{i}.py::test_case PASSED [{i * 2}%]" for i in range(25)),
            "tests/test_mod_25.py::test_bad FAILED",
            "=================================== FAILURES ===================================",
            "___________________________________ test_bad ___________________________________",
            "    def test_bad():",
            ">       assert compute(1, 2) == 4",
            "E       assert 3 == 4",
            "tests/test_mod_25.py:17: AssertionError",
            "=========================== short test summary info ============================",
            "FAILED tests/test_mod_25.py::test_bad",
            "1 failed, 25 passed in 0.42s",
        ]
    )
    assert len(original.splitlines()) < 50
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert out == original


def test_log_like_output_below_headroom_line_threshold_passes_through(tmp_path):
    original = "\n".join(["npm WARN deprecated x"] * 44 + ["npm ERR! something broke"] * 5)
    assert len(original.splitlines()) == 49
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed text" not in out
    assert "compressed log" not in out
    assert out == original


def test_log_output_dedupes_repeated_warnings_without_merging_distinct_messages(tmp_path):
    repeated = [
        f"warning: file /tmp/project/{i}/module.py issue {1000 + i}"
        for i in range(20)
    ]
    distinct = [
        "warning: segfault at 0xdeadbeef in thread main",
        "warning: heap overflow at 0xcafef00d in thread worker",
    ]
    original = "\n".join(
        [
            *(f"INFO setup line {i}" for i in range(60)),
            *repeated,
            *distinct,
            *(f"INFO cleanup line {i}" for i in range(60)),
        ]
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed log" in out
    assert out.count("warning: file /tmp/project/") == 1
    assert "warning: segfault at 0xdeadbeef in thread main" in out
    assert "warning: heap overflow at 0xcafef00d in thread worker" in out
    assert "INFO setup line 30" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_log_output_above_headroom_ccr_ratio_threshold_passes_through(tmp_path):
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
    original = "\n".join(lines)
    assert len(original.splitlines()) == 50
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed log" not in out
    assert "compacted log templates" not in out
    assert out == original


def test_log_output_caps_many_errors_but_keeps_first_last_and_high_signal(tmp_path):
    error_lines = [
        f"ERROR ordinary failure {i}: retryable transient worker issue"
        for i in range(30)
    ]
    error_lines[15] = "ERROR HIGH_SIGNAL_TARGET: payment reconciliation permanently failed for account 7741"
    original = "\n".join(
        [
            *(f"INFO setup line {i}" for i in range(60)),
            *error_lines,
            *(f"INFO cleanup line {i}" for i in range(60)),
        ]
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed log" in out
    assert "ERROR ordinary failure 0: retryable transient worker issue" in out
    assert "ERROR HIGH_SIGNAL_TARGET: payment reconciliation permanently failed for account 7741" in out
    assert "ERROR ordinary failure 29: retryable transient worker issue" in out
    assert out.count("ERROR ordinary failure") < 25
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_log_output_compacts_repeated_templates_without_dropping_variants(tmp_path):
    original = "\n".join(
        f"2026-06-18T12:00:00 INFO worker-{i} processing shared queue"
        for i in range(80)
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compacted log templates" in out
    assert "[Template T1:" in out
    assert "(80 occurrences)" in out
    assert "worker-0" in out
    assert "worker-40" in out
    assert "worker-79" in out
    assert len(out) < len(original)


def test_existing_ccr_marker_is_not_recompressed(tmp_path):
    store = CCRStore(str(tmp_path / "ccr.db"))
    marked = "[CCR:abcdef123456] compressed log: already done\n" + ("line\n" * 500)

    assert compress(marked, store) == marked


def test_uniform_json_array_uses_lossless_table_compaction(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "latency_ms": 10 + i,
            "service": f"svc-{i}",
        }
        for i in range(120)
    ]
    rows[42]["error"] = "FATAL sparse anomaly"
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "[120]{id:int,status:string,latency_ms:int,service:string,error:string?}" in out
    assert "FATAL sparse anomaly" in out
    assert '"119",ok' not in out
    assert "119,ok,129,svc-119," in out
    assert "items -> 15 kept" not in out
    assert "[CCR:" not in out


def test_heterogeneous_json_array_compacts_into_discriminator_buckets(tmp_path):
    rows = []
    for i in range(30):
        rows.append(
            {
                "type": "user",
                "id": i,
                "name": f"user-{i}",
                "email": f"user-{i}@example.com",
            }
        )
        rows.append(
            {
                "type": "order",
                "id": 1000 + i,
                "total": 50 + i,
                "currency": "USD",
            }
        )
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "__buckets:type" in out
    assert "__key:user" in out
    assert "__key:order" in out
    assert "email" in out
    assert "currency" in out
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_json_array_compacts_stringified_json_cells_recursively(tmp_path):
    rows = [
        {
            "event": f"batch-{i}",
            "payload": json.dumps(
                [
                    {"x": i, "status": "ok"},
                    {"x": i + 100, "status": "ok"},
                    {"x": i + 200, "status": "fail" if i == 7 else "ok"},
                ]
            ),
        }
        for i in range(24)
    ]
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "payload" in out
    assert "[3]{x:int,status:string}" in out
    assert "fail" in out
    assert '\\\\\\"x\\\\\\"' not in out
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_json_array_flattens_uniform_nested_object_columns(tmp_path):
    rows = [
        {
            "id": i,
            "service": f"svc-{i % 4}",
            "meta": {
                "region": f"us-{i % 3}",
                "tier": "gold" if i % 2 else "silver",
            },
        }
        for i in range(80)
    ]
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "meta.region" in out
    assert "meta.tier" in out
    assert '"region"' not in out
    assert '"tier"' not in out
    assert "us-2" in out
    assert "gold" in out
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_json_array_marks_sparse_columns_optional(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "optional_note": f"note-{i}" if i in {7, 19} else None,
        }
        for i in range(40)
    ]
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "[40]{id:int,status:string,optional_note:string?}" in out
    assert "note-19" in out
    assert "[CCR:" not in out


def test_json_table_matches_headroom_csv_quoting_for_pathological_values(tmp_path):
    rows = []
    for i in range(40):
        rows.append(
            {
                "id": i,
                "plain": f"value-{i}",
                "with,comma": f"a,b-{i}" if i == 5 else f"ab-{i}",
                "with quote": 'say "hi"' if i == 7 else f"q-{i}",
                "with\nnewline": "line1\nline2" if i == 9 else f"n-{i}",
                "empty": "" if i == 11 else f"e-{i}",
                "spacey": " padded " if i == 13 else f"s-{i}",
            }
        )
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "[40]{empty:string,id:int,plain:string,spacey:string,with\nnewline:string,with quote:string,with,comma:string}" in out
    assert 'e-5,5,value-5,s-5,n-5,q-5,"a,b-5"' in out
    assert 'e-7,7,value-7,s-7,n-7,"say ""hi""",ab-7' in out
    assert 'e-9,9,value-9,s-9,"line1\nline2",q-9,ab-9' in out
    assert ",11,value-11,s-11,n-11,q-11,ab-11" in out
    assert "e-13,13,value-13, padded ,n-13,q-13,ab-13" in out
    assert "[CCR:" not in out


def test_json_array_replaces_opaque_table_cell_with_ccr_marker(tmp_path):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    blob = "".join(alphabet[i % len(alphabet)] for i in range(2048))
    rows = [
        {
            "id": i,
            "artifact": blob if i == 11 else f"artifact-{i}",
            "status": "ok",
        }
        for i in range(30)
    ]
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "<<ccr:" in out
    assert ",base64," in out
    assert '"<<ccr:' not in out
    assert blob not in out
    marker = re.search(r"<<ccr:([0-9a-f]{12}),base64,", out)
    assert marker is not None
    assert marker.group(1) == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    assert store.get(marker.group(1)) == blob
    assert "[CCR:" not in out


def test_heterogeneous_json_array_without_crush_signal_stays_lossless(tmp_path):
    rows = [
        {
            f"unique_{i}": "x" * 40,
            "id": i,
        }
        for i in range(40)
    ]
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed JSON array" not in out
    assert "[CCR:" not in out
    assert "unique_39" in out
    assert '"id":39' in out
    assert len(json.loads(out)) == 40


def test_lossy_json_array_uses_headroom_dropped_rows_sentinel(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "payload": {
                "common": i % 5,
                f"unique_{i}": "y" * 80,
            },
        }
        for i in range(80)
    ]
    original = json.dumps(rows, indent=2)
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    expected_handle = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 16
    assert parsed[-1]["_ccr_dropped"] == f"<<ccr:{expected_handle} 65_rows_offloaded>>"
    assert parsed[-1]["_ccr_summary"].startswith("65 rows omitted")
    assert store.get(expected_handle) == canonical
    assert "compressed JSON array" not in out
    assert "[CCR:" not in out


def test_lossy_json_array_summarizes_dropped_row_categories(tmp_path):
    rows = [
        {
            "id": i,
            "status": "active",
            "name": f"item-{i}",
            "payload": {
                "common": i % 5,
                f"unique_{i}": "y" * 80,
            },
        }
        for i in range(60)
    ]
    rows.extend(
        {
            "id": 100 + i,
            "status": "archived",
            "name": f"old-{i}",
            "payload": {
                "common": i % 5,
                f"archived_unique_{i}": "z" * 80,
            },
        }
        for i in range(20)
    )
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    summary = parsed[-1].get("_ccr_summary", "")
    assert "rows omitted" in summary
    assert "status:" in summary
    assert "active=" in summary or "archived=" in summary
    assert "https://" not in summary


def test_lossy_json_array_preserves_rare_status_outlier(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "payload": {
                "common": i % 5,
                f"unique_{i}": "y" * 80,
            },
        }
        for i in range(80)
    ]
    rows[37] = {
        "id": 37,
        "status": "rate_limited",
        "payload": {
            "common": 2,
            "rare_status_payload": "MUST_KEEP_RATE_LIMITED",
        },
    }
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    assert any(row.get("status") == "rate_limited" for row in parsed if isinstance(row, dict))
    assert "MUST_KEEP_RATE_LIMITED" in out
    assert parsed[-1]["_ccr_dropped"].endswith("_rows_offloaded>>")


def test_lossy_json_array_preserves_top_score_results(tmp_path):
    rows = []
    for i in range(80):
        score = 0.01 + ((i % 20) / 1000)
        title = f"result {i}"
        if 30 <= i <= 41:
            score = 0.99 - ((i - 30) / 1000)
            title = f"TOP_RESULT_{i}"
        rows.append(
            {
                "id": i,
                "title": title,
                "score": score,
                "snippet": {
                    "common": i % 5,
                    f"unique_{i}": "x" * 80,
                },
            }
        )
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    kept_ids = {row.get("id") for row in parsed if isinstance(row, dict)}
    assert set(range(30, 42)).issubset(kept_ids)
    assert all(f"TOP_RESULT_{i}" in out for i in range(30, 42))


def test_lossy_json_array_preserves_time_series_change_point_window(tmp_path):
    start = datetime(2026, 1, 1)
    rows = []
    for i in range(80):
        rows.append(
            {
                "timestamp": (start + timedelta(minutes=i)).isoformat(),
                "metric": 10 if i < 30 else 200,
                "status": "ok",
                "payload": {
                    "common": i % 5,
                    f"unique_{i}": "z" * 80,
                },
                "label": f"row-{i}",
            }
        )
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    kept_indices = {
        int(row["label"].split("-")[1])
        for row in parsed
        if isinstance(row, dict) and "label" in row
    }
    assert parsed[-1]["_ccr_dropped"].endswith("_rows_offloaded>>")
    assert set(range(28, 33)).issubset(kept_indices)


def test_lossy_json_array_preserves_cluster_representatives(tmp_path):
    families = [
        "AUTH_CLUSTER",
        "BILLING_CLUSTER",
        "SEARCH_CLUSTER",
        "SYNC_CLUSTER",
        "CACHE_CLUSTER",
        "EMAIL_CLUSTER",
    ]
    rows = []
    for i in range(96):
        family = families[i // 16]
        prefix = (f"{family} cluster detail shared prefix " + ("x" * 60))[:60]
        rows.append(
            {
                "id": i,
                "status": "ok",
                "level": "info",
                "message": f"{family}: repeated worker event with shared prefix and enough text",
                "detail": f"{prefix} variant {i % 16:02d}",
                "payload": {
                    "common": i % 4,
                    f"unique_{i}": "q" * 60,
                },
            }
        )
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    by_family = {family: [] for family in families}
    for row in parsed:
        if isinstance(row, dict) and "message" in row:
            by_family[row["message"].split(":", 1)[0]].append(row["id"])
    assert parsed[-1]["_ccr_dropped"].endswith("_rows_offloaded>>")
    assert all(len(ids) >= 2 for ids in by_family.values())


def test_json_string_array_samples_and_preserves_errors_and_length_anomalies(tmp_path):
    rows = [f"ordinary event line {i:03d}" for i in range(120)]
    rows[37] = "FATAL failure event MUST_KEEP_STRING_ERROR"
    rows[86] = "length anomaly " + ("X" * 500)
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    assert len(parsed) < len(rows)
    assert "FATAL failure event MUST_KEEP_STRING_ERROR" in parsed
    assert rows[86] in parsed
    assert parsed[0] == rows[0]
    assert parsed[-1] == rows[-1]


def test_json_number_array_samples_and_preserves_outliers(tmp_path):
    rows = [10 for _ in range(320)]
    rows[173] = 5000
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    assert len(parsed) < len(rows)
    assert 5000 in parsed
    assert parsed[0] == 10
    assert parsed[-1] == 10


def test_json_mixed_array_compresses_groups_and_preserves_original_order(tmp_path):
    rows = []
    for i in range(80):
        rows.append(f"ordinary string item {i:03d}")
        rows.append(10)
    rows[37 * 2] = "FATAL mixed string MUST_KEEP"
    rows[53 * 2 + 1] = 9000
    rows.extend([[1, 2, 3], None, True, False])
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    parsed = json.loads(out)
    assert len(parsed) < len(rows)
    assert "FATAL mixed string MUST_KEEP" in parsed
    assert 9000 in parsed
    assert [1, 2, 3] in parsed
    assert None in parsed
    assert True in parsed
    assert False in parsed
    assert parsed.index("FATAL mixed string MUST_KEEP") < parsed.index(9000)


def test_json_array_keeps_rare_high_information_middle_rows(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "service": "api",
            "payload": {"kind": "normal", "bucket": i % 3},
        }
        for i in range(120)
    ]
    rows[73] = {
        "id": 73,
        "status": "ok",
        "service": "api",
        "payload": {"kind": "normal", "bucket": 1},
        "rare_signal": "KEEP-ME-ANCHOR-73",
        "diagnostic": "one-off customer-specific migration note that should survive sampling",
    }
    original = json.dumps(rows)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "KEEP-ME-ANCHOR-73" in out
    assert "rare_signal" in out
    assert "diagnostic" in out
    assert "[120]{id:int,status:string,service:string,payload.bucket:int,payload.kind:string,rare_signal:string?,diagnostic:string?}" in out
    assert "items -> 15 kept" not in out
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_json_object_compacts_nested_tabular_arrays_losslessly(tmp_path):
    original_doc = {
        "kind": "service-report",
        "generated_at": "2026-06-18T00:00:00Z",
        "events": [
            {
                "id": i,
                "service": f"svc-{i % 4}",
                "status": "ok",
                "latency_ms": 20 + i,
            }
            for i in range(80)
        ],
        "summary": "daily event export",
    }
    original = json.dumps(original_doc)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "__buckets:service" in out
    assert "[20]{id:int,latency_ms:int,service:string,status:string}" in out
    assert '79,99,svc-3,ok' in out
    assert "... 77 items omitted" not in out
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_json_document_replaces_nested_opaque_string_with_ccr_marker(tmp_path):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    blob = "".join(alphabet[i % len(alphabet)] for i in range(2048))
    original_doc = {
        "id": "artifact-7",
        "blob": blob,
        "summary": "base64 artifact export",
    }
    original = json.dumps(original_doc)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "<<ccr:" in out
    assert ",base64," in out
    assert blob not in out
    marker = re.search(r"<<ccr:([0-9a-f]{12}),base64,", out)
    assert marker is not None
    assert marker.group(1) == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    assert store.get(marker.group(1)) == blob
    assert len(out) < len(original)
    assert "[CCR:" not in out


def test_pretty_json_object_minifies_losslessly_before_schema_sampling(tmp_path):
    original_doc = {
        f"key_{i:02d}": {
            "id": i,
            "label": f"visible-value-{i:02d}",
            "active": i % 2 == 0,
        }
        for i in range(80)
    }
    original = json.dumps(original_doc, indent=2)
    expected = json.dumps(original_doc, ensure_ascii=False, separators=(",", ":"))
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert out == expected
    assert "visible-value-79" in out
    assert "... keys omitted" not in out
    assert "[CCR:" not in out
    assert json.loads(out) == original_doc


def test_mixed_fenced_content_routes_sections_independently(tmp_path):
    search_lines = "\n".join(f"src/app.py:{i}:def target_{i}(): pass" for i in range(1, 80))
    diff_lines = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -1,70 +1,70 @@",
            *(f" context {i}" for i in range(35)),
            "-old = True",
            "+new = True",
            *(f" context {i}" for i in range(35, 70)),
        ]
    )
    original = f"Search output:\n```text\n{search_lines}\n```\nPatch:\n```diff\n{diff_lines}\n```\n"
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed mixed content" in out
    assert "compressed search results" in out
    assert "compressed diff" in out
    assert "src/app.py (79 matches" in out
    assert "+new = True" in out
    assert "-old = True" in out
    assert store.get(_handle(out)) == original


def test_html_output_extracts_main_content_and_drops_boilerplate(tmp_path):
    article_paragraphs = "\n".join(
        f"<p>Primary article paragraph {i}: retention target ALPHA-{i} explains the implementation.</p>"
        for i in range(40)
    )
    nav = "\n".join(f"<a href='/item-{i}'>Navigation item {i}</a>" for i in range(80))
    original = f"""<!doctype html>
<html>
  <head>
    <title>Compression Route Report</title>
    <style>{'.ad { display: none; }' * 120}</style>
    <script>{'console.log(\"tracking\");' * 120}</script>
  </head>
  <body>
    <header><h1>Site Header</h1>{nav}</header>
    <nav>{nav}</nav>
    <main>
      <article>
        <h1>Compression Route Report</h1>
        {article_paragraphs}
      </article>
    </main>
    <aside>{'Related story teaser ' * 200}</aside>
    <footer>{'Footer legal link ' * 200}</footer>
  </body>
</html>"""
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "extracted HTML content" in out
    assert "Title: Compression Route Report" in out
    assert "Primary article paragraph 0" in out
    assert "retention target ALPHA-39" in out
    assert "console.log" not in out
    assert "Navigation item 79" not in out
    assert "Footer legal link" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_custom_workflow_tags_survive_generic_text_compression(tmp_path):
    protected = (
        '<system-reminder priority="critical">\n'
        "  Do not delete TAG-NEEDLE-7741.\n"
        "  <inner-check>Nested custom instruction survives.</inner-check>\n"
        "</system-reminder>"
    )
    original = (
        ("intro filler line\n" * 120)
        + protected
        + "\n"
        + ("middle filler that would normally be elided\n" * 180)
        + "<marker/>"
        + "\n"
        + ("tail filler line\n" * 120)
    )
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "[CCR:" in out
    assert protected in out
    assert "<marker/>" in out
    assert "TAG-NEEDLE-7741" in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_custom_tag_restore_discards_lost_placeholders_without_reinjection():
    protected = [
        ("{{TAU_TAG_0}}", "<system-reminder>Do not reattach me</system-reminder>"),
        ("{{TAU_TAG_1}}", "<context>keep me</context>"),
    ]

    out = _restore_custom_tags("head {{TAU_TAG_1}} tail", protected)

    assert out == "head <context>keep me</context> tail"
    assert "Do not reattach me" not in out
    assert "Preserved custom tags" not in out


def test_custom_tag_protection_matches_headroom_nested_duplicate_invariants():
    duplicate = (
        "<system-reminder>same</system-reminder> middle "
        "<system-reminder>same</system-reminder>"
    )
    cleaned, protected = _protect_custom_tags(duplicate)

    assert cleaned == "{{TAU_TAG_0}} middle {{TAU_TAG_1}}"
    assert len(protected) == 2
    assert protected[0][0] != protected[1][0]
    assert _restore_custom_tags(cleaned, protected) == duplicate

    nested = "<lvl>" * 60 + "core" + "</lvl>" * 60
    cleaned, protected = _protect_custom_tags(nested)

    assert cleaned == "{{TAU_TAG_0}}"
    assert len(protected) == 1
    assert _restore_custom_tags(cleaned, protected) == nested


def test_custom_tag_protection_handles_self_closing_duplicates_and_placeholder_collision():
    self_closing = "<marker/> middle <marker/>"
    cleaned, protected = _protect_custom_tags(self_closing)

    assert cleaned == "{{TAU_TAG_0}} middle {{TAU_TAG_1}}"
    assert len(protected) == 2
    assert protected[0][0] != protected[1][0]
    assert _restore_custom_tags(cleaned, protected) == self_closing

    collision = (
        "User wrote {{TAU_TAG_0}} on purpose. "
        "<system-reminder>real one</system-reminder>"
    )
    cleaned, protected = _protect_custom_tags(collision)

    assert len(protected) == 1
    assert protected[0][0] != "{{TAU_TAG_0}}"
    assert _restore_custom_tags(cleaned, protected) == collision


def test_generic_text_compression_preserves_middle_headers_and_anchors(tmp_path):
    lines = [f"intro paragraph filler {i}" for i in range(80)]
    lines.extend(
        [
            "## Authentication Recovery",
            "The required migration anchor is AUTH-CRITICAL-7788 and should remain visible.",
            "Use recover_session_token() before retrying the request.",
            "low value explanation that may be elided",
        ]
    )
    lines.extend(f"middle prose filler {i}" for i in range(160))
    lines.extend(["## Final Notes", "Tail anchor RELEASE-9001"])
    lines.extend(f"tail paragraph filler {i}" for i in range(80))
    original = "\n".join(lines)
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "compressed text" in out
    assert "Authentication Recovery" in out
    assert "AUTH-CRITICAL-7788" in out
    assert "recover_session_token()" in out
    assert "middle prose filler 80" not in out
    assert len(out) < len(original)
    assert store.get(_handle(out)) == original


def test_generic_text_compression_honors_target_ratio(tmp_path):
    lines = [f"intro paragraph filler {i}" for i in range(40)]
    lines.extend(f"ANCHOR-{i:04d} important detail line {i}" for i in range(120))
    lines.extend(f"tail paragraph filler {i}" for i in range(40))
    original = "\n".join(lines)

    default_store = CCRStore(str(tmp_path / "default.db"))
    targeted_store = CCRStore(str(tmp_path / "targeted.db"))

    default_out = compress(original, default_store)
    targeted_out = compress(original, targeted_store, target_ratio=0.15)

    assert "compressed text" in default_out
    assert "compressed text" in targeted_out
    assert len(targeted_out) < len(default_out)
    assert targeted_store.get(_handle(targeted_out)) == original


def test_force_compression_bypasses_token_floor_for_read_lifecycle_storage(tmp_path):
    original = "mid exact source snapshot\n" * 24
    store = CCRStore(str(tmp_path / "ccr.db"))

    default_out = compress(original, store)
    forced_out = compress(original, store, force=True)

    assert default_out == original
    assert "compressed text" in forced_out
    assert store.get(_handle(forced_out)) == original


def test_compression_config_controls_token_floor_for_structured_arrays(tmp_path):
    rows = [10 for _ in range(40)]
    rows[21] = 5000
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    default_out = compress(original, store)
    configured_out = compress(original, store, config=CompressionConfig(min_tokens=1))

    assert default_out == original
    assert len(json.loads(configured_out)) < len(rows)
    assert 5000 in json.loads(configured_out)


def test_compression_config_controls_structured_item_budget(tmp_path):
    rows = [f"ordinary event line {i:03d}" for i in range(120)]
    rows[37] = "FATAL failure event MUST_KEEP_STRING_ERROR"
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    default_out = compress(original, store)
    tight_out = compress(original, store, config=CompressionConfig(max_items_after_crush=8))

    assert len(json.loads(tight_out)) < len(json.loads(default_out))
    assert "FATAL failure event MUST_KEEP_STRING_ERROR" in json.loads(tight_out)


def test_compression_config_controls_lossless_threshold_and_ccr_marker(tmp_path):
    rows = [
        {
            "id": i,
            "status": "ok",
            "latency_ms": 10 + i,
            "service": f"svc-{i}",
        }
        for i in range(80)
    ]
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "ccr.db"))

    default_out = compress(original, store)
    lossy_without_marker = compress(
        original,
        store,
        config=CompressionConfig(lossless_min_savings_ratio=1.0, enable_ccr_marker=False),
    )

    assert default_out.startswith("[80]{")
    parsed = json.loads(lossy_without_marker)
    assert len(parsed) < len(rows)
    assert "_ccr_dropped" not in parsed[-1]
    assert "<<ccr:" not in lossy_without_marker


def test_compression_outputs_are_byte_deterministic_for_representative_content(tmp_path):
    search = "\n".join(
        f"src/service.py:{i}:"
        + ("ERROR TARGET_DETERMINISM preserve this search signal" if i == 57 else f"ordinary result {i}")
        for i in range(1, 90)
    )
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
    diff = "\n".join(diff_lines)
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
    cases = {
        "plain": "\n".join(f"plain section {i} repeated operational detail" for i in range(260)),
        "search": search,
        "diff": diff,
        "json": json.dumps(rows, indent=2),
    }

    for name, original in cases.items():
        first_store = CCRStore(str(tmp_path / f"{name}-a.db"))
        second_store = CCRStore(str(tmp_path / f"{name}-b.db"))

        first = compress(original, first_store)
        second = compress(original, second_store)

        assert first != original
        assert first == second
        if "[CCR:" in first:
            assert first_store.get(_handle(first)) == original
            assert second_store.get(_handle(second)) == original
        elif "<<ccr:" in first:
            first_match = re.search(r"<<ccr:([0-9a-f]{12})\s+\d+_rows_offloaded>>", first)
            second_match = re.search(r"<<ccr:([0-9a-f]{12})\s+\d+_rows_offloaded>>", second)
            assert first_match is not None
            assert second_match is not None
            assert json.loads(first_store.get(first_match.group(1)) or "") == json.loads(original)
            assert json.loads(second_store.get(second_match.group(1)) or "") == json.loads(original)
        else:
            assert json.loads(first) == json.loads(second)


def test_provider_context_controls_reach_active_compression_config(tmp_path):
    rows = [10 for _ in range(120)]
    rows[37] = 5000
    original = json.dumps(rows, indent=2)
    store = CCRStore(str(tmp_path / "provider-ccr.db"))
    previous_store = active_compression_runtime._store
    active_compression_runtime._store = store
    pi_ai.reset_compression_stats()
    pi_ai.register_compressor(active_compression_runtime.compress)
    try:
        ctx = Context(
            compression_min_tokens=1,
            compression_max_items_after_crush=8,
            compression_enable_ccr_marker=False,
            messages=[
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=[TextContent(type="text", text=original)],
                    timestamp=0,
                )
            ],
        )
        out = pi_ai.compress_context(ctx)
    finally:
        pi_ai.unregister_compressor()
        active_compression_runtime._store = previous_store

    text = out.messages[0].content[0].text
    parsed = json.loads(text)
    assert len(parsed) < len(rows)
    assert len(parsed) <= 8
    assert 5000 in parsed
    assert not any(isinstance(item, dict) and "_ccr_dropped" in item for item in parsed)
    assert "<<ccr:" not in text
    assert pi_ai.get_compression_stats().compressions_by_strategy == {"text": 1}


def test_html_tags_are_not_treated_as_custom_workflow_tags(tmp_path):
    original = "<div>" + ("ordinary html-ish text\n" * 220) + "</div>"
    store = CCRStore(str(tmp_path / "ccr.db"))

    out = compress(original, store)

    assert "Preserved custom tags" not in out
    assert store.get(_handle(out)) == original
