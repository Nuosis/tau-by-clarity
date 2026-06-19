from __future__ import annotations

import json
from pathlib import Path

from pi_ai.tokenization import count_text_tokens


def test_token_count_matches_headroom_tokenizer_fixtures():
    fixture_dir = Path("/tmp/headroom-src/tests/parity/fixtures/tokenizer")
    assert fixture_dir.exists(), "Headroom tokenizer fixtures are required for parity"
    mismatches = []

    for path in sorted(fixture_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        actual = count_text_tokens(str(data["input"]))
        expected = data["output"]
        if actual != expected:
            mismatches.append(f"{path.name}: expected {expected}, got {actual}")

    assert not mismatches
