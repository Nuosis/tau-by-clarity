from __future__ import annotations

import json
from pathlib import Path

from pi_coding_agent.core.messages import BashExecutionMessage, bash_execution_to_text, convert_to_llm


def _agent_dir(tmp_path: Path, active: bool = True) -> Path:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"active_compression": active}) + "\n")
    return agent_dir


def test_bash_execution_output_is_compressed_when_active_compression_enabled(tmp_path, monkeypatch):
    agent_dir = _agent_dir(tmp_path, active=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("TAU_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.delenv("PI_ACTIVE_COMPRESSION_DISABLED", raising=False)

    msg = BashExecutionMessage(command="pytest -q", output="line\n" * 1000, exit_code=0)

    rendered = bash_execution_to_text(msg)

    assert "Ran `pytest -q`" in rendered
    assert "[CCR:" in rendered
    assert "ccr_retrieve" in rendered
    assert len(rendered) < len(msg.output)


def test_bash_execution_output_is_raw_when_active_compression_disabled(tmp_path, monkeypatch):
    agent_dir = _agent_dir(tmp_path, active=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("TAU_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_ACTIVE_COMPRESSION_DISABLED", "1")

    output = "line\n" * 1000
    msg = BashExecutionMessage(command="pytest -q", output=output, exit_code=0)

    rendered = bash_execution_to_text(msg)

    assert "[CCR:" not in rendered
    assert output in rendered


def test_convert_to_llm_preserves_dict_messages_loaded_from_session():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello from jsonl"}],
            "timestamp": 1,
        },
        {
            "role": "bashExecution",
            "command": "pytest -q",
            "output": "done",
            "exit_code": 0,
            "timestamp": 2,
        },
    ]

    converted = convert_to_llm(messages)

    assert converted[0]["content"][0]["text"] == "hello from jsonl"
    assert converted[1]["role"] == "user"
    assert "Ran `pytest -q`" in converted[1]["content"][0]["text"]
    assert "done" in converted[1]["content"][0]["text"]
