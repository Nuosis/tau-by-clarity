from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pi_coding_agent.clarity_pii.extension import extension_factory


class _FakePi:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def register_flag(self, _name: str, _definition: dict[str, Any]) -> None:
        return None

    def get_flag(self, _name: str) -> None:
        return None

    def on(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def register_command(self, _name: str, _definition: dict[str, Any]) -> None:
        return None


@pytest.mark.asyncio
async def test_provider_pii_filter_preserves_opaque_protocol_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_AGENT_DIR", str(tmp_path))
    pi = _FakePi()
    extension_factory(pi)
    context = SimpleNamespace(session_id="session-provider-id")
    pi.handlers["session_start"]({}, context)

    provider_item_id = "fc_04f4111111111111111a6509"
    provider_call_id = "call_04f4111111111111111a6509"
    payload = {
        "model": "gpt-5.4",
        "input": [
            {
                "type": "function_call",
                "id": provider_item_id,
                "call_id": provider_call_id,
                "name": "read",
                "arguments": '{"card": "4111111111111111"}',
            },
            {
                "type": "function_call_output",
                "call_id": provider_call_id,
                "output": "Call me at 250-555-0123.",
            },
        ],
    }

    result = await pi.handlers["before_provider_request"](
        {"payload": payload},
        context,
    )

    assert result["input"][0]["id"] == provider_item_id
    assert result["input"][0]["call_id"] == provider_call_id
    assert result["input"][1]["call_id"] == provider_call_id
    assert "[PII:CC:1]" in result["input"][0]["arguments"]
    assert "[PII:PHONE:1]" in result["input"][1]["output"]


def test_ccr_evidence_survives_outbound_pii_filter(tmp_path, monkeypatch):
    from pi_coding_agent import active_compression
    from pi_coding_agent.active_compression.ccr import CCRStore
    from pi_coding_agent.active_compression.extension import _retrieve_tool_response
    from pi_coding_agent.clarity_pii.vault import Vault
    from pi_ai import pii
    from pi_ai.types import Context, ToolResultMessage

    store = CCRStore(str(tmp_path / 'ccr.db'))
    monkeypatch.setattr(active_compression, '_store', store)
    original = ('2026-09-14T12:02:22 INFO audit_marker deployment_http_status=503\n'
                '2026-09-14T12:02:23 INFO contact +1 (604) 555-0199\n')
    handle = store.put(original)
    retrieved = _retrieve_tool_response(handle, 'audit_marker', tool_name='ccr_retrieve')
    assert retrieved['details']['kept_items'] > 0
    vault = Vault()
    monkeypatch.setattr(pii, '_factory', lambda: (vault.tokenize, vault.detokenize))
    context = Context(messages=[ToolResultMessage(tool_call_id='retrieve', tool_name='ccr_retrieve',
        content=retrieved['content'], details=retrieved['details'], timestamp=0)])
    protected, _ = pii.protect_context(context)
    text = protected.messages[0].content[0].text
    assert 'audit_marker deployment_http_status=503\n2026-09-14T12:02:23' in text
    assert '+1 (604) 555-0199' not in text
    assert '[PII:PHONE:' in text
    assert vault.detokenize(text) == retrieved['content'][0]['text']
