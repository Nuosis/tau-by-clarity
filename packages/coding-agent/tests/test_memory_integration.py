"""P5 gate — MemoryIntegration bundle: recall + record_turn + compress as a pipeline."""
from __future__ import annotations

import json
import os

from pi_coding_agent.core.memory import CtxBlock, WorkingContextConfig
from pi_coding_agent.core.memory.integration import MemoryIntegration, memory_enabled


class StubLlm:
    def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": True})
        return json.dumps({"decisions": [{
            "title": "reconnect", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
            "memory_type": "file_api", "key": "fileapi:max_reconnect",
            "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}]})


def _integration(tmp_path):
    os.environ["PI_MEMORY_EMBED"] = "deterministic"   # hermetic embeddings
    cfg = WorkingContextConfig(floor_tokens=200, ceiling_tokens=2000,
                               head_tokens=120, tail_tokens=120, recall_k=3)
    return MemoryIntegration(str(tmp_path), llm_fn=StubLlm(), config=cfg)


def test_flag_default_off():
    os.environ.pop("PI_MEMORY_ENABLED", None)
    assert memory_enabled() is False
    os.environ["PI_MEMORY_ENABLED"] = "1"
    assert memory_enabled() is True
    os.environ.pop("PI_MEMORY_ENABLED", None)


def test_record_then_recall(tmp_path):
    from pi_coding_agent.core.memory import Evidence
    mi = _integration(tmp_path)
    written = mi.record_turn([Evidence("e1", "tool_result",
                                        "config/net.py: MAX_RECONNECT_ATTEMPTS = 7741")])
    assert written                                   # curator committed
    block = mi.recall_block("what is MAX_RECONNECT_ATTEMPTS?")
    assert block and "7741" in block                 # recall surfaces it
    mi.close()


def test_compress_uses_store_recall(tmp_path):
    from pi_coding_agent.core.memory import Evidence
    mi = _integration(tmp_path)
    mi.record_turn([Evidence("e1", "tool_result", "MAX_RECONNECT_ATTEMPTS = 7741")])
    blocks = [CtxBlock("user" if i % 2 == 0 else "assistant",
                       f"turn{i} " + "lorem ipsum dolor sit amet " * 14, eid=f"b{i}")
              for i in range(12)]
    out = mi.compress(blocks, "what is MAX_RECONNECT_ATTEMPTS?")
    assert any(o.text.startswith("[compressed") for o in out)   # middle compressed
    assert out[-1].eid == "recall" and "7741" in out[-1].text    # store recall appended
    mi.close()
