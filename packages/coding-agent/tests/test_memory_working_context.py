"""P4 gate — active compression: floor passthrough, position-compress + recall, cache-stable."""
from __future__ import annotations

from pi_coding_agent.core.memory import (
    CtxBlock,
    DeterministicEmbeddingProvider,
    MemoryStore,
    MemoryType,
    SemanticMemory,
    WorkingContextConfig,
    compress_working_context,
)

CFG = WorkingContextConfig(floor_tokens=200, ceiling_tokens=4000,
                           head_tokens=120, tail_tokens=120, recall_k=3)


def _blocks(n: int, words: int = 70) -> list[CtxBlock]:
    # realistic multi-word text so the first-N-words breadcrumb actually truncates
    return [CtxBlock(role="user" if i % 2 == 0 else "assistant",
                     text=f"block{i} " + "lorem ipsum dolor sit amet " * (words // 5),
                     eid=f"b{i}") for i in range(n)]


def _store(tmp_path):
    s = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    s.write_semantic(SemanticMemory(id="", project="", memory_type=MemoryType.FILE_API.value,
                                    title="reconnect", content="MAX_RECONNECT_ATTEMPTS is 7741",
                                    key="fileapi:max_reconnect", provenance="curator"))
    return s


def test_below_floor_passthrough(tmp_path):
    s = _store(tmp_path)
    blocks = _blocks(1)                       # ~100 tok < floor 200
    out = compress_working_context(blocks, s, "anything", CFG)
    assert out == blocks                      # untouched
    s.close()


def test_above_floor_compresses_and_recalls(tmp_path):
    s = _store(tmp_path)
    blocks = _blocks(12)                       # ~1200 tok > floor
    out = compress_working_context(blocks, s, "what is MAX_RECONNECT_ATTEMPTS?", CFG)
    # head + tail verbatim
    assert out[0].text == blocks[0].text
    assert any(o.text == blocks[-1].text for o in out)
    # middle compressed to breadcrumbs
    assert any(o.text.startswith("[compressed") for o in out)
    # recalled memory appended at the tail
    assert out[-1].eid == "recall" and "7741" in out[-1].text
    # net smaller than verbatim transcript
    assert sum(o.tokens for o in out) < sum(b.tokens for b in blocks)
    s.close()


def test_head_is_cache_stable_across_queries(tmp_path):
    s = _store(tmp_path)
    blocks = _blocks(12)
    out_a = compress_working_context(blocks, s, "query about reconnect", CFG)
    out_b = compress_working_context(blocks, s, "totally different question", CFG)
    # the compressed prefix (everything except the query-dependent recall tail) is identical
    pre_a = [o.text for o in out_a if o.eid != "recall"]
    pre_b = [o.text for o in out_b if o.eid != "recall"]
    assert pre_a == pre_b                       # prefix unchanged by query → cache-warm
    s.close()


def test_no_store_still_compresses(tmp_path):
    blocks = _blocks(12)
    out = compress_working_context(blocks, None, "q", CFG)
    assert any(o.text.startswith("[compressed") for o in out)
    assert all(o.eid != "recall" for o in out)   # no recall block without a store
