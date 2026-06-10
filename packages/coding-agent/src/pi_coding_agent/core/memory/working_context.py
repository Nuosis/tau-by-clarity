"""
Active compression — the working-context consumer (P4).

Position-based, cache-friendly (design doc §§6–7): below the floor F leave the context
untouched (append + cache-warm); above F protect a head + tail budget verbatim and
compress the middle to light breadcrumbs, then append the store's recalled atomic
memories at the tail. Durable facts live in the STORE (curator-extracted, §8/P2), so the
transcript breadcrumb can be light — recall is the safety net, not the stub.

Floor/ceiling are per-model, from calibration (`evals/niah_calibrate.py`, §7).
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Scope
from .recall import build_recall_block
from .store import MemoryStore


@dataclass
class WorkingContextConfig:
    floor_tokens: int          # below this: do not compress (cache-warm)
    ceiling_tokens: int        # the model's measured reliable working-set size
    head_tokens: int = 8000    # anchor budget kept verbatim
    tail_tokens: int = 8000    # frontier budget kept verbatim
    recall_k: int = 6
    recall_budget_tokens: int = 1500


# per-model, from §7 calibration (MiniMax-M3: ~250k reliable on dense; floor ~0.5×).
MODEL_PROFILES: dict[str, WorkingContextConfig] = {
    "MiniMax-M3": WorkingContextConfig(floor_tokens=120_000, ceiling_tokens=250_000),
}
DEFAULT_PROFILE = WorkingContextConfig(floor_tokens=20_000, ceiling_tokens=40_000,
                                       head_tokens=4000, tail_tokens=4000)


def profile_for(model: str | None) -> WorkingContextConfig:
    return MODEL_PROFILES.get(model or "", DEFAULT_PROFILE)


@dataclass
class CtxBlock:
    role: str
    text: str
    eid: str = ""

    @property
    def tokens(self) -> int:
        return max(1, len(self.text) // 4)


def _breadcrumb(b: CtxBlock) -> CtxBlock:
    """Light stub — first ~24 words + marker. The store is the fact safety-net."""
    words = b.text.split()
    head = " ".join(words[:24])
    return CtxBlock(role=b.role, eid=b.eid,
                    text=f"[compressed{(' ' + b.eid) if b.eid else ''}] {head}…")


def compress_working_context(
    blocks: list[CtxBlock], store: MemoryStore | None, query: str,
    config: WorkingContextConfig, scope: Scope | None = None,
) -> list[CtxBlock]:
    """Return the working-set blocks. Below floor: unchanged. Above floor: head+tail
    verbatim, middle → breadcrumbs, recalled memories appended at the tail."""
    total = sum(b.tokens for b in blocks)
    if total < config.floor_tokens:
        return list(blocks)

    head, head_tok = [], 0
    for b in blocks:
        if head_tok >= config.head_tokens:
            break
        head.append(b)
        head_tok += b.tokens
    tail, tail_tok = [], 0
    for b in reversed(blocks):
        if tail_tok >= config.tail_tokens:
            break
        tail.append(b)
        tail_tok += b.tokens
    tail.reverse()
    protected_ids = {id(b) for b in head} | {id(b) for b in tail}

    out: list[CtxBlock] = []
    for b in blocks:
        out.append(b if id(b) in protected_ids else _breadcrumb(b))

    if store is not None:
        block = build_recall_block(store, query, scope, k=config.recall_k,
                                   token_budget=config.recall_budget_tokens)
        if block:
            out.append(CtxBlock(role="user", text=block, eid="recall"))
    return out
