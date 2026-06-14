"""
Working-context config + block types.

NOTE (design §12): the position-based middle compression that used to live here
(`compress_working_context` / `_breadcrumb`) was **dropped** — active compression
(Headroom/CCR, `pi_coding_agent.active_compression`) replaces it. What remains is
the per-model floor/ceiling config and the `CtxBlock` type, still used by the
memory recall path. Floor/ceiling are per-model, from calibration
(`evals/niah_calibrate.py`, §7).
"""
from __future__ import annotations

from dataclasses import dataclass


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


# Positional middle compression (`compress_working_context` / `_breadcrumb`) was
# removed here — see the module docstring and design §12. Active compression
# (Headroom/CCR) now owns context reduction; memory owns record + recall.
