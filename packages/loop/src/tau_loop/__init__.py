"""tau_loop — judge-driven outer loop that drives any tau agent to completion.

Loop engineering as harness infrastructure: a deterministic driver wrapped around
genuine LLM judges (no max loops, no deterministic stubs). The agent supplies its
domain rubric and voice via `<agent_dir>/.tau/loop_profile.json`; everything
else is agent-agnostic.
"""

from .loop import run_loop
from .models import (
    ContinuationDecision,
    DecisionEntry,
    IterationOutput,
    IterationRecord,
    JudgeVerdict,
    LedgerState,
    LoopArtifacts,
    LoopResult,
    TerminalReport,
)
from .profile import LoopProfile, load_profile
from .steering import SteeringInbox

__all__ = [
    "run_loop",
    "LoopProfile",
    "load_profile",
    "SteeringInbox",
    "ContinuationDecision",
    "DecisionEntry",
    "IterationOutput",
    "IterationRecord",
    "JudgeVerdict",
    "LedgerState",
    "LoopArtifacts",
    "LoopResult",
    "TerminalReport",
]
