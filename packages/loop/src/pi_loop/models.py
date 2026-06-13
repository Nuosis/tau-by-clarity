"""Pydantic contracts for the outer loop.

Every artifact the loop produces or consumes is a typed model so malformed
judge/generator output is caught at the boundary, not deep in the driver.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

StopCondition = Literal["goal_accomplished", "churn_detector"]


class DecisionEntry(BaseModel):
    """One implementation decision the agent owns (the 'how'), recorded so churn
    can detect re-litigation and user_feedback can report it."""

    decision: str
    rationale: str
    iteration: int


class LedgerState(BaseModel):
    """Durable running state across iterations — the loop's memory."""

    done: list[str] = Field(default_factory=list)
    decided: list[DecisionEntry] = Field(default_factory=list)
    open: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Compact text view fed to judges and generators."""
        done = "\n".join(f"  - {d}" for d in self.done) or "  (nothing yet)"
        decided = (
            "\n".join(f"  - {d.decision} — {d.rationale} (iter {d.iteration})" for d in self.decided)
            or "  (nothing yet)"
        )
        open_ = "\n".join(f"  - {o}" for o in self.open) or "  (nothing open)"
        return f"DONE:\n{done}\n\nDECIDED:\n{decided}\n\nOPEN:\n{open_}"


class JudgeVerdict(BaseModel):
    """A single stop-detector's verdict. `fired=True` means STOP."""

    fired: bool
    rationale: str


class IterationOutput(BaseModel):
    """What the driver independently observed from one agent invocation.

    `final_text` is the agent's user-facing narration. `tool_events` are the
    factual tool results (the observation seam) — judges weight these over
    narration so the agent never grades its own stop condition.
    """

    final_text: str = ""
    tool_events: list[dict] = Field(default_factory=list)
    exit_code: int = 0
    error: Optional[str] = None

    def observed_evidence(self, limit: int = 6000) -> str:
        """Factual tool-output evidence for goal_accomplished, truncated."""
        if not self.tool_events:
            return "(no tool outputs observed this iteration)"
        import json

        text = json.dumps(self.tool_events, ensure_ascii=False, indent=2)
        if len(text) > limit:
            text = text[:limit] + "\n…(truncated)"
        return text


class ContinuationDecision(BaseModel):
    """Output of the continuation suite. Stop if ANY stop-detector fired."""

    stop: bool
    fired_condition: Optional[StopCondition] = None
    rationale: str = ""
    verdicts: dict[str, JudgeVerdict] = Field(default_factory=dict)


class LoopArtifacts(BaseModel):
    """Produced only on continue — the head of the next iteration."""

    ledger: LedgerState
    user_feedback: str
    new_prompt: str


class TerminalReport(BaseModel):
    """Agentic, first-person sign-off emitted once on stop."""

    condition: StopCondition
    message: str


class IterationRecord(BaseModel):
    """Audit trail of one loop turn."""

    index: int
    prompt_sent: str
    final_text: str
    decision: ContinuationDecision


class LoopResult(BaseModel):
    """Final return of a completed loop run."""

    goal: str
    iterations: list[IterationRecord] = Field(default_factory=list)
    terminal: Optional[TerminalReport] = None
    ledger: LedgerState = Field(default_factory=LedgerState)
