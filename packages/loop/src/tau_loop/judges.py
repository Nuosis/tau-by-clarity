"""Continuation suite — the stop-detectors.

Two genuine LLM judges. The continuation suite stops if ANY fires (disjunction).
There is NO iteration cap and NO token cap: churn_detector is the sole runaway
guard, which is exactly why it must be a real semantic judge and never a
deterministic diff/hash heuristic.

The outer loop owns these judges; the agent never grades its own stop condition.
Judges weight the factual tool-output evidence over the agent's narration.

`goal_accomplished` is domain-agnostic: what "done" looks like for a given agent
is supplied by the agent's loop profile (`done_rubric`), not baked in here.
"""

from __future__ import annotations

from .llm import LlmFn, call_json
from .models import (
    ContinuationDecision,
    IterationOutput,
    JudgeVerdict,
    LedgerState,
)

_GOAL_SYSTEM = (
    "You are the goal_accomplished stop-detector for an autonomous agent loop. "
    "You decide ONE thing: is the user's ORIGINAL goal now fully achieved, judged "
    "against observed evidence — not against the agent's optimistic narration. "
    "Weight the factual tool outputs over the agent's prose. If the evidence is "
    "silent, partial, or only the agent's claim, it is NOT achieved. Be "
    "conservative: when unsure, do not fire — the loop has no iteration cap, so a "
    "missed stop is cheap but a false 'done' ships incomplete work.\n\n"
    "WHAT 'DONE' LOOKS LIKE FOR THIS AGENT:\n{done_rubric}\n\n"
    'Return ONLY a JSON object: {{"fired": <bool, true means the goal IS achieved '
    'and the loop should STOP>, "rationale": <one sentence grounded in the '
    'evidence>}}.'
)

_CHURN_SYSTEM = (
    "You are the churn_detector stop-detector for an autonomous agent loop. You "
    "are the ONLY runaway guard — there is no iteration limit. You decide ONE "
    "thing: has the loop stopped making real progress? Fire when you see SEMANTIC "
    "non-progress: the agent re-deciding a choice already settled in the ledger, "
    "oscillating between the same options, repeating the same action without a new "
    "result, or producing no net change to observed state across iterations. Do "
    "NOT fire merely because the work is hard or multi-step — genuine forward "
    "progress (new decisions, new completed items, changing observed state) means "
    "continue, however long it takes. Distinguish 'slow but advancing' from "
    "'spinning'. "
    'Return ONLY a JSON object: {"fired": <bool, true means the loop is STUCK and '
    'should STOP>, "rationale": <one sentence naming the specific churn or the '
    'specific progress that rules it out>}.'
)


def _goal_user(goal: str, ledger: LedgerState, output: IterationOutput) -> str:
    return (
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"LEDGER (durable state):\n{ledger.render()}\n\n"
        f"AGENT'S LATEST NARRATION:\n{output.final_text or '(none)'}\n\n"
        f"OBSERVED TOOL-OUTPUT EVIDENCE (factual):\n{output.observed_evidence()}"
    )


def _churn_user(
    goal: str, ledger: LedgerState, output: IterationOutput, history: list[str]
) -> str:
    recent = "\n\n".join(f"[iter -{i}] {t}" for i, t in enumerate(reversed(history[-4:]), 1)) or "(none)"
    return (
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"LEDGER (durable state):\n{ledger.render()}\n\n"
        f"RECENT AGENT NARRATION (most recent first):\n{recent}\n\n"
        f"THIS ITERATION'S NARRATION:\n{output.final_text or '(none)'}\n\n"
        f"THIS ITERATION'S OBSERVED EVIDENCE:\n{output.observed_evidence()}"
    )


def goal_accomplished(
    llm_fn: LlmFn,
    goal: str,
    ledger: LedgerState,
    output: IterationOutput,
    done_rubric: str,
) -> JudgeVerdict:
    system = _GOAL_SYSTEM.format(done_rubric=done_rubric)
    return call_json(llm_fn, system, _goal_user(goal, ledger, output), JudgeVerdict)


def churn_detector(
    llm_fn: LlmFn,
    goal: str,
    ledger: LedgerState,
    output: IterationOutput,
    history: list[str],
) -> JudgeVerdict:
    return call_json(
        llm_fn, _CHURN_SYSTEM, _churn_user(goal, ledger, output, history), JudgeVerdict
    )


# Stop-detector precedence: success beats stuck when both fire.
_PRECEDENCE: tuple[str, ...] = ("goal_accomplished", "churn_detector")


def aggregate_verdicts(verdicts: dict[str, JudgeVerdict]) -> ContinuationDecision:
    """Pure disjunction: STOP if ANY stop-detector fired. Deterministic — this is
    control logic over already-computed verdicts, not a judgment, so it is the one
    part of the suite that is unit-tested without an LLM."""
    for name in _PRECEDENCE:
        v = verdicts.get(name)
        if v is not None and v.fired:
            return ContinuationDecision(
                stop=True, fired_condition=name, rationale=v.rationale, verdicts=verdicts
            )
    return ContinuationDecision(stop=False, rationale="no stop-detector fired", verdicts=verdicts)


def run_continuation_suite(
    llm_fn: LlmFn,
    goal: str,
    ledger: LedgerState,
    output: IterationOutput,
    history: list[str],
    done_rubric: str,
) -> ContinuationDecision:
    """Evaluate every stop-detector with the live judges, then aggregate."""
    verdicts = {
        "goal_accomplished": goal_accomplished(llm_fn, goal, ledger, output, done_rubric),
        "churn_detector": churn_detector(llm_fn, goal, ledger, output, history),
    }
    return aggregate_verdicts(verdicts)
