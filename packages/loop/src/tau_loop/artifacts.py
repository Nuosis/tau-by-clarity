"""Artifact generators — fire only on CONTINUE (ledger first, it feeds the rest),
plus the terminal report emitted on STOP.

Everything user-facing here is agentic: first-person, in the agent's own voice
(from its loop profile), "the agent is talking to me," never a templated status
dump. All generators are genuine LLM calls.
"""

from __future__ import annotations

from .llm import LlmFn, call_json
from .models import IterationOutput, LedgerState, LoopArtifacts, StopCondition, TerminalReport
from .profile import LoopProfile

# --------------------------------------------------------------------------- #
# Ledger (structured) — the loop's durable memory, updated first each turn.
# --------------------------------------------------------------------------- #

_LEDGER_SYSTEM = (
    "You maintain the durable ledger for an autonomous agent loop. Given the "
    "current ledger and the latest iteration, return the UPDATED ledger. Move "
    "items the evidence shows finished into `done`; record every implementation "
    "decision the agent made (the 'how') in `decided` with a short rationale and "
    "the iteration number; keep genuinely unresolved threads in `open`. Never drop "
    "a settled decision — the loop relies on this to avoid re-deciding. Be terse "
    "and factual; do not invent items not supported by the input. "
    'Return ONLY a JSON object: {"done": [str], "decided": [{"decision": str, '
    '"rationale": str, "iteration": int}], "open": [str]}.'
)


def update_ledger(
    llm_fn: LlmFn, ledger: LedgerState, output: IterationOutput, iteration: int
) -> LedgerState:
    user = (
        f"ITERATION: {iteration}\n\n"
        f"CURRENT LEDGER:\n{ledger.render()}\n\n"
        f"LATEST NARRATION:\n{output.final_text or '(none)'}\n\n"
        f"OBSERVED EVIDENCE:\n{output.observed_evidence()}"
    )
    return call_json(llm_fn, _LEDGER_SYSTEM, user, LedgerState)


# --------------------------------------------------------------------------- #
# user_feedback (prose) — report, never ask. Carries the merged escalation:
# a genuine user-owned 'what' becomes an assume-and-flag line, never a question.
# --------------------------------------------------------------------------- #

_FEEDBACK_SYSTEM = (
    "You are {agent_label}, {voice}, giving the user a short progress update in "
    "your own first-person voice. The user owns the WHAT (intent, customer "
    "experience); you own the HOW (implementation, architecture). Rewrite the raw "
    "iteration into a report, not a request:\n"
    "- Any place the raw text hedges or asks permission about an implementation "
    "choice you own, state it as a decision with a one-clause reason "
    "('I went with TanStack Query because it caches server state cleanly').\n"
    "- If something genuinely depends on a user-owned WHAT you can't know, do NOT "
    "ask and do NOT stop: make your best inference, act on it, and flag it inline "
    "('Assumed monthly billing — tell me if it should be annual').\n"
    "- Lead with what changed/shipped this iteration; keep it concrete and brief; "
    "no headings, no bullet soup, just a couple of plain sentences.\n"
    "Write only the message you'd send the user."
)


def user_feedback(
    llm_fn: LlmFn, goal: str, ledger: LedgerState, output: IterationOutput, profile: LoopProfile
) -> str:
    system = _FEEDBACK_SYSTEM.format(agent_label=profile.agent_label, voice=profile.voice)
    user = (
        f"GOAL:\n{goal}\n\n"
        f"LEDGER:\n{ledger.render()}\n\n"
        f"THIS ITERATION (raw):\n{output.final_text or '(none)'}\n\n"
        f"OBSERVED EVIDENCE:\n{output.observed_evidence()}"
    )
    return (llm_fn(system, user) or "").strip()


# --------------------------------------------------------------------------- #
# new_prompt (prose) — the next instruction to the agent; drift steers it back
# to the original goal. Drift is steering, never a stop condition.
# --------------------------------------------------------------------------- #

_PROMPT_SYSTEM = (
    "You write the next instruction to an autonomous agent to drive the ORIGINAL "
    "goal forward by one concrete step. Use the ledger to avoid re-asking for "
    "settled decisions. If the latest work has drifted away from the original "
    "goal, steer it back explicitly. If the loop is re-treading, tell the agent to "
    "lock the settled choice and move to the next open item. Be specific and "
    "action-oriented; name the next outcome, not a vague 'continue'. Write the "
    "instruction directly to the agent in the imperative, nothing else.\n\n"
    "If USER STEERING is present, it is the user redirecting the work in real "
    "time: honor it FIRST and let it override drift and your own next step, while "
    "still serving the original goal. Fold it into the instruction as the priority."
)


def new_prompt(
    llm_fn: LlmFn,
    goal: str,
    ledger: LedgerState,
    output: IterationOutput,
    steering: list[str] | None = None,
) -> str:
    steer_block = ""
    if steering:
        joined = "\n".join(f"- {s}" for s in steering)
        steer_block = f"\n\nUSER STEERING (incorporate first, overrides drift):\n{joined}"
    user = (
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"LEDGER:\n{ledger.render()}\n\n"
        f"WHAT JUST HAPPENED:\n{output.final_text or '(none)'}\n\n"
        f"OBSERVED EVIDENCE:\n{output.observed_evidence()}"
        f"{steer_block}"
    )
    return (llm_fn(_PROMPT_SYSTEM, user) or "").strip()


def build_continue_artifacts(
    llm_fn: LlmFn,
    goal: str,
    ledger: LedgerState,
    output: IterationOutput,
    iteration: int,
    profile: LoopProfile,
    steering: list[str] | None = None,
) -> LoopArtifacts:
    """Ledger first (feeds the other two), then feedback and the next prompt
    (which folds in any staged user steering)."""
    new_ledger = update_ledger(llm_fn, ledger, output, iteration)
    feedback = user_feedback(llm_fn, goal, new_ledger, output, profile)
    nxt = new_prompt(llm_fn, goal, new_ledger, output, steering)
    return LoopArtifacts(ledger=new_ledger, user_feedback=feedback, new_prompt=nxt)


# --------------------------------------------------------------------------- #
# Terminal report (prose) — agentic sign-off, keyed to the firing condition.
# --------------------------------------------------------------------------- #

_TERMINAL_SYSTEM = (
    "You are {agent_label}, {voice}, signing off on a piece of work in your own "
    "first-person voice — the user should feel you are talking to them, not "
    "reading a status dump. Keep it short and concrete, grounded in the ledger.\n"
    "- If the reason is 'goal_accomplished': report what you shipped and the key "
    "decisions behind it, with quiet confidence. No hedging.\n"
    "- If the reason is 'churn_detector': tell the user plainly that you're "
    "stopping because you're no longer making progress, name exactly where it "
    "stands and what's blocking, and what you'd need to move it forward.\n"
    "Write only the message you'd send the user."
)


def terminal_report(
    llm_fn: LlmFn,
    condition: StopCondition,
    goal: str,
    ledger: LedgerState,
    rationale: str,
    profile: LoopProfile,
) -> TerminalReport:
    system = _TERMINAL_SYSTEM.format(agent_label=profile.agent_label, voice=profile.voice)
    user = (
        f"REASON: {condition}\n"
        f"JUDGE RATIONALE: {rationale}\n\n"
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"FINAL LEDGER:\n{ledger.render()}"
    )
    message = (llm_fn(system, user) or "").strip()
    return TerminalReport(condition=condition, message=message)
