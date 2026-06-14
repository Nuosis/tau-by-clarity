"""The outer driver — drives any tau agent to a judge-determined stop.

Flow:
    agent runs → continuation suite (stop if ANY fired)
        stop     → agentic terminal report → done
        continue → ledger, user_feedback, new_prompt → loop

NO MAX LOOPS. churn_detector is the sole runaway guard. The only deterministic
surface is this control flow; every decision and every user-facing line is a
genuine LLM call. The agent supplies its domain rubric and voice via its loop
profile (`<agent_dir>/.pi-py/loop_profile.json`).
"""

from __future__ import annotations

from typing import Callable, Optional

from .agent_runner import run_agent
from .artifacts import build_continue_artifacts, terminal_report
from .judges import run_continuation_suite
from .llm import LlmFn, resolve_llm
from .models import IterationRecord, LedgerState, LoopResult
from .profile import LoopProfile, load_profile
from .steering import SteeringInbox

Emit = Callable[[str], None]


def _default_emit(text: str) -> None:
    print(text, flush=True)


def run_loop(
    goal: str,
    agent_dir: str,
    *,
    project_root: Optional[str] = None,
    llm_fn: Optional[LlmFn] = None,
    profile: Optional[LoopProfile] = None,
    agent_runner: Callable[..., object] = run_agent,
    emit: Emit = _default_emit,
    iteration_timeout: Optional[int] = None,
    inbox: Optional[SteeringInbox] = None,
) -> LoopResult:
    """Drive `goal` against the agent at `agent_dir` to a judge-determined stop.

    `llm_fn`, `profile`, and `agent_runner` are injectable so tests can exercise
    the driver's branch logic against real captured verdicts/outputs without ever
    stubbing a judge with a deterministic fake.
    """
    llm_fn = llm_fn or resolve_llm(agent_dir)
    profile = profile or load_profile(agent_dir)

    ledger = LedgerState()
    history: list[str] = []
    result = LoopResult(goal=goal, ledger=ledger)

    prompt = goal
    index = 0
    while True:
        index += 1
        output = agent_runner(
            prompt, agent_dir, project_root=project_root, timeout=iteration_timeout
        )

        decision = run_continuation_suite(
            llm_fn, goal, ledger, output, history, profile.done_rubric
        )
        result.iterations.append(
            IterationRecord(
                index=index, prompt_sent=prompt, final_text=output.final_text, decision=decision
            )
        )
        history.append(output.final_text)

        if decision.stop:
            report = terminal_report(
                llm_fn, decision.fired_condition, goal, ledger, decision.rationale, profile
            )
            result.terminal = report
            result.ledger = ledger
            emit(report.message)
            return result

        # Drain any steering the user staged while the agent was working, and
        # inject it into the next prompt (rather than making them wait for the
        # whole loop to finish).
        steering = inbox.drain() if inbox is not None else []
        if steering:
            emit("↪ applying your steering: " + " | ".join(steering))

        artifacts = build_continue_artifacts(
            llm_fn, goal, ledger, output, index, profile, steering=steering
        )
        ledger = artifacts.ledger
        result.ledger = ledger
        emit(artifacts.user_feedback)
        prompt = artifacts.new_prompt
