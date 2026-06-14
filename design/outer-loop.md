# Outer Loop (`tau_loop`) — judge-driven continuation/completion harness

## What this is

"Loop engineering" as **harness infrastructure**: instead of hand-prompting an
agent turn by turn, define a goal and let a loop re-prompt the agent until a
judge-verified done-condition holds. The skill moves from the prompt to the
harness — verification, termination, state, voice.

`tau_loop` is agent-agnostic. It drives **any** tau agent; the agent supplies
only its domain rubric and voice via `<agent_dir>/.tau/loop_profile.json`.
Devin is one consumer, not the owner.

## Three loops — name the one this is

1. **Agent's own internal loop** (e.g. DevFlow drain) — exists.
2. **Agent orchestration loop** — one turn fans out to subagents — exists.
3. **Outer continuation loop** — re-invokes the agent against a standing goal
   until done. **This is `tau_loop`.** Stop/continue is judge-driven, not a
   predicate and not a counter.

## Hard rules (locked)

- **NO MAX LOOPS.** No iteration cap, no token cap. `churn_detector` is the only
  runaway guard — a counter would kill the productive long runs loops exist for.
- **The loop owns the judges.** The agent never grades its own stop condition.
- **Every judge is a genuine LLM judge from day one. NEVER stub a deterministic
  response. EVER.** No heuristic churn detector, no canned verdicts, no
  "just for the unit test" exception.
- **Provider/model are the agent's configured ones**, resolved through
  `pi_ai.get_model` — never hardcoded.
- **Every user-facing artifact is agentic** — first-person, the agent's voice,
  never a templated dump. Includes the terminal report.

## Loop flow

```
agent runs ──► final_text + factual tool_events
        │
        ▼
CONTINUATION SUITE   (stop if ANY fired)
   ├─ goal_accomplished?  (with the agent's done_rubric)
   └─ churn_detector?
        │ none fired (continue)            any fired ──► TERMINAL REPORT
        ▼                                                (agentic, keyed to
   ARTIFACT PHASE                                          firing condition) → STOP
   1. ledger       update done/decided/open
   2. user_feedback   report (assume-and-flag) → user
   3. new_prompt      goal + ledger + drift → agent
        │
        └──► loop
```

## Continuation suite (stop-detectors, disjunction)

- **goal_accomplished** — is the original goal achieved per observed evidence,
  weighting factual tool outputs over narration? Conservative (no false "done").
  Domain-agnostic: the agent's `done_rubric` tells it which observed signals mean
  completion here.
- **churn_detector** — semantic non-progress (re-deciding settled choices,
  oscillation, no net state delta). The sole runaway guard.

Excluded by design: safety judge (subagent gating owns safety), regression judge
(shelved), escalation judge (folded into `user_feedback` as inform-and-continue),
max loops.

## Artifact phase (continue only; ledger first)

- **ledger** — durable done/decided/open; the loop's memory, so churn and drift
  have something to compare against.
- **user_feedback** — narration rewritten to *report*, not ask. Implementation
  hedges become stated decisions; a genuine user-owned *what* becomes an
  assume-and-flag line, never a blocking question. Agentic voice (from profile).
- **new_prompt** — next instruction; drift steers back to the original goal.

## Terminal report (on stop)

First-person sign-off in the agent's voice, keyed to the firing condition
(`goal_accomplished` → what shipped; `churn_detector` → where it's stuck and what
it needs). An LLM artifact, never templated.

## The profile seam

`<agent_dir>/.tau/loop_profile.json`:

```json
{ "done_rubric": "...domain signals that mean done...",
  "voice": "Devin, a friendly engineer",
  "agent_label": "Devin" }
```

All fields optional; generic defaults apply when absent. This is the ONLY place
agent/domain specifics live — the harness never names an agent.

## Package layout

```
packages/loop/src/tau_loop/
  models.py        # Pydantic contracts
  llm.py           # config-driven model via pi_ai.get_model + call_json (parse + 1 re-ask, no fallback)
  profile.py       # load_profile(agent_dir) -> LoopProfile
  judges.py        # goal_accomplished, churn_detector, pure aggregate_verdicts
  artifacts.py     # ledger / user_feedback / new_prompt / terminal_report
  agent_runner.py  # spawn `tau --mode json`, parse events -> IterationOutput
  loop.py          # the driver;  __main__.py the CLI
```

Run: `python -m tau_loop --agent-dir <X> "<goal>"`.

## Tests & evals

- Offline (`packages/loop/tests/`): models, pure aggregation, profile loading,
  and full continue→stop wiring replaying **real captured** verdicts
  (`golden_loop.json`, regenerate with `capture_golden.py`).
- Live (`live_judge_eval.py`): judge discrimination, including realistic
  agent-shaped "claimed done but not really" payloads (runner evidence
  contradicting narration). Genuine model calls; never faked.
