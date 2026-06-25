---
name: agent-build-pattern
description: Use when designing, scoping, or building a new Tau LLM-backed agent or subagent — including creating /Users/marcusswift/agents/<AgentName>, running tau --init, writing OBJECTIVES.md and .tau/SYSTEM.md, building extension tools with Pydantic contracts, adding agent-local skills, defining input/output artifacts, compiling the agent as an importable Python function, and sequencing unit tests plus live evals. Forces prompt-last control-plane discipline, instrumentation before iteration, and clear separation between stub tests and live-LLM evals.
---

# Agent Build Pattern

When the user says "build an agent", "write a new subagent", or "draft a prompt
for X", default to a Tau agent build. The agent directory is the deployment
unit; prompts, objectives, extensions, skills, evals, and compile/test artifacts
belong inside that agent, not scattered through generic application code.

This skill preserves the old control-plane discipline, but the working order is
now the Tau build order. Load `references/control-planes.md` when mapping
behavior to planes. Load `references/build-sequence.md` only when you need the
long-form rationale for prompt-last development.

Use companion skills when they apply:
- `testing-anti-patterns` for unit/eval integrity.
- `eval-sandbox-pattern` for sandboxed live eval execution and trace capture.
- `agent-repair-pattern` when compile, unit, or eval failures require diagnosis.
- `skill-creator` when creating agent-local skills.

## Tau Build Order

1. Create the agent directory under `/Users/marcusswift/agents/<AgentName>`.
2. Initialize Tau in that directory with the project initializer (`tau --init`).
3. Write root `OBJECTIVES.md` with intended user stories and expected
   capabilities.
4. Build extensions that provide the tools the agent needs. Every tool gets
   robust Pydantic input and output models.
5. Decide whether agent-local skills are useful. If yes, compose them with the
   skill-creation workflow and place them under the agent's `.tau/skills/`.
6. Write an extremely brief `.tau/SYSTEM.md`: identity, role, hard rules, and
   voice only.
7. Return to `OBJECTIVES.md` and document input artifacts, output artifacts, and
   subagent handoff contracts. Enforce the input artifact and provide a return
   tool that emits the output artifact.
8. Add eval support. For core agents, copy the root eval extension from
   `/Users/marcusswift/.tau/extensions/evals/` and required dependencies; for
   non-core agents, leverage the core agents/eval harness already available.
9. Compile the agent so it is importable as a Python function object.
10. Write and run basic unit tests.
11. Run eval scenarios and repair failures with evidence, one change at a time.

## File Roles

- `OBJECTIVES.md`: user stories, capabilities, success conditions, artifact
  contracts, subagent handoff contracts, eval expectations.
- `.tau/SYSTEM.md`: concise durable instruction. Do not list tool names,
  artifact schemas, scenario data, project paths, or per-call facts here.
- `.tau/extensions/`: executable tools. Tools own capability and must expose
  typed input/output models.
- `.tau/settings.json`: tool availability and structural access control. An
  empty tools list is a structural denial of default tools, not prompt guidance.
- `.tau/skills/`: agent-local procedural knowledge that is too large,
  reusable, or domain-specific for the system prompt.
- `.tau/evals/`: scenarios that validate the expectations in `OBJECTIVES.md`.

In Tau, a subagent is a full agent. It needs its own initialized directory,
`OBJECTIVES.md`, `.tau/SYSTEM.md`, extensions, settings, tests, and evals.
The parent agent owns delegation policy; the subagent owns its own success
conditions and return artifact.

## Artifact Contracts First

Before writing the system prompt, define the artifact boundary.

- For user-facing agents: write user stories and success conditions in
  `OBJECTIVES.md`.
- For subagents: document who calls the subagent, what input artifact it
  receives, what output artifact it returns, and what failure modes the caller
  must handle.
- Use Pydantic models for every input artifact, output artifact, tool input, and
  tool output.
- If this is a subagent, mirror existing sibling subagent artifact shapes where
  reasonable so parent orchestration stays consistent.
- Enforce input artifacts structurally through the invocation/extension layer.
  Do not rely on prompt prose to make malformed input safe.
- Provide a dedicated return-result tool or extension that returns the output
  artifact. Prefer a forced return tool or strict response schema over asking
  the model to "format JSON correctly."

Validate output schemas against the real configured provider before live evals.
Provider tolerance for deep nesting, recursive refs, unions, and `oneOf` varies.

## Extensions And Tools

Build tools as extensions only after the objective and artifact boundary are
clear.

- Tool availability belongs in `.tau/settings.json` or the runtime tools
  array, not in prompt prose.
- Tool descriptions should be short and operational; the Pydantic schema carries
  field-level constraints.
- Every tool needs typed input and output models with strict enough validation to
  catch malformed calls early.
- Keep parent agents narrow. Prefer a parent orchestrator with explicit subagent
  tools over giving the parent broad read/write/bash access.
- If a capability should not be callable in this turn, do not name the tool in
  `SYSTEM.md`, `OBJECTIVES.md` examples, scenario text, or memory context.
  Models pattern-match dotted tool names and may emit calls even when the tool is
  absent.

## Skills

Create an agent-local skill when behavior is reusable, procedural, or too large
for the system prompt. Do not use skills to hide core contracts.

- Use `skill-creator` for new or updated skills.
- Keep skills under `.tau/skills/` for agent-specific knowledge.
- Put stable domain process in skills; keep per-run facts in the user message or
  input artifact.
- If a skill changes agent behavior, add or update eval scenarios that prove the
  behavior.

## System Prompt Rules

Write `.tau/SYSTEM.md` late and keep it brief.

The system prompt owns:
- identity and role
- hard boundaries the model must self-police
- voice and interaction style

It does not own:
- dynamic task data
- project paths or session variables
- tool names and capability lists
- artifact schemas and field lists
- eval-specific examples

If you are tempted to add a sentence like "do not call X" or "always return
field Y", stop and map the concern to the right plane. Tool access belongs to
the tools/settings plane. Output shape belongs to the artifact model, forced
return tool, or strict schema. Dynamic facts belong to the input artifact or
user message.

## Control-Plane Map

For each required behavior, identify the plane that owns it before coding. The
common planes are system prompt, developer message, user message, tools array,
tool_choice, parallel_tool_calls, response_format/json_schema, max_tokens,
temperature, reasoning effort, seed, cache_control, metadata, and content block
type. See `references/control-planes.md` for details.

Critical distinction:
- Structural enforcement: settings/tools array, forced tool_choice, strict
  response schema, Pydantic validation, max token limits. These reduce eval cost.
- Behavioral enforcement: system prompt, user message, tool descriptions, skill
  prose. These require eval coverage.

## Evals And Tests

Write scenarios that validate each functional expectation in `OBJECTIVES.md`.
Name tests by the plane they prove.

- Unit / contract tests: stub adapter or direct function call proving models,
  extension code, artifact validation, and return-tool plumbing.
- Failure-mode tests: malformed inputs or violating outputs proving the harness
  catches bad behavior.
- Live-LLM evals: real model, real prompt, real tools, real instrumentation.

For core agents, copy the root eval extension from
`/Users/marcusswift/.tau/extensions/evals/` plus dependent directories before
writing scenarios. For non-core agents, use the core agents or existing harness
rather than duplicating infrastructure unnecessarily.

Never claim the agent works from stub/unit tests alone. Compile and unit tests
prove importability and local contracts; live evals prove model behavior.

## Compile And Run Order

Run gates in this order:

1. Compile the agent so it is importable as a Python function object.
2. Run basic unit tests for extensions, artifact models, and return-tool
   behavior.
3. Confirm compile and unit tests are green.
4. Run eval scenarios.
5. Repair failures using captured evidence and one change at a time.

Failures diagnose to their plane:
- Compile failure: packaging/import/config issue.
- Unit failure: extension/model/contract issue.
- Failure-mode eval failure: eval or judge is too permissive.
- Live eval failure: inspect trace, classify the violated plane, then repair.

Do not fix live eval failures by weakening tests or adding reinforcement prose to
the system prompt. Use the `agent-repair-pattern`: capture evidence, find the
smear, make one structural change, re-run the same scenario, keep or revert
based on the trace.

## Instrumentation

Wire trace capture before live eval iteration. Capture the rendered system
prompt, rendered user/input artifact, tools array, tool_choice, response format,
raw response body before parsing, latency, model id, and correlation_id.

When parsing fails, the raw response is the only evidence of what the model
actually returned. Capture it before Pydantic validation or SDK exception
handling truncates the body.

## Anti-patterns this skill blocks

- **Prompt-first development** — writing the system prompt, hoping it covers all behaviors, then writing tests against whatever the model produces.
- **Skipping the Tau scaffold** — hand-building files without `tau --init`,
  then debugging missing settings, extension, skill, or eval
  paths later.
- **Leaving contracts undocumented** — creating extensions or prompts before
  `OBJECTIVES.md` names the user stories, input artifact, output artifact, and
  success conditions.
- **Thin extension schemas** — accepting loose dictionaries or strings where a
  Pydantic model should reject malformed tool calls.
- **Listing dotted tool names in prompt prose without offering them as real tools** — models pattern-match and call them, violating the tools-array contract.
- **Stuffing artifact shape into `.tau/SYSTEM.md`** — output shape belongs to
  Pydantic models, forced return tools, or strict response schemas.
- **Skipping instrumentation until something fails** — by the time you need it, the response that broke is lost.
- **Conflating stub-adapter test green with "the agent works"** — stub tests prove the harness, not the model.
- **Skipping the failure-mode plane** — without it, you can't tell whether your eval passes because the model is correct or because the eval is too permissive.
- **Reaching for `temperature` to fix structural problems** — sampling diversity doesn't fix tool-choice violations, schema mismatches, or boundary violations.
- **Designing the output Pydantic schema without validating against the real provider** — provider tolerance varies; flake here costs you the live eval.
- **Treating the system prompt as the answer to every behavioral question** — it's one plane of fifteen; many behaviors are better enforced structurally.

## Final reporting

When reporting agent build work, include:

- The agent directory and Tau init command used.
- The objective/user-story coverage added to `OBJECTIVES.md`.
- The artifact contracts (Pydantic input/output model names) and return-result
  tool or strict schema used.
- The extensions/tools added and their typed input/output models.
- Any agent-local skills added or intentionally skipped.
- The plane map (which plane owns which behavior).
- Compile status and importable function object path/name.
- Unit, failure-mode, and live eval coverage with counts and pass/fail.
- The correlation_id(s) for live evals, so captured prompts and responses can be
  inspected.
- Any behavior left to behavioral enforcement rather than structural enforcement,
  and why.
