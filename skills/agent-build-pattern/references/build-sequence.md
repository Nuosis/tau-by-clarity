# Build Sequence Reference

Long-form rationale for the pi-py agent build order in `SKILL.md`.

## Why pi-py build order beats prompt-first

Prompt-first agent work fails because it puts the weakest control plane in
charge of the most important boundaries. The system prompt can describe an
agent, but it cannot create the scaffold, enforce artifact schemas, make tools
callable, validate malformed handoffs, or prove model behavior under eval.

The pi-py build order forces the durable pieces into the places that own them:

1. Directory and `.tau/` scaffold establish the deployment unit.
2. `OBJECTIVES.md` defines what the agent is expected to do.
3. Extensions and Pydantic models define executable capability.
4. Agent-local skills hold reusable procedural knowledge.
5. `.tau/SYSTEM.md` stays small because the other planes carry their own
   responsibilities.
6. Artifact contracts and return tools make handoffs machine-checkable.
7. Evals prove the expectations against real model behavior.
8. Compile and unit tests prove the agent can actually be imported and executed.

## Step 1 — Create and initialize the agent

Create `/Users/marcusswift/agents/<AgentName>` and run the pi-py initializer in
that directory. Use the installed command for the environment:

- `py-py --init` when that is the project command.
- `pi-py --init` when that is what the local install exposes.

Do not manually mimic the scaffold unless the initializer is genuinely
unavailable. Missing scaffold details tend to surface later as confusing
settings, extension, skill, or eval path failures.

## Step 2 — Write `OBJECTIVES.md`

Start with user stories for user-facing agents and calling contracts for
subagents. A good objectives file names:

- intended user stories or caller use cases
- expected capabilities
- success conditions
- input artifact shape
- output artifact shape
- subagent handoff expectations, when applicable
- functional expectations that need eval coverage

For subagents, mirror sibling subagent artifacts where possible. Consistency
keeps parent orchestration simple, but do not force a bad fit if the domain
needs a different artifact.

## Step 3 — Build extensions and typed tools

Build extension tools only after the objectives and artifact boundary are clear.

Each tool should have:

- a narrow operational purpose
- a Pydantic input model
- a Pydantic output model
- validation strict enough to reject malformed calls early
- concise descriptions that do not carry hidden policy

Tool availability belongs in `.tau/settings.json` or the runtime tools array.
Do not list non-callable tool names in prompt prose, objectives examples, memory
context, or scenario text. Models pattern-match names and can emit calls for
tools that the current turn did not offer.

## Step 4 — Add skills only when useful

Use agent-local skills for reusable process knowledge, domain workflow, or large
instructions that should not live in `.tau/SYSTEM.md`.

Skills are not a substitute for contracts. If behavior affects the agent's
artifact or capability boundary, document it in `OBJECTIVES.md`, enforce it in
models/settings/tools where possible, and add eval coverage.

## Step 5 — Write `.tau/SYSTEM.md`

Write the system prompt late. Keep it brief.

System prompt owns identity, role, voice, and hard boundaries the model must
self-police. It does not own dynamic task data, project paths, tool names,
artifact schemas, field lists, or eval examples.

If the prompt starts accumulating schema details or tool lists, the design has a
plane smear:

- Output shape belongs in Pydantic models, forced return tools, or strict schema.
- Capability belongs in settings/tools.
- Per-call facts belong in the input artifact or user message.
- Reusable process belongs in a skill.

## Step 6 — Enforce artifact contracts

Return to `OBJECTIVES.md` and make the input and output artifacts explicit.
Then enforce them.

For a subagent, the input artifact is the caller's contract with the subagent.
The output artifact is the caller's contract with the result. Both need model
names, fields, validation rules, and failure modes.

Provide a dedicated return-result extension/tool where appropriate. Prefer a
forced return tool or strict response schema over asking the model to produce
well-formed JSON in prose.

Validate the output schema against the real configured provider before live
evals. Provider tolerance differs for nested models, recursive references,
unions, and `oneOf`.

## Step 7 — Add eval infrastructure and scenarios

For core agents, copy the root eval extension from
`/Users/marcusswift/.tau/extensions/evals/` plus required dependencies. For
non-core agents, reuse the core agents or existing harness when that avoids
duplicating infrastructure.

Write scenarios that map directly to the expectations in `OBJECTIVES.md`.

Keep three proof planes distinct:

- Unit / contract tests prove local models, extensions, artifact validation, and
  return plumbing.
- Failure-mode tests prove the harness catches malformed inputs or violating
  outputs.
- Live-LLM evals prove the real model, prompt, tools, and instrumentation work
  together.

Stub tests are necessary, but they never prove model behavior.

## Step 8 — Compile, unit test, then eval

Run gates in order:

1. Compile the agent so it is importable as a Python function object.
2. Run unit tests for extensions, artifact models, and return-tool behavior.
3. Confirm compile and unit tests are green.
4. Run eval scenarios.
5. Repair failures from captured traces.

Failure diagnosis follows the plane:

- Compile failure: packaging, import, scaffold, or settings issue.
- Unit failure: local code, model, or contract issue.
- Failure-mode eval failure: eval/judge is too permissive.
- Live eval failure: model behavior or a smeared control plane.

For live eval repair, use `agent-repair-pattern`: capture evidence, find the
smear, make one change, re-run the same scenario, and keep or revert based on
the trace.

## Anti-pattern recap

- Prompt-first development.
- Hand-building a scaffold instead of initializing pi-py.
- Writing tools before `OBJECTIVES.md` defines the capability boundary.
- Using loose dictionaries instead of Pydantic models for tool/artifact
  contracts.
- Listing non-callable tool names in prose.
- Stuffing artifact schemas into `.tau/SYSTEM.md`.
- Claiming the agent works from compile/unit tests alone.
- Skipping failure-mode evals.
- Running live evals without raw response instrumentation.
