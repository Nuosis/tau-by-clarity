# Skills

Agent-authoring skills for building agents on Tau by Clarity. The tree is the
source of truth for the **bundled, first-class** skills that ship with the
harness — see `pi_coding_agent.bundled_skills` for the runtime resolver.

- **[agent-build-pattern](agent-build-pattern/SKILL.md)** — the discipline for
  building a new Tau by Clarity agent or subagent: agent-directory layout,
  `OBJECTIVES.md`, a brief `.tau/SYSTEM.md`, extensions with typed Pydantic tool
  contracts, agent-local skills, input/output artifacts, and the prompt-last
  build sequence (compile → unit tests → live evals). See `references/` for the
  control-plane map and the long-form build-sequence rationale.

## Bundling

`agent-build-pattern` is **baked into the harness**. The wheel
`force-include` ships the tree under `pi_coding_agent/bundled_skills/`, the
`DefaultResourceLoader` picks it up on every launch, and the default system
prompt advertises it as the `Agent build discipline` reference. Disable
per-run with `PI_NO_BUNDLED_SKILLS=1`.

Adding another bundled skill: drop `skills/<name>/SKILL.md` here, add a
`force-include` line in the root `pyproject.toml` mapping
`"skills/<name>" = "pi_coding_agent/bundled_skills/<name>"`, and the loader
will surface it automatically.
