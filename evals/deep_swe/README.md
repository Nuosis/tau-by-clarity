# P6 — deep-swe acceptance (baseline vs memory-augmented tau)

Status: **infra validated + adapter scaffolded; the pass@1 micro run is the remaining
step.** Not a fabricated result — the run itself is sustained and has open issues below.

## Validated in this environment
- `docker`: up. `datacurve-pier` installed (`uv tool install datacurve-pier`).
- deep-swe clones cleanly (113 tasks, Harbor format: `instruction.md`, `task.toml`,
  `tests/`, `environment/Dockerfile`, held-out `solution/`).
- Pier custom-agent hook confirmed: `--agent-import-path module:Class`, `--agent-env
  KEY=VAL`, `--model`, `--n-tasks N --sample-seed 0` (deterministic micro subset),
  `pier run -p deep-swe/tasks/<task-id>` (single task).
- Agent interface (`pier.agents.base.BaseAgent`): `name()`, `version()`,
  `async setup(env)`, `async run(instruction, env, context)`; env exposes
  `exec`, `upload_file`, `download_file`. Adapter mirrors
  `pier.agents.installed.mini_swe_agent`.

## Micro run (estimate first; ≈10 tasks, fixed seed)
```
cd /tmp/deep-swe   # git clone https://github.com/datacurve-ai/deep-swe
# baseline (memory off)
pier run -p tasks --n-tasks 10 --sample-seed 0 \
  --agent-import-path evals.deep_swe.pi_py_agent:PiPyAgent \
  --model MiniMax-M3 --agent-env PI_MEMORY_ENABLED=0 -o jobs/baseline
# memory-augmented
pier run -p tasks --n-tasks 10 --sample-seed 0 \
  --agent-import-path evals.deep_swe.pi_py_agent:PiPyAgent \
  --model MiniMax-M3 --agent-env PI_MEMORY_ENABLED=1 -o jobs/memory
# compare pass@1 across the two job dirs; full 113 only if the micro delta looks right.
```

## OPEN ISSUES (must resolve before the run is real/fair)
1. **Headless tau invocation** — confirm the print/non-interactive CLI flags and the
   diff/output contract Pier's verifier expects.
2. **M3 model wiring + network** — wire MiniMax-M3 (provider/base-url/key) inside tau,
   and add `api.minimax.io` to the task's per-agent network allowlist (Pier isolates
   the sandbox).
3. **Embeddings in the sandbox** — memory recall uses local Ollama, absent in the
   isolated env. Decide: tiny in-container embed model, allowlist a remote embedder, or
   deterministic fallback (degrades semantic recall and would *understate* the memory
   arm). Without this the memory arm is not a fair test.

Until these are closed, P6 cannot produce an honest pass@1 delta.
