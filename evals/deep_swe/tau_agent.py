"""
Pier custom-agent adapter for tau (P6) — SCAFFOLD.

Lets deep-swe drive the tau coding agent so we can compare baseline (memory off) vs
memory-augmented (PI_MEMORY_ENABLED=1) on the same tasks/model. Plug in via:

    pier run -p deep-swe/tasks --n-tasks 10 --sample-seed 0 \
        --agent-import-path evals.deep_swe.tau_agent:TauAgent \
        --model MiniMax-M3 --agent-env PI_MEMORY_ENABLED=0    # baseline
    # ...and again with --agent-env PI_MEMORY_ENABLED=1       # memory-augmented

STATUS: scaffold. Mirrors pier.agents.installed.mini_swe_agent (setup installs the
agent into the sandbox; run shells it headless against the instruction via
environment.exec). Three OPEN ISSUES must be resolved before a real pass@1 run — see
evals/deep_swe/README.md:
  1. headless tau invocation (exact print-mode flags + diff/output contract);
  2. M3 model wiring inside tau (provider/base-url/key) + sandbox network allowlist
     so the agent can reach api.minimax.io;
  3. EMBEDDINGS IN SANDBOX — memory recall uses local Ollama, absent in the isolated
     env. Options: run a tiny embed model in-container, allowlist a remote embedder, or
     fall back to deterministic embeddings (degrades semantic recall — would understate
     the memory arm). This must be decided or the memory arm isn't a fair test.
"""
from __future__ import annotations

import os
import shlex

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext


class TauAgent(BaseAgent):
    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "tau"

    def version(self) -> str:
        return "0.0.1-scaffold"

    async def setup(self, environment: BaseEnvironment) -> None:
        # TODO(P6-1): install tau into the sandbox. Either `uv pip install` the built
        # wheel or upload the source tree (environment.upload_file) and `pip install -e`.
        # TODO(P6-3): provision embeddings (Ollama tiny model or deterministic fallback).
        # Memory store is project-local (cwd/.tau/memory) — committed in the task repo,
        # so PI_MEMORY_ENABLED=1 starts empty and the curator populates it during the run.
        await environment.exec("python -V")  # placeholder smoke

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        memory_on = context_env(context).get("PI_MEMORY_ENABLED", "0") == "1"
        model = self.model_name or "MiniMax-M3"
        task = shlex.quote(instruction)
        # TODO(P6-1/2): replace with the real headless tau invocation + model wiring.
        # Shape (mirrors mini-swe-agent): run tau print-mode against the repo with the
        # task as the prompt; PI_MEMORY_ENABLED gates the memory subsystem (P5).
        env_prefix = f"PI_MEMORY_ENABLED={'1' if memory_on else '0'}"
        cmd = f"{env_prefix} tau --print --model {shlex.quote(model)} --task {task}"
        await environment.exec(cmd)  # TODO: confirm tau CLI flags / capture diff


def context_env(context: AgentContext) -> dict:
    """Best-effort read of agent-env vars Pier passes via --agent-env (shape TBD)."""
    for attr in ("env", "agent_env", "environment"):
        value = getattr(context, attr, None)
        if isinstance(value, dict):
            return value
    return dict(os.environ)
