"""Loop profile — the agent-specific seam.

The harness is agent-agnostic: it never names a particular agent or domain. Each
agent supplies its own profile at `<agent_dir>/.tau/loop_profile.json`, which
injects:

  - `done_rubric`: domain hints for goal_accomplished ("what 'done' looks like
    here", e.g. "DevFlow queues drained, runs terminal/healthy"). The judge is
    still conservative and evidence-grounded; the rubric only tells it which
    observed signals mean completion in this agent's world.
  - `voice`: the persona for user_feedback and the terminal report.
  - `agent_label`: how the agent refers to itself ("Devin", or a default).

All fields are optional; sensible generic defaults apply when the file (or any
field) is absent, so an agent with no profile still loops correctly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_DONE_RUBRIC = (
    "The goal is done only when observed evidence shows the requested work is "
    "genuinely complete and verified — not merely asserted. If the evidence is "
    "silent or partial, it is not done."
)
_DEFAULT_VOICE = "a friendly, plainspoken engineer"
_DEFAULT_AGENT_LABEL = "the engineer"


class LoopProfile(BaseModel):
    done_rubric: str = _DEFAULT_DONE_RUBRIC
    voice: str = _DEFAULT_VOICE
    agent_label: str = _DEFAULT_AGENT_LABEL


def load_profile(agent_dir: str) -> LoopProfile:
    """Load the agent's loop profile, or generic defaults if none is present."""
    path = os.path.join(agent_dir, ".tau", "loop_profile.json")
    try:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            return LoopProfile.model_validate(data)
    except Exception:
        pass
    return LoopProfile()
