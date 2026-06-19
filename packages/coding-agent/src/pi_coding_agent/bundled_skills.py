"""Bundled, first-class skills that load with the harness.

The harness ships a small set of skills that are first-class and load
unconditionally for every agent (unless explicitly disabled). They are baked in
— not user/opt-in — so the agent build discipline is always available to the
agent when it authors or modifies a subagent.

The bundled directory layout matches the user/project layout: one subdirectory
per skill, with a `SKILL.md` inside. The first entry is the `agent-build-pattern`
skill: the prompt-last, control-plane discipline the harness itself uses to
build new agents and subagents.

Disable per-run with `PI_NO_BUNDLED_SKILLS=1`.

Path resolution order
---------------------
1. `<package>/bundled_skills/` — wheel-installed location. The wheel's
   `force-include` ships the source-of-truth `skills/` tree from the repo root
   to this path inside `pi_coding_agent/`.
2. `<repo>/skills/` — dev/source layout. Walk up from the package directory to
   the repo root and use the `skills/` tree there. Lets contributors run
   `uv run tau` against a checkout without rebuilding the wheel.
3. `None` — neither location has skills. The harness loads the user/project
   skills only, mirroring the previous behavior.
"""

from __future__ import annotations

import os

DISABLE_ENV = "PI_NO_BUNDLED_SKILLS"

__all__ = [
    "DISABLE_ENV",
    "get_bundled_skills_dir",
    "is_enabled",
]


def is_enabled() -> bool:
    """Bundled skills load by default; off only when the env kill-switch is set."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("1", "true", "yes")


def get_bundled_skills_dir() -> str | None:
    """Absolute path to the bundled skills directory, or None if not found.

    Search order:
    1. `<pi_coding_agent>/bundled_skills/` (wheel-installed).
    2. `<repo>/skills/` (dev layout — `packages/coding-agent/src/pi_coding_agent`
       is four levels below the repo root, so walk up to the root and look for
       `skills/`).
    """
    # 1) Wheel-installed: the file lives next to this module as `bundled_skills/`.
    here = os.path.dirname(os.path.abspath(__file__))
    wheel_dir = os.path.join(here, "bundled_skills")
    if _looks_like_skills_dir(wheel_dir):
        return wheel_dir

    # 2) Dev layout: walk up to the repo root and look for `skills/`.
    #    packages/coding-agent/src/pi_coding_agent/bundled_skills.py
    #    -> packages/coding-agent/src/pi_coding_agent/
    #    -> packages/coding-agent/src/
    #    -> packages/coding-agent/
    #    -> packages/
    #    -> <repo>/
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    repo_skills = os.path.join(repo_root, "skills")
    if _looks_like_skills_dir(repo_skills):
        return repo_skills

    return None


def _looks_like_skills_dir(path: str) -> bool:
    """A skills dir is a directory containing at least one `*/SKILL.md`."""
    if not os.path.isdir(path):
        return False
    try:
        for entry in os.listdir(path):
            if entry.startswith("."):
                continue
            skill_md = os.path.join(path, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                return True
    except OSError:
        return False
    return False
