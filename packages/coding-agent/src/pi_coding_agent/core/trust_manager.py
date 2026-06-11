"""
Project trust store.

Mirrors packages/coding-agent/src/core/trust-manager.ts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

ProjectTrustDecision = bool | None

_CONTEXT_FILE_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")


def _normalize_cwd(cwd: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(cwd)))


def _read_trust_file(path: str) -> dict[str, ProjectTrustDecision]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to read trust store {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid trust store {path}: expected an object")

    data: dict[str, ProjectTrustDecision] = {}
    for key, value in parsed.items():
        if value is not True and value is not False and value is not None:
            raise RuntimeError(
                f"Invalid trust store {path}: value for {key!r} must be true, false, or null"
            )
        data[str(key)] = value
    return data


def _write_trust_file(path: str, data: dict[str, ProjectTrustDecision]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        key: value
        for key, value in sorted(data.items())
        if value is True or value is False or value is None
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
        f.write("\n")


def has_project_config_dir(cwd: str) -> bool:
    return os.path.isdir(os.path.join(_normalize_cwd(cwd), CONFIG_DIR_NAME))


def has_project_trust_inputs(cwd: str) -> bool:
    current = Path(_normalize_cwd(cwd))
    if has_project_config_dir(str(current)):
        return True

    while True:
        for filename in _CONTEXT_FILE_NAMES:
            if (current / filename).exists():
                return True
        if (current / ".agents" / "skills").exists():
            return True

        parent = current.parent
        if parent == current:
            return False
        current = parent


class ProjectTrustStore:
    """Persist project trust decisions in <agent_dir>/trust.json."""

    def __init__(self, agent_dir: str | None = None) -> None:
        self.trust_path = os.path.join(agent_dir or get_agent_dir(), "trust.json")

    def get(self, cwd: str) -> ProjectTrustDecision:
        data = _read_trust_file(self.trust_path)
        value = data.get(_normalize_cwd(cwd))
        return value if value is True or value is False else None

    def set(self, cwd: str, decision: ProjectTrustDecision) -> None:
        data = _read_trust_file(self.trust_path)
        key = _normalize_cwd(cwd)
        if decision is None:
            data.pop(key, None)
        else:
            data[key] = decision
        _write_trust_file(self.trust_path, data)


__all__ = [
    "ProjectTrustDecision",
    "ProjectTrustStore",
    "has_project_config_dir",
    "has_project_trust_inputs",
]
