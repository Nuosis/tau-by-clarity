"""Router tier assignments, resolved at configuration time rather than per call."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pi_agent import ModelInvocationSelection
from pydantic import BaseModel

RouterLevel = Literal["ultra-light", "light", "default", "max"]
ROUTER_LEVELS = ("ultra-light", "light", "default", "max")


class RouterAssignment(BaseModel):
    provider: str
    model: str
    reasoning: str | None

    model_config = {"extra": "forbid", "frozen": True}


# Proposed assignments from the routing sketch; configuring a tier does not enable routing.
ROUTER_DEFAULTS = {
    "ultra-light": RouterAssignment(provider="openai", model="gpt-5.6-luna", reasoning="low"),
    "light": RouterAssignment(provider="openai", model="gpt-5.6-luna", reasoning="high"),
    "default": RouterAssignment(provider="openai", model="gpt-5.6-sol", reasoning="high"),
    "max": RouterAssignment(provider="openai", model="gpt-6-astra", reasoning="medium"),
}


def read_router_assignments(path: str) -> dict[RouterLevel, RouterAssignment]:
    config = _read_config(path)
    router = config.get("router", {})
    if not isinstance(router, dict) or set(router) - {"levels"}:
        raise ValueError("router must contain only a levels mapping")
    levels = router.get("levels", {})
    if not isinstance(levels, dict) or set(levels) - set(ROUTER_LEVELS):
        raise ValueError("Router levels must be ultra-light, light, default, or max")
    return {level: RouterAssignment.model_validate(value) for level, value in levels.items()}


def resolve_router_assignment(assignment: RouterAssignment, registry) -> ModelInvocationSelection:
    """Resolve a configured model and basic reasoning capability once.

    Provider-specific custom effort names remain supported, as in /thinking.
    Authentication is resolved by the invocation loop, not stored here.
    """
    if not assignment.provider.strip() or not assignment.model.strip():
        raise ValueError("Router assignment requires a provider and model")
    model = registry.find(assignment.provider, assignment.model)
    if model is None:
        raise ValueError(f"Unknown router model: {assignment.provider}/{assignment.model}")
    if assignment.reasoning is not None:
        if not assignment.reasoning.strip() or assignment.reasoning == "off":
            raise ValueError("Use null for reasoning off")
        if not model.reasoning:
            raise ValueError(f"Model {assignment.model} does not support reasoning")
    return ModelInvocationSelection(model=model, reasoning=assignment.reasoning)


def load_router_selections(path: str, registry) -> dict[RouterLevel, ModelInvocationSelection]:
    """Load the configured tiers for a router owner to retain and use in its hook."""
    return {level: resolve_router_assignment(value, registry)
            for level, value in read_router_assignments(path).items()}


def store_router_assignment(path: str, level: str, assignment: RouterAssignment, registry) -> None:
    if level not in ROUTER_LEVELS:
        raise ValueError("Router level must be ultra-light, light, default, or max")
    resolve_router_assignment(assignment, registry)
    # Read strictly: malformed existing config must never be replaced silently.
    levels = read_router_assignments(path)
    levels[level] = assignment
    config = _read_config(path)
    config["router"] = {"levels": {key: value.model_dump() for key, value in levels.items()}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_config(path: str) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    config = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be an object")
    return config
