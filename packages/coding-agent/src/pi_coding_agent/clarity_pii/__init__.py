"""Clarity PII — first-class, on-by-default PII tokenization for pi-py.

Ships inside pi_coding_agent and is auto-loaded for every agent via
`resource_loader` (kill-switch: env `PI_CLARITY_PII_DISABLED=1`). Real PII never
leaves the machine; the per-session vault is persisted as a lazy,
session-referenced artifact under `pii_vault/`.
"""

from __future__ import annotations

import os

from .detect import detect
from .vault import ARTIFACT_SCHEMA, Vault, load_artifact, save_artifact

DISABLE_ENV = "PI_CLARITY_PII_DISABLED"

__all__ = [
    "detect",
    "Vault",
    "load_artifact",
    "save_artifact",
    "ARTIFACT_SCHEMA",
    "DISABLE_ENV",
    "is_enabled",
    "builtin_extension_path",
]


def is_enabled() -> bool:
    """Always-on unless explicitly disabled via the kill-switch env var."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("1", "true", "yes")


def builtin_extension_path() -> str:
    """Absolute path to the bundled extension module, for the loader to pick up."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "extension.py")
