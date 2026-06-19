"""Legacy local active compression for Tau.

This package is Tau's hand-rolled CCR/compression implementation. It is not the
Headroom SDK path. Keep changes here isolated while real Headroom integration is
designed and tested.

It currently provides content-aware, reversible compression of tool-output
payloads and installs itself as pi_ai's universal outbound compressor.

Flag: `active_compression` in settings.json — **default ON if absent**. Env
kill-switch `PI_ACTIVE_COMPRESSION_DISABLED=1`. Registered on import (the per-call
compressor honors the flag at call time).

Slice-1 (§12): the compress + CCR-cache + retrieve PATH. The retrieve TRIGGER
(model `ccr_retrieve` tool vs harness-driven rehydration) is the de-risk follow-on
flagged in §12 — `retrieve()` here is the path it will hang off.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .ccr import CCRStore
from .compressor import CompressionConfig
from .compressor import compress as _compress_text

SETTING = "active_compression"
DISABLE_ENV = "PI_ACTIVE_COMPRESSION_DISABLED"

__all__ = [
    "is_enabled",
    "compress",
    "retrieve",
    "mark_expanded",
    "register_with_pi_ai",
    "builtin_extension_path",
    "SETTING",
    "DISABLE_ENV",
    "CompressionConfig",
]


def builtin_extension_path() -> str:
    """Absolute path to the bundled ccr_retrieve extension, for the loader."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "extension.py")


def _settings_paths() -> list[str]:
    paths: list[str] = []
    from ..config import agent_dir_env
    acd = agent_dir_env()
    if acd:
        paths += [os.path.join(acd, "settings.json"), os.path.join(acd, ".tau", "settings.json")]
    paths.append(os.path.join(os.getcwd(), ".tau", "settings.json"))
    return paths


def is_enabled() -> bool:
    """On unless the env kill-switch is set or settings.json sets it false.
    Absent setting → ON (default-on-if-absent)."""
    if os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes"):
        return False
    for p in _settings_paths():
        try:
            data = json.loads(Path(p).read_text())
        except Exception:
            continue
        if isinstance(data, dict) and SETTING in data:
            return bool(data[SETTING])
    return True


_store: CCRStore | None = None


def _ccr() -> CCRStore:
    global _store
    if _store is None:
        base = os.environ.get("PI_AGENT_DIR") or os.path.join(os.path.expanduser("~"), ".tau", "agent")
        _store = CCRStore(os.path.join(base, "ccr.db"))
    return _store


def compress(text: str) -> str:
    if not is_enabled():
        return text
    force = False
    target_ratio = None
    config = CompressionConfig()
    try:
        from pi_ai import (
            get_current_compression_enable_ccr_marker,
            get_current_compression_force_compression,
            get_current_compression_lossless_min_savings_ratio,
            get_current_compression_max_items_after_crush,
            get_current_compression_min_tokens,
            get_current_compression_target_ratio,
        )

        force = get_current_compression_force_compression()
        target_ratio = get_current_compression_target_ratio()
        min_tokens = get_current_compression_min_tokens()
        max_items_after_crush = get_current_compression_max_items_after_crush()
        lossless_min_savings_ratio = get_current_compression_lossless_min_savings_ratio()
        enable_ccr_marker = get_current_compression_enable_ccr_marker()
        config = CompressionConfig(
            min_tokens=config.min_tokens if min_tokens is None else min_tokens,
            max_items_after_crush=(
                config.max_items_after_crush if max_items_after_crush is None else max_items_after_crush
            ),
            lossless_min_savings_ratio=(
                config.lossless_min_savings_ratio
                if lossless_min_savings_ratio is None
                else lossless_min_savings_ratio
            ),
            enable_ccr_marker=config.enable_ccr_marker if enable_ccr_marker is None else enable_ccr_marker,
        )
    except Exception:
        force = False
        target_ratio = None
        config = CompressionConfig()
    return _compress_text(text, _ccr(), target_ratio=target_ratio, force=force, config=config)


def retrieve(handle: str) -> str | None:
    """Fetch a cached original by its CCR handle (the reversibility path)."""
    return _ccr().get(handle)


def mark_expanded(handle: str) -> None:
    """Phase-4 (Context Tracker): mark a handle's original as expanded for the model,
    so the universal compressor stops re-eliding it on subsequent turns."""
    _ccr().mark_expanded(handle)


def register_with_pi_ai() -> None:
    """Install as pi_ai's universal outbound compressor (no-op if pi_ai absent)."""
    try:
        from pi_ai import register_compressor
    except Exception:
        return
    register_compressor(compress)


# Self-register on import.
register_with_pi_ai()
