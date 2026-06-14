"""Tau by Clarity — bidirectional, reversible PII tokenizer for tau.

A first-class, on-by-default feature: real PII values are tokenized
(`[PII:EMAIL:1]`) before any message reaches the model provider, and restored
from the per-session vault on the way back. The vault is persisted as a lazy,
session-referenced artifact (see `vault.save_artifact`).

This module is loaded automatically for every agent (see
`pi_coding_agent.clarity_pii.builtin_extension_path` and its use in
`resource_loader`). Disable with the env var `PI_CLARITY_PII_DISABLED=1` or the
runtime flag.
"""

from __future__ import annotations

import copy
from typing import Any

# Absolute imports: the extension loader execs this file standalone (module name
# `pi_ext_extension`), so relative imports would have no parent package. The
# clarity_pii package is installed/importable, so absolute imports resolve.
from pi_coding_agent.clarity_pii.detect import get_presidio
from pi_coding_agent.clarity_pii.vault import Vault, load_artifact, save_artifact
from pi_coding_agent.clarity_pii.walk import (
    apply_to_message,
    deep_copy_message,
    dict_string_slots,
)

BRAND = "Tau by Clarity"
FLAG_NAME = "pii-filter"


def _vault_dir() -> str:
    import os

    base = os.environ.get("PI_AGENT_DIR") or os.path.join(os.path.expanduser("~"), ".pi-py", "agent")
    return os.path.join(base, "pii_vault")


def _set_flag(pi: Any, name: str, value: bool) -> None:
    runtime = getattr(pi, "_runtime", None)
    if isinstance(runtime, dict):
        runtime.setdefault("flagValues", {})[name] = value


def extension_factory(pi: Any) -> None:
    state: dict[str, Any] = {"vault": Vault(), "session_id": "", "loaded": False}

    def _load(session_id: str) -> None:
        state["session_id"] = session_id
        state["loaded"] = True
        state["vault"] = load_artifact(_vault_dir(), session_id)

    def _save() -> None:
        # Lazy + session-referenced: writes only when the vault has PII.
        save_artifact(_vault_dir(), state["session_id"], state["vault"])

    def _enabled() -> bool:
        val = pi.get_flag(FLAG_NAME)
        return True if val is None else bool(val)

    pi.register_flag(
        FLAG_NAME,
        {
            "description": "Tokenize PII before it reaches the model provider (reversible).",
            "type": "boolean",
            "default": True,
        },
    )

    def on_session_start(event: Any, ctx: Any) -> None:
        _load(getattr(ctx, "session_id", "") or "")

    pi.on("session_start", on_session_start)

    async def on_context(event: dict[str, Any], ctx: Any) -> dict[str, Any] | None:
        if not _enabled():
            return None
        if not state["loaded"]:
            _load(getattr(ctx, "session_id", "") or "")
        vault: Vault = state["vault"]
        messages = event.get("messages") or []
        for msg in messages:
            apply_to_message(msg, vault.detokenize)
        tokenized = [apply_to_message(deep_copy_message(m), vault.tokenize) for m in messages]
        _save()
        return {"messages": tokenized}

    pi.on("context", on_context)

    async def on_before_provider_request(event: dict[str, Any], ctx: Any) -> Any:
        if not _enabled():
            return None
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return None
        vault: Vault = state["vault"]
        for get, set_ in dict_string_slots(payload):
            try:
                set_(vault.tokenize(get()))
            except Exception:
                continue
        _save()
        return payload

    pi.on("before_provider_request", on_before_provider_request)

    async def on_tool_call(event: dict[str, Any], ctx: Any) -> dict[str, Any] | None:
        if not _enabled():
            return None
        args = event.get("input")
        if not isinstance(args, dict) or not args:
            return None
        vault: Vault = state["vault"]
        new_args = copy.deepcopy(args)
        changed = False
        for get, set_ in dict_string_slots(new_args):
            try:
                orig = get()
                restored = vault.detokenize(orig)
                if restored != orig:
                    set_(restored)
                    changed = True
            except Exception:
                continue
        if not changed:
            return None
        return {"arguments": new_args}

    pi.on("tool_call", on_tool_call)

    async def pii_command(args: str, ctx: Any = None) -> str:
        vault: Vault = state["vault"]
        parts = (args or "").strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""

        if sub in ("on", "enable"):
            _set_flag(pi, FLAG_NAME, True)
            return f"{BRAND} enabled."
        if sub in ("off", "disable"):
            _set_flag(pi, FLAG_NAME, False)
            return f"{BRAND} disabled — PII will be sent to the provider in cleartext."
        if sub == "clear":
            state["vault"] = Vault()
            _save()  # empty vault → artifact removed
            return "PII vault cleared."
        if sub == "vault":
            if len(vault) == 0:
                return "PII vault is empty."
            lines = [f"  {tok} → {val}" for tok, val in vault.mappings()]
            return f"PII vault ({len(vault)} entries):\n" + "\n".join(lines)
        if sub == "reveal":
            if not rest:
                return "Usage: /pii reveal <text-with-tokens>"
            return vault.detokenize(rest)
        return (
            f"{BRAND}: {'on' if _enabled() else 'off'} | "
            f"detector: {'presidio+regex' if get_presidio() else 'regex'} | "
            f"vault: {len(vault)} entries | session: {state['session_id'] or '-'}\n"
            "Subcommands: status | on | off | vault | reveal <text> | clear"
        )

    pi.register_command(
        "pii",
        {
            "description": f"Inspect/control {BRAND} (status | on | off | vault | reveal | clear)",
            "handler": pii_command,
        },
    )


# Aliases recognized by the extension loader.
activate = extension_factory
default = extension_factory
