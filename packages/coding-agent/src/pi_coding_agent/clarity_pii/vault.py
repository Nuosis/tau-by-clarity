"""Vault + PII artifact persistence.

The vault is a per-session, bidirectional value↔token store. It is persisted as a
**PII artifact** that is:

  - **Lazy** — written ONLY when the session actually contains PII. A session with
    no PII never creates a file (and an emptied vault deletes its artifact). This
    is what keeps the artifact store from filling with empty no-PII files.
  - **Separate** from the session transcript (its own file under `pii_vault/`),
    but carrying an explicit **reference to the session that produced it**
    (`session_id`) plus `created_at` / `updated_at` / `schema`.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .detect import TOKEN_RE, detect, label_for, make_token

ARTIFACT_SCHEMA = "tau-by-clarity/pii-vault@1"


class Vault:
    """Bidirectional value↔token store with stable tokenization.

    The same raw value always maps to the same token, so re-tokenizing the
    transcript every turn is deterministic and reversible.
    """

    def __init__(self) -> None:
        self._to_token: dict[str, str] = {}
        self._to_value: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, value: str, entity_type: str) -> str:
        existing = self._to_token.get(value)
        if existing is not None:
            return existing
        label = label_for(entity_type)
        self._counters[label] = self._counters.get(label, 0) + 1
        token = make_token(label, self._counters[label])
        self._to_token[value] = token
        self._to_value[token] = value
        return token

    def tokenize(self, text: str) -> str:
        if not text:
            return text
        detected = detect(text)
        for value, etype in sorted(detected, key=lambda p: len(p[0]), reverse=True):
            if value in text:
                text = text.replace(value, self.token_for(value, etype))
        return text

    def detokenize(self, text: str) -> str:
        if not text or "[" + "PII" not in text:
            return text

        def _repl(m: re.Match[str]) -> str:
            token = m.group(0)
            value = self._to_value.get(token)
            if value is not None:
                return value
            if "=" in token:
                value = self._to_value.get(token.replace("=", ":"))
                if value is not None:
                    return value
            return token

        return TOKEN_RE.sub(_repl, text)

    def to_dict(self) -> dict[str, Any]:
        return {"to_value": dict(self._to_value), "counters": dict(self._counters)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vault":
        v = cls()
        v._to_value = dict(data.get("to_value", {}))
        v._to_token = {val: tok for tok, val in v._to_value.items()}
        v._counters = dict(data.get("counters", {}))
        return v

    def __len__(self) -> int:
        return len(self._to_value)

    def mappings(self) -> list[tuple[str, str]]:
        return list(self._to_value.items())


# --------------------------------------------------------------------------- #
# Artifact persistence (lazy, session-referenced)
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_path(vault_dir: str, session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", session_id or "default")
    return os.path.join(vault_dir, f"{safe}.json")


def load_artifact(vault_dir: str, session_id: str) -> Vault:
    """Load the vault for a session, tolerating both the envelope format and the
    legacy flat `{to_value, counters}` shape. Missing file → empty vault."""
    path = artifact_path(vault_dir, session_id)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return Vault()
    if isinstance(data, dict) and isinstance(data.get("vault"), dict):
        return Vault.from_dict(data["vault"])
    if isinstance(data, dict):  # legacy flat artifact
        return Vault.from_dict(data)
    return Vault()


def save_artifact(vault_dir: str, session_id: str, vault: Vault) -> bool:
    """Persist the vault as a session-referenced artifact — LAZILY.

    Returns True if an artifact was written. When the vault is empty, NO artifact
    is created; an existing artifact for this session is removed (covers `/pii
    clear` and any session whose PII was emptied). This is the "only create if the
    session requires it" guarantee.
    """
    path = artifact_path(vault_dir, session_id)

    if len(vault) == 0:
        try:
            os.remove(path)
        except OSError:
            pass
        return False

    created_at = _now_iso()
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and existing.get("created_at"):
            created_at = existing["created_at"]
    except Exception:
        pass

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "session_id": session_id,  # the session that produced this artifact
        "created_at": created_at,
        "updated_at": _now_iso(),
        "vault": vault.to_dict(),
    }
    try:
        os.makedirs(vault_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f)
        return True
    except Exception:
        return False
