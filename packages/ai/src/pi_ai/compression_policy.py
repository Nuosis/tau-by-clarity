"""Headroom-compatible active-compression policy.

The policy maps request auth modes to compression stability defaults. It mirrors
Headroom's Python/Rust field map closely enough that Tau can expose the same
compression-only control surface without importing Headroom's proxy code.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AuthMode(str, Enum):
    PAYG = "payg"
    OAUTH = "oauth"
    SUBSCRIPTION = "subscription"


VOLATILE_TOKEN_THRESHOLD_PAYG = 128
VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION = 32
MAX_LOSSY_RATIO_PAYG = 0.45
MAX_LOSSY_RATIO_SUBSCRIPTION = 0.25
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1
ENFORCEMENT_ENV = "HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    live_zone_only: bool
    cache_aligner_enabled: bool
    volatile_token_threshold: int
    max_lossy_ratio: float
    toin_read_only: bool

    def net_mutation_gain(
        self,
        delta_t: int,
        suffix_tokens: int,
        expected_reads: float,
        p_alive: float,
    ) -> float:
        dt = max(0, delta_t)
        suffix = max(0, suffix_tokens)
        reads = 0.0 if math.isnan(expected_reads) else max(expected_reads, 0.0)
        alive = 1.0 if math.isnan(p_alive) else min(max(p_alive, 0.0), 1.0)
        return float(dt) * (CACHE_WRITE_MULTIPLIER + CACHE_READ_MULTIPLIER * (reads - 1.0)) - alive * (
            CACHE_WRITE_MULTIPLIER - CACHE_READ_MULTIPLIER
        ) * float(suffix + dt)

    def should_mutate_deep(
        self,
        delta_t: int,
        suffix_tokens: int,
        expected_reads: float,
        p_alive: float,
    ) -> bool:
        return self.net_mutation_gain(delta_t, suffix_tokens, expected_reads, p_alive) > 0.0

    def break_even_reads(self, delta_t: int, suffix_tokens: int) -> float:
        if delta_t <= 0:
            return 0.0
        return ((CACHE_WRITE_MULTIPLIER - CACHE_READ_MULTIPLIER) / CACHE_READ_MULTIPLIER) * (
            float(max(0, suffix_tokens)) / float(delta_t)
        )


def _coerce_auth_mode(mode: Any) -> AuthMode:
    if isinstance(mode, AuthMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower().replace("-", "_")
        if normalized in {"payg", "api_key", "api-key"}:
            return AuthMode.PAYG
        if normalized in {"oauth", "o_auth"}:
            return AuthMode.OAUTH
        if normalized in {"subscription", "sub"}:
            return AuthMode.SUBSCRIPTION
    raise ValueError(f"Unhandled AuthMode variant: {mode!r}")


def policy_for_mode(mode: AuthMode | str) -> CompressionPolicy:
    auth_mode = _coerce_auth_mode(mode)
    if auth_mode in {AuthMode.PAYG, AuthMode.OAUTH}:
        return CompressionPolicy(
            live_zone_only=False,
            cache_aligner_enabled=True,
            volatile_token_threshold=VOLATILE_TOKEN_THRESHOLD_PAYG,
            max_lossy_ratio=MAX_LOSSY_RATIO_PAYG,
            toin_read_only=False,
        )
    if auth_mode is AuthMode.SUBSCRIPTION:
        return CompressionPolicy(
            live_zone_only=True,
            cache_aligner_enabled=False,
            volatile_token_threshold=VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION,
            max_lossy_ratio=MAX_LOSSY_RATIO_SUBSCRIPTION,
            toin_read_only=True,
        )
    raise ValueError(f"Unhandled AuthMode variant: {mode!r}")


def policy_default_payg() -> CompressionPolicy:
    return policy_for_mode(AuthMode.PAYG)


def is_enforcement_enabled() -> bool:
    value = os.environ.get(ENFORCEMENT_ENV, "enabled").strip().lower()
    return value not in {"disabled", "off", "false", "0", "no"}


def resolve_policy(auth_mode: AuthMode | str | None) -> CompressionPolicy:
    if auth_mode is None or not is_enforcement_enabled():
        return policy_default_payg()
    return policy_for_mode(auth_mode)
