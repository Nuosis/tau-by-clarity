"""Provider profiles for first-party TUI model/auth flows."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pi_ai.types import Model, ModelCost

ProviderStrength = Literal["strong", "standard", "weak"]
AuthMethod = Literal["subscription", "api_key"]


@dataclass(frozen=True)
class ProviderProfile:
    id: str
    label: str
    default_models: dict[ProviderStrength, str]
    auth_methods: tuple[AuthMethod, ...]
    api: str
    base_url: str
    oauth_provider_id: str | None = None


PROVIDER_PROFILES: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        id="openai",
        label="OpenAI",
        default_models={"strong": "gpt-5.5", "standard": "gpt-5.4", "weak": "gpt-5.4-mini"},
        auth_methods=("subscription", "api_key"),
        api="openai-responses",
        base_url="https://api.openai.com/v1",
        oauth_provider_id="openai-codex",
    ),
    ProviderProfile(
        id="anthropic",
        label="Anthropic",
        default_models={"strong": "claude-opus-4-6", "standard": "claude-sonnet-4-5", "weak": "claude-haiku-4-5"},
        auth_methods=("subscription", "api_key"),
        api="anthropic-messages",
        base_url="https://api.anthropic.com",
        oauth_provider_id="anthropic",
    ),
    ProviderProfile(
        id="google",
        label="Gemini",
        default_models={"strong": "gemini-3-pro-preview", "standard": "gemini-2.5-flash", "weak": "gemini-2.5-flash-lite"},
        auth_methods=("subscription", "api_key"),
        api="google-generative-ai",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        oauth_provider_id="google-gemini-cli",
    ),
    ProviderProfile(
        id="openai-compatible",
        label="OpenAI Compatible",
        default_models={"strong": "gpt-5.5", "standard": "gpt-5.4", "weak": "gpt-5.4-mini"},
        auth_methods=("api_key",),
        api="openai-responses",
        base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
    ),
    ProviderProfile(
        id="anthropic-compatible",
        label="Anthropic Compatible",
        default_models={"strong": "claude-opus-4-6", "standard": "claude-sonnet-4-5", "weak": "claude-haiku-4-5"},
        auth_methods=("api_key",),
        api="anthropic-messages",
        base_url=os.environ.get("ANTHROPIC_COMPATIBLE_BASE_URL", "https://api.anthropic.com"),
    ),
)

STRENGTHS: tuple[ProviderStrength, ...] = ("strong", "standard", "weak")

_ALIASES: dict[str, str] = {
    "oa": "openai",
    "openai": "openai",
    "open-ai": "openai",
    "a": "anthropic",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "g": "google",
    "gemini": "google",
    "google": "google",
    "openai-compatible": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "openai-compatible-api": "openai-compatible",
    "openai-compatible-api-key": "openai-compatible",
    "oa-compatible": "openai-compatible",
    "oai-compatible": "openai-compatible",
    "anthropic-compatible": "anthropic-compatible",
    "anthropic_compatible": "anthropic-compatible",
    "a-compatible": "anthropic-compatible",
    "claude-compatible": "anthropic-compatible",
}


def normalize_provider_id(value: str) -> str:
    key = value.strip().lower().replace(" ", "-")
    return _ALIASES.get(key, key)


def get_provider_profile(provider: str) -> ProviderProfile | None:
    normalized = normalize_provider_id(provider)
    for profile in PROVIDER_PROFILES:
        if profile.id == normalized:
            return profile
    return None


def default_model_for(provider: str, strength: str) -> str | None:
    profile = get_provider_profile(provider)
    if profile is None:
        return None
    if strength not in STRENGTHS:
        return None
    return profile.default_models[strength]  # type: ignore[index]


def synthetic_model(provider: str, model_id: str) -> Model | None:
    profile = get_provider_profile(provider)
    if profile is None:
        return None
    return Model(
        id=model_id,
        name=model_id,
        api=profile.api,
        provider=profile.id,
        base_url=profile.base_url,
        reasoning=profile.api in {"openai-responses", "anthropic-messages", "google-generative-ai"},
        input=["text", "image"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def provider_profile_choices() -> list[tuple[str, str]]:
    return [(profile.id, profile.label) for profile in PROVIDER_PROFILES]


__all__ = [
    "AuthMethod",
    "PROVIDER_PROFILES",
    "ProviderProfile",
    "ProviderStrength",
    "STRENGTHS",
    "default_model_for",
    "get_provider_profile",
    "normalize_provider_id",
    "provider_profile_choices",
    "synthetic_model",
]
