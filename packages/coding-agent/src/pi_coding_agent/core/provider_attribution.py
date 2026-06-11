"""Provider attribution headers."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .telemetry import is_install_telemetry_enabled

OPENROUTER_HOST = "openrouter.ai"
NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
CLOUDFLARE_API_HOST = "api.cloudflare.com"
CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
OPENCODE_HOST = "opencode.ai"


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def matches_host(base_url: str, expected_host: str) -> bool:
    try:
        return urlparse(base_url).hostname == expected_host
    except Exception:
        return False


def _provider(model: Any) -> str:
    return str(_get_attr_or_key(model, "provider", ""))


def _base_url(model: Any) -> str:
    return str(_get_attr_or_key(model, "base_url", _get_attr_or_key(model, "baseUrl", "")) or "")


def is_openrouter_model(model: Any) -> bool:
    return _provider(model) == "openrouter" or OPENROUTER_HOST in _base_url(model)


def is_nvidia_nim_model(model: Any) -> bool:
    return _provider(model) == "nvidia" or matches_host(_base_url(model), NVIDIA_NIM_HOST)


def is_cloudflare_model(model: Any) -> bool:
    return (
        _provider(model) in {"cloudflare-workers-ai", "cloudflare-ai-gateway"}
        or matches_host(_base_url(model), CLOUDFLARE_API_HOST)
        or matches_host(_base_url(model), CLOUDFLARE_AI_GATEWAY_HOST)
    )


def get_default_attribution_headers(model: Any, settings_manager: Any) -> dict[str, str] | None:
    if not is_install_telemetry_enabled(settings_manager):
        return None
    if is_openrouter_model(model):
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }
    if is_nvidia_nim_model(model):
        return {"X-BILLING-INVOKE-ORIGIN": "Pi"}
    if is_cloudflare_model(model):
        return {"User-Agent": "pi-coding-agent"}
    return None


def get_session_headers(model: Any, session_id: str | None) -> dict[str, str] | None:
    if not session_id:
        return None
    if _provider(model) not in {"opencode", "opencode-go"} and not matches_host(_base_url(model), OPENCODE_HOST):
        return None
    return {"x-opencode-session": session_id, "x-opencode-client": "pi"}


def merge_provider_attribution_headers(
    model: Any,
    settings_manager: Any,
    session_id: str | None,
    *header_sources: dict[str, str] | None,
) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    for headers in (
        get_session_headers(model, session_id),
        get_default_attribution_headers(model, settings_manager),
        *header_sources,
    ):
        if headers:
            merged.update(headers)
    return merged or None


__all__ = [
    "get_default_attribution_headers",
    "get_session_headers",
    "is_cloudflare_model",
    "is_nvidia_nim_model",
    "is_openrouter_model",
    "matches_host",
    "merge_provider_attribution_headers",
]
