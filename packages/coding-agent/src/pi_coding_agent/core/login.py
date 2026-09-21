"""Shared native provider-login primitives.

Interactive and RPC callers own presentation.  This module owns the provider
OAuth invocation and credential persistence so headless integrations do not
grow a second authentication implementation.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pi_ai.utils.oauth.types import OAuthAuthInfo, OAuthLoginCallbacks, OAuthPrompt


def subscription_provider(provider: str) -> Any:
    if provider == "openai":
        from pi_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        return openai_codex_oauth_provider
    if provider == "anthropic":
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider

        return anthropic_oauth_provider
    if provider == "google":
        from pi_ai.utils.oauth.google_gemini_cli import gemini_cli_oauth_provider

        return gemini_cli_oauth_provider
    raise ValueError(f"{provider} does not support subscription login")


async def subscription_login(
    provider: str,
    session: Any,
    *,
    on_auth: Callable[[OAuthAuthInfo], None],
    on_prompt: Callable[[OAuthPrompt], Awaitable[str]],
    on_progress: Callable[[str], None],
    oauth_provider: Any | None = None,
) -> None:
    """Run Tau's native OAuth provider and persist its result in session auth."""

    selected = oauth_provider or subscription_provider(provider)
    credentials = await selected.login(
        OAuthLoginCallbacks(
            on_auth=on_auth,
            on_prompt=on_prompt,
            on_progress=lambda message: on_progress(str(message)),
        )
    )
    token = {
        "access_token": credentials.access,
        "refresh_token": credentials.refresh,
        "expires_at": credentials.expires / 1000 if credentials.expires else 0,
        "access": credentials.access,
        "refresh": credentials.refresh,
        "expires": credentials.expires,
        "oauth_provider": getattr(selected, "id", provider),
        **dict(credentials.extra),
    }
    auth = getattr(session, "auth_storage", None) or getattr(session, "_auth_storage", None)
    if auth is None or not hasattr(auth, "set_oauth_token"):
        raise RuntimeError("Session auth storage does not support OAuth tokens")
    auth.set_oauth_token(provider, token)
    if provider == "google":
        auth.set_oauth_token("google-gemini-cli", token)


__all__ = ["subscription_login", "subscription_provider"]
