"""
OpenAI Codex OAuth provider (ChatGPT Plus/Pro).

Implements PKCE + local callback server for ChatGPT OAuth.

Mirrors utils/oauth/openai-codex.ts
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from urllib.parse import urlencode

import httpx

from pi_ai.utils.oauth.pkce import generate_pkce
from pi_ai.utils.oauth.types import OAuthAuthInfo, OAuthCredentials, OAuthLoginCallbacks

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTH_BASE_URL = "https://auth.openai.com"
_AUTHORIZE_URL = f"{_AUTH_BASE_URL}/oauth/authorize"
_TOKEN_URL = f"{_AUTH_BASE_URL}/oauth/token"
_REDIRECT_PORT = 1455
_REDIRECT_PATH = "/auth/callback"
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}{_REDIRECT_PATH}"
_DEVICE_USER_CODE_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
_DEVICE_TOKEN_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/token"
_DEVICE_VERIFICATION_URI = f"{_AUTH_BASE_URL}/codex/device"
_DEVICE_REDIRECT_URI = f"{_AUTH_BASE_URL}/deviceauth/callback"
_DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
_SCOPES = "openid profile email offline_access"


def _get_account_id_from_jwt(access_token: str) -> str | None:
    """Extract account ID from a JWT token's claims."""
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        auth_info = payload.get("https://api.openai.com/auth", {})
        return auth_info.get("chatgpt_account_id") or auth_info.get("user_id") or auth_info.get("sub")
    except Exception:
        return None


def _credentials_from_token_data(data: dict[str, object]) -> OAuthCredentials:
    access = data.get("access_token", "")
    refresh = data.get("refresh_token", "")
    expires_in = data.get("expires_in", 3600)
    expires_at = int(time.time() * 1000) + int(expires_in) * 1000 - 5 * 60 * 1000
    creds = OAuthCredentials(
        refresh=str(refresh),
        access=str(access),
        expires=expires_at,
    )
    account_id = _get_account_id_from_jwt(creds.access)
    if account_id:
        creds.extra["account_id"] = account_id
    return creds


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str,
) -> OAuthCredentials:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
    return _credentials_from_token_data(data)


async def login_openai_codex_device_code(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    """Login with OpenAI Codex OAuth using device-code auth."""
    async with httpx.AsyncClient() as client:
        device_resp = await client.post(
            _DEVICE_USER_CODE_URL,
            json={"client_id": _CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
        if device_resp.status_code == 404:
            raise RuntimeError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use API-key login or verify the server URL."
            )
        device_resp.raise_for_status()
        device_data = device_resp.json()

        device_auth_id = device_data.get("device_auth_id", "")
        user_code = device_data.get("user_code", "")
        interval = device_data.get("interval", 5)
        if isinstance(interval, str):
            interval = float(interval.strip())
        if not device_auth_id or not user_code or not isinstance(interval, int | float) or interval < 0:
            raise RuntimeError(f"Invalid OpenAI Codex device code response: {device_data}")

        callbacks.on_auth(
            OAuthAuthInfo(
                url=_DEVICE_VERIFICATION_URI,
                instructions=f"Enter code: {user_code}",
            )
        )
        if callbacks.on_progress:
            callbacks.on_progress(f"Visit {_DEVICE_VERIFICATION_URI} and enter code: {user_code}")

        deadline = time.monotonic() + _DEVICE_CODE_TIMEOUT_SECONDS
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError("OpenAI Codex device code login timed out")
            await asyncio.sleep(float(interval))
            token_resp = await client.post(
                _DEVICE_TOKEN_URL,
                json={
                    "device_auth_id": device_auth_id,
                    "user_code": user_code,
                },
                headers={"Content-Type": "application/json"},
            )

            if token_resp.is_success:
                token_data = token_resp.json()
                authorization_code = token_data.get("authorization_code")
                code_verifier = token_data.get("code_verifier")
                if not authorization_code or not code_verifier:
                    raise RuntimeError(f"Invalid OpenAI Codex device auth token response: {token_data}")
                return await _exchange_authorization_code(
                    str(authorization_code),
                    str(code_verifier),
                    _DEVICE_REDIRECT_URI,
                )

            if token_resp.status_code in {403, 404}:
                continue

            response_body = token_resp.text
            error_code = ""
            try:
                error = token_resp.json().get("error", "")
                error_code = error.get("code", "") if isinstance(error, dict) else str(error)
            except Exception:
                pass
            if error_code == "deviceauth_authorization_pending":
                continue
            if error_code == "slow_down":
                interval = float(interval) + 5
                continue
            raise RuntimeError(
                f"OpenAI Codex device auth failed with status {token_resp.status_code}"
                f"{': ' + response_body if response_body else ''}"
            )


async def login_openai_codex(callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
    """Login with OpenAI Codex OAuth (PKCE + local callback server)."""
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    auth_params = urlencode({
        "client_id": _CLIENT_ID,
        "redirect_uri": _REDIRECT_URI,
        "response_type": "code",
        "scope": _SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli",
        "state": state,
    })
    auth_url = f"{_AUTHORIZE_URL}?{auth_params}"
    ready: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    callback_task = asyncio.create_task(_wait_for_callback_code(ready))
    await ready
    callbacks.on_auth(OAuthAuthInfo(url=auth_url, instructions="Visit the URL to authorize ChatGPT access."))

    code, returned_state = await callback_task
    if returned_state != state:
        raise ValueError("OAuth state mismatch during OpenAI login")

    return await _exchange_authorization_code(code, verifier, _REDIRECT_URI)


async def refresh_openai_codex_token(credentials: OAuthCredentials) -> OAuthCredentials:
    """Refresh OpenAI Codex OAuth token."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID,
                "refresh_token": credentials.refresh,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()

    data.setdefault("refresh_token", credentials.refresh)
    data.setdefault("access_token", credentials.access)
    creds = _credentials_from_token_data(data)
    creds.extra.update(credentials.extra)
    account_id = _get_account_id_from_jwt(creds.access)
    if account_id:
        creds.extra["account_id"] = account_id
    return creds


async def _wait_for_callback_code(ready: asyncio.Future[None] | None = None) -> tuple[str, str]:
    """Start a local HTTP server to receive the OAuth callback."""
    code_future: asyncio.Future[tuple[str, str]] = asyncio.get_event_loop().create_future()

    try:
        from aiohttp import web  # type: ignore[import]

        async def handle(request: web.Request) -> web.Response:
            code = request.query.get("code", "")
            state = request.query.get("state", "")
            if code and not code_future.done():
                code_future.set_result((code, state))
            return web.Response(
                text="<html><body>Authorization complete! You can close this window.</body></html>",
                content_type="text/html",
            )

        app = web.Application()
        app.router.add_get(_REDIRECT_PATH, handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", _REDIRECT_PORT)
        await site.start()
        if ready is not None and not ready.done():
            ready.set_result(None)
        try:
            return await asyncio.wait_for(code_future, timeout=300)
        finally:
            await runner.cleanup()

    except ImportError:
        if ready is not None and not ready.done():
            ready.set_result(None)
        code = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter the authorization code from the callback URL: "
        )
        state = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter the state value from the callback URL: "
        )
        return code, state
    except Exception as exc:
        if ready is not None and not ready.done():
            ready.set_exception(exc)
        raise


class _OpenAICodexOAuthProvider:
    id = "openai-codex"
    name = "OpenAI Codex (ChatGPT Plus/Pro)"
    uses_callback_server = False

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        return await login_openai_codex_device_code(callbacks)

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return await refresh_openai_codex_token(credentials)

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials.access

    def modify_models(self, models, credentials):
        account_id = credentials.extra.get("account_id", "")
        if account_id:
            for model in models:
                if hasattr(model, "base_url") and model.base_url:
                    pass  # Would update base URL with account ID
        return models


openai_codex_oauth_provider = _OpenAICodexOAuthProvider()
