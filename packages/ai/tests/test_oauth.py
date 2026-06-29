"""
Tests for the OAuth subpackage.

Tests PKCE generation, provider registry, and token refresh (mocked HTTP).
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------

class TestPKCE:
    def test_generate_pkce_returns_two_strings(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_pkce_verifier_length(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, _ = generate_pkce()
        # 32 bytes base64url encoded = 43 chars (without padding)
        assert len(verifier) >= 40

    def test_pkce_challenge_length(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        _, challenge = generate_pkce()
        assert len(challenge) >= 40

    def test_pkce_is_different_each_call(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2
        assert c1 != c2

    def test_pkce_base64url_no_padding(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        assert "=" not in verifier
        assert "=" not in challenge

    def test_pkce_challenge_is_sha256_of_verifier(self):
        """Verify challenge is correct SHA-256 base64url of verifier."""
        import base64
        import hashlib

        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert challenge == expected


# ---------------------------------------------------------------------------
# OAuth Types
# ---------------------------------------------------------------------------

class TestOAuthCredentials:
    def test_to_dict_round_trip(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        creds = OAuthCredentials(refresh="r", access="a", expires=12345)
        d = creds.to_dict()
        assert d["refresh"] == "r"
        assert d["access"] == "a"
        assert d["expires"] == 12345

    def test_from_dict(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        d = {"refresh": "r", "access": "a", "expires": 99999}
        creds = OAuthCredentials.from_dict(d)
        assert creds.refresh == "r"
        assert creds.access == "a"
        assert creds.expires == 99999

    def test_from_dict_with_extra_fields(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        d = {"refresh": "r", "access": "a", "expires": 0, "account_id": "u123"}
        creds = OAuthCredentials.from_dict(d)
        assert creds.extra.get("account_id") == "u123"


# ---------------------------------------------------------------------------
# OAuth Registry
# ---------------------------------------------------------------------------

class TestOAuthRegistry:
    def test_get_oauth_provider_anthropic(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("anthropic")
        assert p is not None
        assert p.id == "anthropic"
        assert p.name

    def test_get_oauth_provider_github_copilot(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("github-copilot")
        assert p is not None

    def test_get_oauth_provider_gemini_cli(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("google-gemini-cli")
        assert p is not None

    def test_get_oauth_provider_antigravity(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("google-antigravity")
        assert p is not None

    def test_get_oauth_provider_openai_codex(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("openai-codex")
        assert p is not None

    def test_get_unknown_provider_returns_none(self):
        from pi_ai.utils.oauth import get_oauth_provider
        assert get_oauth_provider("nonexistent-provider") is None

    def test_get_oauth_providers_returns_all(self):
        from pi_ai.utils.oauth import get_oauth_providers
        providers = get_oauth_providers()
        assert len(providers) >= 5
        ids = [p.id for p in providers]
        assert "anthropic" in ids
        assert "github-copilot" in ids

    def test_register_custom_provider(self):
        from pi_ai.utils.oauth import get_oauth_provider, register_oauth_provider

        class _Custom:
            id = "custom-test-provider-xyz"
            name = "Custom Test"
            uses_callback_server = False

            async def login(self, callbacks): ...
            async def refresh_token(self, credentials): return credentials
            def get_api_key(self, credentials): return credentials.access
            def modify_models(self, models, credentials): return models

        register_oauth_provider(_Custom())
        p = get_oauth_provider("custom-test-provider-xyz")
        assert p is not None
        assert p.name == "Custom Test"


class TestOpenAICodexOAuthProvider:
    @pytest.mark.asyncio
    async def test_device_code_login_uses_device_auth_without_callback_server(self):
        from pi_ai.utils.oauth.openai_codex import login_openai_codex_device_code
        from pi_ai.utils.oauth.types import OAuthLoginCallbacks

        auth_infos = []
        progress = []

        device_resp = MagicMock()
        device_resp.status_code = 200
        device_resp.json.return_value = {
            "device_auth_id": "device-123",
            "user_code": "USER-CODE",
            "interval": 1,
        }

        pending_resp = MagicMock()
        pending_resp.status_code = 403
        pending_resp.is_success = False
        pending_resp.text = ""

        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.is_success = True
        token_resp.json.return_value = {
            "authorization_code": "auth-code",
            "code_verifier": "device-verifier",
        }

        exchange_resp = MagicMock()
        exchange_resp.status_code = 200
        exchange_resp.json.return_value = {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

        with (
            patch("pi_ai.utils.oauth.openai_codex.asyncio.sleep", AsyncMock()),
            patch("pi_ai.utils.oauth.openai_codex.httpx.AsyncClient") as MockClient,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(side_effect=[device_resp, pending_resp, token_resp, exchange_resp])
            MockClient.return_value = mock_ctx

            creds = await login_openai_codex_device_code(
                OAuthLoginCallbacks(
                    on_auth=lambda info: auth_infos.append(info),
                    on_prompt=AsyncMock(return_value=""),
                    on_progress=lambda message: progress.append(message),
                )
            )

        assert auth_infos[0].url == "https://auth.openai.com/codex/device"
        assert auth_infos[0].instructions == "Enter code: USER-CODE"
        assert progress == ["Visit https://auth.openai.com/codex/device and enter code: USER-CODE"]
        assert creds.access == "access"
        assert creds.refresh == "refresh"

        calls = mock_ctx.post.call_args_list
        assert calls[0].args[0] == "https://auth.openai.com/api/accounts/deviceauth/usercode"
        assert calls[1].args[0] == "https://auth.openai.com/api/accounts/deviceauth/token"
        assert calls[2].args[0] == "https://auth.openai.com/api/accounts/deviceauth/token"
        assert calls[3].args[0] == "https://auth.openai.com/oauth/token"
        assert calls[3].kwargs["data"]["redirect_uri"] == "https://auth.openai.com/deviceauth/callback"
        assert calls[3].kwargs["data"]["code_verifier"] == "device-verifier"

    def test_registered_provider_uses_device_code_login(self):
        from pi_ai.utils.oauth.openai_codex import openai_codex_oauth_provider

        assert openai_codex_oauth_provider.uses_callback_server is False

    @pytest.mark.asyncio
    async def test_login_uses_current_authorize_endpoint_and_state(self):
        from pi_ai.utils.oauth.openai_codex import login_openai_codex
        from pi_ai.utils.oauth.types import OAuthLoginCallbacks

        auth_urls: list[str] = []
        async def wait_for_callback(ready=None):
            if ready is not None and not ready.done():
                ready.set_result(None)
            return "code", "state-123"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

        with (
            patch("pi_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("pi_ai.utils.oauth.openai_codex.secrets.token_urlsafe", return_value="state-123"),
            patch("pi_ai.utils.oauth.openai_codex._wait_for_callback_code", AsyncMock(side_effect=wait_for_callback)),
            patch("pi_ai.utils.oauth.openai_codex.httpx.AsyncClient") as MockClient,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_ctx

            creds = await login_openai_codex(
                OAuthLoginCallbacks(
                    on_auth=lambda info: auth_urls.append(info.url),
                    on_prompt=AsyncMock(return_value=""),
                )
            )

        parsed = urlparse(auth_urls[0])
        query = parse_qs(parsed.query)
        assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://auth.openai.com/oauth/authorize"
        assert query["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
        assert query["redirect_uri"] == ["http://localhost:1455/auth/callback"]
        assert query["scope"] == ["openid profile email offline_access"]
        assert query["code_challenge"] == ["challenge"]
        assert query["code_challenge_method"] == ["S256"]
        assert query["id_token_add_organizations"] == ["true"]
        assert query["codex_cli_simplified_flow"] == ["true"]
        assert query["originator"] == ["codex_cli"]
        assert query["state"] == ["state-123"]
        assert creds.access == "access"
        assert creds.refresh == "refresh"

    @pytest.mark.asyncio
    async def test_login_rejects_callback_state_mismatch(self):
        from pi_ai.utils.oauth.openai_codex import login_openai_codex
        from pi_ai.utils.oauth.types import OAuthLoginCallbacks

        async def wait_for_callback(ready=None):
            if ready is not None and not ready.done():
                ready.set_result(None)
            return "code", "wrong-state"

        with (
            patch("pi_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("pi_ai.utils.oauth.openai_codex.secrets.token_urlsafe", return_value="state-123"),
            patch("pi_ai.utils.oauth.openai_codex._wait_for_callback_code", AsyncMock(side_effect=wait_for_callback)),
        ):
            with pytest.raises(ValueError, match="OAuth state mismatch"):
                await login_openai_codex(
                    OAuthLoginCallbacks(
                        on_auth=lambda info: None,
                        on_prompt=AsyncMock(return_value=""),
                    )
                )

    @pytest.mark.asyncio
    async def test_login_opens_browser_only_after_callback_server_ready(self):
        from pi_ai.utils.oauth.openai_codex import login_openai_codex
        from pi_ai.utils.oauth.types import OAuthLoginCallbacks

        order: list[str] = []

        async def wait_for_callback(ready=None):
            order.append("server-start")
            assert order == ["server-start"]
            if ready is not None and not ready.done():
                order.append("server-ready")
                ready.set_result(None)
            return "code", "state-123"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

        with (
            patch("pi_ai.utils.oauth.openai_codex.generate_pkce", return_value=("verifier", "challenge")),
            patch("pi_ai.utils.oauth.openai_codex.secrets.token_urlsafe", return_value="state-123"),
            patch("pi_ai.utils.oauth.openai_codex._wait_for_callback_code", AsyncMock(side_effect=wait_for_callback)),
            patch("pi_ai.utils.oauth.openai_codex.httpx.AsyncClient") as MockClient,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_ctx

            await login_openai_codex(
                OAuthLoginCallbacks(
                    on_auth=lambda info: order.append("browser-open"),
                    on_prompt=AsyncMock(return_value=""),
                )
            )

        assert order == ["server-start", "server-ready", "browser-open"]


# ---------------------------------------------------------------------------
# get_oauth_api_key
# ---------------------------------------------------------------------------

class TestGetOAuthApiKey:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_credentials(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        result = await get_oauth_api_key("anthropic", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_api_key_when_credentials_valid(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        future_ts = int(time.time() * 1000) + 3600 * 1000  # 1 hour from now
        creds_dict = {"refresh": "r", "access": "valid_token", "expires": future_ts}
        result = await get_oauth_api_key("anthropic", {"anthropic": creds_dict})
        assert result is not None
        assert result["api_key"] == "valid_token"

    @pytest.mark.asyncio
    async def test_refreshes_expired_token(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        from pi_ai.utils.oauth.types import OAuthCredentials

        # Expired credentials
        expired_creds = {"refresh": "r", "access": "old_token", "expires": 0}
        new_creds = OAuthCredentials(refresh="new_r", access="new_token", expires=int(time.time() * 1000) + 3600000)

        with patch("pi_ai.utils.oauth.get_oauth_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.refresh_token = AsyncMock(return_value=new_creds)
            mock_provider.get_api_key = lambda c: c.access
            mock_get.return_value = mock_provider

            result = await get_oauth_api_key("anthropic", {"anthropic": expired_creds})
            assert result is not None
            assert result["api_key"] == "new_token"

    @pytest.mark.asyncio
    async def test_raises_for_unknown_provider(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        with pytest.raises(ValueError, match="Unknown OAuth provider"):
            await get_oauth_api_key("fake-provider", {})


# ---------------------------------------------------------------------------
# Anthropic OAuth Provider (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAnthropicOAuthProvider:
    def test_provider_id(self):
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider
        assert anthropic_oauth_provider.id == "anthropic"

    def test_get_api_key(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider
        creds = OAuthCredentials(refresh="r", access="tok_123", expires=99999)
        assert anthropic_oauth_provider.get_api_key(creds) == "tok_123"

    @pytest.mark.asyncio
    async def test_refresh_token_mocked(self):
        from pi_ai.utils.oauth.anthropic import refresh_anthropic_token

        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }

        with patch("pi_ai.utils.oauth.anthropic.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_ctx

            creds = await refresh_anthropic_token("old_refresh")
            assert creds.access == "new_access"
            assert creds.refresh == "new_refresh"


# ---------------------------------------------------------------------------
# GitHub Copilot OAuth (unit tests)
# ---------------------------------------------------------------------------

class TestGitHubCopilotOAuth:
    def test_normalize_domain(self):
        from pi_ai.utils.oauth.github_copilot import normalize_domain
        assert normalize_domain("github.example.com") == "github.example.com"
        assert normalize_domain("https://github.example.com") == "github.example.com"
        assert normalize_domain("  github.com  ") == "github.com"

    def test_get_base_url_default(self):
        from pi_ai.utils.oauth.github_copilot import get_github_copilot_base_url
        assert get_github_copilot_base_url() == "https://api.github.com"

    def test_get_base_url_custom_domain(self):
        from pi_ai.utils.oauth.github_copilot import get_github_copilot_base_url
        url = get_github_copilot_base_url("my-corp.github.com")
        assert "my-corp.github.com" in url
