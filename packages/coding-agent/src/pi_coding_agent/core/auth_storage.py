"""
API key and OAuth credential storage — mirrors packages/coding-agent/src/core/auth-storage.ts
"""
from __future__ import annotations

import json
import os
import base64
import hashlib
import hmac
import secrets
from pi_coding_agent.config import CONFIG_DIR_NAME
from pathlib import Path
from typing import Any, Callable, Awaitable


LockResult = dict[str, Any]


class AuthStorageBackend:
    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        raise NotImplementedError

    async def with_lock_async(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:
        raise NotImplementedError


class FileAuthStorageBackend(AuthStorageBackend):
    def __init__(self, auth_path: str | None = None) -> None:
        from pi_coding_agent.config import get_auth_path

        self.auth_path = os.path.abspath(os.path.expanduser(auth_path or get_auth_path()))
        self.encrypted = True

    def _ensure_file(self) -> None:
        os.makedirs(os.path.dirname(self.auth_path), mode=0o700, exist_ok=True)
        if not os.path.exists(self.auth_path):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(self.auth_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("{}")

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        self._ensure_file()
        with open(self.auth_path, encoding="utf-8") as f:
            current = f.read()
        result = fn(current)
        if "next" in result:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(self.auth_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(result["next"])
        return result.get("result")

    async def with_lock_async(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:
        self._ensure_file()
        with open(self.auth_path, encoding="utf-8") as f:
            current = f.read()
        result = await fn(current)
        if "next" in result:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(self.auth_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(result["next"])
        return result.get("result")


class InMemoryAuthStorageBackend(AuthStorageBackend):
    def __init__(self, value: str | None = None) -> None:
        self.value = value

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> Any:
        result = fn(self.value)
        if "next" in result:
            self.value = result["next"]
        return result.get("result")

    async def with_lock_async(self, fn: Callable[[str | None], Awaitable[LockResult]]) -> Any:
        result = await fn(self.value)
        if "next" in result:
            self.value = result["next"]
        return result.get("result")


class AuthStorage:
    """
    Stores API keys and OAuth credentials securely on disk.
    Mirrors AuthStorage in TypeScript.

    Storage: ~/.pi/agent/auth.json
    """

    AUTH_DIR = os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME, "agent")
    AUTH_FILE = os.path.join(AUTH_DIR, "auth.json")

    def __init__(self, storage: AuthStorageBackend | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._runtime_overrides: dict[str, str] = {}  # Runtime API key overrides (in-memory only)
        self._fallback_resolver: Callable[[str], str | None] | None = None
        self._errors: list[Exception] = []
        self._storage = storage

    @classmethod
    def create(cls, auth_path: str | None = None) -> "AuthStorage":
        return cls(FileAuthStorageBackend(auth_path))

    @classmethod
    def from_storage(cls, storage: AuthStorageBackend) -> "AuthStorage":
        return cls(storage)

    @classmethod
    def in_memory(cls, data: dict[str, Any] | None = None) -> "AuthStorage":
        storage = InMemoryAuthStorageBackend(json.dumps(data or {}, indent=2))
        return cls.from_storage(storage)

    def _record_error(self, error: Any) -> None:
        self._errors.append(error if isinstance(error, Exception) else Exception(str(error)))

    def _parse_data(self, content: str | None) -> dict[str, Any]:
        if not content:
            return {}
        raw = json.loads(content)
        if not isinstance(raw, dict):
            return {}
        if raw.get("encrypted") is True:
            return self._decrypt_storage(raw)
        return raw

    def _storage_path(self) -> str:
        if self._storage is not None and hasattr(self._storage, "auth_path"):
            return str(getattr(self._storage, "auth_path"))
        return self.AUTH_FILE

    def _storage_uses_encryption(self) -> bool:
        if self._storage is None:
            return True
        return bool(getattr(self._storage, "encrypted", False))

    def _key_path(self) -> str:
        return f"{self._storage_path()}.key"

    def _load_or_create_key(self) -> bytes:
        path = self._key_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                key = f.read().strip()
            if key:
                return base64.urlsafe_b64decode(key)
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        key = secrets.token_bytes(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.urlsafe_b64encode(key))
        return key

    def _keystream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        chunks: list[bytes] = []
        counter = 0
        while sum(len(chunk) for chunk in chunks) < length:
            chunks.append(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
            counter += 1
        return b"".join(chunks)[:length]

    def _encrypt_storage(self, data: dict[str, Any]) -> str:
        plaintext = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        key = self._load_or_create_key()
        nonce = secrets.token_bytes(16)
        stream = self._keystream(key, nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
        mac = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
        envelope = {
            "version": 2,
            "encrypted": True,
            "cipher": "pi-py-xor-hmac-sha256",
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
            "data": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            "mac": base64.urlsafe_b64encode(mac).decode("ascii"),
        }
        return json.dumps(envelope, indent=2)

    def _decrypt_storage(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if envelope.get("cipher") != "pi-py-xor-hmac-sha256":
            raise ValueError("Unsupported encrypted auth storage format")
        key = self._load_or_create_key()
        nonce = base64.urlsafe_b64decode(str(envelope.get("nonce", "")))
        ciphertext = base64.urlsafe_b64decode(str(envelope.get("data", "")))
        expected = base64.urlsafe_b64decode(str(envelope.get("mac", "")))
        actual = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Encrypted auth storage integrity check failed")
        stream = self._keystream(key, nonce, len(ciphertext))
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, stream))
        decoded = json.loads(plaintext.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else {}

    def _read_storage(self) -> dict[str, Any]:
        if self._storage is not None:
            content_holder: dict[str, str | None] = {"content": None}
            self._storage.with_lock(lambda current: (content_holder.update(content=current) or {"result": None}))
            return self._parse_data(content_holder["content"])
        if os.path.exists(self.AUTH_FILE):
            with open(self.AUTH_FILE, encoding="utf-8") as f:
                return self._parse_data(f.read())
        return {}

    def _write_storage(self, data: dict[str, Any]) -> None:
        serialized = self._encrypt_storage(data) if self._storage_uses_encryption() else json.dumps(data, indent=2)
        if self._storage is not None:
            self._storage.with_lock(lambda current: {"result": None, "next": serialized})
            return
        os.makedirs(self.AUTH_DIR, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(self.AUTH_FILE, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        try:
            self._data = self._read_storage()
        except Exception as exc:
            self._data = {}
            self._record_error(exc)
        self._loaded = True

    def _save(self) -> None:
        try:
            self._write_storage(self._data)
        except Exception as exc:
            self._record_error(exc)
            raise

    def reload(self) -> None:
        self._loaded = False
        self._load()

    def get(self, provider: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        credential = self._data.get(provider)
        # OAuth/subscription tokens take precedence over API keys.  Older
        # auth.json files can contain both a current provider credential and
        # legacy api_keys/oauth_tokens entries, so apply the same precedence
        # here that resolve_api_key() applies.
        if provider in self._data.get("oauth_tokens", {}):
            return {"type": "oauth", **self._data["oauth_tokens"][provider]}
        if isinstance(credential, dict) and credential.get("type") == "oauth":
            return dict(credential)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            return dict(credential)
        if provider in self._data.get("api_keys", {}):
            return {"type": "api_key", "key": self._data["api_keys"][provider]}
        return None

    def set(self, provider: str, credential: dict[str, Any]) -> None:
        self._ensure_loaded()
        self._data[provider] = dict(credential)
        if credential.get("type") == "api_key":
            self._data.setdefault("api_keys", {})[provider] = credential.get("key")
        elif credential.get("type") == "oauth":
            oauth = dict(credential)
            oauth.pop("type", None)
            self._data.setdefault("oauth_tokens", {})[provider] = oauth
        self._save()

    def remove(self, provider: str) -> None:
        self._ensure_loaded()
        self._data.pop(provider, None)
        self._data.get("api_keys", {}).pop(provider, None)
        self._data.get("oauth_tokens", {}).pop(provider, None)
        self._save()

    def list(self) -> list[str]:
        self._ensure_loaded()
        providers: set[str] = set()
        for provider, value in self._data.items():
            if provider not in {"api_keys", "oauth_tokens"} and isinstance(value, dict):
                providers.add(provider)
        providers.update(self._data.get("api_keys", {}).keys())
        providers.update(self._data.get("oauth_tokens", {}).keys())
        return sorted(providers)

    def has(self, provider: str) -> bool:
        return self.get(provider) is not None

    def has_auth(self, provider: str) -> bool:
        if provider in self._runtime_overrides:
            return True
        if self.has(provider):
            return True
        from pi_ai.env_api_keys import get_env_api_key
        if get_env_api_key(provider):
            return True
        return bool(self._fallback_resolver and self._fallback_resolver(provider))

    def get_auth_status(self, provider: str) -> dict[str, Any]:
        if self.has(provider):
            return {"configured": True, "source": "stored"}
        if provider in self._runtime_overrides:
            return {"configured": False, "source": "runtime", "label": "--api-key"}
        from pi_ai.env_api_keys import get_env_api_key
        if get_env_api_key(provider):
            return {"configured": False, "source": "environment"}
        if self._fallback_resolver and self._fallback_resolver(provider):
            return {"configured": False, "source": "fallback", "label": "custom provider config"}
        return {"configured": False}

    def get_all(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {provider: credential for provider in self.list() if (credential := self.get(provider))}

    def drain_errors(self) -> list[Exception]:
        drained = list(self._errors)
        self._errors = []
        return drained

    def get_api_key(self, provider: str) -> str | None:
        """Get the stored API key for a provider."""
        self._ensure_loaded()
        api_keys = self._data.get("api_keys", {})
        if isinstance(api_keys, dict) and provider in api_keys:
            return api_keys.get(provider)
        credential = self._data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            return credential.get("key")
        return None

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key for a provider."""
        self.set(provider, {"type": "api_key", "key": api_key})

    def delete_api_key(self, provider: str) -> None:
        """Delete the stored API key for a provider."""
        self._ensure_loaded()
        api_keys = self._data.get("api_keys", {})
        if isinstance(api_keys, dict):
            api_keys.pop(provider, None)
        credential = self._data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            self._data.pop(provider, None)
        self._save()

    def get_oauth_token(self, provider: str) -> dict[str, Any] | None:
        """Get the stored OAuth token for a provider."""
        self._ensure_loaded()
        oauth_tokens = self._data.get("oauth_tokens", {})
        if isinstance(oauth_tokens, dict) and provider in oauth_tokens:
            return dict(oauth_tokens[provider])
        credential = self._data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "oauth":
            token = dict(credential)
            token.pop("type", None)
            return token
        return None

    def set_oauth_token(self, provider: str, token: dict[str, Any]) -> None:
        """Store an OAuth token for a provider."""
        self.set(provider, {"type": "oauth", **token})

    def delete_oauth_token(self, provider: str) -> None:
        """Delete the stored OAuth token for a provider."""
        self._ensure_loaded()
        oauth_tokens = self._data.get("oauth_tokens", {})
        if isinstance(oauth_tokens, dict):
            oauth_tokens.pop(provider, None)
        credential = self._data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "oauth":
            self._data.pop(provider, None)
        self._save()

    def set_runtime_api_key(self, provider: str, api_key: str) -> None:
        """
        Set runtime API key override (in-memory only, not persisted to disk).
        Mirrors TypeScript AuthStorage.setRuntimeApiKey().
        
        Runtime overrides have the highest priority in resolve_api_key().
        Used by openclaw to inject API keys from auth-profiles.json.
        """
        self._runtime_overrides[provider] = api_key

    def remove_runtime_api_key(self, provider: str) -> None:
        self._runtime_overrides.pop(provider, None)

    def set_fallback_resolver(self, resolver: Callable[[str], str | None] | None) -> None:
        self._fallback_resolver = resolver

    def resolve_api_key(self, provider: str) -> str | None:
        """
        Resolve API key for a provider.
        Priority (aligned with TypeScript):
        1. Runtime override (CLI --api-key, openclaw injection) - highest
        2. OAuth token from auth.json (auto-refresh)
        3. Stored API key from auth.json
        4. Environment variable
        """
        # 1. Runtime override takes highest priority
        if provider in self._runtime_overrides:
            return self._runtime_overrides[provider]
        
        # 2. Try OAuth token
        oauth = self.get_oauth_token(provider)
        if oauth:
            access_token = oauth.get("access_token")
            if access_token:
                # Check expiry
                expires_at = oauth.get("expires_at", 0)
                import time
                if expires_at and expires_at > time.time():
                    return access_token
                # Token expired, try refresh
                refreshed = self._refresh_oauth_token(provider)
                if refreshed:
                    return refreshed

        # 3. Stored key from auth.json
        stored = self.get_api_key(provider)
        if stored:
            return stored
        
        # 4. Environment variable fallback
        from pi_ai.env_api_keys import get_env_api_key
        env_key = get_env_api_key(provider)
        if env_key:
            return env_key
        return self._fallback_resolver(provider) if self._fallback_resolver else None

    def is_using_oauth(self, provider: str) -> bool:
        """Check if provider uses OAuth authentication."""
        return self.get_oauth_token(provider) is not None

    async def login(
        self,
        provider: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_url: str | None = None,
        token_url: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Perform OAuth login for a provider.
        Returns the token dict on success, None on failure.
        Mirrors login() in TypeScript AuthStorage.
        """
        import webbrowser
        import urllib.parse
        import secrets
        import asyncio

        state = secrets.token_urlsafe(32)
        redirect_port = 19747
        redirect_uri = f"http://localhost:{redirect_port}/oauth/callback"

        if not auth_url or not token_url:
            return None

        params = {
            "client_id": client_id or "",
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(scopes or []),
        }
        full_url = f"{auth_url}?{urllib.parse.urlencode(params)}"

        auth_code: str | None = None
        received_state: str | None = None

        from http.server import HTTPServer, BaseHTTPRequestHandler

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self_handler):
                nonlocal auth_code, received_state
                parsed = urllib.parse.urlparse(self_handler.path)
                qs = urllib.parse.parse_qs(parsed.query)
                auth_code = qs.get("code", [None])[0]
                received_state = qs.get("state", [None])[0]
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.end_headers()
                self_handler.wfile.write(b"<html><body><h1>Login successful. You can close this tab.</h1></body></html>")

            def log_message(self_handler, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", redirect_port), CallbackHandler)
        server.timeout = 120

        webbrowser.open(full_url)

        # Wait for callback in a thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.handle_request)
        server.server_close()

        if not auth_code or received_state != state:
            return None

        # Exchange code for token
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(token_url, data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri,
                "client_id": client_id or "",
                "client_secret": client_secret or "",
            })
            if resp.status_code != 200:
                return None
            token_data = resp.json()

        import time
        expires_in = token_data.get("expires_in", 3600)
        token_data["expires_at"] = time.time() + expires_in

        self.set_oauth_token(provider, token_data)
        return token_data

    def logout(self, provider: str, credential_type: str | None = None) -> None:
        """Logout from a provider by removing selected stored credentials."""
        if credential_type in {None, "all"}:
            self.remove(provider)
        elif credential_type == "api_key":
            self.delete_api_key(provider)
        elif credential_type in {"token", "oauth", "subscription"}:
            self.delete_oauth_token(provider)
        else:
            raise ValueError(f"Unknown credential type: {credential_type}")

    def _refresh_oauth_token(self, provider: str) -> str | None:
        """Attempt to refresh an expired OAuth token synchronously."""
        oauth = self.get_oauth_token(provider)
        if not oauth:
            return None
        refresh_token = oauth.get("refresh_token")
        token_url = oauth.get("token_url")
        client_id = oauth.get("client_id")
        if not refresh_token or not token_url:
            return None

        try:
            import httpx
            with httpx.Client() as client:
                resp = client.post(token_url, data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id or "",
                })
                if resp.status_code != 200:
                    return None
                token_data = resp.json()

            import time
            expires_in = token_data.get("expires_in", 3600)
            token_data["expires_at"] = time.time() + expires_in
            token_data["refresh_token"] = refresh_token
            token_data["token_url"] = token_url
            token_data["client_id"] = client_id

            self.set_oauth_token(provider, token_data)
            return token_data.get("access_token")
        except Exception:
            return None

    def get_oauth_providers(self) -> list[str]:
        """Get list of providers with OAuth credentials."""
        return [provider for provider in self.list() if self.is_using_oauth(provider)]

    def list_stored_providers(self) -> list[str]:
        """List all providers with stored credentials."""
        return self.list()


AuthStorage.fromStorage = AuthStorage.from_storage
AuthStorage.inMemory = AuthStorage.in_memory
AuthStorage.setRuntimeApiKey = AuthStorage.set_runtime_api_key
AuthStorage.removeRuntimeApiKey = AuthStorage.remove_runtime_api_key
AuthStorage.setFallbackResolver = AuthStorage.set_fallback_resolver
AuthStorage.resolveApiKey = AuthStorage.resolve_api_key
AuthStorage.getApiKey = AuthStorage.get_api_key
AuthStorage.setApiKey = AuthStorage.set_api_key
AuthStorage.deleteApiKey = AuthStorage.delete_api_key
AuthStorage.getAuthStatus = AuthStorage.get_auth_status
AuthStorage.hasAuth = AuthStorage.has_auth
AuthStorage.getAll = AuthStorage.get_all
AuthStorage.drainErrors = AuthStorage.drain_errors
AuthStorage.listStoredProviders = AuthStorage.list_stored_providers
AuthStorage.getOAuthProviders = AuthStorage.get_oauth_providers

__all__ = [
    "AuthStorage",
    "AuthStorageBackend",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
]
