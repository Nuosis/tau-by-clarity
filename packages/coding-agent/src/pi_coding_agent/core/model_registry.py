"""
Model registry — mirrors packages/coding-agent/src/core/model-registry.ts

Manages built-in and custom models, provides API key resolution.
Supports loading custom models from ~/.pi/agent/models.json,
applying provider-level and per-model overrides.
"""
from __future__ import annotations

import json
import os
from pi_coding_agent.config import CONFIG_DIR_NAME
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from pi_ai import get_model, get_models, get_providers
from pi_ai.types import Model
from pi_coding_agent.core.provider_profiles import PROVIDER_PROFILES, synthetic_model


# ─── Config schema (runtime validation without AJV) ───────────────────────────

def _validate_models_config(config: Any) -> str | None:
    """Validate models.json structure. Returns error string or None."""
    if not isinstance(config, dict):
        return "models.json must be a JSON object"
    providers = config.get("providers")
    if providers is None:
        return '"providers" key is required in models.json'
    if not isinstance(providers, dict):
        return '"providers" must be an object'

    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            return f"Provider {provider_name}: config must be an object"

        models = provider_config.get("models") or []
        model_overrides = provider_config.get("modelOverrides") or {}

        if not models:
            # Override-only: needs baseUrl or modelOverrides
            if not provider_config.get("baseUrl") and not model_overrides and not provider_config.get("tiers"):
                return (
                    f'Provider {provider_name}: must specify "baseUrl", '
                    '"modelOverrides", "tiers", or "models".'
                )
        else:
            # Custom models: needs baseUrl + apiKey
            if not provider_config.get("baseUrl"):
                return f'Provider {provider_name}: "baseUrl" is required when defining custom models.'
            # apiKey may be omitted when the key is stored in encrypted auth.json.

        for model_def in models:
            provider_api = provider_config.get("api")
            model_api = model_def.get("api")
            if not provider_api and not model_api:
                mid = model_def.get("id", "?")
                return (
                    f'Provider {provider_name}, model {mid}: no "api" specified. '
                    "Set at provider or model level."
                )
            if not model_def.get("id"):
                return f'Provider {provider_name}: model missing "id"'
            cw = model_def.get("contextWindow")
            if cw is not None and cw <= 0:
                return f'Provider {provider_name}, model {model_def["id"]}: invalid contextWindow'
            mt = model_def.get("maxTokens")
            if mt is not None and mt <= 0:
                return f'Provider {provider_name}, model {model_def["id"]}: invalid maxTokens'

    return None


def _resolve_config_value(value: str) -> str | None:
    """
    Resolve a config value: env var reference (e.g. "$MY_KEY" or "$(cmd)")
    or a plain string.
    Mirrors resolveConfigValue() in TypeScript.
    """
    if not value:
        return None

    # Shell command substitution: $(cmd)
    m = re.match(r"^\$\((.+)\)$", value.strip())
    if m:
        try:
            result = subprocess.run(
                m.group(1), shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    # Environment variable: $VAR_NAME or ${VAR_NAME}
    m2 = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$", value.strip())
    if m2:
        return os.environ.get(m2.group(1))

    # Plain string
    return value


def _resolve_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Resolve header values (env vars, shell commands)."""
    if not headers:
        return None
    resolved = {}
    for k, v in headers.items():
        val = _resolve_config_value(v)
        if val is not None:
            resolved[k] = val
    return resolved or None


def _apply_model_override(model: Model, override: dict[str, Any]) -> Model:
    """
    Deep merge a model override into a model.
    Mirrors applyModelOverride() in TypeScript.
    """
    result = Model(
        id=model.id,
        name=model.name,
        api=model.api,
        provider=model.provider,
        reasoning=model.reasoning,
        input=list(model.input),
        cost=dict(model.cost) if hasattr(model.cost, "__iter__") else model.cost,
        context_window=model.context_window,
        max_tokens=model.max_tokens,
        base_url=getattr(model, "base_url", None),
        headers=dict(model.headers) if model.headers else None,
        compat=dict(model.compat) if model.compat else None,
    )

    if override.get("name") is not None:
        result = Model(**{**result.__dict__, "name": override["name"]})
    if override.get("reasoning") is not None:
        result = Model(**{**result.__dict__, "reasoning": override["reasoning"]})
    if override.get("input") is not None:
        result = Model(**{**result.__dict__, "input": override["input"]})
    if override.get("contextWindow") is not None:
        result = Model(**{**result.__dict__, "context_window": override["contextWindow"]})
    if override.get("maxTokens") is not None:
        result = Model(**{**result.__dict__, "max_tokens": override["maxTokens"]})

    # Merge cost
    if override.get("cost"):
        base_cost = dict(result.cost) if result.cost else {}
        ov_cost = override["cost"]
        new_cost = {
            "input": ov_cost.get("input", base_cost.get("input", 0)),
            "output": ov_cost.get("output", base_cost.get("output", 0)),
            "cacheRead": ov_cost.get("cacheRead", base_cost.get("cacheRead", 0)),
            "cacheWrite": ov_cost.get("cacheWrite", base_cost.get("cacheWrite", 0)),
        }
        result = Model(**{**result.__dict__, "cost": new_cost})

    # Merge headers
    if override.get("headers"):
        resolved = _resolve_headers(override["headers"])
        if resolved:
            base_headers = dict(result.headers) if result.headers else {}
            result = Model(**{**result.__dict__, "headers": {**base_headers, **resolved}})

    # Merge compat
    if override.get("compat"):
        base_compat = dict(result.compat) if result.compat else {}
        result = Model(**{**result.__dict__, "compat": {**base_compat, **override["compat"]}})

    return result


@dataclass
class ProviderConfig:
    """Configuration for a registered provider."""
    name: str
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    headers: dict[str, str] | None = None
    auth_header: bool = False
    models: list[dict[str, Any]] = field(default_factory=list)
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class ModelRegistry:
    """
    Model registry — loads and manages models, resolves API keys.
    Mirrors ModelRegistry in TypeScript.

    Supports:
    - Built-in models from pi_ai
    - Custom models/overrides from ~/.pi/agent/models.json
    - Runtime provider registration (for extensions)
    - API key resolution via env vars / shell commands
    """

    def __init__(
        self,
        auth_storage: Any = None,
        models_json_path: str | None = None,
    ) -> None:
        self._auth_storage = auth_storage
        if models_json_path is None:
            models_json_path = os.path.join(
                os.path.expanduser("~"), CONFIG_DIR_NAME, "agent", "models.json"
            )
        self._models_json_path = models_json_path
        self._models: list[Model] = []
        self._custom_provider_api_keys: dict[str, str] = {}
        self._registered_providers: dict[str, ProviderConfig] = {}
        self._config_providers: dict[str, ProviderConfig] = {}
        self._load_error: str | None = None
        self._extra_models: list[Model] = []

        self._load_models()

    def _load_models(self) -> None:
        """(Re)load built-in + custom models."""
        self._custom_provider_api_keys.clear()
        self._config_providers.clear()
        self._load_error = None

        custom_models, overrides, model_overrides, error = self._load_custom_models()
        if error:
            self._load_error = error

        built_in = self._load_built_in_models(overrides, model_overrides)
        combined = self._merge_custom_models(built_in, custom_models)

        # Apply OAuth provider modifications if auth_storage supports it
        if self._auth_storage and hasattr(self._auth_storage, "get_oauth_providers"):
            for provider in self._auth_storage.get_oauth_providers():
                provider_id = getattr(provider, "id", provider)
                cred = self._auth_storage.get(provider_id)
                cred_type = cred.get("type") if isinstance(cred, dict) else getattr(cred, "type", None)
                if cred and cred_type == "oauth":
                    modify = getattr(provider, "modify_models", None)
                    if callable(modify):
                        combined = modify(combined, cred)

        # Apply registered extension providers
        for prov_name, prov_config in self._registered_providers.items():
            combined = self._apply_provider_config_to_models(combined, prov_name, prov_config)

        combined = self._apply_oauth_transport_overrides(combined)

        self._models = combined + self._extra_models

    def _load_built_in_models(
        self,
        overrides: dict[str, dict[str, Any]],
        model_overrides: dict[str, dict[str, dict[str, Any]]],
    ) -> list[Model]:
        """Load built-in models and apply provider/model overrides."""
        result: list[Model] = []
        for provider_name in get_providers():
            try:
                models = get_models(provider_name)
            except Exception:
                continue

            prov_override = overrides.get(provider_name, {})
            per_model = model_overrides.get(provider_name, {})

            for model in models:
                m = model

                # Apply provider-level baseUrl/headers
                if prov_override.get("baseUrl") or prov_override.get("headers"):
                    resolved_hdrs = _resolve_headers(prov_override.get("headers"))
                    base_url = prov_override.get("baseUrl", getattr(m, "base_url", None))
                    updates: dict[str, Any] = {"base_url": base_url}
                    if resolved_hdrs:
                        base_hdrs = dict(m.headers) if m.headers else {}
                        updates["headers"] = {**base_hdrs, **resolved_hdrs}
                    try:
                        m = Model(**{**m.__dict__, **updates})
                    except Exception:
                        pass

                # Apply per-model override
                if m.id in per_model:
                    try:
                        m = _apply_model_override(m, per_model[m.id])
                    except Exception:
                        pass

                result.append(m)

        existing = {(m.provider, m.id) for m in result}
        for profile in PROVIDER_PROFILES:
            if profile.id in get_providers():
                continue
            for model_id in profile.default_models.values():
                key = (profile.id, model_id)
                if key in existing:
                    continue
                model = synthetic_model(profile.id, model_id)
                if model is not None:
                    result.append(model)
                    existing.add(key)

        return result

    def _merge_custom_models(self, built_in: list[Model], custom: list[Model]) -> list[Model]:
        """Merge custom models into built-in list (custom wins on ID collision)."""
        merged = list(built_in)
        for custom_model in custom:
            idx = next(
                (i for i, m in enumerate(merged)
                 if m.provider == custom_model.provider and m.id == custom_model.id),
                -1,
            )
            if idx >= 0:
                merged[idx] = custom_model
            else:
                merged.append(custom_model)
        return merged

    def _load_custom_models(
        self,
    ) -> tuple[list[Model], dict[str, Any], dict[str, Any], str | None]:
        """
        Load and parse models.json.
        Returns (models, provider_overrides, model_overrides, error).
        """
        path = self._models_json_path
        if not path or not os.path.exists(path):
            return [], {}, {}, None

        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            return [], {}, {}, f"Failed to parse models.json: {e}\n\nFile: {path}"
        except OSError as e:
            return [], {}, {}, f"Failed to load models.json: {e}\n\nFile: {path}"

        err = _validate_models_config(config)
        if err:
            return [], {}, {}, f"Invalid models.json schema:\n  - {err}\n\nFile: {path}"

        overrides: dict[str, Any] = {}
        model_overrides: dict[str, dict[str, Any]] = {}
        models: list[Model] = []

        for provider_name, prov_cfg in config.get("providers", {}).items():
            self._config_providers[provider_name] = ProviderConfig(
                name=prov_cfg.get("name") or provider_name,
                base_url=prov_cfg.get("baseUrl"),
                api_key=prov_cfg.get("apiKey"),
                api=prov_cfg.get("api"),
                headers=prov_cfg.get("headers"),
                auth_header=prov_cfg.get("authHeader", False),
                models=prov_cfg.get("models") or [],
                model_overrides=prov_cfg.get("modelOverrides") or {},
            )

            # Provider-level overrides for built-in models
            if prov_cfg.get("baseUrl") or prov_cfg.get("headers") or prov_cfg.get("apiKey"):
                overrides[provider_name] = {
                    "baseUrl": prov_cfg.get("baseUrl"),
                    "headers": prov_cfg.get("headers"),
                    "apiKey": prov_cfg.get("apiKey"),
                }

            if prov_cfg.get("apiKey"):
                self._custom_provider_api_keys[provider_name] = prov_cfg["apiKey"]

            if prov_cfg.get("modelOverrides"):
                model_overrides[provider_name] = prov_cfg["modelOverrides"]

            # Custom model definitions
            for model_def in prov_cfg.get("models") or []:
                api = model_def.get("api") or prov_cfg.get("api")
                if not api:
                    continue

                # Resolve headers
                prov_headers = _resolve_headers(prov_cfg.get("headers")) or {}
                model_headers = _resolve_headers(model_def.get("headers")) or {}
                merged_headers: dict[str, str] | None = {**prov_headers, **model_headers} or None

                # Auth header injection
                if prov_cfg.get("authHeader") and prov_cfg.get("apiKey"):
                    resolved_key = _resolve_config_value(prov_cfg["apiKey"])
                    if resolved_key:
                        merged_headers = {**(merged_headers or {}), "Authorization": f"Bearer {resolved_key}"}

                default_cost = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}

                try:
                    m = Model(
                        id=model_def["id"],
                        name=model_def.get("name") or model_def["id"],
                        api=api,
                        provider=provider_name,
                        reasoning=model_def.get("reasoning", False),
                        input=model_def.get("input", ["text"]),
                        cost=model_def.get("cost") or default_cost,
                        context_window=model_def.get("contextWindow", 128000),
                        max_tokens=model_def.get("maxTokens", 16384),
                        base_url=prov_cfg.get("baseUrl"),
                        headers=merged_headers,
                        compat=model_def.get("compat"),
                    )
                    models.append(m)
                except Exception:
                    pass

        return models, overrides, model_overrides, None

    def _apply_provider_config_to_models(
        self,
        models: list[Model],
        provider_name: str,
        prov_config: ProviderConfig,
    ) -> list[Model]:
        """Apply a registered provider config to the model list."""
        new_models = list(models)
        for model_def in prov_config.models:
            api = model_def.get("api") or prov_config.api
            if not api:
                continue
            default_cost = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
            provider_headers = _resolve_headers(prov_config.headers) or {}
            model_headers = _resolve_headers(model_def.get("headers")) or {}
            merged_headers = {**provider_headers, **model_headers} or None
            try:
                m = Model(
                    id=model_def["id"],
                    name=model_def.get("name") or model_def["id"],
                    api=api,
                    provider=provider_name,
                    reasoning=model_def.get("reasoning", False),
                    input=model_def.get("input", ["text"]),
                    cost=model_def.get("cost") or default_cost,
                    context_window=model_def.get("contextWindow", 128000),
                    max_tokens=model_def.get("maxTokens", 16384),
                    base_url=prov_config.base_url,
                    headers=merged_headers,
                    compat=model_def.get("compat"),
                )
                # Check if it replaces an existing entry
                idx = next(
                    (i for i, em in enumerate(new_models)
                     if em.provider == provider_name and em.id == m.id),
                    -1,
                )
                if idx >= 0:
                    new_models[idx] = m
                else:
                    new_models.append(m)
            except Exception:
                pass
        return new_models

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload models from disk (built-in + custom from models.json)."""
        self._load_models()

    def get_error(self) -> str | None:
        """Get any error from loading models.json."""
        return self._load_error

    def get_all(self) -> list[Model]:
        """Get all models (built-in + custom)."""
        return list(self._models)

    def get_all_models(self) -> list[Model]:
        """Alias for get_all()."""
        return self.get_all()

    async def get_available(self) -> list[Model]:
        """
        Get only models that have auth configured.
        Mirrors getAvailable() in TypeScript.
        """
        if self._auth_storage and hasattr(self._auth_storage, "has_auth"):
            return [m for m in self._models if self._auth_storage.has_auth(m.provider)]
        # Fallback: check environment variables
        return [m for m in self._models if self._has_env_auth(m.provider)]

    def _has_env_auth(self, provider: str) -> bool:
        """Check if a provider has auth via environment variable."""
        env_map = {
            "anthropic": ["ANTHROPIC_API_KEY"],
            "anthropic-compatible": ["ANTHROPIC_COMPATIBLE_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "openai-compatible": ["OPENAI_COMPATIBLE_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "bedrock": ["AWS_ACCESS_KEY_ID"],
            "vertex": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
        }
        keys = env_map.get(provider.lower(), [])
        return any(os.environ.get(k) for k in keys) or bool(
            self._custom_provider_api_keys.get(provider)
        )

    def get_api_key(self, provider: str) -> str | None:
        """
        Resolve API key for a provider.
        Checks custom config first, then auth_storage, then env vars.
        """
        # Custom key from models.json
        key_config = self._custom_provider_api_keys.get(provider)
        if key_config:
            return _resolve_config_value(key_config)

        # Auth storage. Use the canonical resolver so subscription/OAuth tokens
        # take precedence over stored or environment API keys.
        if self._auth_storage:
            if hasattr(self._auth_storage, "resolve_api_key"):
                return self._auth_storage.resolve_api_key(provider)
            if hasattr(self._auth_storage, "get_api_key"):
                return self._auth_storage.get_api_key(provider)
            cred = self._auth_storage.get(provider) if hasattr(self._auth_storage, "get") else None
            if cred and hasattr(cred, "api_key"):
                return cred.api_key

        # Environment variables — delegate to the canonical resolver in pi_ai
        from pi_ai.env_api_keys import get_env_api_key
        return get_env_api_key(provider)

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        """Resolve API key for a provider. Mirrors getApiKeyForProvider()."""
        return self.get_api_key(provider)

    def resolve_headers(self, model: Model) -> dict[str, str] | None:
        """
        Resolve headers for a model, interpolating env vars and shell commands.
        Mirrors resolveHeaders() in TypeScript.
        """
        if not model.headers:
            return None
        return _resolve_headers(model.headers)

    async def get_api_key_and_headers(self, model: Model) -> dict[str, Any]:
        """Return Node-style request auth for a model."""
        try:
            api_key = self.get_api_key(model.provider)
            headers = dict(model.headers or {})
            resolved_headers = _resolve_headers(headers) if headers else None
            if resolved_headers is not None:
                headers = resolved_headers

            provider_config = self._registered_providers.get(model.provider) or self._config_providers.get(model.provider)
            if provider_config:
                provider_headers = _resolve_headers(provider_config.headers) or {}
                headers = {**headers, **provider_headers}
                if provider_config.auth_header:
                    if not api_key:
                        return {"ok": False, "error": f'No API key found for "{model.provider}"'}
                    headers["Authorization"] = f"Bearer {api_key}"

            return {
                "ok": True,
                "apiKey": api_key,
                "headers": headers or None,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def has_configured_auth(self, model: Model | str) -> bool:
        """Fast auth availability check. Mirrors hasConfiguredAuth()."""
        provider = model if isinstance(model, str) else model.provider
        return self._has_env_auth(provider) or bool(self.get_api_key(provider))

    def get_provider_auth_status(self, provider: str) -> dict[str, Any]:
        """Return auth status including models.json/runtime provider API keys."""
        if self._auth_storage and hasattr(self._auth_storage, "get_auth_status"):
            status = self._auth_storage.get_auth_status(provider)
            if status.get("source"):
                return status
        else:
            status = {"configured": False}

        provider_key = self._custom_provider_api_keys.get(provider)
        if not provider_key:
            return status

        stripped = provider_key.strip()
        if stripped.startswith("$(") and stripped.endswith(")"):
            return {"configured": True, "source": "models_json_command"}
        env_match = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$", stripped)
        if env_match:
            env_name = env_match.group(1)
            return (
                {"configured": True, "source": "environment", "label": env_name}
                if os.environ.get(env_name)
                else {"configured": False}
            )
        return {"configured": True, "source": "models_json_key"}

    def get_provider_display_name(self, provider: str) -> str:
        """Get display name for a provider."""
        registered = self._registered_providers.get(provider)
        if registered and registered.name:
            return registered.name
        configured = self._config_providers.get(provider)
        if configured and configured.name:
            return configured.name
        if self._auth_storage and hasattr(self._auth_storage, "get_oauth_providers"):
            for oauth_provider in self._auth_storage.get_oauth_providers():
                if getattr(oauth_provider, "id", None) == provider and getattr(oauth_provider, "name", None):
                    return oauth_provider.name
        from pi_coding_agent.core.provider_display_names import get_provider_display_name
        return get_provider_display_name(provider)

    def is_using_oauth(self, model: Model) -> bool:
        """Check if a model is using OAuth credentials."""
        if self._auth_storage and hasattr(self._auth_storage, "is_using_oauth"):
            return bool(self._auth_storage.is_using_oauth(model.provider))
        cred = self._auth_storage.get(model.provider) if self._auth_storage and hasattr(self._auth_storage, "get") else None
        return bool(cred and cred.get("type") == "oauth")

    def register_provider(self, name: str, config: dict[str, Any]) -> None:
        """
        Register a provider from an extension.
        Mirrors registerProvider() in TypeScript.
        """
        prov = ProviderConfig(
            name=config.get("name") or name,
            base_url=config.get("baseUrl"),
            api_key=config.get("apiKey"),
            api=config.get("api"),
            headers=config.get("headers"),
            auth_header=config.get("authHeader", False),
            models=config.get("models") or [],
            model_overrides=config.get("modelOverrides") or {},
        )
        self._registered_providers[name] = prov

        if prov.api_key:
            self._custom_provider_api_keys[name] = prov.api_key

        # Re-merge with new provider
        new_models = self._apply_provider_config_to_models(self._models, name, prov)
        self._models = new_models

    def unregister_provider(self, name: str) -> None:
        """Remove a runtime-registered provider and rebuild available models."""
        self._registered_providers.pop(name, None)
        self._custom_provider_api_keys.pop(name, None)
        self._load_models()

    def reset_registered_providers(self) -> None:
        """Clear runtime provider registrations and rebuild available models."""
        self._registered_providers.clear()
        self._load_models()

    def register_model(self, model: Model) -> None:
        """Register an individual extra model."""
        self._extra_models.append(model)
        self._models.append(model)

    def get_model(self, provider: str, model_id: str) -> Model:
        """Get a model by provider and ID."""
        oauth_model = self._oauth_backed_model(provider, model_id)
        if oauth_model is not None:
            return oauth_model
        config_loaded = self._find_config_loaded_model(provider, model_id)
        if config_loaded is not None:
            return config_loaded
        configured = self._synthetic_config_model(provider, model_id)
        if configured is not None:
            return configured
        for m in self._models:
            if m.provider == provider and m.id == model_id:
                return m
        model = get_model(provider, model_id)
        if model is not None:
            return model
        synthetic = synthetic_model(provider, model_id)
        if synthetic is not None:
            return synthetic
        raise RuntimeError(f"Unknown model {provider}/{model_id}")

    def find(self, provider: str, model_id: str) -> Model | None:
        """Find a model or return None."""
        oauth_model = self._oauth_backed_model(provider, model_id)
        if oauth_model is not None:
            return oauth_model
        config_loaded = self._find_config_loaded_model(provider, model_id)
        if config_loaded is not None:
            return config_loaded
        configured = self._synthetic_config_model(provider, model_id)
        if configured is not None:
            return configured
        for m in self._models:
            if m.provider == provider and m.id == model_id:
                return m
        try:
            model = get_model(provider, model_id)
            if model is not None:
                return model
        except Exception:
            pass
        synthetic = synthetic_model(provider, model_id)
        if synthetic is not None:
            return synthetic
        return None

    def _oauth_backed_model(self, provider: str, model_id: str) -> Model | None:
        if provider != "openai" or not self._has_oauth_token("openai"):
            return None
        model = get_model("openai-codex", model_id)
        if model is None:
            return None
        return Model(**{**model.__dict__, "provider": "openai"})

    def _has_oauth_token(self, provider: str) -> bool:
        if not self._auth_storage:
            return False
        get_token = getattr(self._auth_storage, "get_oauth_token", None)
        return bool(callable(get_token) and get_token(provider))

    def _apply_oauth_transport_overrides(self, models: list[Model]) -> list[Model]:
        if not self._has_oauth_token("openai"):
            return models
        overridden: list[Model] = []
        for model in models:
            if model.provider != "openai":
                overridden.append(model)
                continue
            oauth_model = self._oauth_backed_model("openai", model.id)
            overridden.append(oauth_model or model)
        return overridden

    def _find_config_loaded_model(self, provider: str, model_id: str) -> Model | None:
        config = self._config_providers.get(provider) or self._registered_providers.get(provider)
        if config is None:
            return None
        explicit_model_ids = {model.get("id") for model in config.models if isinstance(model, dict)}
        if model_id not in explicit_model_ids:
            return None
        for m in self._models:
            if m.provider != provider or m.id != model_id:
                continue
            if config.base_url and getattr(m, "base_url", None) == config.base_url:
                return m
            if config.api and getattr(m, "api", None) == config.api and getattr(m, "base_url", None):
                return m
        return None

    def _synthetic_config_model(self, provider: str, model_id: str) -> Model | None:
        config = self._config_providers.get(provider) or self._registered_providers.get(provider)
        if config is None or not config.api:
            return None
        return Model(
            id=model_id,
            name=model_id,
            api=config.api,
            provider=provider,
            base_url=config.base_url,
            headers=config.headers,
            reasoning=config.api in {"openai-responses", "anthropic-messages"},
            input=["text", "image"],
            cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            context_window=128000,
            max_tokens=8192,
        )

    def get_providers(self) -> list[str]:
        """Get all available providers."""
        return list({m.provider for m in self._models} | set(self._config_providers))

    def resolve_model(
        self,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> Model:
        """Resolve a model by optional ID and provider."""
        if model_id and provider:
            m = self.find(provider, model_id)
            if m:
                return m

        if model_id:
            for m in self._models:
                if m.id == model_id:
                    return m

        # Auto-select default based on which providers have API keys configured.
        # Prefer OpenAI (gpt-5.5) when OPENAI_API_KEY is present, then Anthropic,
        # then Google, then whatever is available. gpt-5.5 is the default when no
        # global default model is set.
        _default_preference = [
            ("openai", "gpt-5.5"),
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("google", "gemini-2.5-pro"),
        ]
        for prov, mid in _default_preference:
            if self._has_env_auth(prov):
                m = self.find(prov, mid)
                if m:
                    return m

        # Last resort: default to OpenAI gpt-5.5 (never a non-deterministic
        # first-in-catalog pick). If that model isn't in the catalog, fall back
        # to whatever is available.
        m = self.find("openai", "gpt-5.5")
        if m:
            return m
        if self._models:
            return self._models[0]
        raise RuntimeError("No models available")

    getAll = get_all
    getAvailable = get_available
    getError = get_error
    getApiKey = get_api_key
    getApiKeyForProvider = get_api_key_for_provider
    getApiKeyAndHeaders = get_api_key_and_headers
    hasConfiguredAuth = has_configured_auth
    getProviderAuthStatus = get_provider_auth_status
    getProviderDisplayName = get_provider_display_name
    isUsingOAuth = is_using_oauth
    registerProvider = register_provider
    unregisterProvider = unregister_provider
    resetApiProviders = reset_registered_providers
