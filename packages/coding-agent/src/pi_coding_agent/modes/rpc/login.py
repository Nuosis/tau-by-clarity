"""Correlated, headless presentation for Tau's native provider login."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from pi_coding_agent.core.login import subscription_login
from pi_coding_agent.core.provider_profiles import PROVIDER_PROFILES, get_provider_profile

_RPC_PROVIDERS = tuple(
    profile for profile in PROVIDER_PROFILES
    if profile.id in {"openai", "anthropic", "google"}
)


class RpcLoginController:
    """Own one login exchange for one RPC session at a time."""

    def __init__(
        self,
        output: Callable[[dict[str, Any]], None],
        *,
        subscription_runner: Callable[..., Any] = subscription_login,
    ) -> None:
        self.output = output
        self.subscription_runner = subscription_runner
        self.login_id: str | None = None
        self.session_id: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._request_id: str | None = None
        self._response: asyncio.Future[str | None] | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, session: Any, *, provider: str | None = None, method: str | None = None) -> dict[str, str]:
        if self.active:
            raise ValueError("A login is already active for this RPC session")
        self.login_id = str(uuid.uuid4())
        self.session_id = str(session.session_id)
        self._task = asyncio.create_task(self._run(session, provider=provider, method=method))
        return {"loginId": self.login_id, "sessionId": self.session_id}

    def respond(self, *, login_id: str, request_id: str, value: Any = None, cancelled: bool = False) -> None:
        self._require_flow(login_id)
        if request_id != self._request_id or self._response is None or self._response.done():
            raise ValueError("Login response does not match the active request")
        self._response.set_result(None if cancelled else str(value or ""))

    def cancel(self, *, login_id: str) -> None:
        self._require_flow(login_id)
        if self._response is not None and not self._response.done():
            self._response.set_result(None)
        elif self._task is not None:
            self._task.cancel()

    def _require_flow(self, login_id: str) -> None:
        if not self.active or login_id != self.login_id:
            raise ValueError("Login flow is absent or belongs to another RPC session")

    def _emit(self, event: str, **data: Any) -> None:
        self.output({
            "type": "login_event",
            "event": event,
            "loginId": self.login_id,
            "sessionId": self.session_id,
            **data,
        })

    async def _ask(
        self,
        *,
        kind: str,
        message: str,
        options: list[dict[str, str]] | None = None,
        sensitive: bool = False,
        allow_empty: bool = False,
    ) -> str:
        request_id = str(uuid.uuid4())
        response: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._request_id = request_id
        self._response = response
        self._emit(
            "request",
            requestId=request_id,
            requestKind=kind,
            message=message,
            options=options or [],
            sensitive=sensitive,
        )
        value = await response
        self._request_id = None
        self._response = None
        if value is None:
            raise asyncio.CancelledError
        if not allow_empty and not value.strip():
            raise ValueError("A value is required")
        return value

    async def _run(self, session: Any, *, provider: str | None, method: str | None) -> None:
        try:
            self._emit("started", message="Choose a provider to sign in to Tau.")
            if provider is None:
                provider = await self._ask(
                    kind="select",
                    message="Provider",
                    options=[{"value": p.id, "label": p.label} for p in _RPC_PROVIDERS],
                )
            profile = get_provider_profile(provider)
            if profile not in _RPC_PROVIDERS:
                raise ValueError(f"Provider is not available for RPC login: {provider}")
            if method is None:
                if len(profile.auth_methods) == 1:
                    method = profile.auth_methods[0]
                else:
                    labels = {"subscription": "Subscription", "api_key": "API key"}
                    method = await self._ask(
                        kind="select",
                        message="Authentication method",
                        options=[{"value": item, "label": labels[item]} for item in profile.auth_methods],
                    )
            if method not in profile.auth_methods:
                raise ValueError(f"{profile.label} does not support {method} login")

            if method == "subscription":
                def on_auth(info: Any) -> None:
                    self._emit(
                        "authorization",
                        message=str(getattr(info, "instructions", None) or "Open this link to authorize Tau."),
                        url=str(info.url),
                    )

                async def on_prompt(prompt: Any) -> str:
                    return await self._ask(
                        kind="input",
                        message=str(prompt.message),
                        sensitive=True,
                        allow_empty=bool(getattr(prompt, "allow_empty", False)),
                    )

                await self.subscription_runner(
                    profile.id,
                    session,
                    on_auth=on_auth,
                    on_prompt=on_prompt,
                    on_progress=lambda message: self._emit("progress", message=str(message)),
                )
            else:
                key = await self._ask(
                    kind="input",
                    message=f"{profile.label} API key",
                    sensitive=True,
                )
                session.login_api_key(profile.id, key)

            settings = getattr(session, "settings_manager", None)
            save_project = getattr(settings, "save_project", None)
            if callable(save_project):
                save_project("defaultProvider", profile.id)
            self._emit("completed", message=f"Login stored for {profile.label}.", provider=profile.id)
        except asyncio.CancelledError:
            self._emit("cancelled", message="Login cancelled.")
        except Exception as exc:
            self._emit("failed", message=f"Login failed: {exc}")
        finally:
            self._request_id = None
            self._response = None


__all__ = ["RpcLoginController"]
