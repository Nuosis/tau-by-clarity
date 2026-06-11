"""Auth selection and login dialog components."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from .selectors import fuzzy_filter
from .text_input import TextInput


@dataclass
class AuthSelectorProvider:
    id: str
    name: str
    auth_type: str


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class OAuthSelectorComponent:
    def __init__(
        self,
        mode: str,
        auth_storage: Any,
        providers: list[Any],
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        get_auth_status: Callable[[str], Any] | None = None,
    ) -> None:
        self.mode = mode
        self.auth_storage = auth_storage
        self.all_providers = providers
        self.filtered_providers = list(providers)
        self.selected_index = 0
        self.on_select = on_select or (lambda provider_id: None)
        self.on_cancel = on_cancel or (lambda: None)
        self.get_auth_status = get_auth_status or (lambda provider_id: {"source": "none"})
        self.search_input = TextInput(on_submit=lambda _: self._submit())

    def _provider_text(self, provider: Any) -> str:
        return " ".join(
            str(_get_attr_or_key(provider, key, ""))
            for key in ("name", "id", "auth_type", "authType")
        )

    def filter_providers(self, query: str) -> None:
        self.search_input.set_value(query)
        self.filtered_providers = (
            fuzzy_filter(self.all_providers, query, self._provider_text)
            if query.strip()
            else list(self.all_providers)
        )
        self.selected_index = max(0, min(self.selected_index, max(0, len(self.filtered_providers) - 1)))

    def selected_provider(self) -> Any | None:
        if not self.filtered_providers:
            return None
        return self.filtered_providers[self.selected_index]

    def _submit(self) -> None:
        provider = self.selected_provider()
        if provider is not None:
            self.on_select(str(_get_attr_or_key(provider, "id")))

    def handle_input(self, key_data: str) -> None:
        if key_data in {"up", "k", "\x1b[A"} and self.filtered_providers:
            self.selected_index = max(0, self.selected_index - 1)
        elif key_data in {"down", "j", "\x1b[B"} and self.filtered_providers:
            self.selected_index = min(len(self.filtered_providers) - 1, self.selected_index + 1)
        elif key_data in {"\n", "enter", "return"}:
            self._submit()
        elif key_data in {"escape", "esc", "\x1b"}:
            self.on_cancel()
        else:
            self.search_input.handle_input(key_data)
            self.filter_providers(self.search_input.get_value())

    def format_status_indicator(self, provider: Any) -> str:
        provider_id = str(_get_attr_or_key(provider, "id"))
        provider_auth_type = str(_get_attr_or_key(provider, "auth_type", _get_attr_or_key(provider, "authType", "")))
        credential = self.auth_storage.get(provider_id) if self.auth_storage is not None else None
        credential_type = _get_attr_or_key(credential, "type")
        if credential_type == provider_auth_type:
            return "configured"
        if credential is not None:
            return "subscription configured" if credential_type == "oauth" else "API key configured"
        if provider_auth_type != "api_key":
            return "unconfigured"
        status = self.get_auth_status(provider_id)
        source = _get_attr_or_key(status, "source", "none")
        return {
            "environment": "env",
            "runtime": "runtime API key",
            "fallback": "custom API key",
            "models_json_key": "key in models.json",
            "models_json_command": "command in models.json",
        }.get(source, "unconfigured")

    def render(self) -> list[str]:
        title = "Select provider to configure:" if self.mode == "login" else "Select provider to logout:"
        lines = [title, *self.search_input.render()]
        if not self.filtered_providers:
            lines.append("No providers available" if not self.all_providers else "No matching providers")
            return lines
        for idx, provider in enumerate(self.filtered_providers[:8]):
            prefix = "→ " if idx == self.selected_index else "  "
            lines.append(f"{prefix}{_get_attr_or_key(provider, 'name')} - {self.format_status_indicator(provider)}")
        return lines


class LoginDialogComponent:
    def __init__(
        self,
        provider_id: str,
        on_complete: Callable[[bool, str | None], None] | None = None,
        provider_name_override: str | None = None,
        title_override: str | None = None,
        tui: Any | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.provider_name = provider_name_override or provider_id
        self.title = title_override or f"Login to {self.provider_name}"
        self.on_complete = on_complete or (lambda success, message=None: None)
        self.tui = tui
        self.cancelled = False
        self.content_lines: list[str] = []
        self.input = TextInput(on_submit=self._resolve_input, on_escape=self.cancel)
        self._future: asyncio.Future[str] | None = None

    @property
    def signal(self) -> dict[str, bool]:
        return {"aborted": self.cancelled}

    def _request_render(self) -> None:
        if self.tui is not None and hasattr(self.tui, "request_render"):
            self.tui.request_render()

    def cancel(self) -> None:
        self.cancelled = True
        if self._future is not None and not self._future.done():
            self._future.set_exception(RuntimeError("Login cancelled"))
        self.on_complete(False, "Login cancelled")

    def _resolve_input(self, value: str) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(value)
            self._future = None

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        self.content_lines = [url]
        if instructions:
            self.content_lines.append(instructions)
        self._request_render()

    def show_device_code(self, info: Any) -> None:
        verification_uri = _get_attr_or_key(info, "verification_uri", _get_attr_or_key(info, "verificationUri"))
        user_code = _get_attr_or_key(info, "user_code", _get_attr_or_key(info, "userCode"))
        self.content_lines = [str(verification_uri), f"Enter code: {user_code}"]
        self._request_render()

    async def show_manual_input(self, prompt: str) -> str:
        return await self.show_prompt(prompt)

    async def show_prompt(self, message: str, placeholder: str | None = None) -> str:
        self.content_lines.append(message)
        if placeholder:
            self.content_lines.append(f"e.g., {placeholder}")
        self.input.set_value("")
        self._future = asyncio.get_running_loop().create_future()
        self._request_render()
        return await self._future

    def show_info(self, lines: list[str]) -> None:
        self.content_lines = list(lines)
        self._request_render()

    def show_waiting(self, message: str) -> None:
        self.content_lines.append(message)
        self._request_render()

    def show_progress(self, message: str) -> None:
        self.content_lines.append(message)
        self._request_render()

    def handle_input(self, key_data: str) -> None:
        if key_data in {"escape", "esc", "\x1b"}:
            self.cancel()
        else:
            self.input.handle_input(key_data)

    def render(self) -> list[str]:
        return [self.title, *self.content_lines, *self.input.render()]
