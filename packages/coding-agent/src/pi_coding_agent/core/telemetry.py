"""Telemetry setting helpers."""
from __future__ import annotations

import os
from typing import Any


def is_truthy_env_flag(value: str | None) -> bool:
    return bool(value) and value in {"1", "true", "yes", "TRUE", "True", "YES", "Yes"}


def _get_setting(settings_manager: Any, name: str, default: bool = False) -> bool:
    getter_name = "".join(part.capitalize() for part in name.split("_"))
    for candidate in (
        f"get_{name}",
        f"get{name[0].upper()}{name[1:]}",
        f"get{getter_name}",
    ):
        method = getattr(settings_manager, candidate, None)
        if callable(method):
            return bool(method())
    if isinstance(settings_manager, dict):
        return bool(settings_manager.get(name, settings_manager.get("enableInstallTelemetry", default)))
    return bool(getattr(settings_manager, name, getattr(settings_manager, "enableInstallTelemetry", default)))


def is_install_telemetry_enabled(settings_manager: Any, telemetry_env: str | None = None) -> bool:
    env_value = os.environ.get("PI_TELEMETRY") if telemetry_env is None else telemetry_env
    if env_value is not None:
        return is_truthy_env_flag(env_value)
    return _get_setting(settings_manager, "enable_install_telemetry", False)


__all__ = ["is_install_telemetry_enabled", "is_truthy_env_flag"]
