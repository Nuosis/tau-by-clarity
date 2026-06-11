"""Best-effort platform browser launcher."""
from __future__ import annotations

import platform
import subprocess


def browser_command(target: str, system: str | None = None) -> tuple[str, list[str]]:
    name = (system or platform.system()).lower()
    if name == "darwin":
        return "open", [target]
    if name == "windows":
        return "rundll32", ["url.dll,FileProtocolHandler", target]
    return "xdg-open", [target]


def open_browser(target: str) -> subprocess.Popen | None:
    cmd, args = browser_command(target)
    try:
        return subprocess.Popen([cmd, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return None


__all__ = ["browser_command", "open_browser"]
