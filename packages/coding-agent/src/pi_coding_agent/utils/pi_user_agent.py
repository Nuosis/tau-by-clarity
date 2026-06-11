"""Pi user-agent helper."""
from __future__ import annotations

import platform
import sys


def get_pi_user_agent(version: str) -> str:
    runtime = f"python/{platform.python_version()}"
    return f"pi/{version} ({sys.platform}; {runtime}; {platform.machine()})"


__all__ = ["get_pi_user_agent"]
