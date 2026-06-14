"""Mid-loop steering.

A user can submit a message while the agent is working. It is STAGED (and
acknowledged immediately, so it never silently disappears) and drained by the
driver at the next iteration boundary, where it is injected into the agent's next
prompt — instead of waiting for the entire outer loop to finish, which can be a
very long time.
"""

from __future__ import annotations

import threading


class SteeringInbox:
    """Thread-safe staging buffer for steering messages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._staged: list[str] = []

    def submit(self, message: str) -> str:
        """Stage a steering message. Returns an ack string (empty if blank)."""
        msg = (message or "").strip()
        if not msg:
            return ""
        with self._lock:
            self._staged.append(msg)
            n = len(self._staged)
        return f"✎ staged — will steer the agent at the next iteration ({n} pending)"

    def drain(self) -> list[str]:
        """Return and clear all staged messages."""
        with self._lock:
            out = self._staged[:]
            self._staged.clear()
        return out

    def pending(self) -> int:
        with self._lock:
            return len(self._staged)
