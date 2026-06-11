"""Countdown timer state for dialog components."""
from __future__ import annotations

import math
from typing import Callable


class CountdownTimer:
    def __init__(
        self,
        timeout_ms: int,
        on_tick: Callable[[int], None] | None = None,
        on_expire: Callable[[], None] | None = None,
    ) -> None:
        self.remaining_seconds = max(0, math.ceil(timeout_ms / 1000))
        self.on_tick = on_tick or (lambda seconds: None)
        self.on_expire = on_expire or (lambda: None)
        self.disposed = False
        self.on_tick(self.remaining_seconds)

    def tick(self, seconds: int = 1) -> None:
        if self.disposed:
            return
        self.remaining_seconds = max(0, self.remaining_seconds - seconds)
        self.on_tick(self.remaining_seconds)
        if self.remaining_seconds <= 0:
            self.dispose()
            self.on_expire()

    def dispose(self) -> None:
        self.disposed = True
