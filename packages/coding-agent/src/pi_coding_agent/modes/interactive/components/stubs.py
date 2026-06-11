"""Interactive component stub ledger."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentStub:
    name: str


_NAMES: list[str] = []

STUB_COMPONENTS: dict[str, ComponentStub] = {name: ComponentStub(name=name) for name in _NAMES}
