#!/usr/bin/env python3
"""Validation: the universal PII filter captures PII on EVERY call, any source.

Imports pi_coding_agent (which self-registers the real Clarity PII filter), then
drives complete_simple through a local fake provider from several "sources" with
several PII types, asserting: real PII never reaches the provider (wire is
tokenized) and the caller transparently gets cleartext back.

    python evals/pii_coverage_check.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages", "coding-agent", "src"))

import pi_coding_agent  # noqa: F401,E402  (self-registers the real filter)
import pi_ai  # noqa: E402
from pi_ai.api_registry import register_api_provider  # noqa: E402
from pi_ai.types import (  # noqa: E402
    AssistantMessage,
    Context,
    EventDone,
    Model,
    TextContent,
    UserMessage,
)

PII_SAMPLES = {
    "email": "jane.doe@acme.example",
    "ssn": "123-45-6789",
    "phone": "415-555-0199",
    "credit_card": "4111 1111 1111 1111",
    "ip": "192.168.14.22",
}
_TOKEN_RE = re.compile(r"\[PII:[A-Z_]+:\d+\]")


class _Wire:
    def __init__(self):
        self.seen = ""

    async def stream_simple(self, model, context, options):
        self.seen = context.messages[-1].content
        yield EventDone(reason="stop", message=AssistantMessage(
            content=[TextContent(text=f"ack {context.messages[-1].content}")],
            api=model.api, provider=model.provider, model=model.id, timestamp=0))

    async def stream(self, m, c, o):
        async for e in self.stream_simple(m, c, o):
            yield e


def main() -> int:
    wire = _Wire()
    register_api_provider("pii-check", wire, source_id="piichk")
    model = Model(id="m", name="m", api="pii-check", provider="fake", base_url="http://x")
    print("PII coverage — universal filter:", "registered" if pi_ai.has_pii_filter() else "MISSING")
    print("-" * 64)

    fails = []
    for label, value in PII_SAMPLES.items():
        prompt = f"my {label} is {value}, store it"
        ctx = Context(messages=[UserMessage(content=prompt, timestamp=0)])
        reply = asyncio.run(pi_ai.complete_simple(model, ctx))
        on_wire = wire.seen
        leaked = value in on_wire
        tokenized = bool(_TOKEN_RE.search(on_wire))
        restored = value in "".join(b.text for b in reply.content if b.type == "text")
        ok = (not leaked) and tokenized and restored
        print(f"  {label:<12} wire-tokenized: {tokenized!s:<5} leaked: {leaked!s:<5} "
              f"reply-restored: {restored!s:<5} {'✓' if ok else '✗'}")
        if not ok:
            fails.append(label)

    print("-" * 64)
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
