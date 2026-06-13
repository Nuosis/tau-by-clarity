"""Genuine LLM access for the loop's judges and generators.

Provider/model are NOT hardcoded: we read the TARGET AGENT's configured
`defaultProvider`/`defaultModel` from its `.pi-py/settings.json` and resolve the
model — wire (`api`), base_url, limits — through `pi_ai.get_model`, the exact
registry the agent runs on. The call goes through `pi_ai.complete_simple`, so the
loop's judges run on the same model the agent does, with the correct transport
selected automatically. The key comes from `AuthStorage` (the same decryptor the
agent uses) or the standard `*_API_KEY` env var.

HARD RULE: there is NO deterministic stub in this module. `call_json` retries the
model on a parse miss; it never fabricates a verdict.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

LlmFn = Callable[[str, str], str]

# Last-resort fallbacks ONLY if settings.json is unreadable; normal operation
# always uses the agent's configured values.
_FALLBACK_PROVIDER = "minimax"
_FALLBACK_MODEL = "MiniMax-M3"

T = TypeVar("T", bound=BaseModel)


def _settings(agent_dir: str) -> dict:
    try:
        return json.loads(Path(os.path.join(agent_dir, ".pi-py", "settings.json")).read_text())
    except Exception:
        return {}


def _resolve_key(provider: str) -> str:
    env = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("PI_LOOP_API_KEY")
    if env:
        return env.strip()
    try:
        from pi_coding_agent.core.auth_storage import AuthStorage

        key = AuthStorage().resolve_api_key(provider)
        if key:
            return key.strip()
    except Exception:
        pass
    raise RuntimeError(
        f"No API key for provider {provider!r}: set {provider.upper()}_API_KEY "
        f"or log in so AuthStorage can resolve it."
    )


def resolve_llm(agent_dir: str, *, model: str | None = None) -> LlmFn:
    """Return `llm_fn(system, user) -> str`, a real call on the agent's model.

    `model` overrides the configured `defaultModel` only when explicitly passed.
    """
    from pi_ai import Context, complete_simple, get_model
    from pi_ai.types import SimpleStreamOptions, UserMessage

    settings = _settings(agent_dir)
    provider = (settings.get("defaultProvider") or _FALLBACK_PROVIDER).lower()
    model_id = model or settings.get("defaultModel") or _FALLBACK_MODEL

    resolved = get_model(provider, model_id)
    if resolved is None:
        raise RuntimeError(
            f"Configured model {provider}/{model_id} not found in the pi_ai registry."
        )
    key = _resolve_key(provider)

    def llm_fn(system: str, user: str) -> str:
        async def _go() -> str:
            ctx = Context(
                system_prompt=system,
                messages=[UserMessage(content=user, timestamp=int(time.time() * 1000))],
            )
            msg = await complete_simple(resolved, ctx, SimpleStreamOptions(api_key=key))
            return "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            )

        return asyncio.run(_go())

    return llm_fn


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating fences / preamble."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


_JSON_REASK = (
    "\n\nYour previous reply did not parse as the required JSON object. "
    "Reply with ONLY a single valid JSON object matching the requested fields. "
    "No prose, no code fences."
)


def call_json(llm_fn: LlmFn, system: str, user: str, model_cls: Type[T]) -> T:
    """Call the model and validate into `model_cls`.

    On a parse/validation miss, re-ask the model ONCE with a corrective hint.
    If it still fails, raise — never fabricate a result (no deterministic stub).
    """
    raw = llm_fn(system, user)
    parsed = extract_json(raw)
    try:
        return model_cls.model_validate(parsed)
    except ValidationError:
        pass

    raw2 = llm_fn(system + _JSON_REASK, user)
    parsed2 = extract_json(raw2)
    try:
        return model_cls.model_validate(parsed2)
    except ValidationError as e:
        raise RuntimeError(
            f"{model_cls.__name__} did not parse after one retry. "
            f"Last raw reply: {raw2[:500]!r}"
        ) from e
