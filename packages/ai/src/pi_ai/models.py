"""
Model registry and utilities — mirrors packages/ai/src/models.ts
"""
from __future__ import annotations

from .models_generated import MODELS
from .types import Model, Usage, UsageCost


def get_model(provider: str, model_id: str) -> Model | None:
    """Get a model by provider and model ID. Returns None if not found."""
    key = f"{provider}/{model_id}"
    return MODELS.get(key)


def get_providers() -> list[str]:
    """Return list of all registered providers."""
    seen: set[str] = set()
    result: list[str] = []
    for model in MODELS.values():
        if model.provider not in seen:
            seen.add(model.provider)
            result.append(model.provider)
    return sorted(result)


def get_models(provider: str | None = None) -> list[Model]:
    """Return all models, optionally filtered by provider."""
    models = list(MODELS.values())
    if provider is not None:
        models = [m for m in models if m.provider == provider]
    return models


def calculate_cost(model: Model, usage: Usage) -> float:
    """Calculate total cost in USD from usage and model pricing. Also mutates usage.cost."""
    input_cost = usage.input / 1_000_000 * model.cost.input
    output_cost = usage.output / 1_000_000 * model.cost.output
    cache_read_cost = usage.cache_read / 1_000_000 * model.cost.cache_read
    cache_write_cost = usage.cache_write / 1_000_000 * model.cost.cache_write
    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    usage.cost = UsageCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        cache_write=cache_write_cost,
        total=total,
    )
    return total


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports xhigh reasoning."""
    if model.provider == "openai" and model.id == "gpt-6-astra":
        return True
    if any(version in model.id for version in ("gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.2")):
        return True
    if model.api == "anthropic-messages":
        return "opus-4-6" in model.id or "opus-4.6" in model.id
    return False


def supports_adaptive(model: Model | None) -> bool:
    """Check if a model exposes Anthropic adaptive thinking."""
    if model is None or model.api != "anthropic-messages":
        return False
    model_id = model.id.lower()
    return any(
        family in model_id
        for family in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")
    )


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check if two models are equal by comparing both id and provider."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
