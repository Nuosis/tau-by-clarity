"""Ask Jev whether a new user input belongs to the active intention."""
from __future__ import annotations

import hashlib
import os

import aiohttp

from .turn_review import Intention

JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
# Jev 1.13, per Marcus 2026-10-17: ~typesafe/jev-latest was never authorized.
JEV_MODEL = "typesafe/jev-1.13"
ALIGNMENT_QUESTION = {
    "type": "choice",
    "instructions": (
        "Can `latest_input` still be considered part of `active_intention`? "
        "Judge the requested outcome in context, including clarification, correction, "
        "implementation, and verification of the same work. Choose a new intention "
        "only when the input asks for an independent or replacing outcome."
    ),
    "criteria": {
        "continues_active_intention": (
            "The input develops, clarifies, corrects, implements, or verifies the active work."
        ),
        "starts_new_intention": (
            "The input requests a separate outcome or replaces the active work."
        ),
    },
}


async def judge_intention_alignment(
    active_intention: Intention,
    latest_input: str,
    *,
    get_api_key,
    record,
    endpoint: str = JEV_ENDPOINT,
    route=None,
) -> str:
    """Return Jev's typed choice; record the decision without duplicating user text."""
    # Same Jev route as the churn guard: TAU_JEV_ENDPOINT (a credentialed proxy), else the
    # configured OpenRouter provider route and its key.
    # An override endpoint carries its own credential; route headers stay with their own route.
    override = os.environ.get("TAU_JEV_ENDPOINT")
    route_endpoint, route_headers = route() if (route is not None and not override) else (endpoint, {})
    endpoint = override or route_endpoint
    headers = {**(route_headers or {}), "Content-Type": "application/json"}
    if not os.environ.get("TAU_JEV_ENDPOINT") and not any(k.lower() == "authorization" for k in headers):
        key = await get_api_key("openrouter")
        if not key:
            raise RuntimeError("OpenRouter API key is required for Jev intention alignment")
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": JEV_MODEL,
        "state": {
            "active_intention": active_intention.model_dump(),
            "latest_input": latest_input,
        },
        "questions": {"alignment": ALIGNMENT_QUESTION},
    }
    digest = hashlib.sha256(latest_input.encode("utf-8")).hexdigest()
    record("tau.intention_alignment.started", metadata={
        "model": JEV_MODEL, "latest_input_sha256": digest,
    })
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=15, sock_read=None)
        ) as client:
            async with client.post(endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        answer = result["answers"]["alignment"]
        choice = answer["choice"]
        if answer.get("type") != "choice" or choice not in ALIGNMENT_QUESTION["criteria"]:
            raise ValueError("Jev returned an invalid intention alignment choice")
        record("tau.intention_alignment.completed", metadata={
            "model": result.get("model", JEV_MODEL),
            "latest_input_sha256": digest,
            "choice": choice,
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
        })
        return choice
    except Exception as exc:
        record("tau.intention_alignment.failed", metadata={
            "model": JEV_MODEL, "latest_input_sha256": digest,
            "error": type(exc).__name__,
        })
        raise
