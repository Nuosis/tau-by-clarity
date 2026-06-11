"""
Custom Provider Extension

Registers an extra model provider at startup. Unlike the Node
custom-provider-anthropic example — which hand-wires the Anthropic SDK — the
Python runtime registers providers *declaratively*: you hand pi.register_provider()
a config dict and the model registry wires the HTTP client for you, using the
`api` field to pick the wire protocol ("openai", "anthropic", …).

This example registers a local OpenAI-compatible server (e.g. LM Studio, vLLM,
Ollama's OpenAI endpoint). Point base_url at your server.

Usage:
1. Copy this file to ~/.pi/agent/extensions/ or your project's .pi/extensions/
2. Start pi and select one of the registered models.

Provider config keys:
    name, baseUrl, apiKey, api, headers, authHeader, models, modelOverrides
Model config keys:
    id, name, api, reasoning, input, cost, contextWindow, maxTokens, headers, compat

Port of examples/extensions/custom-provider-anthropic from the Node reference.
"""

PROVIDER_NAME = "local-openai"
BASE_URL = "http://localhost:1234/v1"


def extension_factory(pi):
    pi.register_provider(
        PROVIDER_NAME,
        {
            "baseUrl": BASE_URL,
            "apiKey": "not-needed-for-local",
            "api": "openai",  # use the OpenAI wire protocol
            "models": [
                {
                    "id": "local-model",
                    "name": "Local Model (OpenAI-compatible)",
                    "contextWindow": 128000,
                    "maxTokens": 16384,
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                },
            ],
        },
    )


# ── Async form ────────────────────────────────────────────────────────────────
# The factory may be async: pi awaits it before startup continues, so providers
# discovered at runtime (e.g. by querying the server's /models endpoint) are
# available immediately. Rename this to `extension_factory` to use it instead.
async def extension_factory_async(pi):
    models = []
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for entry in data.get("data", []):
            models.append(
                {
                    "id": entry["id"],
                    "name": entry["id"],
                    "contextWindow": 128000,
                    "maxTokens": 16384,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                }
            )
    except Exception:
        # Server not running — register nothing rather than crash startup.
        return

    if models:
        pi.register_provider(
            PROVIDER_NAME,
            {"baseUrl": BASE_URL, "apiKey": "not-needed-for-local", "api": "openai", "models": models},
        )
