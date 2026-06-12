"""Built-in provider display names."""
from __future__ import annotations

BUILT_IN_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "amazon-bedrock": "Amazon Bedrock",
    "ant-ling": "Ant Ling",
    "azure-openai-responses": "Azure OpenAI Responses",
    "cerebras": "Cerebras",
    "cloudflare-ai-gateway": "Cloudflare AI Gateway",
    "cloudflare-workers-ai": "Cloudflare Workers AI",
    "deepseek": "DeepSeek",
    "fireworks": "Fireworks",
    "google": "Google Gemini",
    "google-vertex": "Google Vertex AI",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "kimi-coding": "Kimi For Coding",
    "mistral": "Mistral",
    "minimax": "MiniMax",
    "minimax-cn": "MiniMax (China)",
    "moonshotai": "Moonshot AI",
    "moonshotai-cn": "Moonshot AI (China)",
    "nvidia": "NVIDIA NIM",
    "opencode": "OpenCode Zen",
    "opencode-go": "OpenCode Go",
    "openai": "OpenAI",
    "openai-compatible": "OpenAI Compatible",
    "openrouter": "OpenRouter",
    "together": "Together AI",
    "vercel-ai-gateway": "Vercel AI Gateway",
    "xai": "xAI",
    "zai": "ZAI",
    "zai-coding-cn": "ZAI Coding Plan (China)",
    "xiaomi": "Xiaomi MiMo",
    "xiaomi-token-plan-cn": "Xiaomi MiMo Token Plan (China)",
    "xiaomi-token-plan-ams": "Xiaomi MiMo Token Plan (Amsterdam)",
    "xiaomi-token-plan-sgp": "Xiaomi MiMo Token Plan (Singapore)",
    "anthropic-compatible": "Anthropic Compatible",
}


def get_provider_display_name(provider: str) -> str:
    return BUILT_IN_PROVIDER_DISPLAY_NAMES.get(provider, provider)


__all__ = ["BUILT_IN_PROVIDER_DISPLAY_NAMES", "get_provider_display_name"]
