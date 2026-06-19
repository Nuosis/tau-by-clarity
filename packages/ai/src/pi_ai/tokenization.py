from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def _encoding_for_model(model: str):
    import tiktoken

    if model.startswith("gpt-4o"):
        return tiktoken.get_encoding("o200k_base")
    if model.startswith(("gpt-4", "gpt-3.5", "text-embedding")):
        return tiktoken.get_encoding("cl100k_base")
    return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str, *, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    try:
        return len(_encoding_for_model(model).encode(text))
    except Exception:
        return max(1, len(text) // 4)
