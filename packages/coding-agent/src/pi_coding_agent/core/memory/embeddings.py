"""
Embedding providers for memory recall — LOCAL only.

Per design doc §8/§10: embeddings run locally (Ollama nomic-embed-text, 768-d); never
ship project content to a cloud embedder (Anthropic has no embeddings endpoint anyway).
A deterministic hashing provider backs tests so they need no network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from typing import Protocol

EMBED_DIM = 768


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingProvider:
    """Local Ollama embeddings (default model nomic-embed-text, 768-d).

    Raises on connection failure — the caller decides whether to fall back.
    Use ``utils.ollama.embed_with_fallback`` for the auto-degrade path.
    """

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return resp["embeddings"]


class DeterministicEmbeddingProvider:
    """Hashing-based pseudo-embeddings — reproducible, no network. Tests/dev only.

    Bag-of-token hashing into EMBED_DIM buckets, L2-normalized. Good enough that exact
    and near-lexical matches rank above unrelated text; NOT a substitute for real
    semantic similarity (paraphrase will not match)."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in _tokens(t):
                h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def _tokens(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) >= 3]


def embedding_provider_from_env() -> EmbeddingProvider:
    """Ollama by default; deterministic when PI_MEMORY_EMBED=deterministic (tests/CI)."""
    if os.environ.get("PI_MEMORY_EMBED", "").lower() == "deterministic":
        return DeterministicEmbeddingProvider()
    return OllamaEmbeddingProvider(
        model=os.environ.get("PI_MEMORY_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def cosine(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return d / (na * nb)
