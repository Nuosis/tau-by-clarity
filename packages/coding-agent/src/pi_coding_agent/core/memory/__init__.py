"""Project-local atomic memory for pi-py (design doc §8). P0: store + scaffolding."""
from .embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    cosine,
    embedding_provider_from_env,
)
from .models import (
    ConversationTurn,
    MemoryHit,
    MemoryStatus,
    MemoryType,
    Scope,
    ScopeType,
    SemanticMemory,
)
from .store import MemoryStore

__all__ = [
    "MemoryStore", "SemanticMemory", "MemoryHit", "ConversationTurn", "Scope",
    "MemoryType", "ScopeType", "MemoryStatus", "EmbeddingProvider",
    "OllamaEmbeddingProvider", "DeterministicEmbeddingProvider",
    "embedding_provider_from_env", "cosine",
]
