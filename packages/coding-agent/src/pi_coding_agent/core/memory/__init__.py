"""Project-local atomic memory for tau (design doc §8). P0: store + scaffolding."""
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
from .curator import CommitDecision, Curator, Evidence
from .recall import build_recall_block, latest_user_query
from .store import MemoryStore
from .working_context import (
    CtxBlock,
    WorkingContextConfig,
    profile_for,
)

__all__ = [
    "MemoryStore", "SemanticMemory", "MemoryHit", "ConversationTurn", "Scope",
    "MemoryType", "ScopeType", "MemoryStatus", "EmbeddingProvider",
    "OllamaEmbeddingProvider", "DeterministicEmbeddingProvider",
    "embedding_provider_from_env", "cosine",
    "Curator", "Evidence", "CommitDecision",
    "build_recall_block", "latest_user_query",
    "WorkingContextConfig", "CtxBlock", "profile_for",
]
