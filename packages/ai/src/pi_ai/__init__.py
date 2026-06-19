"""
pi_ai — Unified LLM API
Python mirror of @mariozechner/pi-ai
"""

# Core types
# API registry
from .api_registry import (
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)

# Universal active-compression hook (registered by tau-by-clarity; no-op otherwise)
from .compression import (
    CacheAlignmentStats,
    CompressionCacheStats,
    CompressionEvent,
    CompressionLearningStats,
    CompressionStats,
    UnitOutcomeStats,
    VolatileFinding,
    cache_alignment_score,
    compress_context,
    detect_volatile_content,
    get_cache_alignment_stats,
    get_compression_cache_stats,
    get_compression_learning_stats,
    get_compression_circuit_breaker_state,
    get_compression_stats,
    get_unit_outcome_stats,
    get_current_compression_enable_ccr_marker,
    get_current_compression_image_optimize,
    get_current_compression_force_compression,
    get_current_compression_lossless_min_savings_ratio,
    get_current_compression_max_items_after_crush,
    get_current_compression_min_tokens,
    get_current_compression_policy,
    get_current_compression_target_ratio,
    has_compressor,
    register_compression_observer,
    register_compressor,
    reset_cache_alignment_stats,
    reset_compression_cache,
    reset_compression_circuit_breaker,
    reset_compression_learning_stats,
    reset_compression_stats,
    unregister_compression_observer,
    unregister_compressor,
)
from .compression_policy import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    AuthMode,
    CompressionPolicy,
    policy_default_payg,
    policy_for_mode,
    resolve_policy,
)

# Environment API keys
from .env_api_keys import get_env_api_key

# Model registry
from .models import calculate_cost, get_model, get_models, get_providers, models_are_equal, supports_xhigh

# Universal PII filter hook (registered by clarity_pii; no-op otherwise)
from .pii import has_pii_filter, register_pii_filter, unregister_pii_filter

# Streaming functions
from .stream import complete, complete_simple, stream, stream_simple
from .types import (
    Api,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    CacheRetention,
    Context,
    EventDone,
    EventError,
    EventStart,
    EventTextDelta,
    EventTextEnd,
    EventTextStart,
    EventThinkingDelta,
    EventThinkingEnd,
    EventThinkingStart,
    EventToolCallDelta,
    EventToolCallEnd,
    EventToolCallStart,
    ImageContent,
    KnownApi,
    KnownProvider,
    Message,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    OpenRouterRouting,
    Provider,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    Transport,
    Usage,
    UsageCost,
    UserMessage,
    VercelGatewayRouting,
)

# Utilities
from .utils.event_stream import AssistantMessageEventStream as AssistantMessageEventStreamClass
from .utils.event_stream import EventStream, create_assistant_message_event_stream
from .utils.json_parse import (
    StreamingJsonParseResult,
    parse_partial_json,
    parse_streaming_json,
    parse_streaming_json_result,
)
from .utils.overflow import get_overflow_patterns, is_context_overflow
from .utils.sanitize_unicode import sanitize_surrogates
from .utils.validation import validate_tool_arguments, validate_tool_call


def unregister_request_context_manager() -> None:
    """Backward-compatible cleanup hook for callers/tests from older pi_ai.

    Tau does not currently install a request-context manager at this layer, so
    unregistering it is an exact no-op.
    """
    return None


__all__ = [
    # Types
    "Api", "KnownApi", "KnownProvider", "Provider",
    "ThinkingLevel", "ThinkingBudgets", "CacheRetention", "Transport", "StopReason",
    "StreamOptions", "SimpleStreamOptions",
    "TextContent", "ThinkingContent", "ImageContent", "ToolCall",
    "Usage", "UsageCost",
    "UserMessage", "AssistantMessage", "ToolResultMessage", "Message",
    "Tool", "Context", "Model", "ModelCost",
    "OpenAICompletionsCompat", "OpenAIResponsesCompat", "OpenRouterRouting", "VercelGatewayRouting",
    "AssistantMessageEvent", "AssistantMessageEventStream",
    "EventStart", "EventTextStart", "EventTextDelta", "EventTextEnd",
    "EventThinkingStart", "EventThinkingDelta", "EventThinkingEnd",
    "EventToolCallStart", "EventToolCallDelta", "EventToolCallEnd",
    "EventDone", "EventError",
    # Models
    "get_model", "get_providers", "get_models", "calculate_cost", "supports_xhigh", "models_are_equal",
    # Registry
    "register_api_provider", "get_api_provider", "get_api_providers", "unregister_api_providers", "clear_api_providers",
    # Keys
    "get_env_api_key",
    # Streaming
    "stream", "complete", "stream_simple", "complete_simple",
    # PII filter hook
    "register_pii_filter", "unregister_pii_filter", "has_pii_filter",
    "unregister_request_context_manager",
    # Active-compression hook
    "CacheAlignmentStats", "CompressionCacheStats", "CompressionEvent", "CompressionLearningStats",
    "CompressionStats", "UnitOutcomeStats",
    "VolatileFinding", "detect_volatile_content", "cache_alignment_score",
    "register_compressor", "unregister_compressor", "has_compressor", "compress_context",
    "get_current_compression_force_compression",
    "get_current_compression_min_tokens",
    "get_current_compression_max_items_after_crush",
    "get_current_compression_lossless_min_savings_ratio",
    "get_current_compression_enable_ccr_marker",
    "get_current_compression_image_optimize",
    "get_current_compression_target_ratio",
    "get_current_compression_policy",
    "AuthMode", "CompressionPolicy", "policy_for_mode", "policy_default_payg",
    "resolve_policy", "CACHE_WRITE_MULTIPLIER", "CACHE_READ_MULTIPLIER",
    "register_compression_observer", "unregister_compression_observer",
    "get_compression_stats", "reset_compression_stats",
    "get_unit_outcome_stats",
    "get_cache_alignment_stats", "reset_cache_alignment_stats",
    "get_compression_cache_stats", "reset_compression_cache",
    "get_compression_learning_stats", "reset_compression_learning_stats",
    "get_compression_circuit_breaker_state", "reset_compression_circuit_breaker",
    # Utils
    "EventStream", "AssistantMessageEventStreamClass", "create_assistant_message_event_stream",
    "parse_partial_json", "parse_streaming_json", "parse_streaming_json_result", "StreamingJsonParseResult",
    "validate_tool_arguments", "validate_tool_call",
    "is_context_overflow", "get_overflow_patterns",
    "sanitize_surrogates",
]
