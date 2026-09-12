"""
pi_agent — Agent loop and state management
Python mirror of @mariozechner/pi-agent-core
"""

from .agent import Agent, AgentOptions
from .agent_loop import agent_loop, agent_loop_continue
from .generation_health import DegenerateGeneration, detect_degenerate_generation
from .proxy import stream_proxy
from .types import (
    AgentContext,
    AgentEvent,
    AgentEventAgentEnd,
    AgentEventAgentStart,
    AgentEventMessageEnd,
    AgentEventMessageStart,
    AgentEventMessageUpdate,
    AgentEventRunState,
    AgentEventToolEnd,
    AgentEventToolStart,
    AgentEventToolUpdate,
    AgentEventTurnEnd,
    AgentEventTurnStart,
    AgentLoopConfig,
    BeforeModelInvocation,
    ModelInvocation,
    ModelInvocationSelection,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolExecutionPolicy,
    AgentToolResult,
    AgentToolUpdateCallback,
    CustomAgentMessages,
    StreamFn,
    ThinkingLevel,
)

__all__ = [
    # Agent class
    "Agent",
    "AgentOptions",
    # Loop functions
    "agent_loop",
    "agent_loop_continue",
    "DegenerateGeneration",
    "detect_degenerate_generation",
    # Proxy
    "stream_proxy",
    # Types
    "AgentContext",
    "AgentEvent",
    "AgentEventAgentEnd",
    "AgentEventAgentStart",
    "AgentEventMessageEnd",
    "AgentEventMessageStart",
    "AgentEventMessageUpdate",
    "AgentEventRunState",
    "AgentEventToolEnd",
    "AgentEventToolStart",
    "AgentEventToolUpdate",
    "AgentEventTurnEnd",
    "AgentEventTurnStart",
    "AgentLoopConfig",
    "BeforeModelInvocation",
    "ModelInvocation",
    "ModelInvocationSelection",
    "AgentMessage",
    "AgentState",
    "AgentTool",
    "AgentToolExecutionPolicy",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "CustomAgentMessages",
    "StreamFn",
    "ThinkingLevel",
]
