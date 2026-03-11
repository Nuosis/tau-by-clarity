"""
pi_coding_agent — Coding agent CLI
Python mirror of @mariozechner/pi-coding-agent
"""

from .core.agent_session import AgentSession
from .core.sdk import (
    create_agent_session,
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
)
from .core.session_manager import SessionManager, build_session_context, CURRENT_SESSION_VERSION
from .core.settings_manager import (
    Settings,
    SettingsManager,
    CompactionSettings,
    ImageSettings,
    RetrySettings,
)
from .core.auth_storage import AuthStorage
from .core.model_registry import ModelRegistry
from .core.system_prompt import build_system_prompt
from .core.tools import (
    create_bash_tool, bash_tool,
    create_edit_tool, edit_tool,
    create_find_tool, find_tool,
    create_grep_tool, grep_tool,
    create_ls_tool, ls_tool,
    create_read_tool, read_tool,
    create_write_tool, write_tool,
)
from .core.tools.truncate import (
    truncate_head,
    truncate_tail,
    truncate_line,
    format_size,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
)
from .core.compaction import (
    compact_context as compact,
    should_compact,
)
from .core.compaction.compaction import DEFAULT_COMPACTION_SETTINGS

__all__ = [
    # Core
    "AgentSession",
    "create_agent_session",
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    # Session
    "SessionManager",
    "build_session_context",
    "CURRENT_SESSION_VERSION",
    # Settings
    "Settings",
    "SettingsManager",
    "CompactionSettings",
    "ImageSettings",
    "RetrySettings",
    # Auth
    "AuthStorage",
    # Model
    "ModelRegistry",
    # System prompt
    "build_system_prompt",
    # Tools
    "create_bash_tool", "bash_tool",
    "create_edit_tool", "edit_tool",
    "create_find_tool", "find_tool",
    "create_grep_tool", "grep_tool",
    "create_ls_tool", "ls_tool",
    "create_read_tool", "read_tool",
    "create_write_tool", "write_tool",
    # Tool utilities
    "truncate_head",
    "truncate_tail",
    "truncate_line",
    "format_size",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    # Compaction
    "compact",
    "should_compact",
    "DEFAULT_COMPACTION_SETTINGS",
]
