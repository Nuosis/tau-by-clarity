"""
Extension system — mirrors packages/coding-agent/src/core/extensions/

Provides a plugin system for the coding agent with:
- Extension discovery and loading
- Event-based lifecycle hooks
- Custom tool registration
- Command registration
- Context modification
"""

from .types import (
    Extension,
    ExtensionAPI,
    ExtensionContext,
    ExtensionFactory,
    PiManifest,
    RegisteredCommand,
    RegisteredTool,
    ToolDefinition,
)
from .runner import ExtensionRunner
from .loader import discover_and_load_extensions, load_extensions

__all__ = [
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionFactory",
    "ExtensionRunner",
    "PiManifest",
    "RegisteredCommand",
    "RegisteredTool",
    "ToolDefinition",
    "discover_and_load_extensions",
    "load_extensions",
]
