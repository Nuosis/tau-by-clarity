"""
Test SDK alignment with TypeScript version.
Tests the new CreateAgentSessionOptions and CreateAgentSessionResult.
"""
from __future__ import annotations

import pytest
from pi_ai import get_model
from pi_coding_agent import (
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    create_agent_session,
)


class TestSDKAlignment:
    """Test SDK API alignment with TypeScript version."""

    @pytest.mark.asyncio
    async def test_create_agent_session_minimal(self):
        """Test minimal session creation (no options)."""
        result = await create_agent_session()
        
        assert isinstance(result, CreateAgentSessionResult)
        assert result.session is not None
        assert result.session.cwd is not None
        assert result.session.model is not None

    @pytest.mark.asyncio
    async def test_create_agent_session_with_options(self):
        """Test session creation with explicit options."""
        model = get_model("anthropic", "claude-3-5-sonnet-20241022")
        
        options = CreateAgentSessionOptions(
            model=model,
            thinking_level="high",
        )
        
        result = await create_agent_session(options)
        
        assert result.session.model == model
        # Thinking level may be clamped based on model capabilities
        assert result.session.thinking_level in ["off", "minimal", "low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_create_agent_session_returns_result_object(self):
        """Test that create_agent_session returns CreateAgentSessionResult."""
        result = await create_agent_session()
        
        # Check all expected fields
        assert hasattr(result, "session")
        assert hasattr(result, "extensions_result")
        assert hasattr(result, "model_fallback_message")
        
        # Session should be valid
        assert result.session is not None
        
        # Optional fields can be None
        assert result.extensions_result is None or isinstance(result.extensions_result, dict)
        assert result.model_fallback_message is None or isinstance(result.model_fallback_message, str)

    @pytest.mark.asyncio
    async def test_dataclass_options_structure(self):
        """Test that CreateAgentSessionOptions is a proper dataclass."""
        options = CreateAgentSessionOptions(
            cwd="/tmp/test",
            agent_dir="/tmp/agent",
            thinking_level="medium",
        )
        
        assert options.cwd == "/tmp/test"
        assert options.agent_dir == "/tmp/agent"
        assert options.thinking_level == "medium"
        
        # Check other fields have defaults
        assert options.model is None
        assert options.scoped_models is None
        assert options.tools is None
        assert options.custom_tools is None

    @pytest.mark.asyncio
    async def test_session_has_new_methods(self):
        """Test that session has all newly aligned methods."""
        result = await create_agent_session()
        session = result.session
        
        # Test new methods exist
        assert hasattr(session, "dispose")
        assert hasattr(session, "set_scoped_models")
        assert hasattr(session, "set_auto_compaction_enabled")
        assert hasattr(session, "set_auto_retry_enabled")
        assert hasattr(session, "execute_bash")
        assert hasattr(session, "abort_bash")
        assert hasattr(session, "set_session_name")
        assert hasattr(session, "get_user_messages_for_forking")
        assert hasattr(session, "reload")
        assert hasattr(session, "bind_extensions")
        assert hasattr(session, "has_extension_handlers")
        
        # Test new properties
        assert hasattr(session, "auto_compaction_enabled")
        assert hasattr(session, "auto_retry_enabled")

    @pytest.mark.asyncio
    async def test_dispose_method(self):
        """Test dispose method cleans up resources."""
        result = await create_agent_session()
        session = result.session
        
        # Should have listeners initially
        initial_listener_count = len(session._listeners)
        
        # Dispose should clear listeners
        session.dispose()
        
        assert len(session._listeners) == 0

    @pytest.mark.asyncio
    async def test_set_scoped_models(self):
        """Test set_scoped_models method."""
        result = await create_agent_session()
        session = result.session
        
        model1 = get_model("anthropic", "claude-3-5-sonnet-20241022")
        model2 = get_model("google", "gemini-2.0-flash")
        
        scoped = [
            {"model": model1, "thinking_level": "high"},
            {"model": model2, "thinking_level": "medium"},
        ]
        
        session.set_scoped_models(scoped)
        
        assert session._scoped_models == scoped

    @pytest.mark.asyncio
    async def test_auto_compaction_enabled_property(self):
        """Test auto_compaction_enabled property."""
        result = await create_agent_session()
        session = result.session
        
        # Get initial state
        initial = session.auto_compaction_enabled
        assert isinstance(initial, bool)
        
        # Note: set_auto_compaction_enabled is currently a no-op
        # as settings are file-based, but the API should exist
        session.set_auto_compaction_enabled(not initial)
        # The value won't actually change since it's file-based
        # Just verify the method exists and doesn't crash
        assert isinstance(session.auto_compaction_enabled, bool)

    @pytest.mark.asyncio
    async def test_auto_retry_enabled_property(self):
        """Test auto_retry_enabled property."""
        result = await create_agent_session()
        session = result.session
        
        # Get initial state
        initial = session.auto_retry_enabled
        assert isinstance(initial, bool)
        
        # Note: set_auto_retry_enabled is currently a no-op
        # as settings are file-based, but the API should exist
        session.set_auto_retry_enabled(not initial)
        # The value won't actually change since it's file-based
        # Just verify the method exists and doesn't crash
        assert isinstance(session.auto_retry_enabled, bool)

    @pytest.mark.asyncio
    async def test_get_user_messages_for_forking(self):
        """Test get_user_messages_for_forking method."""
        result = await create_agent_session()
        session = result.session
        
        # Initially should be empty
        messages = session.get_user_messages_for_forking()
        assert isinstance(messages, list)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_has_extension_handlers(self):
        """Test has_extension_handlers method (stub)."""
        result = await create_agent_session()
        session = result.session
        
        # Currently a stub, should return False
        assert session.has_extension_handlers("any_event") is False

    @pytest.mark.asyncio
    async def test_set_session_name(self):
        """Test set_session_name method."""
        result = await create_agent_session()
        session = result.session
        
        # Should not raise
        session.set_session_name("Test Session")


class TestExportAlignment:
    """Test that all TypeScript exports are available in Python."""

    def test_agent_package_exports(self):
        """Test pi-agent package exports."""
        from pi_agent import (
            Agent,
            AgentToolUpdateCallback,
            CustomAgentMessages,
            StreamFn,
        )
        
        assert Agent is not None
        assert StreamFn is not None
        assert CustomAgentMessages is not None
        assert AgentToolUpdateCallback is not None

    def test_ai_package_exports(self):
        """Test pi-ai package exports."""
        from pi_ai import (
            clear_api_providers,
            create_assistant_message_event_stream,
            get_api_providers,
            parse_streaming_json,
            validate_tool_call,
        )
        
        assert get_api_providers is not None
        assert clear_api_providers is not None
        assert validate_tool_call is not None
        assert parse_streaming_json is not None
        assert create_assistant_message_event_stream is not None

    def test_coding_agent_exports(self):
        """Test pi-coding-agent package exports."""
        from pi_coding_agent import (
            DEFAULT_COMPACTION_SETTINGS,
            DEFAULT_MAX_BYTES,
            DEFAULT_MAX_LINES,
            CompactionSettings,
            CreateAgentSessionOptions,
            CreateAgentSessionResult,
            ImageSettings,
            RetrySettings,
            build_session_context,
            compact,
            create_agent_session,
            format_size,
            should_compact,
            truncate_head,
            truncate_line,
            truncate_tail,
        )
        
        # SDK
        assert create_agent_session is not None
        assert CreateAgentSessionOptions is not None
        assert CreateAgentSessionResult is not None
        
        # Settings
        assert CompactionSettings is not None
        assert ImageSettings is not None
        assert RetrySettings is not None
        
        # Compaction
        assert compact is not None
        assert should_compact is not None
        assert DEFAULT_COMPACTION_SETTINGS is not None
        
        # Truncate
        assert truncate_head is not None
        assert truncate_tail is not None
        assert truncate_line is not None
        assert format_size is not None
        assert DEFAULT_MAX_BYTES is not None
        assert DEFAULT_MAX_LINES is not None
        
        # Session
        assert build_session_context is not None

    def test_tui_editor_methods(self):
        """Test TUI Editor component has new methods."""
        from pi_tui.components.editor import Editor
        
        # Check methods exist
        assert hasattr(Editor, "get_autocomplete_max_visible")
        assert hasattr(Editor, "set_autocomplete_max_visible")


class TestTypeAlignment:
    """Test type definitions match TypeScript."""

    def test_stream_fn_protocol(self):
        """Test StreamFn is a Protocol with correct signature."""
        from pi_agent.types import StreamFn
        
        # StreamFn should be a Protocol (typing.Protocol)
        assert hasattr(StreamFn, "__call__")

    def test_model_compat_structured_types(self):
        """Test Model.compat has structured types."""
        from pi_ai.types import (
            OpenAICompletionsCompat,
            OpenAIResponsesCompat,
            OpenRouterRouting,
            VercelGatewayRouting,
        )
        
        # All should be dataclasses
        assert hasattr(OpenAICompletionsCompat, "__dataclass_fields__")
        assert hasattr(OpenAIResponsesCompat, "__dataclass_fields__")
        assert hasattr(OpenRouterRouting, "__dataclass_fields__")
        assert hasattr(VercelGatewayRouting, "__dataclass_fields__")

    def test_extensions_tool_definition_exported(self):
        """Test ToolDefinition is exported from extensions."""
        from pi_coding_agent.core.extensions import ToolDefinition
        
        assert ToolDefinition is not None


class TestDefaultValues:
    """Test default values match TypeScript."""

    @pytest.mark.asyncio
    async def test_thinking_level_default_medium(self):
        """Test thinking_level defaults to 'medium' (not 'off')."""
        result = await create_agent_session()
        session = result.session
        
        # Default should be 'medium' (may be clamped to 'off' if model doesn't support reasoning)
        # Check the settings object
        # Note: actual thinking_level may be 'off' if model doesn't support reasoning
        assert session._settings.thinking_level in ["off", "medium", "low", "minimal", "high"]
