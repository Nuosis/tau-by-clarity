"""
pi_coding_agent — Coding agent CLI
Python mirror of @mariozechner/pi-coding-agent
"""

from .cli_sub.args import Args, parse_args, print_help
from .config import (
    CONFIG_DIR_NAME,
    ENV_AGENT_DIR,
    ENV_SESSION_DIR,
    VERSION,
    get_agent_dir,
    get_docs_path,
    get_examples_path,
    get_package_dir,
    get_readme_path,
)
from .core.agent_session import AgentSession
from .core.sdk import (
    create_agent_session,
    CreateAgentSessionOptions,
    CreateAgentSessionResult,
    AgentSessionRuntime,
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionRuntimeFactory,
    CreateAgentSessionRuntimeResult,
    CreateAgentSessionServicesOptions,
    SessionImportFileNotFoundError,
    create_agent_session_from_services,
    create_agent_session_runtime,
    create_agent_session_services,
    create_all_tools,
    create_coding_tools,
    create_read_only_tools,
)
from .core.session_manager import SessionManager, build_session_context, CURRENT_SESSION_VERSION
from .core.settings_manager import (
    Settings,
    SettingsManager,
    CompactionSettings,
    FileSettingsStorage,
    ImageSettings,
    InMemorySettingsStorage,
    RetrySettings,
    SettingsStorage,
)
from .core.auth_storage import AuthStorage, AuthStorageBackend, FileAuthStorageBackend, InMemoryAuthStorageBackend
from .core.model_registry import ModelRegistry
from .core.system_prompt import build_system_prompt
from .core.trust_manager import (
    ProjectTrustStore,
    has_project_config_dir,
    has_project_trust_inputs,
)
from .core.tools import (
    create_bash_tool, bash_tool,
    create_bash_tool_definition,
    create_edit_tool, edit_tool,
    create_edit_tool_definition,
    create_find_tool, find_tool,
    create_find_tool_definition,
    create_grep_tool, grep_tool,
    create_grep_tool_definition,
    create_ls_tool, ls_tool,
    create_ls_tool_definition,
    create_read_tool, read_tool,
    create_read_tool_definition,
    create_write_tool, write_tool,
    create_write_tool_definition,
    create_tool,
    create_tool_definition,
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
from .modes.interactive.theme import (
    Theme,
    get_language_from_path,
    get_markdown_theme,
    get_select_list_theme,
    get_settings_list_theme,
    highlight_code,
    init_theme,
)
from .modes import (
    InteractiveMode,
    PrintModeOptions,
    RpcClient,
    RpcClientOptions,
    runPrintMode,
    runRpcMode,
    run_interactive_mode,
    run_print_mode,
    run_rpc_mode,
)
from .modes.interactive.components import (
    ArminComponent,
    AssistantMessageComponent,
    BashExecutionComponent,
    BorderedLoader,
    BranchSummaryMessageComponent,
    CompactionSummaryMessageComponent,
    CustomEditor,
    CustomMessageComponent,
    DaxnutsComponent,
    DynamicBorder,
    ExtensionEditorComponent,
    ExtensionInputComponent,
    ExtensionSelectorComponent,
    FooterComponent,
    LoginDialogComponent,
    ModelSelectorComponent,
    OAuthSelectorComponent,
    ScopedModelsSelectorComponent,
    SessionSelectorComponent,
    SettingsSelectorComponent,
    ShowImagesSelectorComponent,
    SkillInvocationMessageComponent,
    ThemeSelectorComponent,
    ThinkingSelectorComponent,
    ToolExecutionComponent,
    TreeSelectorComponent,
    TrustSelectorComponent,
    UserMessageComponent,
    UserMessageSelectorComponent,
    VisualTruncateResult,
    key_hint,
    keyHint,
    key_text,
    keyText,
    raw_key_hint,
    rawKeyHint,
    render_diff,
    renderDiff,
    truncate_to_visual_lines,
    truncateToVisualLines,
)
from .core.messages import convert_to_llm
from .core.package_manager import ConfiguredPackage, DefaultPackageManager
from .core.resource_loader import DefaultResourceLoader, load_project_context_files
from .core.skills import format_skills_for_prompt, load_skills, load_skills_from_dir
from .core.source_info import create_synthetic_source_info
from .utils.clipboard import copy_to_clipboard
from .utils.frontmatter import parse_frontmatter, strip_frontmatter
from .utils.image_convert import convert_to_png
from .utils.image_resize import ResizedImage, format_dimension_note, resize_image
from .utils.shell import get_shell_config

createAgentSession = create_agent_session
createAgentSessionFromServices = create_agent_session_from_services
createAgentSessionRuntime = create_agent_session_runtime
createAgentSessionServices = create_agent_session_services
createAllTools = create_all_tools
createBashTool = create_bash_tool
createBashToolDefinition = create_bash_tool_definition
createCodingTools = create_coding_tools
createEditTool = create_edit_tool
createEditToolDefinition = create_edit_tool_definition
createFindTool = create_find_tool
createFindToolDefinition = create_find_tool_definition
createGrepTool = create_grep_tool
createGrepToolDefinition = create_grep_tool_definition
createLsTool = create_ls_tool
createLsToolDefinition = create_ls_tool_definition
createReadOnlyTools = create_read_only_tools
createReadTool = create_read_tool
createReadToolDefinition = create_read_tool_definition
createSyntheticSourceInfo = create_synthetic_source_info
createTool = create_tool
createToolDefinition = create_tool_definition
createWriteTool = create_write_tool
createWriteToolDefinition = create_write_tool_definition
convertToLlm = convert_to_llm
copyToClipboard = copy_to_clipboard
formatDimensionNote = format_dimension_note
formatSkillsForPrompt = format_skills_for_prompt
getShellConfig = get_shell_config
loadProjectContextFiles = load_project_context_files
loadSkills = load_skills
loadSkillsFromDir = load_skills_from_dir
parseFrontmatter = parse_frontmatter
resizeImage = resize_image
stripFrontmatter = strip_frontmatter
convertToPng = convert_to_png
getAgentDir = get_agent_dir
getDocsPath = get_docs_path
getExamplesPath = get_examples_path
getPackageDir = get_package_dir
getReadmePath = get_readme_path
parseArgs = parse_args

__all__ = [
    # Core
    "Args",
    "AgentSession",
    "CONFIG_DIR_NAME",
    "ENV_AGENT_DIR",
    "ENV_SESSION_DIR",
    "VERSION",
    "create_agent_session",
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "AgentSessionRuntime",
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "CreateAgentSessionServicesOptions",
    "SessionImportFileNotFoundError",
    "create_agent_session_from_services",
    "create_agent_session_runtime",
    "create_agent_session_services",
    "createAgentSession",
    "createAgentSessionFromServices",
    "createAgentSessionRuntime",
    "createAgentSessionServices",
    "parse_args",
    "parseArgs",
    "print_help",
    # Session
    "SessionManager",
    "build_session_context",
    "CURRENT_SESSION_VERSION",
    # Settings
    "Settings",
    "SettingsManager",
    "CompactionSettings",
    "FileSettingsStorage",
    "ImageSettings",
    "InMemorySettingsStorage",
    "RetrySettings",
    "SettingsStorage",
    # Auth
    "AuthStorage",
    "AuthStorageBackend",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "ConfiguredPackage",
    "DefaultPackageManager",
    "DefaultResourceLoader",
    # Trust
    "ProjectTrustStore",
    "has_project_config_dir",
    "has_project_trust_inputs",
    # Model
    "ModelRegistry",
    # System prompt
    "build_system_prompt",
    "get_agent_dir",
    "get_docs_path",
    "get_examples_path",
    "get_package_dir",
    "get_readme_path",
    "getAgentDir",
    "getDocsPath",
    "getExamplesPath",
    "getPackageDir",
    "getReadmePath",
    # Tools
    "create_bash_tool", "bash_tool",
    "create_bash_tool_definition",
    "create_edit_tool", "edit_tool",
    "create_edit_tool_definition",
    "create_find_tool", "find_tool",
    "create_find_tool_definition",
    "create_grep_tool", "grep_tool",
    "create_grep_tool_definition",
    "create_ls_tool", "ls_tool",
    "create_ls_tool_definition",
    "create_read_tool", "read_tool",
    "create_read_tool_definition",
    "create_write_tool", "write_tool",
    "create_write_tool_definition",
    "create_all_tools", "create_coding_tools", "create_read_only_tools",
    "create_tool", "create_tool_definition",
    "createAllTools", "createBashTool", "createBashToolDefinition",
    "createCodingTools", "createEditTool", "createEditToolDefinition",
    "createFindTool", "createFindToolDefinition", "createGrepTool",
    "createGrepToolDefinition", "createLsTool", "createLsToolDefinition",
    "createReadOnlyTools", "createReadTool", "createReadToolDefinition",
    "createTool", "createToolDefinition", "createWriteTool",
    "createWriteToolDefinition",
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
    "convert_to_llm",
    "convertToLlm",
    "create_synthetic_source_info",
    "createSyntheticSourceInfo",
    "format_skills_for_prompt",
    "formatSkillsForPrompt",
    "load_project_context_files",
    "loadProjectContextFiles",
    "load_skills",
    "loadSkills",
    "load_skills_from_dir",
    "loadSkillsFromDir",
    # Theme utilities
    "Theme",
    "get_language_from_path",
    "get_markdown_theme",
    "get_select_list_theme",
    "get_settings_list_theme",
    "highlight_code",
    "init_theme",
    # Run modes
    "InteractiveMode",
    "PrintModeOptions",
    "RpcClient",
    "RpcClientOptions",
    "runPrintMode",
    "runRpcMode",
    "run_interactive_mode",
    "run_print_mode",
    "run_rpc_mode",
    # UI components
    "ArminComponent",
    "AssistantMessageComponent",
    "BashExecutionComponent",
    "BorderedLoader",
    "BranchSummaryMessageComponent",
    "CompactionSummaryMessageComponent",
    "CustomEditor",
    "CustomMessageComponent",
    "DaxnutsComponent",
    "DynamicBorder",
    "ExtensionEditorComponent",
    "ExtensionInputComponent",
    "ExtensionSelectorComponent",
    "FooterComponent",
    "LoginDialogComponent",
    "ModelSelectorComponent",
    "OAuthSelectorComponent",
    "ScopedModelsSelectorComponent",
    "SessionSelectorComponent",
    "SettingsSelectorComponent",
    "ShowImagesSelectorComponent",
    "SkillInvocationMessageComponent",
    "ThemeSelectorComponent",
    "ThinkingSelectorComponent",
    "ToolExecutionComponent",
    "TreeSelectorComponent",
    "TrustSelectorComponent",
    "UserMessageComponent",
    "UserMessageSelectorComponent",
    "VisualTruncateResult",
    "key_hint",
    "keyHint",
    "key_text",
    "keyText",
    "raw_key_hint",
    "rawKeyHint",
    "render_diff",
    "renderDiff",
    "truncate_to_visual_lines",
    "truncateToVisualLines",
    # Utilities
    "ResizedImage",
    "convert_to_png",
    "convertToPng",
    "copy_to_clipboard",
    "copyToClipboard",
    "format_dimension_note",
    "formatDimensionNote",
    "get_shell_config",
    "getShellConfig",
    "parse_frontmatter",
    "parseFrontmatter",
    "resize_image",
    "resizeImage",
    "strip_frontmatter",
    "stripFrontmatter",
]


# Install Clarity PII as pi_ai's universal outbound filter on import, so every
# LLM call made anywhere in this distribution tokenizes PII regardless of source.
try:
    from .clarity_pii import register_with_pi_ai as _register_clarity_pii

    _register_clarity_pii()
except Exception:
    pass

# Install active compression (content-aware + CCR) as pi_ai's universal outbound
# compressor; default-on via the active_compression settings flag.
try:
    from .active_compression import register_with_pi_ai as _register_active_compression

    _register_active_compression()
except Exception:
    pass
