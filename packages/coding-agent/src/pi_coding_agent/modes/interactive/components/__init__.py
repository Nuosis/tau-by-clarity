"""Interactive mode UI components — mirrors TypeScript component namespace."""

from .auth_components import AuthSelectorProvider, LoginDialogComponent, OAuthSelectorComponent
from .bash_execution import BashExecutionComponent
from .config_selector import (
    ConfigSelectorComponent,
    ResourceGroup,
    ResourceItem,
    ResourceSubgroup,
    build_resource_groups,
    get_group_label,
)
from .countdown_timer import CountdownTimer
from .extension_components import (
    CustomEditor,
    ExtensionEditorComponent,
    ExtensionInputComponent,
    ExtensionSelectorComponent,
)
from .easter_eggs import ArminComponent, DaxnutsComponent
from .footer import (
    FooterComponent,
    format_cwd_for_footer,
    format_tokens,
    sanitize_status_text,
    truncate_to_width,
    visible_width,
)
from .messages import (
    AssistantMessageComponent,
    BranchSummaryMessageComponent,
    CompactionSummaryMessageComponent,
    CustomMessageComponent,
    SkillInvocationMessageComponent,
    UserMessageComponent,
)
from .rendering import (
    BorderedLoader,
    DynamicBorder,
    ToolExecutionComponent,
    VisualTruncateResult,
    format_key_text,
    get_text_output,
    key_display_text,
    key_hint,
    key_text,
    parse_diff_line,
    raw_key_hint,
    render_diff,
    render_intra_line_diff,
    replace_tabs,
    truncate_to_visual_lines,
)
from .selectors import (
    ModelSelectorComponent,
    ScopedModelsSelectorComponent,
    SelectItem,
    SelectList,
    ShowImagesSelectorComponent,
    ThemeSelectorComponent,
    ThinkingSelectorComponent,
    UserMessageItem,
    UserMessageSelectorComponent,
    fuzzy_filter,
    fuzzy_score,
    models_are_equal,
)
from .session_selector_search import (
    filterAndSortSessions,
    filter_and_sort_sessions,
    getSessionSearchText,
    get_session_search_text,
    hasSessionName,
    has_session_name,
    matchSession,
    match_session,
    parseSearchQuery,
    parse_search_query,
)
from .session_selector import SessionSelectorComponent
from .settings_selector import SettingItem, SettingsSelectorComponent
from .stubs import ComponentStub, STUB_COMPONENTS
from .tree_selector import (
    SessionTreeNode,
    TreeSelectorComponent,
    build_tree_from_entries,
)
from .trust_selector import TrustSelectorComponent, format_decision

keyHint = key_hint
keyText = key_text
rawKeyHint = raw_key_hint
renderDiff = render_diff
truncateToVisualLines = truncate_to_visual_lines

__all__ = [
    "AuthSelectorProvider",
    "ArminComponent",
    "AssistantMessageComponent",
    "BashExecutionComponent",
    "BorderedLoader",
    "BranchSummaryMessageComponent",
    "CompactionSummaryMessageComponent",
    "ComponentStub",
    "ConfigSelectorComponent",
    "CountdownTimer",
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
    "ResourceGroup",
    "ResourceItem",
    "ResourceSubgroup",
    "ScopedModelsSelectorComponent",
    "SelectItem",
    "SelectList",
    "SessionSelectorComponent",
    "STUB_COMPONENTS",
    "SessionTreeNode",
    "SettingItem",
    "SettingsSelectorComponent",
    "ShowImagesSelectorComponent",
    "SkillInvocationMessageComponent",
    "ThemeSelectorComponent",
    "ThinkingSelectorComponent",
    "ToolExecutionComponent",
    "TreeSelectorComponent",
    "TrustSelectorComponent",
    "UserMessageItem",
    "UserMessageComponent",
    "UserMessageSelectorComponent",
    "VisualTruncateResult",
    "build_tree_from_entries",
    "build_resource_groups",
    "filterAndSortSessions",
    "filter_and_sort_sessions",
    "format_key_text",
    "format_cwd_for_footer",
    "format_decision",
    "format_tokens",
    "fuzzy_filter",
    "fuzzy_score",
    "getSessionSearchText",
    "get_session_search_text",
    "get_text_output",
    "get_group_label",
    "hasSessionName",
    "has_session_name",
    "key_display_text",
    "key_hint",
    "keyHint",
    "key_text",
    "keyText",
    "matchSession",
    "match_session",
    "models_are_equal",
    "parse_diff_line",
    "parseSearchQuery",
    "parse_search_query",
    "raw_key_hint",
    "rawKeyHint",
    "render_diff",
    "renderDiff",
    "render_intra_line_diff",
    "replace_tabs",
    "sanitize_status_text",
    "truncate_to_width",
    "truncate_to_visual_lines",
    "truncateToVisualLines",
    "visible_width",
]
