"""
Settings management — mirrors packages/coding-agent/src/core/settings-manager.ts

Manages global (~/.pi/agent/settings.json) and project (.pi/settings.json) settings.
Supports deep merge, write queue, settings migration, and all getter/setter methods.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


# ─── Sub-settings dataclasses ─────────────────────────────────────────────────

@dataclass
class CompactionSettings:
    enabled: bool | None = None          # default: True
    reserve_tokens: int | None = None    # default: 16384
    keep_recent_tokens: int | None = None  # default: 20000


@dataclass
class BranchSummarySettings:
    reserve_tokens: int | None = None    # default: 16384


@dataclass
class RetrySettings:
    enabled: bool | None = None          # default: True
    max_retries: int | None = None       # default: 3
    base_delay_ms: int | None = None     # default: 2000
    max_delay_ms: int | None = None      # default: 60000


@dataclass
class TerminalSettings:
    show_images: bool | None = None      # default: True
    clear_on_shrink: bool | None = None  # default: False


@dataclass
class ImageSettings:
    auto_resize: bool | None = None      # default: True
    block_images: bool | None = None     # default: False


@dataclass
class ThinkingBudgetsSettings:
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


@dataclass
class MarkdownSettings:
    code_block_indent: str | None = None  # default: "  "


@dataclass
class WarningSettings:
    permissions: bool | None = None


# ─── Full settings ─────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Agent settings — mirrors the Settings interface in TypeScript.
    All fields optional (merged from global + project).
    """
    # Core
    last_changelog_version: str | None = None
    # Optional per-instance agent name. When set, the TUI labels assistant turns
    # "<name>:" instead of the generic "Assistant:".
    name: str | None = None
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking_level: str | None = None  # off|minimal|low|medium|high|xhigh
    transport: str | None = None               # "sse" | "websocket"
    steering_mode: str | None = "one-at-a-time"  # "all" | "one-at-a-time"
    follow_up_mode: str | None = "one-at-a-time"  # "all" | "one-at-a-time"
    theme: str | None = None
    hide_thinking_block: bool | None = None
    shell_path: str | None = None
    quiet_startup: bool | None = None
    shell_command_prefix: str | None = None
    collapse_changelog: bool | None = None
    enable_skill_commands: bool | None = None  # default: True
    double_escape_action: str | None = None    # "fork" | "tree" | "none"
    editor_padding_x: int | None = None
    autocomplete_max_visible: int | None = None
    show_hardware_cursor: bool | None = None
    tree_filter_mode: str | None = None        # "default"|"no-tools"|"user-only"|"labeled-only"|"all"

    # Nested settings objects (stored as dicts in JSON)
    compaction: dict[str, Any] | None = None
    branch_summary: dict[str, Any] | None = None
    retry: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    images: dict[str, Any] | None = None
    thinking_budgets: dict[str, Any] | None = None
    markdown: dict[str, Any] | None = None
    warnings: dict[str, Any] | None = None
    session_vars: dict[str, str] | None = None

    # Array fields
    packages: list[Any] | None = None
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None
    enabled_models: list[str] | None = None
    tools: list[str] | None = None

    # Project-local memory (off by default; kill-switch). See
    # design/context-and-memory-management.md. Env PI_MEMORY_ENABLED=1 also forces on.
    memory_enabled: bool = False

    # Legacy/compat fields
    thinking_level: str = "off"
    auto_compact: bool = True
    compact_threshold: float = 0.8
    max_retries: int = 3
    retry_delay_ms: int = 1000
    include_images: bool = True
    max_image_size_kb: int = 5000
    model_id: str | None = None
    provider: str | None = None
    image_auto_resize: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def merge(self, other: "Settings") -> "Settings":
        """Merge another Settings into this one (other wins for non-None values)."""
        base = self.to_dict()
        for k, v in other.to_dict().items():
            if v is None:
                continue
            # Deep merge nested dicts
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return Settings.from_dict(base)


class SettingsStorage:
    def with_lock(self, scope: str, fn: Any) -> None:
        raise NotImplementedError


class FileSettingsStorage(SettingsStorage):
    def __init__(self, cwd: str, agent_dir: str) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        self.global_settings_path = os.path.join(os.path.abspath(os.path.expanduser(agent_dir)), "settings.json")
        self.project_settings_path = os.path.join(
            os.path.abspath(os.path.expanduser(cwd)),
            CONFIG_DIR_NAME,
            "settings.json",
        )

    def with_lock(self, scope: str, fn: Any) -> None:
        path = self.global_settings_path if scope == "global" else self.project_settings_path
        current: str | None = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                current = f.read()
        next_value = fn(current)
        if next_value is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(next_value)


class InMemorySettingsStorage(SettingsStorage):
    def __init__(self, global_value: str | None = None, project_value: str | None = None) -> None:
        self.global_value = global_value
        self.project_value = project_value

    def with_lock(self, scope: str, fn: Any) -> None:
        current = self.global_value if scope == "global" else self.project_value
        next_value = fn(current)
        if next_value is not None:
            if scope == "global":
                self.global_value = next_value
            else:
                self.project_value = next_value


# ─── Deep merge helper ────────────────────────────────────────────────────────

# Global settings keys that are resource-path arrays — these stay project-local
# (only inherited from global with --inherit). Everything else in the global
# file (provider/model/thinking/theme/transport/…) is always inherited.
_GLOBAL_RESOURCE_KEYS = {"extensions", "skills", "prompts", "themes"}


def deep_merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge override into base. Nested objects merge recursively.
    Arrays and primitives: override wins.
    Mirrors deepMergeSettings() in TypeScript.
    """
    result = dict(base)
    for key, val in override.items():
        if val is None:
            continue
        base_val = result.get(key)
        if (
            isinstance(val, dict)
            and isinstance(base_val, dict)
        ):
            result[key] = {**base_val, **val}
        else:
            result[key] = val
    return result


def _strip_json_trailing_commas(content: str) -> str:
    """Remove JSON object/array trailing commas without touching string values."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(content):
        ch = content[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(content) and content[j] in " \t\r\n":
                j += 1
            if j < len(content) and content[j] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def _loads_settings_json(content: str) -> dict[str, Any]:
    raw = json.loads(content)
    if isinstance(raw, dict):
        return raw
    return {}


def _loads_settings_json_lenient(content: str) -> dict[str, Any]:
    try:
        return _loads_settings_json(content)
    except json.JSONDecodeError:
        stripped = _strip_json_trailing_commas(content)
        if stripped == content:
            raise
        return _loads_settings_json(stripped)


# ─── Settings migration ───────────────────────────────────────────────────────

def migrate_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate old settings format to current.
    Mirrors migrateSettings() in TypeScript.
    """
    # queueMode → steeringMode
    if "queueMode" in raw and "steeringMode" not in raw:
        raw["steeringMode"] = raw.pop("queueMode")

    # legacy websockets bool → transport enum
    if "transport" not in raw and "websockets" in raw:
        raw["transport"] = "websocket" if raw.pop("websockets") else "sse"

    # Old skills object → array
    skills = raw.get("skills")
    if isinstance(skills, dict) and skills is not None:
        ec = skills.get("enableSkillCommands")
        dirs = skills.get("customDirectories")
        if ec is not None and "enableSkillCommands" not in raw:
            raw["enableSkillCommands"] = ec
        if isinstance(dirs, list) and dirs:
            raw["skills"] = dirs
        else:
            raw.pop("skills", None)

    return raw


# ─── SettingsManager ──────────────────────────────────────────────────────────

class SettingsManager:
    """
    Manages global and project-level settings.
    Mirrors SettingsManager in TypeScript.

    Global settings:  ~/.pi/agent/settings.json
    Project settings: <project_root>/.pi/settings.json

    Provides:
    - deep merge (project overrides global)
    - asyncio.Lock write queue to prevent race conditions
    - settings migration on load
    - all getter/setter methods matching TypeScript API
    """

    from pi_coding_agent.config import CONFIG_DIR_NAME as _CFG_DIR
    GLOBAL_SETTINGS_DIR = os.path.join(os.path.expanduser("~"), _CFG_DIR, "agent")

    def __init__(
        self,
        project_root: str | None = None,
        global_settings_file: str | None = None,
        storage: SettingsStorage | None = None,
        project_trusted: bool = True,
        inherit_global: bool = True,
        keep_agent_resources: bool = False,
    ) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        self.project_root = project_root or os.getcwd()
        self._project_settings_file = os.path.join(self.project_root, CONFIG_DIR_NAME, "settings.json")
        self._global_settings_file = (
            global_settings_file
            or os.path.join(self.GLOBAL_SETTINGS_DIR, "settings.json")
        )
        self._storage = storage
        self._project_trusted = project_trusted
        # When False, the global settings file is NOT merged (project-local only).
        self._inherit_global = inherit_global
        self._keep_agent_resources = keep_agent_resources
        self._global_raw: dict[str, Any] = {}
        self._project_raw: dict[str, Any] = {}
        self._merged: dict[str, Any] = {}
        self._errors: list[dict[str, Any]] = []
        self._write_lock = asyncio.Lock()
        self._runtime_overrides: dict[str, Any] = {}
        self._loaded = False

    @classmethod
    def create(
        cls,
        cwd: str | None = None,
        agent_dir: str | None = None,
        options: dict[str, Any] | None = None,
        inherit_global: bool = True,
    ) -> "SettingsManager":
        """Factory matching TypeScript SettingsManager.create(cwd, agentDir).

        inherit_global=False loads project-local settings only (the harness
        default; --inherit flips it on).
        """
        from pi_coding_agent.config import get_agent_dir

        opts = options or {}
        project_trusted = opts.get("projectTrusted", opts.get("project_trusted", True))
        keep_agent_resources = bool(opts.get("keepAgentResources", opts.get("keep_agent_resources", False)))
        storage = FileSettingsStorage(cwd or os.getcwd(), agent_dir or get_agent_dir())
        mgr = cls(
            project_root=cwd,
            storage=storage,
            project_trusted=bool(project_trusted),
            inherit_global=inherit_global,
            keep_agent_resources=keep_agent_resources,
        )
        mgr.load()
        return mgr

    @classmethod
    def from_storage(
        cls,
        storage: SettingsStorage,
        options: dict[str, Any] | None = None,
    ) -> "SettingsManager":
        project_trusted = (options or {}).get("projectTrusted", (options or {}).get("project_trusted", True))
        mgr = cls(storage=storage, project_trusted=bool(project_trusted))
        mgr.load()
        return mgr

    @classmethod
    def in_memory(cls, settings: dict[str, Any] | None = None) -> "SettingsManager":
        """Create an in-memory settings manager (no file I/O)."""
        storage = InMemorySettingsStorage(json.dumps(migrate_settings(dict(settings or {})), indent=2))
        return cls.from_storage(storage)

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load settings from disk and run migration.

        The GLOBAL file is always read, but when inherit_global is False only its
        scalar *preference* keys (defaultProvider/Model/ThinkingLevel, theme, …)
        are kept — the resource-path arrays (skills/prompts/extensions/themes)
        are dropped so those stay project-local. So a new agent dir inherits your
        global provider/model defaults without pulling in global resources.
        """
        if self._storage is not None:
            self._global_raw = self._load_scope("global", self._global_raw if self._loaded else None)
            self._project_raw = (
                self._load_scope("project", self._project_raw if self._loaded else None)
                if self._project_trusted
                else {}
            )
        else:
            self._global_raw = self._load_file(
                self._global_settings_file,
                "global",
                self._global_raw if self._loaded else None,
            )
            self._project_raw = (
                self._load_file(
                    self._project_settings_file,
                    "project",
                    self._project_raw if self._loaded else None,
                )
                if self._project_trusted
                else {}
            )
        if not self._inherit_global and not self._keep_agent_resources:
            self._global_raw = {
                k: v for k, v in self._global_raw.items() if k not in _GLOBAL_RESOURCE_KEYS
            }
        self._rebuild()
        self._loaded = True

    def _load_scope(self, scope: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        content_holder: dict[str, str | None] = {"content": None}
        try:
            assert self._storage is not None
            self._storage.with_lock(scope, lambda current: (content_holder.update(content=current) or None))
            content = content_holder["content"]
            if not content:
                return {}
            raw = _loads_settings_json_lenient(content)
            return migrate_settings(raw)
        except Exception as e:
            self._errors.append({"scope": scope, "error": str(e)})
            return dict(previous or {})

    def _load_file(
        self,
        path: str,
        scope: str,
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load raw settings dict from JSON file, running migration."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                raw = _loads_settings_json_lenient(f.read())
            return migrate_settings(raw)
        except (json.JSONDecodeError, OSError) as e:
            self._errors.append({"scope": scope, "error": str(e)})
            return dict(previous or {})

    def _rebuild(self) -> None:
        """Recompute merged settings from global + project + runtime overrides."""
        self._merged = deep_merge_settings(self._global_raw, self._project_raw)
        if self._runtime_overrides:
            self._merged = deep_merge_settings(self._merged, self._runtime_overrides)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def reload(self) -> None:
        """Reload settings from disk."""
        self.load()

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        trusted = bool(trusted)
        if self._project_trusted == trusted:
            return
        self._project_trusted = trusted
        if trusted:
            self._project_raw = (
                self._load_scope("project", self._project_raw if self._loaded else None)
                if self._storage is not None
                else self._load_file(
                    self._project_settings_file,
                    "project",
                    self._project_raw if self._loaded else None,
                )
            )
        else:
            self._project_raw = {}
        self._rebuild()

    def _write_file(self, path: str, data: dict[str, Any]) -> None:
        """Write settings dict to JSON file, creating dirs as needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def _write_scope(self, scope: str, data: dict[str, Any]) -> None:
        if self._storage is not None:
            self._storage.with_lock(scope, lambda current: json.dumps(data, indent=2))
            return
        path = self._global_settings_file if scope == "global" else self._project_settings_file
        self._write_file(path, data)

    def save_global(self, key: str, value: Any) -> None:
        """Update and persist a single global settings key."""
        self._ensure_loaded()
        self._global_raw[key] = value
        self._rebuild()
        self._write_scope("global", self._global_raw)

    def save_project(self, key: str, value: Any) -> None:
        """Update and persist a single project settings key."""
        self._ensure_loaded()
        if not self._project_trusted:
            raise RuntimeError("Project is not trusted; refusing to write project settings")
        self._project_raw[key] = value
        self._rebuild()
        self._write_scope("project", self._project_raw)

    async def save_global_async(self, key: str, value: Any) -> None:
        """Thread-safe async version of save_global."""
        async with self._write_lock:
            self.save_global(key, value)

    async def save_project_async(self, key: str, value: Any) -> None:
        """Thread-safe async version of save_project."""
        async with self._write_lock:
            self.save_project(key, value)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply runtime overrides (not persisted to disk)."""
        self._runtime_overrides = deep_merge_settings(self._runtime_overrides, overrides)
        self._rebuild()

    # ── Read access ───────────────────────────────────────────────────────────

    def get(self) -> Settings:
        """Get merged Settings object (project overrides global)."""
        self._ensure_loaded()
        merged = self._map_raw_to_settings(self._merged)
        return Settings.from_dict(merged)

    def get_merged_raw(self) -> dict[str, Any]:
        """Get merged raw dict."""
        self._ensure_loaded()
        return dict(self._merged)

    def get_global_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._global_raw)

    def get_project_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._project_raw)

    def drain_errors(self) -> list[dict[str, Any]]:
        """Drain and return all accumulated settings errors."""
        drained = list(self._errors)
        self._errors = []
        return drained

    @staticmethod
    def _map_raw_to_settings(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert camelCase JSON keys to snake_case for Settings dataclass."""
        mapping = {
            "defaultProvider": "default_provider",
            "defaultModel": "default_model",
            "defaultThinkingLevel": "default_thinking_level",
            "steeringMode": "steering_mode",
            "followUpMode": "follow_up_mode",
            "hideThinkingBlock": "hide_thinking_block",
            "shellPath": "shell_path",
            "quietStartup": "quiet_startup",
            "shellCommandPrefix": "shell_command_prefix",
            "collapseChangelog": "collapse_changelog",
            "enableSkillCommands": "enable_skill_commands",
            "doubleEscapeAction": "double_escape_action",
            "editorPaddingX": "editor_padding_x",
            "autocompleteMaxVisible": "autocomplete_max_visible",
            "showHardwareCursor": "show_hardware_cursor",
            "treeFilterMode": "tree_filter_mode",
            "branchSummary": "branch_summary",
            "thinkingBudgets": "thinking_budgets",
            "enabledModels": "enabled_models",
            "lastChangelogVersion": "last_changelog_version",
        }
        result = {}
        for k, v in raw.items():
            py_key = mapping.get(k, k)
            result[py_key] = v
        return result

    # ── Typed getters (matching TypeScript API) ────────────────────────────────

    def get_default_provider(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultProvider") or self._merged.get("default_provider")

    def get_default_model(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultModel") or self._merged.get("default_model")

    def get_default_thinking_level(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("defaultThinkingLevel") or self._merged.get("default_thinking_level")

    def get_agent_name(self) -> str | None:
        """Optional per-instance agent name (used as the assistant turn label)."""
        self._ensure_loaded()
        val = self._merged.get("name")
        return val if isinstance(val, str) and val.strip() else None

    def get_theme(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("theme")

    def get_transport(self) -> str:
        self._ensure_loaded()
        return self._merged.get("transport", "auto")

    def get_steering_mode(self) -> str:
        self._ensure_loaded()
        return self._merged.get("steeringMode") or self._merged.get("steering_mode", "one-at-a-time")

    def get_follow_up_mode(self) -> str:
        self._ensure_loaded()
        return self._merged.get("followUpMode") or self._merged.get("follow_up_mode", "one-at-a-time")

    def get_quiet_startup(self) -> bool:
        self._ensure_loaded()
        return bool(self._merged.get("quietStartup") or self._merged.get("quiet_startup", False))

    def get_shell_path(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("shellPath") or self._merged.get("shell_path")

    def get_shell_command_prefix(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("shellCommandPrefix") or self._merged.get("shell_command_prefix")

    def get_enable_skill_commands(self) -> bool:
        self._ensure_loaded()
        val = self._merged.get("enableSkillCommands") or self._merged.get("enable_skill_commands")
        return val if val is not None else True

    def get_double_escape_action(self) -> str:
        self._ensure_loaded()
        return self._merged.get("doubleEscapeAction") or self._merged.get("double_escape_action", "tree")

    def get_compaction_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"enabled": True, "reserveTokens": 16384, "keepRecentTokens": 20000}
        override = self._merged.get("compaction") or {}
        return {**defaults, **override}

    def set_compaction_enabled(self, enabled: bool) -> None:
        """Persist global auto-compaction enabled state."""
        self._ensure_loaded()
        compaction = dict(self._global_raw.get("compaction") or {})
        compaction["enabled"] = bool(enabled)
        self.save_global("compaction", compaction)

    def get_block_images(self) -> bool:
        """Get whether images should be blocked from being sent to LLM providers."""
        self._ensure_loaded()
        images = self._merged.get("images") or {}
        return bool(images.get("blockImages", False))

    def set_block_images(self, blocked: bool) -> None:
        """Set whether images should be blocked from being sent to LLM providers."""
        self._ensure_loaded()
        if "images" not in self._global_raw:
            self._global_raw["images"] = {}
        self._global_raw["images"]["blockImages"] = blocked
        self.save_global("images", self._global_raw["images"])

    def get_retry_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"enabled": True, "maxRetries": 3, "baseDelayMs": 2000, "maxDelayMs": 60000}
        override = self._merged.get("retry") or {}
        return {**defaults, **override}

    def set_retry_enabled(self, enabled: bool) -> None:
        """Persist global auto-retry enabled state."""
        self._ensure_loaded()
        retry = dict(self._global_raw.get("retry") or {})
        retry["enabled"] = bool(enabled)
        self.save_global("retry", retry)

    def get_terminal_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"showImages": True, "clearOnShrink": False}
        override = self._merged.get("terminal") or {}
        return {**defaults, **override}

    def get_image_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"autoResize": True, "blockImages": False}
        override = self._merged.get("images") or {}
        return {**defaults, **override}

    def get_image_auto_resize(self) -> bool:
        return bool(self.get_image_settings().get("autoResize", True))

    def get_thinking_budgets(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._merged.get("thinkingBudgets") or self._merged.get("thinking_budgets") or {})

    def get_markdown_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"codeBlockIndent": "  "}
        override = self._merged.get("markdown") or {}
        return {**defaults, **override}

    def get_branch_summary_settings(self) -> dict[str, Any]:
        self._ensure_loaded()
        defaults: dict[str, Any] = {"reserveTokens": 16384, "skipPrompt": False}
        override = self._merged.get("branchSummary") or self._merged.get("branch_summary") or {}
        return {**defaults, **override}

    def get_branch_summary_skip_prompt(self) -> bool:
        return bool(self.get_branch_summary_settings().get("skipPrompt", False))

    _TREE_FILTER_VALID = {"default", "no-tools", "user-only", "labeled-only", "all"}

    def get_tree_filter_mode(self) -> str:
        self._ensure_loaded()
        mode = self._merged.get("treeFilterMode") or self._merged.get("tree_filter_mode")
        return mode if mode in self._TREE_FILTER_VALID else "default"

    def get_enabled_models(self) -> list[str] | None:
        self._ensure_loaded()
        val = self._merged.get("enabledModels") or self._merged.get("enabled_models")
        return list(val) if isinstance(val, list) else None

    def get_packages(self) -> list[Any]:
        self._ensure_loaded()
        val = self._merged.get("packages") or []
        return list(val) if isinstance(val, list) else []

    def get_extensions(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("extensions") or []
        return list(val) if isinstance(val, list) else []

    def get_tools(self) -> list[str] | None:
        self._ensure_loaded()
        val = self._merged.get("tools")
        if isinstance(val, list):
            return [str(item) for item in val if isinstance(item, str)]
        return None

    def get_skills(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("skills") or []
        return list(val) if isinstance(val, list) else []

    def get_prompts(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("prompts") or []
        return list(val) if isinstance(val, list) else []

    def get_themes(self) -> list[str]:
        self._ensure_loaded()
        val = self._merged.get("themes") or []
        return list(val) if isinstance(val, list) else []

    def _get_nested(self, section: str, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        data = self._merged.get(section) or {}
        if not isinstance(data, dict):
            return default
        return data.get(key, default)

    def _set_nested_global(self, section: str, key: str, value: Any) -> None:
        self._ensure_loaded()
        data = self._global_raw.get(section)
        if not isinstance(data, dict):
            data = {}
        data[key] = value
        self._global_raw[section] = data
        self._rebuild()
        self._write_scope("global", self._global_raw)

    def _set_global(self, key: str, value: Any) -> None:
        self.save_global(key, value)

    def get_last_changelog_version(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("lastChangelogVersion") or self._merged.get("last_changelog_version")

    def set_last_changelog_version(self, version: str) -> None:
        self._set_global("lastChangelogVersion", version)

    def get_session_dir(self) -> str | None:
        self._ensure_loaded()
        return self._merged.get("sessionDir") or self._merged.get("session_dir")

    def set_default_provider(self, provider: str) -> None:
        self._set_global("defaultProvider", provider)

    def set_default_model(self, model_id: str) -> None:
        self._set_global("defaultModel", model_id)

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._ensure_loaded()
        self._global_raw["defaultProvider"] = provider
        self._global_raw["defaultModel"] = model_id
        self._rebuild()
        self._write_scope("global", self._global_raw)

    def set_steering_mode(self, mode: str) -> None:
        self._set_global("steeringMode", mode)

    def set_follow_up_mode(self, mode: str) -> None:
        self._set_global("followUpMode", mode)

    def set_theme(self, theme: str) -> None:
        self._set_global("theme", theme)

    def set_default_thinking_level(self, level: str) -> None:
        self._set_global("defaultThinkingLevel", level)

    def set_transport(self, transport: str) -> None:
        self._set_global("transport", transport)

    def get_compaction_enabled(self) -> bool:
        return bool(self._get_nested("compaction", "enabled", True))

    def get_compaction_reserve_tokens(self) -> int:
        return int(self._get_nested("compaction", "reserveTokens", 16384))

    def get_compaction_keep_recent_tokens(self) -> int:
        return int(self._get_nested("compaction", "keepRecentTokens", 20000))

    # ── Typed setters ─────────────────────────────────────────────────────────

    def set_packages(self, packages: list[Any]) -> None:
        """Set global package sources list."""
        self.save_global("packages", list(packages))

    def set_project_packages(self, packages: list[Any]) -> None:
        """Set project package sources list."""
        self.save_project("packages", list(packages))

    def get_retry_enabled(self) -> bool:
        return bool(self._get_nested("retry", "enabled", True))

    def get_http_idle_timeout_ms(self) -> int:
        self._ensure_loaded()
        value = self._merged.get("httpIdleTimeoutMs") or self._merged.get("http_idle_timeout_ms")
        return int(value) if isinstance(value, int | float) and value >= 0 else 300_000

    def set_http_idle_timeout_ms(self, timeout_ms: int) -> None:
        if timeout_ms < 0:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {timeout_ms}")
        self._set_global("httpIdleTimeoutMs", int(timeout_ms))

    def get_web_socket_connect_timeout_ms(self) -> int | None:
        self._ensure_loaded()
        value = self._merged.get("websocketConnectTimeoutMs") or self._merged.get("websocket_connect_timeout_ms")
        return int(value) if isinstance(value, int | float) and value >= 0 else None

    def get_provider_retry_settings(self) -> dict[str, Any]:
        provider = self._get_nested("retry", "provider", {})
        if not isinstance(provider, dict):
            provider = {}
        return {
            "timeoutMs": provider.get("timeoutMs"),
            "maxRetries": provider.get("maxRetries"),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs", 60000),
        }

    def get_hide_thinking_block(self) -> bool:
        self._ensure_loaded()
        return bool(self._merged.get("hideThinkingBlock") or self._merged.get("hide_thinking_block", False))

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._set_global("hideThinkingBlock", bool(hide))

    def set_shell_path(self, path: str | None) -> None:
        self._set_global("shellPath", path)

    def set_quiet_startup(self, quiet: bool) -> None:
        self._set_global("quietStartup", bool(quiet))

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._set_global("shellCommandPrefix", prefix)

    def get_npm_command(self) -> list[str] | None:
        self._ensure_loaded()
        command = self._merged.get("npmCommand") or self._merged.get("npm_command")
        return list(command) if isinstance(command, list) else None

    def set_npm_command(self, command: list[str] | None) -> None:
        self._set_global("npmCommand", list(command) if command is not None else None)

    def get_collapse_changelog(self) -> bool:
        self._ensure_loaded()
        return bool(self._merged.get("collapseChangelog") or self._merged.get("collapse_changelog", False))

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._set_global("collapseChangelog", bool(collapse))

    def get_enable_install_telemetry(self) -> bool:
        self._ensure_loaded()
        return bool(self._merged.get("enableInstallTelemetry") if "enableInstallTelemetry" in self._merged else True)

    def set_enable_install_telemetry(self, enabled: bool) -> None:
        self._set_global("enableInstallTelemetry", bool(enabled))

    def get_extension_paths(self) -> list[str]:
        return self.get_extensions()

    def set_extension_paths(self, paths: list[str]) -> None:
        self._set_global("extensions", list(paths))

    def set_project_extension_paths(self, paths: list[str]) -> None:
        self.save_project("extensions", list(paths))

    def get_skill_paths(self) -> list[str]:
        return self.get_skills()

    def set_skill_paths(self, paths: list[str]) -> None:
        self._set_global("skills", list(paths))

    def set_project_skill_paths(self, paths: list[str]) -> None:
        self.save_project("skills", list(paths))

    def get_prompt_template_paths(self) -> list[str]:
        return self.get_prompts()

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._set_global("prompts", list(paths))

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        self.save_project("prompts", list(paths))

    def get_theme_paths(self) -> list[str]:
        return self.get_themes()

    def set_theme_paths(self, paths: list[str]) -> None:
        self._set_global("themes", list(paths))

    def set_project_theme_paths(self, paths: list[str]) -> None:
        self.save_project("themes", list(paths))

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._set_global("enableSkillCommands", bool(enabled))

    def get_show_images(self) -> bool:
        return bool(self._get_nested("terminal", "showImages", True))

    def set_show_images(self, show: bool) -> None:
        self._set_nested_global("terminal", "showImages", bool(show))

    def get_image_width_cells(self) -> int:
        value = self._get_nested("terminal", "imageWidthCells", 60)
        return max(1, int(value)) if isinstance(value, int | float) else 60

    def set_image_width_cells(self, width: int) -> None:
        self._set_nested_global("terminal", "imageWidthCells", max(1, int(width)))

    def get_clear_on_shrink(self) -> bool:
        configured = self._get_nested("terminal", "clearOnShrink", None)
        if configured is not None:
            return bool(configured)
        return os.environ.get("PI_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._set_nested_global("terminal", "clearOnShrink", bool(enabled))

    def get_show_terminal_progress(self) -> bool:
        return bool(self._get_nested("terminal", "showTerminalProgress", False))

    def set_show_terminal_progress(self, enabled: bool) -> None:
        self._set_nested_global("terminal", "showTerminalProgress", bool(enabled))

    def set_image_auto_resize(self, enabled: bool) -> None:
        self._set_nested_global("images", "autoResize", bool(enabled))

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._set_global("enabledModels", list(patterns) if patterns is not None else None)

    def set_double_escape_action(self, action: str) -> None:
        self._set_global("doubleEscapeAction", action)

    def set_tree_filter_mode(self, mode: str) -> None:
        self._set_global("treeFilterMode", mode)

    def get_show_hardware_cursor(self) -> bool:
        self._ensure_loaded()
        if "showHardwareCursor" in self._merged:
            return bool(self._merged["showHardwareCursor"])
        return os.environ.get("PI_HARDWARE_CURSOR") == "1"

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._set_global("showHardwareCursor", bool(enabled))

    def get_editor_padding_x(self) -> int:
        self._ensure_loaded()
        value = self._merged.get("editorPaddingX") or self._merged.get("editor_padding_x", 0)
        return max(0, min(3, int(value))) if isinstance(value, int | float) else 0

    def set_editor_padding_x(self, padding: int) -> None:
        self._set_global("editorPaddingX", max(0, min(3, int(padding))))

    def get_autocomplete_max_visible(self) -> int:
        self._ensure_loaded()
        value = self._merged.get("autocompleteMaxVisible") or self._merged.get("autocomplete_max_visible", 5)
        return max(3, min(20, int(value))) if isinstance(value, int | float) else 5

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        self._set_global("autocompleteMaxVisible", max(3, min(20, int(max_visible))))

    def get_code_block_indent(self) -> str:
        value = self._get_nested("markdown", "codeBlockIndent", "  ")
        return value if isinstance(value, str) else "  "

    def get_warnings(self) -> dict[str, Any]:
        self._ensure_loaded()
        warnings = self._merged.get("warnings") or {}
        return dict(warnings) if isinstance(warnings, dict) else {}

    def set_warnings(self, warnings: dict[str, Any]) -> None:
        self._set_global("warnings", dict(warnings))

    def update_global(self, **kwargs: Any) -> None:
        """Update specific global settings fields."""
        self._ensure_loaded()
        for k, v in kwargs.items():
            self._global_raw[k] = v
        self._rebuild()
        self._write_scope("global", self._global_raw)

    def update_project(self, **kwargs: Any) -> None:
        """Update specific project settings fields."""
        self._ensure_loaded()
        if not self._project_trusted:
            raise RuntimeError("Project is not trusted; refusing to write project settings")
        for k, v in kwargs.items():
            self._project_raw[k] = v
        self._rebuild()
        self._write_scope("project", self._project_raw)


SettingsManager.fromStorage = SettingsManager.from_storage
SettingsManager.inMemory = SettingsManager.in_memory
SettingsManager.isProjectTrusted = SettingsManager.is_project_trusted
SettingsManager.setProjectTrusted = SettingsManager.set_project_trusted
SettingsManager.getGlobalSettings = SettingsManager.get_global_settings
SettingsManager.getProjectSettings = SettingsManager.get_project_settings
SettingsManager.drainErrors = SettingsManager.drain_errors
SettingsManager.applyOverrides = SettingsManager.apply_overrides
SettingsManager.getLastChangelogVersion = SettingsManager.get_last_changelog_version
SettingsManager.setLastChangelogVersion = SettingsManager.set_last_changelog_version
SettingsManager.getSessionDir = SettingsManager.get_session_dir
SettingsManager.getDefaultProvider = SettingsManager.get_default_provider
SettingsManager.getDefaultModel = SettingsManager.get_default_model
SettingsManager.setDefaultProvider = SettingsManager.set_default_provider
SettingsManager.setDefaultModel = SettingsManager.set_default_model
SettingsManager.setDefaultModelAndProvider = SettingsManager.set_default_model_and_provider
SettingsManager.getDefaultThinkingLevel = SettingsManager.get_default_thinking_level
SettingsManager.setDefaultThinkingLevel = SettingsManager.set_default_thinking_level
SettingsManager.getTheme = SettingsManager.get_theme
SettingsManager.setTheme = SettingsManager.set_theme
SettingsManager.getTransport = SettingsManager.get_transport
SettingsManager.setTransport = SettingsManager.set_transport
SettingsManager.getSteeringMode = SettingsManager.get_steering_mode
SettingsManager.setSteeringMode = SettingsManager.set_steering_mode
SettingsManager.getFollowUpMode = SettingsManager.get_follow_up_mode
SettingsManager.setFollowUpMode = SettingsManager.set_follow_up_mode
SettingsManager.getQuietStartup = SettingsManager.get_quiet_startup
SettingsManager.setQuietStartup = SettingsManager.set_quiet_startup
SettingsManager.getShellPath = SettingsManager.get_shell_path
SettingsManager.setShellPath = SettingsManager.set_shell_path
SettingsManager.getShellCommandPrefix = SettingsManager.get_shell_command_prefix
SettingsManager.setShellCommandPrefix = SettingsManager.set_shell_command_prefix
SettingsManager.getNpmCommand = SettingsManager.get_npm_command
SettingsManager.setNpmCommand = SettingsManager.set_npm_command
SettingsManager.getCompactionSettings = SettingsManager.get_compaction_settings
SettingsManager.getCompactionEnabled = SettingsManager.get_compaction_enabled
SettingsManager.getCompactionReserveTokens = SettingsManager.get_compaction_reserve_tokens
SettingsManager.getCompactionKeepRecentTokens = SettingsManager.get_compaction_keep_recent_tokens
SettingsManager.setCompactionEnabled = SettingsManager.set_compaction_enabled
SettingsManager.getBranchSummarySettings = SettingsManager.get_branch_summary_settings
SettingsManager.getBranchSummarySkipPrompt = SettingsManager.get_branch_summary_skip_prompt
SettingsManager.getRetrySettings = SettingsManager.get_retry_settings
SettingsManager.getRetryEnabled = SettingsManager.get_retry_enabled
SettingsManager.setRetryEnabled = SettingsManager.set_retry_enabled
SettingsManager.getHttpIdleTimeoutMs = SettingsManager.get_http_idle_timeout_ms
SettingsManager.setHttpIdleTimeoutMs = SettingsManager.set_http_idle_timeout_ms
SettingsManager.getProviderRetrySettings = SettingsManager.get_provider_retry_settings
SettingsManager.getWebSocketConnectTimeoutMs = SettingsManager.get_web_socket_connect_timeout_ms
SettingsManager.getHideThinkingBlock = SettingsManager.get_hide_thinking_block
SettingsManager.setHideThinkingBlock = SettingsManager.set_hide_thinking_block
SettingsManager.getCollapseChangelog = SettingsManager.get_collapse_changelog
SettingsManager.setCollapseChangelog = SettingsManager.set_collapse_changelog
SettingsManager.getEnableInstallTelemetry = SettingsManager.get_enable_install_telemetry
SettingsManager.setEnableInstallTelemetry = SettingsManager.set_enable_install_telemetry
SettingsManager.getTerminalSettings = SettingsManager.get_terminal_settings
SettingsManager.getImageSettings = SettingsManager.get_image_settings
SettingsManager.getImageAutoResize = SettingsManager.get_image_auto_resize
SettingsManager.setImageAutoResize = SettingsManager.set_image_auto_resize
SettingsManager.getBlockImages = SettingsManager.get_block_images
SettingsManager.setBlockImages = SettingsManager.set_block_images
SettingsManager.getPackages = SettingsManager.get_packages
SettingsManager.setPackages = SettingsManager.set_packages
SettingsManager.setProjectPackages = SettingsManager.set_project_packages
SettingsManager.getExtensionPaths = SettingsManager.get_extension_paths
SettingsManager.setExtensionPaths = SettingsManager.set_extension_paths
SettingsManager.setProjectExtensionPaths = SettingsManager.set_project_extension_paths
SettingsManager.getSkillPaths = SettingsManager.get_skill_paths
SettingsManager.setSkillPaths = SettingsManager.set_skill_paths
SettingsManager.setProjectSkillPaths = SettingsManager.set_project_skill_paths
SettingsManager.getPromptTemplatePaths = SettingsManager.get_prompt_template_paths
SettingsManager.setPromptTemplatePaths = SettingsManager.set_prompt_template_paths
SettingsManager.setProjectPromptTemplatePaths = SettingsManager.set_project_prompt_template_paths
SettingsManager.getThemePaths = SettingsManager.get_theme_paths
SettingsManager.setThemePaths = SettingsManager.set_theme_paths
SettingsManager.setProjectThemePaths = SettingsManager.set_project_theme_paths
SettingsManager.getEnableSkillCommands = SettingsManager.get_enable_skill_commands
SettingsManager.setEnableSkillCommands = SettingsManager.set_enable_skill_commands
SettingsManager.getThinkingBudgets = SettingsManager.get_thinking_budgets
SettingsManager.getShowImages = SettingsManager.get_show_images
SettingsManager.setShowImages = SettingsManager.set_show_images
SettingsManager.getImageWidthCells = SettingsManager.get_image_width_cells
SettingsManager.setImageWidthCells = SettingsManager.set_image_width_cells
SettingsManager.getClearOnShrink = SettingsManager.get_clear_on_shrink
SettingsManager.setClearOnShrink = SettingsManager.set_clear_on_shrink
SettingsManager.getShowTerminalProgress = SettingsManager.get_show_terminal_progress
SettingsManager.setShowTerminalProgress = SettingsManager.set_show_terminal_progress
SettingsManager.getEnabledModels = SettingsManager.get_enabled_models
SettingsManager.setEnabledModels = SettingsManager.set_enabled_models
SettingsManager.getDoubleEscapeAction = SettingsManager.get_double_escape_action
SettingsManager.setDoubleEscapeAction = SettingsManager.set_double_escape_action
SettingsManager.getTreeFilterMode = SettingsManager.get_tree_filter_mode
SettingsManager.setTreeFilterMode = SettingsManager.set_tree_filter_mode
SettingsManager.getShowHardwareCursor = SettingsManager.get_show_hardware_cursor
SettingsManager.setShowHardwareCursor = SettingsManager.set_show_hardware_cursor
SettingsManager.getEditorPaddingX = SettingsManager.get_editor_padding_x
SettingsManager.setEditorPaddingX = SettingsManager.set_editor_padding_x
SettingsManager.getAutocompleteMaxVisible = SettingsManager.get_autocomplete_max_visible
SettingsManager.setAutocompleteMaxVisible = SettingsManager.set_autocomplete_max_visible
SettingsManager.getCodeBlockIndent = SettingsManager.get_code_block_indent
SettingsManager.getWarnings = SettingsManager.get_warnings
SettingsManager.setWarnings = SettingsManager.set_warnings

__all__ = [
    "BranchSummarySettings",
    "CompactionSettings",
    "FileSettingsStorage",
    "ImageSettings",
    "InMemorySettingsStorage",
    "MarkdownSettings",
    "RetrySettings",
    "Settings",
    "SettingsManager",
    "SettingsStorage",
    "TerminalSettings",
    "ThinkingBudgetsSettings",
    "WarningSettings",
    "WarningSettings",
    "deep_merge_settings",
    "migrate_settings",
]
