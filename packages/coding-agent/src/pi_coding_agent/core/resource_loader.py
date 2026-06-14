"""
Resource loader for skills, prompts, themes, extensions, and AGENTS.md files.

Mirrors core/resource-loader.ts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from pi_coding_agent.core.diagnostics import ResourceDiagnostic
from pi_coding_agent.core.prompt_templates import (
    LoadPromptTemplatesOptions,
    PromptTemplate,
    load_prompt_templates,
)
from pi_coding_agent.core.skills import LoadSkillsOptions, Skill, load_skills


@dataclass
class PathMetadata:
    source: str
    scope: str  # "user" | "project" | "temporary"
    origin: str  # "package" | "top-level"
    base_dir: str | None = None


@dataclass
class ResourceExtensionPaths:
    skill_paths: list[dict[str, Any]] = field(default_factory=list)
    prompt_paths: list[dict[str, Any]] = field(default_factory=list)
    theme_paths: list[dict[str, Any]] = field(default_factory=list)


_CONTEXT_CANDIDATES = ["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"]


def get_extension_discovery_paths(
    cwd: str,
    agent_dir: str | None = None,
    *,
    inherit_global: bool = True,
) -> list[str]:
    """Return extension module paths discovered from configured extensions dirs."""
    from pi_coding_agent.config import CONFIG_DIR_NAME, agent_dir_env, get_agent_dir
    from pi_coding_agent.core.extensions.loader import discover_extensions_in_dir

    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    resolved_agent_dir = os.path.abspath(os.path.expanduser(agent_dir or get_agent_dir()))
    explicit_agent_dir = bool(agent_dir_env())

    project_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "extensions")
    dirs: list[str] = []

    if explicit_agent_dir:
        dirs.append(os.path.join(resolved_agent_dir, "extensions"))
    elif inherit_global:
        # get_agent_dir() is ~/.tau/agent; Python extensions live in
        # ~/.tau/extensions so they are not mixed with agent internals.
        global_dir = os.path.join(os.path.dirname(resolved_agent_dir), "extensions")
        dirs.append(global_dir)

    if not explicit_agent_dir:
        dirs.append(project_dir)

    paths: list[str] = []
    seen: set[str] = set()

    # First-class, on-by-default bundled extensions: their extension modules load
    # for every agent unless explicitly disabled (each owns its own kill-switch).
    for _module in ("pi_coding_agent.clarity_pii", "pi_coding_agent.active_compression"):
        try:
            import importlib

            mod = importlib.import_module(_module)
            if mod.is_enabled():
                bpath = mod.builtin_extension_path()
                resolved = os.path.abspath(bpath)
                if os.path.exists(bpath) and resolved not in seen:
                    seen.add(resolved)
                    paths.append(bpath)
        except Exception:
            pass

    for directory in dirs:
        for path in discover_extensions_in_dir(directory):
            resolved = os.path.abspath(path)
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def _extensions_result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        normalized = dict(result)
    else:
        normalized = {
            "extensions": list(getattr(result, "extensions", []) or []),
            "diagnostics": list(
                getattr(result, "diagnostics", None)
                or getattr(result, "errors", [])
                or []
            ),
        }
        runtime = getattr(result, "runtime", None)
        if isinstance(runtime, dict):
            normalized["runtime"] = runtime
    normalized.setdefault("extensions", [])
    normalized.setdefault("diagnostics", [])
    normalized.setdefault("runtime", {"flagValues": {}})
    normalized["runtime"].setdefault("flagValues", {})
    normalized["runtime"].setdefault("pendingProviderRegistrations", [])
    return normalized


def _load_context_file_from_dir(dir_path: str) -> dict[str, str] | None:
    """Load the first AGENTS.md or CLAUDE.md found in dir_path."""
    for filename in _CONTEXT_CANDIDATES:
        fpath = os.path.join(dir_path, filename)
        if os.path.exists(fpath):
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    return {"path": fpath, "content": f.read()}
            except OSError as e:
                print(f"Warning: Could not read {fpath}: {e}")
    return None


def _load_project_context_files(
    cwd: str | None = None,
    agent_dir: str | None = None,
    project_trusted: bool | None = True,
    walk_ancestors: bool = True,
) -> list[dict[str, str]]:
    """Load AGENTS.md / CLAUDE.md from global and project ancestors.

    When walk_ancestors is False the search is contained to the launch dir
    (cwd) — context from parent directories is not pulled in.
    """
    from pi_coding_agent.config import get_agent_dir

    resolved_cwd = cwd or os.getcwd()
    resolved_agent_dir = agent_dir or get_agent_dir()

    context_files: list[dict[str, str]] = []
    seen: set[str] = set()

    global_ctx = _load_context_file_from_dir(resolved_agent_dir)
    if global_ctx:
        context_files.append(global_ctx)
        seen.add(global_ctx["path"])

    if project_trusted is False:
        return context_files

    ancestor_files: list[dict[str, str]] = []
    current = resolved_cwd
    root = os.path.abspath("/")

    while True:
        ctx = _load_context_file_from_dir(current)
        if ctx and ctx["path"] not in seen:
            ancestor_files.insert(0, ctx)
            seen.add(ctx["path"])

        if not walk_ancestors:
            break  # contained to the launch dir
        if os.path.abspath(current) == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    context_files.extend(ancestor_files)
    return context_files


def load_project_context_files(
    cwd: str | None = None,
    agent_dir: str | None = None,
    project_trusted: bool | None = True,
) -> list[dict[str, str]]:
    """Load AGENTS.md / CLAUDE.md context files for the global and project scopes."""
    return _load_project_context_files(cwd, agent_dir, project_trusted)


def _resolve_prompt_input(input_path: str | None, description: str) -> str | None:
    if not input_path:
        return None
    if os.path.exists(input_path):
        try:
            with open(input_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as e:
            print(f"Warning: Could not read {description} file {input_path}: {e}")
            return input_path
    return input_path


@dataclass
class Theme:
    """A loaded theme."""
    name: str
    path: str
    colors: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefaultResourceLoaderOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    settings_manager: Any = None
    event_bus: Any = None
    additional_extension_paths: list[str] = field(default_factory=list)
    additional_skill_paths: list[str] = field(default_factory=list)
    additional_prompt_template_paths: list[str] = field(default_factory=list)
    additional_theme_paths: list[str] = field(default_factory=list)
    extension_factories: list[Any] = field(default_factory=list)
    no_extensions: bool = False
    no_skills: bool = False
    no_prompt_templates: bool = False
    no_themes: bool = False
    no_context_files: bool = False
    # When False (the harness default), skills/prompts/extensions/themes are
    # discovered from the project dir only; the global agent dir is not scanned.
    inherit_global: bool = True
    system_prompt: str | None = None
    append_system_prompt: str | list[str] | None = None
    extensions_override: Callable | None = None
    skills_override: Callable | None = None
    prompts_override: Callable | None = None
    themes_override: Callable | None = None
    agents_files_override: Callable | None = None
    system_prompt_override: Callable | None = None
    append_system_prompt_override: Callable | None = None


class DefaultResourceLoader:
    """Loads and manages agent resources (skills, prompts, themes, extensions, AGENTS files)."""

    def __init__(self, options: DefaultResourceLoaderOptions | None = None) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir, get_project_agent_dir

        opts = options or DefaultResourceLoaderOptions()
        self._cwd = opts.cwd or os.getcwd()
        # Default: project-local discovery only. When a caller explicitly sets
        # the agent-dir env var (PI_CODING_AGENT_DIR, or its TAU_ alias), that
        # directory is the agent under test and must own its resources even
        # without --inherit.
        from pi_coding_agent.config import agent_dir_env
        explicit_agent_dir = bool(agent_dir_env())
        self._explicit_agent_dir = explicit_agent_dir
        if opts.inherit_global or explicit_agent_dir:
            self._agent_dir = opts.agent_dir or get_agent_dir()
        else:
            self._agent_dir = get_project_agent_dir(self._cwd)
        self._inherit_global = opts.inherit_global
        self._config_dir_name = CONFIG_DIR_NAME
        self._settings_manager = opts.settings_manager
        self._event_bus = opts.event_bus
        self._additional_extension_paths = list(opts.additional_extension_paths)
        self._additional_skill_paths = list(opts.additional_skill_paths)
        self._additional_prompt_paths = list(opts.additional_prompt_template_paths)
        self._additional_theme_paths = list(opts.additional_theme_paths)
        self._extension_factories = list(opts.extension_factories)
        self._no_extensions = opts.no_extensions
        self._no_skills = opts.no_skills
        self._no_prompt_templates = opts.no_prompt_templates
        self._no_themes = opts.no_themes
        self._no_context_files = opts.no_context_files
        self._system_prompt_source = opts.system_prompt
        self._append_system_prompt_source = opts.append_system_prompt
        self._extensions_override = opts.extensions_override
        self._skills_override = opts.skills_override
        self._prompts_override = opts.prompts_override
        self._themes_override = opts.themes_override
        self._agents_files_override = opts.agents_files_override
        self._system_prompt_override = opts.system_prompt_override
        self._append_system_prompt_override = opts.append_system_prompt_override

        self._extensions_result: dict[str, Any] = {"extensions": [], "diagnostics": []}
        self._skills: list[Skill] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._themes: list[Theme] = []
        self._theme_diagnostics: list[ResourceDiagnostic] = []
        self._agents_files: list[dict[str, str]] = []
        self._system_prompt: str | None = None
        self._append_system_prompt: list[str] = []
        self._path_metadata: dict[str, PathMetadata] = {}
        self._last_skill_paths: list[str] = []
        self._last_prompt_paths: list[str] = []
        self._last_theme_paths: list[str] = []
        self._last_extension_paths: list[str] = []

    def get_extensions(self) -> dict[str, Any]:
        """Get loaded extensions result. Mirrors getExtensions() in TypeScript."""
        return dict(self._extensions_result)

    def get_skills(self) -> dict[str, Any]:
        return {"skills": self._skills, "diagnostics": self._skill_diagnostics}

    def get_prompts(self) -> dict[str, Any]:
        return {"prompts": self._prompts, "diagnostics": self._prompt_diagnostics}

    def get_themes(self) -> dict[str, Any]:
        """Get loaded themes. Mirrors getThemes() in TypeScript."""
        return {"themes": self._themes, "diagnostics": self._theme_diagnostics}

    def get_agents_files(self) -> dict[str, Any]:
        return {"agentsFiles": self._agents_files, "agents_files": self._agents_files}

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    def get_path_metadata(self) -> dict[str, PathMetadata]:
        return self._path_metadata

    getExtensions = get_extensions
    getSkills = get_skills
    getPrompts = get_prompts
    getThemes = get_themes
    getAgentsFiles = get_agents_files
    getSystemPrompt = get_system_prompt
    getAppendSystemPrompt = get_append_system_prompt

    def extend_resources(self, paths: ResourceExtensionPaths) -> None:
        skill_paths = getattr(paths, "skill_paths", None)
        prompt_paths = getattr(paths, "prompt_paths", None)
        theme_paths = getattr(paths, "theme_paths", None)
        if isinstance(paths, dict):
            skill_paths = paths.get("skillPaths") or paths.get("skill_paths") or skill_paths
            prompt_paths = paths.get("promptPaths") or paths.get("prompt_paths") or prompt_paths
            theme_paths = paths.get("themePaths") or paths.get("theme_paths") or theme_paths

        if skill_paths:
            new_paths = [entry["path"] if isinstance(entry, dict) else getattr(entry, "path") for entry in skill_paths]
            self._last_skill_paths = self._merge_paths(self._last_skill_paths, new_paths)
            self._update_skills_from_paths(self._last_skill_paths)

        if prompt_paths:
            new_paths = [entry["path"] if isinstance(entry, dict) else getattr(entry, "path") for entry in prompt_paths]
            self._last_prompt_paths = self._merge_paths(self._last_prompt_paths, new_paths)
            self._update_prompts_from_paths(self._last_prompt_paths)

        if theme_paths:
            new_paths = [entry["path"] if isinstance(entry, dict) else getattr(entry, "path") for entry in theme_paths]
            self._last_theme_paths = self._merge_paths(self._last_theme_paths, new_paths)
            self._update_themes_from_paths(self._last_theme_paths)

    extendResources = extend_resources

    async def reload(self) -> None:
        """Reload all resources from disk."""
        self._path_metadata = {}

        # Load extensions (from extension dirs + additional paths)
        if not self._no_extensions:
            await self._load_extensions()

        # Load skills
        skill_paths = self._resolve_resource_paths_from_settings("skills") + self._additional_skill_paths
        merged_skill_paths = self._merge_paths([], skill_paths)
        self._last_skill_paths = merged_skill_paths
        self._update_skills_from_paths(merged_skill_paths)

        # Load prompt templates
        prompt_paths = self._resolve_resource_paths_from_settings("prompts") + self._additional_prompt_paths
        merged_prompt_paths = self._merge_paths([], prompt_paths)
        self._last_prompt_paths = merged_prompt_paths
        self._update_prompts_from_paths(merged_prompt_paths)

        # Load themes
        if not self._no_themes:
            theme_paths = self._resolve_resource_paths_from_settings("themes") + self._additional_theme_paths
            merged_theme_paths = self._merge_paths([], theme_paths)
            self._last_theme_paths = merged_theme_paths
            self._update_themes_from_paths(merged_theme_paths)

        # Load AGENTS.md context files
        project_trusted = True
        if self._settings_manager and hasattr(self._settings_manager, "is_project_trusted"):
            project_trusted = bool(self._settings_manager.is_project_trusted())
        context_files = (
            []
            if self._no_context_files
            else _load_project_context_files(
                self._cwd,
                self._agent_dir,
                False if self._explicit_agent_dir else project_trusted,
                walk_ancestors=self._inherit_global,
            )
        )
        agents_files_base = {"agentsFiles": context_files, "agents_files": context_files}
        resolved = (
            self._agents_files_override(agents_files_base)
            if self._agents_files_override
            else agents_files_base
        )
        self._agents_files = resolved.get("agentsFiles") or resolved.get("agents_files", [])

        # System prompt
        base_system = _resolve_prompt_input(
            self._system_prompt_source or self._discover_system_prompt_file(),
            "system prompt",
        )
        self._system_prompt = (
            self._system_prompt_override(base_system)
            if self._system_prompt_override
            else base_system
        )

        append_source = self._append_system_prompt_source
        if append_source is None:
            append_source = self._discover_append_system_prompt_file()
        append_sources = append_source if isinstance(append_source, list) else [append_source]
        base_append = [
            resolved
            for source in append_sources
            if source
            for resolved in [_resolve_prompt_input(source, "append system prompt")]
            if resolved
        ]
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override
            else base_append
        )

    def _resolve_resource_paths_from_settings(self, resource_type: str) -> list[str]:
        """Get additional resource paths from settings manager."""
        if not self._settings_manager:
            return []
        try:
            getter = getattr(self._settings_manager, f"get_{resource_type}", None)
            if callable(getter):
                val = getter()
                if isinstance(val, list):
                    return [str(p) for p in val if isinstance(p, str)]
        except Exception:
            pass
        return []

    async def _load_extensions(self) -> None:
        """
        Load extensions from discovered extension directories + additional paths.
        Detects conflicts in tool/command/flag names.
        Mirrors extension loading in DefaultResourceLoader.reload() in TypeScript.
        """
        ext_paths = (
            get_extension_discovery_paths(
                self._cwd,
                self._agent_dir,
                inherit_global=self._inherit_global,
            )
            + self._additional_extension_paths
        )
        ext_paths = self._merge_paths([], ext_paths)

        if not ext_paths and not self._extension_factories:
            base_result: dict[str, Any] = {"extensions": [], "diagnostics": []}
        else:
            try:
                from pi_coding_agent.core.extensions.loader import load_extensions
                event_bus = self._event_bus
                if event_bus is None:
                    from pi_coding_agent.core.event_bus import create_event_bus
                    event_bus = create_event_bus()
                loaded_result = await load_extensions(
                    ext_paths,
                    self._cwd,
                    event_bus,
                )
                base_result = _extensions_result_to_dict(loaded_result)
                # Detect conflicts
                base_result = self._detect_extension_conflicts(base_result)
            except Exception as e:
                base_result = {"extensions": [], "diagnostics": [{"type": "error", "message": str(e)}]}

        resolved = (
            self._extensions_override(base_result)
            if self._extensions_override
            else base_result
        )
        resolved = _extensions_result_to_dict(resolved)
        self._extensions_result = resolved

    def _detect_extension_conflicts(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Detect name collisions in tools, commands, and flags across extensions.
        Mirrors conflict detection in TypeScript.
        """
        seen_tools: dict[str, str] = {}
        seen_commands: dict[str, str] = {}
        seen_flags: dict[str, str] = {}
        diagnostics: list[dict[str, Any]] = list(result.get("diagnostics", []))
        extensions: list[Any] = result.get("extensions", [])

        for ext in extensions:
            ext_path = getattr(ext, "path", "") or ""

            tools = getattr(ext, "tools", {}) or {}
            tool_values = tools.values() if isinstance(tools, dict) else tools
            for tool in tool_values or []:
                name = getattr(tool, "name", "") or ""
                if name in seen_tools:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Tool name "{name}" collision between {seen_tools[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_tools[name] = ext_path

            commands = getattr(ext, "commands", {}) or {}
            command_values = commands.values() if isinstance(commands, dict) else commands
            for cmd in command_values or []:
                name = getattr(cmd, "name", "") or ""
                if name in seen_commands:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Command name "{name}" collision between {seen_commands[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_commands[name] = ext_path

            flags = getattr(ext, "flags", {}) or {}
            flag_values = flags.values() if isinstance(flags, dict) else flags
            for flag in flag_values or []:
                name = getattr(flag, "name", "") or ""
                if name in seen_flags:
                    diagnostics.append({
                        "type": "collision",
                        "message": f'Flag name "{name}" collision between {seen_flags[name]} and {ext_path}',
                        "path": ext_path,
                    })
                else:
                    seen_flags[name] = ext_path

        return {**result, "diagnostics": diagnostics}

    def _update_themes_from_paths(self, theme_paths: list[str]) -> None:
        """Load themes from paths."""
        themes: list[Theme] = []
        diagnostics: list[ResourceDiagnostic] = []

        for path in theme_paths:
            if not os.path.exists(path):
                continue
            try:
                if os.path.isfile(path) and path.endswith(".json"):
                    import json
                    with open(path, encoding="utf-8") as f:
                        colors = json.load(f)
                    name = os.path.splitext(os.path.basename(path))[0]
                    themes.append(Theme(name=name, path=path, colors=colors))
                elif os.path.isdir(path):
                    for fname in os.listdir(path):
                        if fname.endswith(".json"):
                            fpath = os.path.join(path, fname)
                            try:
                                import json
                                with open(fpath, encoding="utf-8") as f:
                                    colors = json.load(f)
                                name = os.path.splitext(fname)[0]
                                themes.append(Theme(name=name, path=fpath, colors=colors))
                            except Exception as e:
                                diagnostics.append(ResourceDiagnostic(
                                    type="error",
                                    message=f"Failed to load theme {fpath}: {e}",
                                    path=fpath,
                                ))
            except Exception as e:
                diagnostics.append(ResourceDiagnostic(
                    type="error",
                    message=f"Failed to load theme from {path}: {e}",
                    path=path,
                ))

        themes_result: dict[str, Any] = {"themes": themes, "diagnostics": diagnostics}
        resolved = self._themes_override(themes_result) if self._themes_override else themes_result
        self._themes = resolved["themes"]
        self._theme_diagnostics = resolved.get("diagnostics", [])

    def _update_skills_from_paths(self, skill_paths: list[str]) -> None:
        if self._no_skills and not skill_paths:
            skills_result = {"skills": [], "diagnostics": []}
        else:
            result = load_skills(
                LoadSkillsOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    skill_paths=skill_paths,
                    include_defaults=not self._no_skills,
                )
            )
            skills_result = {"skills": result.skills, "diagnostics": result.diagnostics}

        resolved = (
            self._skills_override(skills_result)
            if self._skills_override
            else skills_result
        )
        self._skills = resolved["skills"]
        self._skill_diagnostics = resolved["diagnostics"]

    def _update_prompts_from_paths(self, prompt_paths: list[str]) -> None:
        if self._no_prompt_templates and not prompt_paths:
            prompts_result: dict[str, Any] = {"prompts": [], "diagnostics": []}
        else:
            all_prompts = load_prompt_templates(
                LoadPromptTemplatesOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    prompt_paths=prompt_paths,
                    include_defaults=not self._no_prompt_templates,
                )
            )
            deduped = self._dedupe_prompts(all_prompts)
            prompts_result = deduped

        resolved = (
            self._prompts_override(prompts_result)
            if self._prompts_override
            else prompts_result
        )
        self._prompts = resolved["prompts"]
        self._prompt_diagnostics = resolved.get("diagnostics", [])

    def _dedupe_prompts(
        self, prompts: list[PromptTemplate]
    ) -> dict[str, Any]:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for prompt in prompts:
            if prompt.name in seen:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "/{prompt.name}" collision',
                        path=prompt.file_path,
                    )
                )
            else:
                seen[prompt.name] = prompt
        return {"prompts": list(seen.values()), "diagnostics": diagnostics}

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for p in primary + additional:
            resolved = self._resolve_resource_path(p)
            if resolved not in seen:
                seen.add(resolved)
                merged.append(resolved)
        return merged

    def _resolve_resource_path(self, p: str) -> str:
        home = os.path.expanduser("~")
        t = p.strip()
        if t == "~":
            expanded = home
        elif t.startswith("~/"):
            expanded = os.path.join(home, t[2:])
        elif t.startswith("~"):
            expanded = os.path.join(home, t[1:])
        else:
            expanded = t
        return os.path.abspath(os.path.join(self._cwd, expanded))

    def _discover_system_prompt_file(self) -> str | None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        global_path = os.path.join(self._agent_dir, "SYSTEM.md")
        if self._explicit_agent_dir and os.path.exists(global_path):
            return global_path
        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        if os.path.exists(global_path):
            return global_path
        return None

    def _discover_append_system_prompt_file(self) -> str | None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        global_path = os.path.join(self._agent_dir, "APPEND_SYSTEM.md")
        if self._explicit_agent_dir and os.path.exists(global_path):
            return global_path
        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "APPEND_SYSTEM.md")
        if os.path.exists(project_path):
            return project_path
        if os.path.exists(global_path):
            return global_path
        return None
