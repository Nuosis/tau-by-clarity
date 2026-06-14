"""
Package manager for installing, removing, and resolving agent extension packages.

Supports local paths, npm packages (via npm CLI), and git repositories.
Mirrors core/package-manager.ts
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pi_coding_agent.utils.git import GitSource, parse_git_url


SourceScope = Literal["user", "project", "temporary"]
ResourceType = Literal["extensions", "skills", "prompts", "themes"]
MissingSourceAction = Literal["install", "skip", "error"]

RESOURCE_TYPES: list[ResourceType] = ["extensions", "skills", "prompts", "themes"]

_FILE_PATTERNS: dict[ResourceType, str] = {
    "extensions": r"\.(ts|js)$",
    "skills": r"\.md$",
    "prompts": r"\.md$",
    "themes": r"\.json$",
}

_IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass
class PathMetadata:
    source: str
    scope: SourceScope
    origin: Literal["package", "top-level"]
    base_dir: str | None = None


@dataclass
class ResolvedResource:
    path: str
    enabled: bool
    metadata: PathMetadata


@dataclass
class ResolvedPaths:
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)


@dataclass
class ConfiguredPackage:
    source: str
    scope: Literal["user", "project"]
    filtered: bool
    installed_path: str | None = None


@dataclass
class ProgressEvent:
    type: Literal["start", "progress", "complete", "error"]
    action: Literal["install", "remove", "update", "clone", "pull"]
    source: str
    message: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class _NpmSource:
    type: Literal["npm"] = "npm"
    spec: str = ""
    name: str = ""
    pinned: bool = False


@dataclass
class _LocalSource:
    type: Literal["local"] = "local"
    path: str = ""


_ParsedSource = _NpmSource | GitSource | _LocalSource


@dataclass
class _PiManifest:
    extensions: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)


def _is_override_pattern(s: str) -> bool:
    return s.startswith("!") or s.startswith("+") or s.startswith("-")


def _is_glob_pattern(s: str) -> bool:
    return "*" in s or "?" in s


def _collect_files(dir_path: str, pattern: str) -> list[str]:
    """Recursively collect files matching regex pattern in dir_path."""
    result: list[str] = []
    if not os.path.isdir(dir_path):
        return result
    rx = re.compile(pattern)
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
        for fname in files:
            if rx.search(fname):
                result.append(os.path.join(root, fname))
    return result


def _collect_skill_entries(dir_path: str, include_root_files: bool = True) -> list[str]:
    entries: list[str] = []
    if not os.path.isdir(dir_path):
        return entries
    try:
        for item in sorted(os.scandir(dir_path), key=lambda e: e.name):
            if item.name.startswith(".") or item.name in ("node_modules", "__pycache__"):
                continue
            full = item.path
            is_dir = item.is_dir(follow_symlinks=True)
            is_file = item.is_file(follow_symlinks=True)

            if is_dir:
                entries.extend(_collect_skill_entries(full, False))
            elif is_file:
                if include_root_files and item.name.endswith(".md"):
                    entries.append(full)
                elif not include_root_files and item.name == "SKILL.md":
                    entries.append(full)
    except OSError:
        pass
    return entries


def _collect_resource_files(dir_path: str, resource_type: ResourceType) -> list[str]:
    if resource_type == "skills":
        return _collect_skill_entries(dir_path)
    return _collect_files(dir_path, _FILE_PATTERNS[resource_type])


def _find_git_repo_root(start_dir: str) -> str | None:
    current = os.path.realpath(os.path.abspath(os.path.expanduser(start_dir)))
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _collect_ancestor_agents_skill_dirs(start_dir: str) -> list[str]:
    skill_dirs: list[str] = []
    current = os.path.realpath(os.path.abspath(os.path.expanduser(start_dir)))
    git_root = _find_git_repo_root(current)
    while True:
        skill_dirs.append(os.path.join(current, ".agents", "skills"))
        if git_root and current == git_root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return skill_dirs


def _apply_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> set[str]:
    includes: list[str] = []
    excludes: list[str] = []
    force_includes: list[str] = []
    force_excludes: list[str] = []

    for p in patterns:
        if p.startswith("+"):
            force_includes.append(p[1:])
        elif p.startswith("-"):
            force_excludes.append(p[1:])
        elif p.startswith("!"):
            excludes.append(p[1:])
        else:
            includes.append(p)

    def _matches(path: str, pats: list[str]) -> bool:
        rel = os.path.relpath(path, base_dir)
        name = os.path.basename(path)
        return any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat)
            for pat in pats
        )

    result = list(all_paths) if not includes else [p for p in all_paths if _matches(p, includes)]
    if excludes:
        result = [p for p in result if not _matches(p, excludes)]
    if force_includes:
        for fp in all_paths:
            if fp not in result and _matches(fp, force_includes):
                result.append(fp)
    if force_excludes:
        result = [p for p in result if not _matches(p, force_excludes)]
    return set(result)


def _split_patterns(entries: list[str]) -> tuple[list[str], list[str]]:
    plain: list[str] = []
    patterns: list[str] = []
    for entry in entries:
        if _is_override_pattern(entry):
            patterns.append(entry)
        else:
            plain.append(entry)
    return plain, patterns


class DefaultPackageManager:
    """Resolves, installs, updates, and removes agent resource packages."""

    def __init__(
        self,
        cwd: str,
        agent_dir: str,
        settings_manager: Any = None,
    ) -> None:
        from pi_coding_agent.config import CONFIG_DIR_NAME

        self._cwd = cwd
        self._agent_dir = agent_dir
        self._settings_manager = settings_manager
        self._config_dir_name = CONFIG_DIR_NAME
        self._progress_callback: ProgressCallback | None = None

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self._progress_callback = callback

    def _emit(self, event: ProgressEvent) -> None:
        if self._progress_callback:
            self._progress_callback(event)

    async def _with_progress(
        self,
        action: str,
        source: str,
        message: str,
        operation: Any,
    ) -> None:
        self._emit(ProgressEvent(type="start", action=action, source=source, message=message))
        try:
            await operation()
            self._emit(ProgressEvent(type="complete", action=action, source=source))
        except Exception as e:
            self._emit(ProgressEvent(type="error", action=action, source=source, message=str(e)))
            raise

    def _parse_source(self, source: str) -> _ParsedSource:
        if source.startswith("npm:"):
            spec = source[4:].strip()
            name, version = self._parse_npm_spec(spec)
            return _NpmSource(spec=spec, name=name, pinned=bool(version))

        trimmed = source.strip()
        is_win_abs = re.match(r"^[A-Za-z]:[\\/]|^\\\\", trimmed)
        is_local = (
            trimmed.startswith(".")
            or trimmed.startswith("/")
            or trimmed in ("~", )
            or trimmed.startswith("~/")
            or bool(is_win_abs)
        )
        if is_local:
            return _LocalSource(path=source)

        git_parsed = parse_git_url(source)
        if git_parsed:
            return git_parsed

        return _LocalSource(path=source)

    def _parse_npm_spec(self, spec: str) -> tuple[str, str | None]:
        m = re.match(r"^(@?[^@]+(?:/[^@]+)?)(?:@(.+))?$", spec)
        if not m:
            return spec, None
        return m.group(1) or spec, m.group(2)

    def _resolve_path(self, input_path: str) -> str:
        home = os.path.expanduser("~")
        t = input_path.strip()
        if t == "~":
            return home
        if t.startswith("~/"):
            return os.path.join(home, t[2:])
        if t.startswith("~"):
            return os.path.join(home, t[1:])
        return os.path.abspath(os.path.join(self._cwd, t))

    def _resolve_path_from_base(self, input_path: str, base_dir: str) -> str:
        home = os.path.expanduser("~")
        t = input_path.strip()
        if t == "~":
            return home
        if t.startswith("~/"):
            return os.path.join(home, t[2:])
        if t.startswith("~"):
            return os.path.join(home, t[1:])
        return os.path.abspath(os.path.join(base_dir, t))

    def _get_base_dir_for_scope(self, scope: SourceScope) -> str:
        if scope == "project":
            return os.path.join(self._cwd, self._config_dir_name)
        if scope == "user":
            return self._agent_dir
        return self._cwd

    def _get_npm_install_root(self, scope: SourceScope, temporary: bool) -> str:
        if temporary:
            return self._get_temporary_dir("npm")
        if scope == "project":
            return os.path.join(self._cwd, self._config_dir_name, "npm")
        return os.path.expanduser("~/.npm")

    def _get_npm_install_path(self, source: _NpmSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return os.path.join(self._get_temporary_dir("npm"), "node_modules", source.name)
        if scope == "project":
            return os.path.join(
                self._cwd, self._config_dir_name, "npm", "node_modules", source.name
            )
        npm_root = self._get_global_npm_root()
        return os.path.join(npm_root, source.name)

    def _get_global_npm_root(self) -> str:
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return os.path.expanduser("~/.npm/lib/node_modules")

    def _get_git_install_path(self, source: GitSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return self._get_temporary_dir(f"git-{source.host}", source.path)
        if scope == "project":
            return os.path.join(
                self._cwd, self._config_dir_name, "git", source.host, source.path
            )
        return os.path.join(self._agent_dir, "git", source.host, source.path)

    def _get_temporary_dir(self, prefix: str, suffix: str | None = None) -> str:
        key = f"{prefix}-{suffix or ''}"
        hash_val = hashlib.sha256(key.encode()).hexdigest()[:8]
        return os.path.join(tempfile.gettempdir(), "pi-extensions", prefix, hash_val, suffix or "")

    def get_installed_path(self, source: str, scope: SourceScope) -> str | None:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            path = self._get_npm_install_path(parsed, scope)
            return path if os.path.exists(path) else None
        if isinstance(parsed, GitSource):
            path = self._get_git_install_path(parsed, scope)
            return path if os.path.exists(path) else None
        if isinstance(parsed, _LocalSource):
            base = self._get_base_dir_for_scope(scope)
            path = self._resolve_path_from_base(parsed.path, base)
            return path if os.path.exists(path) else None
        return None

    async def resolve(
        self,
        on_missing: Callable[[str], Any] | None = None,
    ) -> ResolvedPaths:
        """Resolve all configured packages to resource paths."""
        accumulator: dict[ResourceType, dict[str, tuple[PathMetadata, bool]]] = {
            "extensions": {}, "skills": {}, "prompts": {}, "themes": {}
        }

        if self._settings_manager:
            global_settings = self._settings_manager.get_global_settings()
            project_settings = self._settings_manager.get_project_settings()

            all_packages: list[tuple[Any, SourceScope]] = []
            for pkg in project_settings.get("packages", []):
                all_packages.append((pkg, "project"))
            for pkg in global_settings.get("packages", []):
                all_packages.append((pkg, "user"))

            for pkg, scope in self._dedupe_packages(all_packages):
                source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
                package_filter = pkg if isinstance(pkg, dict) else None
                await self._resolve_source_to_accumulator(
                    source_str, scope, accumulator, on_missing, package_filter
                )

            global_base_dir = self._agent_dir
            project_base_dir = os.path.join(self._cwd, self._config_dir_name)
            for rt in RESOURCE_TYPES:
                project_entries = project_settings.get(rt, []) or []
                global_entries = global_settings.get(rt, []) or []
                self._resolve_local_entries(
                    project_entries,
                    rt,
                    accumulator[rt],
                    PathMetadata(source="local", scope="project", origin="top-level", base_dir=project_base_dir),
                    project_base_dir,
                )
                self._resolve_local_entries(
                    global_entries,
                    rt,
                    accumulator[rt],
                    PathMetadata(source="local", scope="user", origin="top-level", base_dir=global_base_dir),
                    global_base_dir,
                )

            self._add_auto_discovered_resources(
                accumulator,
                global_settings,
                project_settings,
                global_base_dir,
                project_base_dir,
            )

        return self._to_resolved_paths(accumulator)

    async def resolve_extension_sources(
        self,
        sources: list[str],
        options: dict[str, Any] | None = None,
    ) -> ResolvedPaths:
        opts = options or {}
        scope: SourceScope = "temporary" if opts.get("temporary") else ("project" if opts.get("local") else "user")
        accumulator: dict[ResourceType, dict[str, tuple[PathMetadata, bool]]] = {
            "extensions": {}, "skills": {}, "prompts": {}, "themes": {}
        }
        for source in sources:
            await self._resolve_source_to_accumulator(source, scope, accumulator)
        return self._to_resolved_paths(accumulator)

    async def _resolve_source_to_accumulator(
        self,
        source_str: str,
        scope: SourceScope,
        accumulator: dict,
        on_missing: Callable | None = None,
        package_filter: dict[str, list[str]] | None = None,
    ) -> None:
        parsed = self._parse_source(source_str)
        metadata = PathMetadata(source=source_str, scope=scope, origin="package")

        if isinstance(parsed, _LocalSource):
            base = self._get_base_dir_for_scope(scope)
            resolved = self._resolve_path_from_base(parsed.path, base)
            if os.path.exists(resolved):
                if os.path.isfile(resolved):
                    metadata.base_dir = os.path.dirname(resolved)
                    self._add_to_accumulator(accumulator["extensions"], resolved, metadata, True)
                elif os.path.isdir(resolved):
                    metadata.base_dir = resolved
                    collected = self._collect_package_resources(resolved, accumulator, metadata, package_filter)
                    if not collected:
                        self._add_to_accumulator(accumulator["extensions"], resolved, metadata, True)
            return

        if isinstance(parsed, _NpmSource):
            install_path = self._get_npm_install_path(parsed, scope)
            if not os.path.exists(install_path):
                if on_missing:
                    action = await on_missing(source_str)
                    if action == "skip":
                        return
                    if action == "error":
                        raise RuntimeError(f"Missing source: {source_str}")
                await self._install_npm(parsed, scope, scope == "temporary")
            metadata.base_dir = install_path
            self._collect_package_resources(install_path, accumulator, metadata, package_filter)
            return

        if isinstance(parsed, GitSource):
            install_path = self._get_git_install_path(parsed, scope)
            if not os.path.exists(install_path):
                if on_missing:
                    action = await on_missing(source_str)
                    if action == "skip":
                        return
                    if action == "error":
                        raise RuntimeError(f"Missing source: {source_str}")
                await self._install_git(parsed, scope)
            metadata.base_dir = install_path
            self._collect_package_resources(install_path, accumulator, metadata, package_filter)

    def _collect_package_resources(
        self,
        package_root: str,
        accumulator: dict,
        metadata: PathMetadata,
        package_filter: dict[str, list[str]] | None = None,
    ) -> bool:
        if package_filter is not None:
            for rt in RESOURCE_TYPES:
                patterns = package_filter.get(rt)
                if patterns is None:
                    self._collect_default_resources(package_root, rt, accumulator[rt], metadata)
                else:
                    self._apply_package_filter(package_root, patterns, rt, accumulator[rt], metadata)
            return True

        manifest = self._read_pi_manifest(package_root)
        if manifest:
            for rt in RESOURCE_TYPES:
                entries = getattr(manifest, rt, [])
                self._add_manifest_entries(entries, package_root, rt, accumulator[rt], metadata)
            return True

        has_any = False
        for rt in RESOURCE_TYPES:
            d = os.path.join(package_root, rt)
            if os.path.isdir(d):
                files = _collect_resource_files(d, rt)
                for f in files:
                    self._add_to_accumulator(accumulator[rt], f, metadata, True)
                has_any = True
        return has_any

    def _collect_default_resources(
        self,
        package_root: str,
        resource_type: ResourceType,
        target: dict,
        metadata: PathMetadata,
    ) -> None:
        manifest = self._read_pi_manifest(package_root)
        entries = getattr(manifest, resource_type, None) if manifest else None
        if entries is not None:
            self._add_manifest_entries(entries, package_root, resource_type, target, metadata)
            return
        d = os.path.join(package_root, resource_type)
        if os.path.isdir(d):
            for f in _collect_resource_files(d, resource_type):
                self._add_to_accumulator(target, f, metadata, True)

    def _collect_files_from_paths(
        self,
        entries: list[str],
        root: str,
        resource_type: ResourceType,
    ) -> list[str]:
        files: list[str] = []
        for entry in entries:
            resolved = os.path.abspath(os.path.join(root, entry))
            if os.path.isfile(resolved):
                files.append(resolved)
            elif os.path.isdir(resolved):
                files.extend(_collect_resource_files(resolved, resource_type))
            elif _is_glob_pattern(entry):
                import glob

                for match in glob.glob(os.path.join(root, entry), recursive=True):
                    if os.path.isfile(match):
                        files.append(os.path.abspath(match))
                    elif os.path.isdir(match):
                        files.extend(_collect_resource_files(match, resource_type))
        return list(dict.fromkeys(files))

    def _resolve_local_entries(
        self,
        entries: list[str],
        resource_type: ResourceType,
        target: dict,
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        if not entries:
            return
        plain, patterns = _split_patterns(entries)
        all_files = self._collect_files_from_paths(plain, base_dir, resource_type)
        enabled = _apply_patterns(all_files, patterns, base_dir)
        for f in all_files:
            self._add_to_accumulator(target, f, metadata, f in enabled)

    def _is_enabled_by_overrides(self, file_path: str, patterns: list[str], base_dir: str) -> bool:
        overrides = [pattern for pattern in patterns if _is_override_pattern(pattern)]
        excludes = [pattern[1:] for pattern in overrides if pattern.startswith("!")]
        force_includes = [pattern[1:] for pattern in overrides if pattern.startswith("+")]
        force_excludes = [pattern[1:] for pattern in overrides if pattern.startswith("-")]

        def _matches(path: str, pats: list[str]) -> bool:
            rel = os.path.relpath(path, base_dir)
            name = os.path.basename(path)
            return any(
                fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat)
                for pat in pats
            )

        enabled = True
        if excludes and _matches(file_path, excludes):
            enabled = False
        if force_includes and _matches(file_path, force_includes):
            enabled = True
        if force_excludes and _matches(file_path, force_excludes):
            enabled = False
        return enabled

    def _add_auto_resources(
        self,
        resource_type: ResourceType,
        paths: list[str],
        target: dict,
        metadata: PathMetadata,
        overrides: list[str],
        base_dir: str,
    ) -> None:
        for path in paths:
            self._add_to_accumulator(
                target,
                path,
                metadata,
                self._is_enabled_by_overrides(path, overrides, base_dir),
            )

    def _add_auto_discovered_resources(
        self,
        accumulator: dict,
        global_settings: dict[str, Any],
        project_settings: dict[str, Any],
        global_base_dir: str,
        project_base_dir: str,
    ) -> None:
        user_metadata = PathMetadata(source="auto", scope="user", origin="top-level", base_dir=global_base_dir)
        project_metadata = PathMetadata(source="auto", scope="project", origin="top-level", base_dir=project_base_dir)
        project_trusted = self._settings_manager is None or self._settings_manager.is_project_trusted()
        for rt in RESOURCE_TYPES:
            project_dir = os.path.join(project_base_dir, rt)
            if project_trusted:
                self._add_auto_resources(
                    rt,
                    _collect_resource_files(project_dir, rt),
                    accumulator[rt],
                    project_metadata,
                    project_settings.get(rt, []) or [],
                    project_base_dir,
                )
            user_dir = os.path.join(global_base_dir, rt)
            self._add_auto_resources(
                rt,
                _collect_resource_files(user_dir, rt),
                accumulator[rt],
                user_metadata,
                global_settings.get(rt, []) or [],
                global_base_dir,
            )

        user_agents_skills_dir = os.path.join(os.path.expanduser("~"), ".agents", "skills")
        user_agents_base_dir = os.path.dirname(user_agents_skills_dir)
        user_agents_metadata = PathMetadata(
            source="auto",
            scope="user",
            origin="top-level",
            base_dir=user_agents_base_dir,
        )
        self._add_auto_resources(
            "skills",
            _collect_skill_entries(user_agents_skills_dir),
            accumulator["skills"],
            user_agents_metadata,
            global_settings.get("skills", []) or [],
            user_agents_base_dir,
        )

        if project_trusted:
            user_agents_real = os.path.realpath(user_agents_skills_dir)
            for agents_skills_dir in _collect_ancestor_agents_skill_dirs(self._cwd):
                if os.path.realpath(agents_skills_dir) == user_agents_real:
                    continue
                agents_base_dir = os.path.dirname(agents_skills_dir)
                agents_metadata = PathMetadata(
                    source="auto",
                    scope="project",
                    origin="top-level",
                    base_dir=agents_base_dir,
                )
                self._add_auto_resources(
                    "skills",
                    _collect_skill_entries(agents_skills_dir),
                    accumulator["skills"],
                    agents_metadata,
                    project_settings.get("skills", []) or [],
                    agents_base_dir,
                )

    def _collect_manifest_files(self, package_root: str, resource_type: ResourceType) -> list[str]:
        manifest = self._read_pi_manifest(package_root)
        entries = getattr(manifest, resource_type, []) if manifest else []
        if entries:
            plain, patterns = _split_patterns(entries)
            all_files = self._collect_files_from_paths(plain, package_root, resource_type)
            if patterns:
                return sorted(_apply_patterns(all_files, patterns, package_root))
            return all_files
        d = os.path.join(package_root, resource_type)
        return _collect_resource_files(d, resource_type) if os.path.isdir(d) else []

    def _apply_package_filter(
        self,
        package_root: str,
        patterns: list[str],
        resource_type: ResourceType,
        target: dict,
        metadata: PathMetadata,
    ) -> None:
        all_files = self._collect_manifest_files(package_root, resource_type)
        enabled = _apply_patterns(all_files, patterns, package_root) if patterns else set()
        for f in all_files:
            self._add_to_accumulator(target, f, metadata, f in enabled)

    def _add_manifest_entries(
        self,
        entries: list[str],
        root: str,
        resource_type: ResourceType,
        target: dict,
        metadata: PathMetadata,
    ) -> None:
        if not entries:
            return
        plain, patterns = _split_patterns(entries)
        all_files = self._collect_files_from_paths(plain, root, resource_type)

        enabled = _apply_patterns(all_files, patterns, root)
        for f in all_files:
            if f in enabled:
                self._add_to_accumulator(target, f, metadata, True)

    def _read_pi_manifest(self, package_root: str) -> _PiManifest | None:
        pkg_json = os.path.join(package_root, "package.json")
        if not os.path.exists(pkg_json):
            return None
        try:
            with open(pkg_json) as f:
                data = json.load(f)
            pi_data = data.get("pi", {})
            if not isinstance(pi_data, dict):
                return None
            return _PiManifest(
                extensions=pi_data.get("extensions", []),
                skills=pi_data.get("skills", []),
                prompts=pi_data.get("prompts", []),
                themes=pi_data.get("themes", []),
            )
        except Exception:
            return None

    def _add_to_accumulator(
        self,
        target: dict,
        path: str,
        metadata: PathMetadata,
        enabled: bool,
    ) -> None:
        if path and path not in target:
            target[path] = (metadata, enabled)

    def _resource_precedence_rank(self, metadata: PathMetadata) -> int:
        if metadata.origin == "package":
            return 4
        scope_base = 0 if metadata.scope == "project" else 2
        return scope_base + (0 if metadata.source == "local" else 1)

    def _dedupe_packages(self, packages: list[tuple[Any, SourceScope]]) -> list[tuple[Any, SourceScope]]:
        seen: dict[str, tuple[Any, SourceScope]] = {}
        for pkg, scope in packages:
            source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if not source_str:
                continue
            identity = self._get_package_identity(source_str, scope)
            existing = seen.get(identity)
            if existing is None or (scope == "project" and existing[1] == "user"):
                seen[identity] = (pkg, scope)
        return list(seen.values())

    def _to_resolved_paths(self, accumulator: dict) -> ResolvedPaths:
        def _build(entries: dict) -> list[ResolvedResource]:
            resolved = [
                ResolvedResource(path=p, enabled=e, metadata=m)
                for p, (m, e) in entries.items()
            ]
            resolved.sort(key=lambda item: self._resource_precedence_rank(item.metadata))
            seen: set[str] = set()
            deduped: list[ResolvedResource] = []
            for item in resolved:
                canonical_path = os.path.realpath(os.path.abspath(item.path))
                if canonical_path in seen:
                    continue
                seen.add(canonical_path)
                deduped.append(item)
            return deduped

        return ResolvedPaths(
            extensions=_build(accumulator["extensions"]),
            skills=_build(accumulator["skills"]),
            prompts=_build(accumulator["prompts"]),
            themes=_build(accumulator["themes"]),
        )

    async def install(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        await self._with_progress("install", source, f"Installing {source}...", lambda: self._do_install(parsed, scope))

    async def _do_install(self, parsed: _ParsedSource, scope: SourceScope) -> None:
        if isinstance(parsed, _NpmSource):
            await self._install_npm(parsed, scope, False)
        elif isinstance(parsed, GitSource):
            await self._install_git(parsed, scope)

    async def remove(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        await self._with_progress("remove", source, f"Removing {source}...", lambda: self._do_remove(parsed, scope))

    async def _do_remove(self, parsed: _ParsedSource, scope: SourceScope) -> None:
        if isinstance(parsed, _NpmSource):
            await self._uninstall_npm(parsed, scope)
        elif isinstance(parsed, GitSource):
            install_path = self._get_git_install_path(parsed, scope)
            if os.path.exists(install_path):
                shutil.rmtree(install_path, ignore_errors=True)

    async def update(self, source: str | None = None) -> None:
        """Update all configured package sources or a specific source."""
        if not self._settings_manager:
            return

        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        target_identity = self._get_package_identity(source) if source else None
        matched = False

        for pkg in global_settings.get("packages", []):
            source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if not source_str:
                continue
            if target_identity and self._get_package_identity(source_str, "user") != target_identity:
                continue
            matched = True
            await self._update_source_for_scope(source_str, "user")

        for pkg in project_settings.get("packages", []):
            source_str = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if not source_str:
                continue
            if target_identity and self._get_package_identity(source_str, "project") != target_identity:
                continue
            matched = True
            await self._update_source_for_scope(source_str, "project")

        if source and not matched:
            raise RuntimeError(f"No matching package found for {source}")

    async def self_update(self, force: bool = False, package_name: str = "tau-by-clarity") -> None:
        """Update the Python CLI package that provides this harness."""
        args = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
        if force:
            args.insert(-1, "--force-reinstall")

        async def _op() -> None:
            await self._run_command(args)

        await self._with_progress("update", package_name, f"Updating {package_name}...", _op)

    async def _update_source_for_scope(self, source: str, scope: SourceScope) -> None:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            if parsed.pinned:
                return

            async def _op() -> None:
                await self._install_npm(parsed, scope, False)

            await self._with_progress("update", source, f"Updating {source}...", _op)
            return

        if isinstance(parsed, GitSource):
            if parsed.pinned:
                return

            async def _op() -> None:
                install_path = self._get_git_install_path(parsed, scope)
                if os.path.exists(install_path):
                    await self._run_command(["git", "pull", "--ff-only"], cwd=install_path)
                else:
                    await self._install_git(parsed, scope)

            await self._with_progress("update", source, f"Updating {source}...", _op)

    def _get_package_identity(self, source: str, scope: SourceScope = "user") -> str:
        parsed = self._parse_source(source)
        if isinstance(parsed, _NpmSource):
            return f"npm:{parsed.name}"
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        base_dir = self._get_base_dir_for_scope(scope)
        resolved = self._resolve_path_from_base(parsed.path, base_dir)
        return f"local:{resolved}"

    async def _install_npm(self, source: _NpmSource, scope: SourceScope, temporary: bool) -> None:
        if scope == "user" and not temporary:
            await self._run_command(["npm", "install", "-g", source.spec])
        else:
            install_root = self._get_npm_install_root(scope, temporary)
            os.makedirs(install_root, exist_ok=True)
            await self._run_command(["npm", "install", source.spec, "--prefix", install_root])

    async def _uninstall_npm(self, source: _NpmSource, scope: SourceScope) -> None:
        if scope == "user":
            await self._run_command(["npm", "uninstall", "-g", source.name])
        else:
            install_root = self._get_npm_install_root(scope, False)
            if os.path.exists(install_root):
                await self._run_command(["npm", "uninstall", source.name, "--prefix", install_root])

    async def _install_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if os.path.exists(target_dir):
            return
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        await self._run_command(["git", "clone", source.repo, target_dir])
        if source.ref:
            await self._run_command(["git", "checkout", source.ref], cwd=target_dir)
        pkg_json = os.path.join(target_dir, "package.json")
        if os.path.exists(pkg_json):
            await self._run_command(["npm", "install"], cwd=target_dir)

    async def _run_command(self, args: list[str], cwd: str | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(args)} failed with code {proc.returncode}: {stderr.decode()}"
            )

    def add_source_to_settings(self, source: str, options: dict[str, Any] | None = None) -> bool:
        if not self._settings_manager:
            return False
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        packages = settings.get("packages", [])
        if any(
            (p if isinstance(p, str) else p.get("source", "")) == source
            for p in packages
        ):
            return False
        packages.append(source)
        if scope == "project":
            self._settings_manager.set_project_packages(packages)
        else:
            self._settings_manager.set_packages(packages)
        return True

    def remove_source_from_settings(self, source: str, options: dict[str, Any] | None = None) -> bool:
        if not self._settings_manager:
            return False
        scope: SourceScope = "project" if (options or {}).get("local") else "user"
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        packages = settings.get("packages", [])
        new_packages = [
            p for p in packages
            if (p if isinstance(p, str) else p.get("source", "")) != source
        ]
        if len(new_packages) == len(packages):
            return False
        if scope == "project":
            self._settings_manager.set_project_packages(new_packages)
        else:
            self._settings_manager.set_packages(new_packages)
        return True

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        if not self._settings_manager:
            return []
        configured: list[ConfiguredPackage] = []
        for pkg in self._settings_manager.get_global_settings().get("packages", []) or []:
            source = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if source:
                configured.append(
                    ConfiguredPackage(
                        source=source,
                        scope="user",
                        filtered=isinstance(pkg, dict),
                        installed_path=self.get_installed_path(source, "user"),
                    )
                )
        for pkg in self._settings_manager.get_project_settings().get("packages", []) or []:
            source = pkg if isinstance(pkg, str) else pkg.get("source", "")
            if source:
                configured.append(
                    ConfiguredPackage(
                        source=source,
                        scope="project",
                        filtered=isinstance(pkg, dict),
                        installed_path=self.get_installed_path(source, "project"),
                    )
                )
        return configured

    async def install_and_persist(self, source: str, options: dict[str, Any] | None = None) -> None:
        await self.install(source, options)
        self.add_source_to_settings(source, options)

    async def remove_and_persist(self, source: str, options: dict[str, Any] | None = None) -> bool:
        await self.remove(source, options)
        return self.remove_source_from_settings(source, options)


DefaultPackageManager.installAndPersist = DefaultPackageManager.install_and_persist
DefaultPackageManager.removeAndPersist = DefaultPackageManager.remove_and_persist
DefaultPackageManager.selfUpdate = DefaultPackageManager.self_update


def _package_manager_add_source_to_settings(self, source, options=None):
    return self.add_source_to_settings(source, options)


def _package_manager_get_installed_path(self, source, scope):
    return self.get_installed_path(source, scope)


def _package_manager_list_configured_packages(self):
    return self.list_configured_packages()


def _package_manager_remove_source_from_settings(self, source, options=None):
    return self.remove_source_from_settings(source, options)


async def _package_manager_resolve_extension_sources(self, sources, options=None):
    return await self.resolve_extension_sources(sources, options)


def _package_manager_set_progress_callback(self, callback):
    return self.set_progress_callback(callback)


DefaultPackageManager.addSourceToSettings = _package_manager_add_source_to_settings
DefaultPackageManager.getInstalledPath = _package_manager_get_installed_path
DefaultPackageManager.resolveExtensionSources = _package_manager_resolve_extension_sources
DefaultPackageManager.removeSourceFromSettings = _package_manager_remove_source_from_settings
DefaultPackageManager.setProgressCallback = _package_manager_set_progress_callback
DefaultPackageManager.listConfiguredPackages = _package_manager_list_configured_packages
