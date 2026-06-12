"""
Extension loader — mirrors packages/coding-agent/src/core/extensions/loader.ts

Discovers and loads extensions from:
1. Global: ~/.pi-py/extensions/
2. Local: <cwd>/.pi-py/extensions/
3. Explicit paths from CLI/API options
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Extension, ExtensionAPI, ExtensionFactory


@dataclass
class LoadExtensionsResult:
    extensions: list[Extension] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)


def read_pi_manifest(directory: str) -> dict[str, Any] | None:
    """Read pi manifest from pyproject.toml or package.json."""
    pyproject = os.path.join(directory, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                return None
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            pi_config = data.get("tool", {}).get("pi", {})
            if pi_config:
                return pi_config
        except Exception:
            pass

    pkg_json = os.path.join(directory, "package.json")
    if os.path.exists(pkg_json):
        import json
        try:
            with open(pkg_json, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("pi")
        except Exception:
            pass

    return None


def discover_extensions_in_dir(directory: str) -> list[str]:
    """Discover extension paths in a directory."""
    if not os.path.isdir(directory):
        return []

    paths: list[str] = []
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isfile(full) and entry.endswith(".py") and not entry.startswith("_"):
            paths.append(full)
        elif os.path.isdir(full):
            index = os.path.join(full, "__init__.py")
            if os.path.exists(index):
                paths.append(full)
            else:
                manifest = read_pi_manifest(full)
                if manifest and manifest.get("extensions"):
                    for ext_path in manifest["extensions"]:
                        resolved = os.path.join(full, ext_path)
                        if os.path.exists(resolved):
                            paths.append(resolved)

    return paths


def _load_extension_module(path: str) -> ExtensionFactory | None:
    """Load an extension module and return its factory function."""
    try:
        module_name = f"pi_ext_{Path(path).stem}"
        if os.path.isdir(path):
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(path, "__init__.py"),
                submodule_search_locations=[path],
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)

        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        factory = (
            getattr(module, "extension_factory", None)
            or getattr(module, "activate", None)
            or getattr(module, "default", None)
        )
        if callable(factory):
            return factory

        return None
    except Exception:
        return None


def create_extension_runtime() -> dict[str, Any]:
    """Create the extension runtime context (mirrors createExtensionRuntime in TS)."""
    return {
        "extensions": [],
        "commands": {},
        "tools": {},
        "flags": {},
        "flagValues": {},
        "pendingProviderRegistrations": [],
    }


async def load_extension_from_path(
    path: str,
    cwd: str,
    event_bus: Any,
    runtime: dict[str, Any] | None = None,
) -> Extension:
    """
    Load a single extension from a file path.
    Raises FileNotFoundError if the path doesn't exist.
    Raises ImportError if no factory function is found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extension not found: {path}")

    factory = _load_extension_module(path)
    if factory is None:
        raise ImportError(f"No extension_factory(), activate(), or default() export in: {path}")

    resolved = os.path.abspath(path)
    runtime = runtime or create_extension_runtime()
    ext = Extension(path=path, resolved_path=resolved)
    api = ExtensionAPI(ext, runtime)

    import asyncio
    import inspect
    ret = factory(api)
    if inspect.isawaitable(ret):
        await ret

    return ext


async def load_extensions(
    paths: list[str],
    cwd: str = "",
    event_bus: Any = None,
    runtime: dict[str, Any] | None = None,
) -> LoadExtensionsResult:
    """Load extensions from explicit paths (async version)."""
    runtime = runtime or create_extension_runtime()
    result = LoadExtensionsResult(runtime=runtime)

    for path in paths:
        resolved = os.path.abspath(path)
        factory = _load_extension_module(resolved)
        if factory is None:
            result.errors.append({"path": resolved, "error": "No extension_factory(), activate(), or default() export"})
            continue

        ext = Extension(path=path, resolved_path=resolved)
        api = ExtensionAPI(ext, runtime)

        try:
            import asyncio
            import inspect
            ret = factory(api)
            if inspect.isawaitable(ret):
                await ret
        except Exception as e:
            result.errors.append({"path": resolved, "error": str(e)})
            continue

        result.extensions.append(ext)

    return result


def load_extensions_sync(paths: list[str]) -> LoadExtensionsResult:
    """Load extensions synchronously from explicit paths."""
    runtime = create_extension_runtime()
    result = LoadExtensionsResult(runtime=runtime)

    for path in paths:
        resolved = os.path.abspath(path)
        factory = _load_extension_module(resolved)
        if factory is None:
            result.errors.append({"path": resolved, "error": "No extension_factory(), activate(), or default() export"})
            continue

        ext = Extension(path=path, resolved_path=resolved)
        api = ExtensionAPI(ext, runtime)

        try:
            import asyncio
            import inspect
            ret = factory(api)
            if inspect.isawaitable(ret):
                try:
                    loop = asyncio.get_running_loop()
                    loop.run_until_complete(ret)
                except RuntimeError:
                    asyncio.run(ret)
        except Exception as e:
            result.errors.append({"path": resolved, "error": str(e)})
            continue

        result.extensions.append(ext)

    return result


def discover_and_load_extensions(
    cwd: str,
    agent_dir: str | None = None,
    extra_paths: list[str] | None = None,
) -> LoadExtensionsResult:
    """Discover and load all extensions synchronously."""
    agent_dir = agent_dir or os.path.join(os.path.expanduser("~"), ".pi", "agent")
    all_paths: list[str] = []

    # Global extensions
    global_dir = os.path.join(agent_dir, "extensions")
    all_paths.extend(discover_extensions_in_dir(global_dir))

    # Local project extensions
    local_dir = os.path.join(cwd, ".pi", "extensions")
    all_paths.extend(discover_extensions_in_dir(local_dir))

    # Explicit paths
    if extra_paths:
        for p in extra_paths:
            resolved = os.path.abspath(p) if not os.path.isabs(p) else p
            if os.path.exists(resolved):
                all_paths.append(resolved)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in all_paths:
        rp = os.path.abspath(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return load_extensions_sync(unique)
