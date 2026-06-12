"""
Configuration paths and package detection.

Mirrors packages/coding-agent/src/config.ts
"""
from __future__ import annotations

import os
import shutil
import subprocess
from importlib.metadata import version as _pkg_version
from pathlib import Path


# App metadata (mirrors piConfig in package.json)
APP_NAME: str = "pi"
# The Python harness uses its OWN config dir so it never reads the Node `.pi`
# tree (whose extensions are .ts and can't load here). Auth/models are migrated
# from the legacy dir on first run; see migrate_legacy_global_config().
CONFIG_DIR_NAME: str = ".pi-py"
LEGACY_CONFIG_DIR_NAME: str = ".pi"

try:
    VERSION: str = _pkg_version("clarity-pi")
except Exception:
    try:
        VERSION = _pkg_version("pi-coding-agent")
    except Exception:
        VERSION = "0.54.3"

ENV_AGENT_DIR: str = f"{APP_NAME.upper()}_CODING_AGENT_DIR"
ENV_SESSION_DIR: str = f"{APP_NAME.upper()}_CODING_AGENT_SESSION_DIR"


# ============================================================================
# User Config Paths (~/.pi/agent/*)
# ============================================================================


def get_agent_dir() -> str:
    """Get the agent config directory (e.g., ~/.pi/agent/)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        home = os.path.expanduser("~")
        if env_dir == "~":
            return home
        if env_dir.startswith("~/"):
            return home + env_dir[1:]
        return env_dir
    return os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME, "agent")


def get_prompts_dir() -> str:
    """Get path to prompt templates directory."""
    return os.path.join(get_agent_dir(), "prompts")


def get_themes_dir() -> str:
    """Get path to bundled themes directory."""
    return os.path.join(get_package_dir(), "modes", "interactive", "theme")


def get_custom_themes_dir() -> str:
    """Get path to user custom themes directory."""
    return os.path.join(get_agent_dir(), "themes")


def get_bin_dir() -> str:
    """Get path to managed binaries directory (fd, rg)."""
    return os.path.join(get_agent_dir(), "bin")


def get_sessions_dir() -> str:
    """Get path to sessions directory."""
    return os.path.join(get_agent_dir(), "sessions")


def get_models_path() -> str:
    """Get path to models.json."""
    return os.path.join(get_agent_dir(), "models.json")


def get_auth_path() -> str:
    """Get path to auth.json."""
    return os.path.join(get_agent_dir(), "auth.json")


def get_settings_path() -> str:
    """Get path to settings.json."""
    return os.path.join(get_agent_dir(), "settings.json")


def get_debug_log_path(cwd: str | None = None) -> str:
    """Get path to debug log file.

    Debug logs are process diagnostics, not agent resources. Keep them under
    the launched project/workspace dir so subagents still report into the parent
    workspace log instead of writing inside PI_CODING_AGENT_DIR.
    """
    return os.path.join(os.path.abspath(cwd or os.getcwd()), ".debug.log")


def get_package_dir() -> str:
    """Get package root directory."""
    return str(Path(__file__).resolve().parents[2])


def get_docs_path() -> str:
    """Get path to package docs directory."""
    return os.path.join(get_package_dir(), "docs")


def get_examples_path() -> str:
    """Get path to package examples directory."""
    return os.path.join(get_package_dir(), "examples")


def get_readme_path() -> str:
    """Get path to package README."""
    return os.path.join(get_package_dir(), "README.md")


def get_changelog_path() -> str:
    """Get path to package changelog."""
    return os.path.join(get_package_dir(), "CHANGELOG.md")


def get_tools_dir() -> str:
    """Get path to managed tool binaries directory."""
    return get_bin_dir()


def get_share_viewer_url(gist_id: str) -> str:
    base_url = os.environ.get("PI_SHARE_VIEWER_URL", "https://pi.dev/session/")
    return f"{base_url}#{gist_id}"


# ============================================================================
# Legacy aliases kept for backward compatibility
# ============================================================================


def get_global_config_dir() -> str:
    """Get the global Pi config directory (~/.pi)."""
    return os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME)


def get_global_agent_dir() -> str:
    """Get the global Pi agent directory (~/.pi/agent)."""
    return get_agent_dir()


def get_global_sessions_dir() -> str:
    """Get the sessions directory (~/.pi/agent/sessions)."""
    return get_sessions_dir()


def get_project_config_dir(cwd: str | None = None) -> str:
    """Get the project-local Pi config directory (<cwd>/.pi-py)."""
    base = cwd or os.getcwd()
    return os.path.join(base, CONFIG_DIR_NAME)


def get_project_agent_dir(cwd: str | None = None) -> str:
    """Project-local agent dir (<cwd>/.pi-py/agent) — the default config root
    when not inheriting global config."""
    return os.path.join(get_project_config_dir(cwd), "agent")


def get_project_sessions_dir(cwd: str | None = None) -> str:
    """Project-local sessions dir (<cwd>/.pi-py/agent/sessions) — sessions are
    contained per-project by default."""
    return os.path.join(get_project_agent_dir(cwd), "sessions")


def migrate_legacy_global_config() -> None:
    """One-time: seed the new global dir (~/.pi-py/agent) with auth + models
    from the legacy Node dir (~/.pi/agent) so existing API keys keep working.

    Only copies a file if the destination does not already exist; never deletes
    or overwrites. Safe to call on every startup.
    """
    import shutil

    home = os.path.expanduser("~")
    legacy = os.path.join(home, LEGACY_CONFIG_DIR_NAME, "agent")
    new = get_agent_dir()
    if os.path.abspath(legacy) == os.path.abspath(new):
        return
    try:
        os.makedirs(new, mode=0o700, exist_ok=True)
    except OSError:
        return
    for name in ("auth.json", "models.json"):
        src = os.path.join(legacy, name)
        dst = os.path.join(new, name)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass


def _global_default_seed() -> dict:
    """Pull the default provider/model/thinking from the global
    ~/.pi-py/agent/settings.json so a new project file shows what it actually
    runs. Empty dict if no global defaults exist yet (pre-/login)."""
    import json

    seed: dict = {}
    gdata: dict = {}
    global_settings = get_settings_path()  # ~/.pi-py/agent/settings.json
    if os.path.exists(global_settings):
        try:
            with open(global_settings, encoding="utf-8") as gf:
                loaded = json.load(gf)
            if isinstance(loaded, dict):
                gdata = loaded
            for key in ("defaultProvider", "defaultModel", "defaultThinkingLevel"):
                if gdata.get(key) is not None:
                    seed[key] = gdata[key]
        except (OSError, json.JSONDecodeError):
            pass
    # Always surface the memory toggle in a fresh project file so it's visible
    # and one edit away from on — the flag's home is settings.json (env var only
    # forces on for tests/CI). Inherit a global preference if one is set; else
    # fall back to the product default (off / kill-switch retained).
    seed["memory_enabled"] = bool(gdata.get("memory_enabled", False))
    return seed


def ensure_project_settings(cwd: str | None = None) -> str | None:
    """Guarantee <cwd>/.pi-py/settings.json exists, seeded from the global
    defaults, on EVERY launch — so a normal `pi-py` run never leaves a
    half-empty .pi-py (sessions dir but no visible config). Returns the path if
    it was just created, else None. Never overwrites an existing file."""
    import json

    base = cwd or os.getcwd()
    config_dir = get_project_config_dir(base)
    settings_path = os.path.join(config_dir, "settings.json")
    if os.path.exists(settings_path):
        return None
    os.makedirs(config_dir, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_global_default_seed(), indent=2) + "\n")
    return settings_path


def ensure_memory_store(cwd: str | None = None) -> str | None:
    """Guarantee the project-local memory store exists at
    <cwd>/.pi-py/memory/memory.db (dir + schema'd SQLite db). Returns the db
    path if it was just created, else None. Never touches an existing db.

    Offline: instantiating the store creates the dir and schema only — no
    embedding/Ollama call happens until something is actually written.
    """
    base = cwd or os.getcwd()
    mem_dir = os.path.join(get_project_config_dir(base), "memory")
    db_path = os.path.join(mem_dir, "memory.db")
    if os.path.exists(db_path):
        return None
    try:
        # Local import: core.memory.store imports CONFIG_DIR_NAME from this
        # module, so a top-level import would be circular.
        from pi_coding_agent.core.memory.store import MemoryStore

        store = MemoryStore(base)
        store.close()
    except Exception:
        # Memory is an optional capability; never let a store hiccup block init.
        return None
    return db_path if os.path.exists(db_path) else None


def ensure_project_agent_config(cwd: str | None = None) -> list[str]:
    """Seed <cwd>/.pi-py/agent with global auth/model config.

    This makes an initialized project self-contained when launched with
    PI_CODING_AGENT_DIR pointing at its local agent dir. Idempotent: only
    copies missing files and never overwrites project-local config.
    """
    import shutil

    base = cwd or os.getcwd()
    global_agent_dir = get_agent_dir()
    project_agent_dir = get_project_agent_dir(base)
    if os.path.abspath(global_agent_dir) == os.path.abspath(project_agent_dir):
        return []

    created: list[str] = []
    os.makedirs(project_agent_dir, mode=0o700, exist_ok=True)
    for name in ("auth.json", "auth.json.key", "models.json"):
        src = os.path.join(global_agent_dir, name)
        dst = os.path.join(project_agent_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                created.append(dst)
            except OSError:
                pass
    return created


def ensure_project_uv_runner(cwd: str | None = None) -> list[str]:
    """Create a project-local uv environment for pi-py.

    For fresh agent directories, this writes a root ``pyproject.toml`` so plain
    ``uv run pi-py`` works from the active directory. It is intentionally
    non-destructive: existing project files and venvs are left alone.
    """
    base = cwd or os.getcwd()
    pyproject_path = os.path.join(base, "pyproject.toml")
    created: list[str] = []

    if not os.path.exists(pyproject_path):
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "[project]",
                        'name = "pi-py-agent-runner"',
                        'version = "0.1.0"',
                        'requires-python = ">=3.11,<3.14"',
                        f'dependencies = ["clarity-pi=={VERSION}"]',
                        "",
                        "[tool.uv]",
                        "package = false",
                        "",
                    ]
                )
            )
        created.append(pyproject_path)

    venv_dir = os.path.join(base, ".venv")
    uv_bin = shutil.which("uv")
    if uv_bin and not os.path.exists(venv_dir):
        try:
            subprocess.run(
                [uv_bin, "sync", "--project", base, "--quiet"],
                cwd=base,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if os.path.exists(venv_dir):
                created.append(venv_dir)
        except Exception:
            pass

    return created


def scaffold_project(cwd: str | None = None) -> list[str]:
    """Create the project-local .pi-py structure for a new agent dir.

    Idempotent: only creates what's missing; never overwrites. Returns the list
    of paths created.
    """
    import json

    base = cwd or os.getcwd()
    config_dir = get_project_config_dir(base)  # <cwd>/.pi-py
    created: list[str] = []

    for sub in ("skills", "prompts", "extensions"):
        d = os.path.join(config_dir, sub)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            created.append(d)

    settings_path = os.path.join(config_dir, "settings.json")
    if not os.path.exists(settings_path):
        os.makedirs(config_dir, exist_ok=True)
        # Seed the project file with the resolved defaults from the global
        # ~/.pi-py/agent/settings.json so the agent's config is visible and
        # self-contained right here in its own dir (not an empty {} that hides
        # what it's actually running). If no global defaults exist yet, this
        # stays empty until /login writes them.
        with open(settings_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_global_default_seed(), indent=2) + "\n")
        created.append(settings_path)

    # Stand up the project-local memory store so it's present and git-trackable
    # from day one (the design commits ./.pi-py/memory/ to git). Creating the
    # store writes an empty schema'd memory.db; it's offline (no Ollama call
    # until something is actually embedded), so this is safe even with
    # memory_enabled=false.
    mem_db = ensure_memory_store(base)
    if mem_db:
        created.append(mem_db)

    created.extend(ensure_project_agent_config(base))
    created.extend(ensure_project_uv_runner(base))

    agents_path = os.path.join(base, "AGENTS.md")
    if not os.path.exists(agents_path):
        name = os.path.basename(os.path.abspath(base))
        with open(agents_path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n<!-- Project context for this agent. -->\n")
        created.append(agents_path)

    return created


def find_project_root(cwd: str | None = None) -> str:
    """Find the project root by looking for known markers."""
    current = Path(cwd or os.getcwd())
    markers = {".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"}

    while True:
        for marker in markers:
            if (current / marker).exists():
                return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    return cwd or os.getcwd()


def is_git_repo(cwd: str | None = None) -> bool:
    """Check if the directory is inside a git repo."""
    current = Path(cwd or os.getcwd())
    while True:
        if (current / ".git").exists():
            return True
        parent = current.parent
        if parent == current:
            break
        current = parent
    return False
