"""Init scaffolding seeds the memory toggle; non-inherit launches keep global
resource arrays (extensions/skills/prompts/themes) out of project sessions.

Covers two regressions observed on a fresh `pi-py` init under a cwd whose
ancestor home holds a global ~/.tau with an `extensions` entry:
  1. The memory flag was absent from the generated settings.json.
  2. The runtime-built interactive session re-introduced the global extension
     even without --inherit, because its SettingsManager defaulted to
     inherit_global=True.
"""
from __future__ import annotations

import json
import os

import pytest


def _write_global(agent_dir: str, data: dict) -> None:
    os.makedirs(agent_dir, exist_ok=True)
    with open(os.path.join(agent_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_scaffold_seeds_memory_enabled_key(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    _write_global(str(home / ".tau" / "agent"), {"defaultProvider": "minimax"})

    proj = tmp_path / "proj"
    proj.mkdir()
    created = config.scaffold_project(str(proj))
    settings_path = os.path.join(str(proj), ".tau", "settings.json")
    assert settings_path in created

    data = json.loads(open(settings_path, encoding="utf-8").read())
    # The toggle is always present and visible, defaulting off (kill-switch).
    assert data["memory_enabled"] is False
    assert data["defaultProvider"] == "minimax"


def test_scaffold_inherits_global_memory_preference(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    _write_global(str(home / ".tau" / "agent"), {"memory_enabled": True})

    proj = tmp_path / "proj"
    proj.mkdir()
    config.ensure_project_settings(str(proj))
    data = json.loads(open(os.path.join(str(proj), ".tau", "settings.json"), encoding="utf-8").read())
    assert data["memory_enabled"] is True


def test_scaffold_stands_up_memory_store(tmp_path) -> None:
    import sqlite3

    from pi_coding_agent import config

    proj = tmp_path / "proj"
    proj.mkdir()
    created = config.scaffold_project(str(proj))
    db = os.path.join(str(proj), ".tau", "memory", "memory.db")
    assert db in created
    assert os.path.exists(db)
    tables = {r[0] for r in sqlite3.connect(db).execute(
        "select name from sqlite_master where type='table'").fetchall()}
    assert {"semantic_memory", "conversation_memory"} <= tables
    # Idempotent: the db is not recreated/reported on a second pass.
    assert config.ensure_memory_store(str(proj)) is None


def test_scaffold_creates_empty_memory_db_not_copy_of_root_memory(tmp_path, monkeypatch) -> None:
    import sqlite3

    from pi_coding_agent import config
    from pi_coding_agent.core.memory.models import ConversationTurn
    from pi_coding_agent.core.memory.store import MemoryStore

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)

    root_store = MemoryStore(str(home))
    root_store.append_turn(ConversationTurn(
        id="",
        project=str(home),
        role="user",
        content="root memory must not be cloned",
    ))
    root_store.close()

    proj = tmp_path / "proj"
    proj.mkdir()
    config.scaffold_project(str(proj))

    root_db = home / ".tau" / "memory" / "memory.db"
    project_db = proj / ".tau" / "memory" / "memory.db"
    assert root_db.exists()
    assert project_db.exists()

    root_count = sqlite3.connect(root_db).execute("select count(*) from conversation_memory").fetchone()[0]
    project_conn = sqlite3.connect(project_db)
    project_counts = dict(project_conn.execute(
        "select 'semantic_memory', count(*) from semantic_memory "
        "union all select 'conversation_memory', count(*) from conversation_memory "
        "union all select 'summary_memory', count(*) from summary_memory "
        "union all select 'tool_log_memory', count(*) from tool_log_memory"
    ).fetchall())

    assert root_count == 1
    assert project_counts == {
        "semantic_memory": 0,
        "conversation_memory": 0,
        "summary_memory": 0,
        "tool_log_memory": 0,
    }


def test_scaffold_copies_root_skills_without_overwrite(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)

    root_skills = home / ".tau" / "skills"
    (root_skills / "agent-build-pattern" / "references").mkdir(parents=True)
    (root_skills / "agent-build-pattern" / "SKILL.md").write_text("root skill\n", encoding="utf-8")
    (root_skills / "agent-build-pattern" / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    (root_skills / ".DS_Store").write_text("skip\n", encoding="utf-8")

    proj = tmp_path / "proj"
    proj.mkdir()
    project_skill = proj / ".tau" / "skills" / "existing-skill"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text("local skill\n", encoding="utf-8")
    root_existing = root_skills / "existing-skill"
    root_existing.mkdir()
    (root_existing / "SKILL.md").write_text("root should not overwrite\n", encoding="utf-8")

    created = config.scaffold_project(str(proj))

    copied_skill = proj / ".tau" / "skills" / "agent-build-pattern"
    assert str(copied_skill) in created
    assert (copied_skill / "SKILL.md").read_text(encoding="utf-8") == "root skill\n"
    assert (copied_skill / "references" / "notes.md").read_text(encoding="utf-8") == "notes\n"
    assert not (proj / ".tau" / "skills" / ".DS_Store").exists()
    assert (project_skill / "SKILL.md").read_text(encoding="utf-8") == "local skill\n"
    assert str(project_skill) not in created


def test_scaffold_copies_global_auth_and_models_without_overwrite(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config
    from pi_coding_agent.core.auth_storage import AuthStorage

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)

    global_agent = home / ".tau" / "agent"
    global_agent.mkdir(parents=True)
    AuthStorage.create(str(global_agent / "auth.json")).set_api_key("minimax", "global-minimax-key")
    (global_agent / "models.json").write_text(json.dumps({"providers": {"minimax": {}}}), encoding="utf-8")

    proj = tmp_path / "proj"
    proj.mkdir()
    created = config.scaffold_project(str(proj))
    local_agent = proj / ".tau" / "agent"

    assert str(local_agent / "auth.json") in created
    assert str(local_agent / "auth.json.key") in created
    assert str(local_agent / "models.json") in created
    assert AuthStorage.create(str(local_agent / "auth.json")).get_api_key("minimax") == "global-minimax-key"

    (local_agent / "models.json").write_text('{"local": true}', encoding="utf-8")
    AuthStorage.create(str(local_agent / "auth.json")).set_api_key("minimax", "local-minimax-key")

    created_again = config.scaffold_project(str(proj))
    assert str(local_agent / "auth.json") not in created_again
    assert str(local_agent / "auth.json.key") not in created_again
    assert str(local_agent / "models.json") not in created_again
    assert (local_agent / "models.json").read_text(encoding="utf-8") == '{"local": true}'
    assert AuthStorage.create(str(local_agent / "auth.json")).get_api_key("minimax") == "local-minimax-key"


def test_scaffold_uses_root_global_config_even_when_agent_dir_env_points_local(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config
    from pi_coding_agent.core.auth_storage import AuthStorage

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)

    root_agent = home / ".tau" / "agent"
    root_agent.mkdir(parents=True)
    _write_global(str(root_agent), {
        "defaultProvider": "openai",
        "defaultModel": "gpt-5",
        "defaultThinkingLevel": "medium",
    })
    AuthStorage.create(str(root_agent / "auth.json")).set_api_key("openai", "root-openai-key")
    (root_agent / "models.json").write_text(json.dumps({"providers": {"openai": {}}}), encoding="utf-8")

    proj = tmp_path / "proj"
    proj.mkdir()
    local_agent = proj / ".tau" / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(local_agent))

    created = config.scaffold_project(str(proj))
    settings = json.loads((proj / ".tau" / "settings.json").read_text(encoding="utf-8"))

    assert settings["defaultProvider"] == "openai"
    assert settings["defaultModel"] == "gpt-5"
    assert settings["defaultThinkingLevel"] == "medium"
    assert str(local_agent / "auth.json") in created
    assert str(local_agent / "auth.json.key") in created
    assert str(local_agent / "models.json") in created
    assert AuthStorage.create(str(local_agent / "auth.json")).get_api_key("openai") == "root-openai-key"


def test_existing_project_settings_fill_missing_root_defaults_without_overwrite(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    _write_global(str(home / ".tau" / "agent"), {
        "defaultProvider": "openai",
        "defaultModel": "gpt-5",
        "defaultThinkingLevel": "high",
        "memory_enabled": True,
    })

    proj = tmp_path / "proj"
    settings_path = proj / ".tau" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"defaultProvider": "anthropic"}), encoding="utf-8")

    updated = config.ensure_project_settings(str(proj))
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    assert updated == str(settings_path)
    assert settings["defaultProvider"] == "anthropic"
    assert settings["defaultModel"] == "gpt-5"
    assert settings["defaultThinkingLevel"] == "high"
    assert settings["memory_enabled"] is True


def test_legacy_migration_seeds_root_tau_settings(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    legacy_root = home / ".pi-py"
    legacy_agent = legacy_root / "agent"
    legacy_agent.mkdir(parents=True)
    (legacy_root / "settings.json").write_text(
        json.dumps({"defaultProvider": "minimax", "defaultModel": "MiniMax-M3"}),
        encoding="utf-8",
    )
    (legacy_agent / "settings.json").write_text(
        json.dumps({"defaultProvider": "openai", "defaultModel": "gpt-5"}),
        encoding="utf-8",
    )

    config.migrate_legacy_global_config()
    settings_path = home / ".tau" / "agent" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    assert settings == {"defaultProvider": "minimax", "defaultModel": "MiniMax-M3"}


def test_scaffold_creates_project_uv_runner_in_active_directory(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    calls: list[tuple[list[str], str]] = []

    def fake_run(args, cwd=None, **kwargs):
        calls.append((list(args), str(cwd)))
        (tmp_path / "proj" / ".venv").mkdir(parents=True)
        return object()

    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(config.subprocess, "run", fake_run)

    proj = tmp_path / "proj"
    proj.mkdir()
    created = config.scaffold_project(str(proj))
    pyproject = proj / "pyproject.toml"

    assert str(pyproject) in created
    assert str(proj / ".venv") in created
    text = pyproject.read_text(encoding="utf-8")
    assert 'dependencies = ["tau-by-clarity==' in text
    assert "package = false" in text
    assert calls == [(["/usr/local/bin/uv", "sync", "--project", str(proj), "--quiet"], str(proj))]


def test_scaffold_preserves_existing_project_uv_runner(tmp_path, monkeypatch) -> None:
    from pi_coding_agent import config

    proj = tmp_path / "proj"
    proj.mkdir()
    pyproject = proj / "pyproject.toml"
    pyproject.write_text("# custom runner\n", encoding="utf-8")
    (proj / ".venv").mkdir()

    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(config.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not sync")))

    created = config.scaffold_project(str(proj))

    assert str(pyproject) not in created
    assert str(proj / ".venv") not in created
    assert pyproject.read_text(encoding="utf-8") == "# custom runner\n"


def test_agent_name_label(tmp_path) -> None:
    from types import SimpleNamespace

    from pi_coding_agent.core.settings_manager import SettingsManager
    from pi_coding_agent.modes.interactive.tui import _assistant_label

    named = SimpleNamespace(settings_manager=SettingsManager.in_memory({"name": "Devin"}))
    assert _assistant_label(named) == "Devin:"

    unnamed = SimpleNamespace(settings_manager=SettingsManager.in_memory({}))
    assert _assistant_label(unnamed) == "Assistant:"

    # Blank/whitespace name falls back to the generic label.
    blank = SimpleNamespace(settings_manager=SettingsManager.in_memory({"name": "  "}))
    assert _assistant_label(blank) == "Assistant:"


def test_non_inherit_drops_global_extensions(tmp_path, monkeypatch) -> None:
    from pi_coding_agent.config import get_agent_dir
    from pi_coding_agent.core.settings_manager import SettingsManager

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    ext = str(home / ".tau" / "extensions" / "pii_filter.py")
    _write_global(get_agent_dir(), {"extensions": [ext]})

    proj = tmp_path / "proj"
    proj.mkdir()

    # Default launch (no --inherit): the global extension array is dropped.
    sm = SettingsManager.create(str(proj), get_agent_dir(), inherit_global=False)
    assert sm.get_extensions() == []

    # --inherit: the global extension is pulled in.
    sm_inherit = SettingsManager.create(str(proj), get_agent_dir(), inherit_global=True)
    assert sm_inherit.get_extensions() == [ext]


def test_explicit_agent_dir_resources_load_without_inherit(tmp_path, monkeypatch) -> None:
    """PI_CODING_AGENT_DIR is an explicit agent root, not inherited global state."""
    import asyncio

    from pi_coding_agent.config import get_agent_dir
    from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
    from pi_coding_agent.core.settings_manager import SettingsManager

    agent_dir = tmp_path / "agent"
    project = tmp_path / "project"
    ext = agent_dir / "extensions" / "demo.py"
    ext.parent.mkdir(parents=True)
    ext.write_text("def activate(api):\n    pass\n", encoding="utf-8")
    project.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    _write_global(get_agent_dir(), {"extensions": [str(ext)], "tools": ["demo_tool"]})

    sm = SettingsManager.create(
        str(project),
        get_agent_dir(),
        {"keepAgentResources": True},
        inherit_global=False,
    )
    assert sm.get_extensions() == [str(ext)]
    assert sm.get_tools() == ["demo_tool"]

    loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(
            cwd=str(project),
            agent_dir=get_agent_dir(),
            settings_manager=sm,
            inherit_global=False,
        )
    )
    asyncio.run(loader.reload())
    assert loader.get_extensions()["diagnostics"] == []
    assert len(loader.get_extensions()["extensions"]) == 1


def test_extensions_discovered_from_user_directory_not_settings(tmp_path, monkeypatch) -> None:
    import asyncio

    from pi_coding_agent.config import get_agent_dir
    from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
    from pi_coding_agent.core.settings_manager import SettingsManager

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)

    extensions_dir = home / ".tau" / "extensions"
    file_ext = extensions_dir / "file_ext.py"
    package_ext = extensions_dir / "package_ext" / "__init__.py"
    settings_ext = home / ".tau" / "settings_only.py"
    file_ext.parent.mkdir(parents=True)
    package_ext.parent.mkdir(parents=True)
    file_ext.write_text("def activate(api):\n    pass\n", encoding="utf-8")
    package_ext.write_text("def activate(api):\n    pass\n", encoding="utf-8")
    settings_ext.write_text("def activate(api):\n    raise RuntimeError('settings entry loaded')\n", encoding="utf-8")
    _write_global(get_agent_dir(), {"extensions": [str(settings_ext)]})

    project = tmp_path / "project"
    project.mkdir()
    sm = SettingsManager.create(str(project), get_agent_dir(), inherit_global=True)
    assert sm.get_extensions() == [str(settings_ext)]

    loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(
            cwd=str(project),
            agent_dir=get_agent_dir(),
            settings_manager=sm,
            inherit_global=True,
        )
    )
    asyncio.run(loader.reload())

    loaded = {os.path.basename(getattr(ext, "path", "")) for ext in loader.get_extensions()["extensions"]}
    assert loaded == {"file_ext.py", "package_ext"}
    assert loader.get_extensions()["diagnostics"] == []


def test_explicit_agent_dir_owns_resources_over_project_cwd(tmp_path, monkeypatch) -> None:
    import asyncio

    from pi_coding_agent.config import get_agent_dir
    from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
    from pi_coding_agent.core.settings_manager import SettingsManager

    monkeypatch.setenv("PI_CLARITY_PII_DISABLED", "1")
    monkeypatch.setenv("PI_ACTIVE_COMPRESSION_DISABLED", "1")
    child_agent = tmp_path / "project" / ".tau" / "subagents" / "runner" / ".tau"
    project = tmp_path / "project"
    child_agent.mkdir(parents=True)
    (project / ".tau" / "extensions").mkdir(parents=True)
    (project / ".tau" / "skills" / "project-skill").mkdir(parents=True)
    (child_agent / "extensions").mkdir(parents=True)
    (child_agent / "skills" / "child-skill").mkdir(parents=True)

    (project / ".tau" / "SYSTEM.md").write_text("PROJECT SYSTEM\n", encoding="utf-8")
    (child_agent / "SYSTEM.md").write_text("CHILD SYSTEM\n", encoding="utf-8")
    (project / ".tau" / "extensions" / "project_ext.py").write_text(
        "def activate(api):\n    pass\n",
        encoding="utf-8",
    )
    (child_agent / "extensions" / "child_ext.py").write_text(
        "def activate(api):\n    pass\n",
        encoding="utf-8",
    )
    (project / ".tau" / "skills" / "project-skill" / "SKILL.md").write_text(
        "---\nname: project-skill\ndescription: project\n---\nproject\n",
        encoding="utf-8",
    )
    (child_agent / "skills" / "child-skill" / "SKILL.md").write_text(
        "---\nname: child-skill\ndescription: child\n---\nchild\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TAU_CODING_AGENT_DIR", str(child_agent))
    sm = SettingsManager.create(str(project), get_agent_dir(), inherit_global=False)
    loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(
            cwd=str(project),
            agent_dir=get_agent_dir(),
            settings_manager=sm,
            inherit_global=False,
        )
    )
    asyncio.run(loader.reload())

    assert loader.get_system_prompt() == "CHILD SYSTEM\n"
    loaded_extensions = [
        os.path.basename(getattr(ext, "path", ""))
        for ext in loader.get_extensions()["extensions"]
    ]
    assert loaded_extensions == ["child_ext.py"]
    loaded_skills = {skill.name for skill in loader.get_skills()["skills"]}
    assert "child-skill" in loaded_skills
    assert "project-skill" not in loaded_skills


def test_extensions_discovered_from_project_directory_without_inherit(tmp_path) -> None:
    import asyncio

    from pi_coding_agent.config import get_agent_dir
    from pi_coding_agent.core.resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions
    from pi_coding_agent.core.settings_manager import SettingsManager

    project = tmp_path / "project"
    ext = project / ".tau" / "extensions" / "project_ext.py"
    ext.parent.mkdir(parents=True)
    ext.write_text("def activate(api):\n    pass\n", encoding="utf-8")

    sm = SettingsManager.create(str(project), get_agent_dir(), inherit_global=False)
    loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(
            cwd=str(project),
            agent_dir=get_agent_dir(),
            settings_manager=sm,
            inherit_global=False,
        )
    )
    asyncio.run(loader.reload())

    loaded = [os.path.basename(getattr(ext, "path", "")) for ext in loader.get_extensions()["extensions"]]
    assert loaded == ["project_ext.py"]


@pytest.mark.parametrize("inherit,expected", [(False, []), (True, "ext")])
def test_create_runtime_host_respects_inherit(tmp_path, monkeypatch, inherit, expected) -> None:
    """_create_runtime_host's factory rebuilds the session; the resource_loader
    it hands to create_agent_session must reflect the launch-time --inherit
    choice, not silently default to inherit_global=True."""
    import asyncio
    from types import SimpleNamespace

    from pi_coding_agent import main as main_mod
    from pi_coding_agent.config import get_agent_dir

    home = tmp_path / "home"
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)) if p.startswith("~") else p)
    ext = str(home / ".tau" / "extensions" / "pii_filter.py")
    os.makedirs(os.path.dirname(ext), exist_ok=True)
    with open(ext, "w", encoding="utf-8") as f:
        f.write("def activate(api):\n    pass\n")
    _write_global(get_agent_dir(), {"extensions": [ext]})

    proj = tmp_path / "proj"
    proj.mkdir()

    captured: dict = {}

    async def fake_create_agent_session(opts):
        captured["loader"] = opts.resource_loader
        return SimpleNamespace(session=SimpleNamespace(), extensions_result={}, model_fallback_message=None)

    monkeypatch.setattr(main_mod, "create_agent_session", fake_create_agent_session)

    parsed = SimpleNamespace(
        inherit=inherit, extensions=None, skills=None, prompt_templates=None, themes=None,
        no_extensions=False, no_skills=False, no_prompt_templates=False, no_themes=False,
        no_context_files=False, system_prompt=None, append_system_prompt=None,
        tools=None, exclude_tools=None, no_tools=False, no_builtin_tools=False,
    )
    session = SimpleNamespace(session_manager=SimpleNamespace())

    async def build():
        return await main_mod._create_runtime_host(
            parsed, cwd=str(proj), session=session,
            auth_storage=SimpleNamespace(), model_registry=SimpleNamespace(),
            settings_manager=SimpleNamespace(), resource_loader=SimpleNamespace(),
            resolved_model=None, thinking=None,
        )

    asyncio.run(build())

    loaded = [getattr(e, "path", e) for e in captured["loader"].get_extensions().get("extensions", [])]
    if expected == "ext":
        assert loaded == [ext]
    else:
        assert loaded == []
