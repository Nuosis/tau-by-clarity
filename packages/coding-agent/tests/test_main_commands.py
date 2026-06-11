from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_parse_package_command_install_local() -> None:
    from pi_coding_agent.main import _parse_package_command

    parsed = _parse_package_command(["install", "npm:@foo/bar", "--local", "--approve"])
    assert parsed is not None
    assert parsed["command"] == "install"
    assert parsed["source"] == "npm:@foo/bar"
    assert parsed["local"] is True
    assert parsed["project_trust_override"] is True


def test_parse_package_command_invalid() -> None:
    from pi_coding_agent.main import _parse_package_command

    assert _parse_package_command(["chat", "hello"]) is None


def test_parse_package_command_node_update_targets() -> None:
    from pi_coding_agent.main import _parse_package_command

    parsed = _parse_package_command(["uninstall", "npm:@foo/bar", "-l"])
    assert parsed is not None
    assert parsed["command"] == "remove"
    assert parsed["local"] is True

    parsed = _parse_package_command(["update", "--extension", "npm:@foo/bar", "--force"])
    assert parsed is not None
    assert parsed["force"] is True
    assert parsed["update_target"] == {"type": "extensions", "source": "npm:@foo/bar"}

    parsed = _parse_package_command(["update", "pi", "--extensions"])
    assert parsed is not None
    assert parsed["update_target"] == {"type": "all", "source": None}

    parsed = _parse_package_command(["update", "--self"])
    assert parsed is not None
    assert parsed["update_target"] == {"type": "self", "source": None}


def test_parse_package_command_node_diagnostics() -> None:
    from pi_coding_agent.main import _parse_package_command

    parsed = _parse_package_command(["update", "--extension"])
    assert parsed is not None
    assert parsed["missing_option_value"] == "--extension"

    parsed = _parse_package_command(["update", "--extension", "a", "--extension", "b"])
    assert parsed is not None
    assert parsed["conflicting_options"] == "--extension can only be provided once"

    parsed = _parse_package_command(["update", "--extension", "a", "--self"])
    assert parsed is not None
    assert parsed["conflicting_options"] == "--extension cannot be combined with --self or --extensions"

    parsed = _parse_package_command(["install", "a", "b"])
    assert parsed is not None
    assert parsed["invalid_argument"] == "b"


def test_package_help_mentions_uninstall_alias(capsys) -> None:
    from pi_coding_agent.main import _print_package_help

    _print_package_help("remove")

    out = capsys.readouterr().out
    assert "Alias: pi uninstall <source> [-l]" in out


@pytest.mark.asyncio
async def test_handle_package_command_list(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    class _Settings:
        def drain_errors(self):
            return []

        def get_global_settings(self):
            return {"packages": ["npm:@foo/bar"]}

        def get_project_settings(self):
            return {"packages": ["./local-ext"]}

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        def list_configured_packages(self):
            return [
                SimpleNamespace(
                    source="npm:@foo/bar",
                    scope="user",
                    filtered=False,
                    installed_path="/tmp/user/npm:@foo_bar",
                ),
                SimpleNamespace(
                    source="./local-ext",
                    scope="project",
                    filtered=False,
                    installed_path="/tmp/project/._local-ext",
                ),
            ]

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(lambda *_: _Settings())}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["list"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert "User packages:" in out
    assert "Project packages:" in out
    assert "npm:@foo/bar" in out
    assert "./local-ext" in out


@pytest.mark.asyncio
async def test_handle_package_command_update_extension_target(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    calls: list[str | None] = []

    class _Settings:
        def drain_errors(self):
            return []

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        async def update(self, source=None):
            calls.append(source)

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(lambda *_: _Settings())}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["update", "--extension", "npm:@foo/bar"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert calls == ["npm:@foo/bar"]
    assert "Updated npm:@foo/bar" in out


@pytest.mark.asyncio
async def test_handle_package_command_update_self_runs_self_update(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    calls: list[tuple[str, object]] = []

    class _Settings:
        def drain_errors(self):
            return []

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        async def self_update(self, force=False):
            calls.append(("self_update", force))

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(lambda *_: _Settings())}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["update", "--self", "--force"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert calls == [("self_update", True)]
    assert "Updated pi" in out


@pytest.mark.asyncio
async def test_handle_package_command_update_all_updates_extensions_then_self(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    calls: list[tuple[str, object]] = []

    class _Settings:
        def drain_errors(self):
            return []

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        async def update(self, source=None):
            calls.append(("update", source))

        async def self_update(self, force=False):
            calls.append(("self_update", force))

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(lambda *_: _Settings())}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["update"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert calls == [("update", None), ("self_update", False)]
    assert "Updated packages" in out
    assert "Updated pi" in out


@pytest.mark.asyncio
async def test_package_manager_self_update_invokes_python_pip(monkeypatch, tmp_path) -> None:
    import sys
    from pi_coding_agent.core.package_manager import DefaultPackageManager

    commands: list[list[str]] = []

    class _PackageManager(DefaultPackageManager):
        async def _run_command(self, args, cwd=None):
            commands.append(args)

    manager = _PackageManager(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), settings_manager=object())

    await manager.self_update(force=True)

    assert commands == [[
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "pi-coding-agent",
    ]]


@pytest.mark.asyncio
async def test_handle_package_command_rejects_untrusted_local_write(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    class _TrustStore:
        def __init__(self, _agent_dir):
            pass

        def get(self, _cwd):
            return False

    monkeypatch.setattr(main_mod, "has_project_trust_inputs", lambda _cwd: True)
    monkeypatch.setattr(main_mod, "ProjectTrustStore", _TrustStore)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")

    handled, code = await main_mod._handle_package_command(["install", "./local", "--local"])
    err = capsys.readouterr().err

    assert handled is True
    assert code == 1
    assert "Project is not trusted" in err


@pytest.mark.asyncio
async def test_handle_package_command_no_approve_ignores_project_packages(monkeypatch, capsys) -> None:
    from pi_coding_agent import main as main_mod

    created_options: list[dict[str, object]] = []

    class _Settings:
        def __init__(self, project_trusted):
            self.project_trusted = project_trusted

        def drain_errors(self):
            return []

        def get_global_settings(self):
            return {"packages": ["npm:@foo/bar"]}

        def get_project_settings(self):
            return {"packages": ["./local-ext"]} if self.project_trusted else {}

    class _PkgMgr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_progress_callback(self, _cb):
            return None

        def list_configured_packages(self):
            packages = [
                SimpleNamespace(
                    source="npm:@foo/bar",
                    scope="user",
                    filtered=False,
                    installed_path="/tmp/user/npm:@foo_bar",
                )
            ]
            if self.kwargs["settings_manager"].project_trusted:
                packages.append(
                    SimpleNamespace(
                        source="./local-ext",
                        scope="project",
                        filtered=False,
                        installed_path="/tmp/project/._local-ext",
                    )
                )
            return packages

    def create_settings(_cwd, _agent_dir=None, options=None):
        created_options.append(options or {})
        return _Settings(bool((options or {}).get("projectTrusted", True)))

    monkeypatch.setattr(main_mod, "SettingsManager", type("S", (), {"create": staticmethod(create_settings)}))
    monkeypatch.setattr(main_mod, "DefaultPackageManager", _PkgMgr)
    monkeypatch.setattr(main_mod, "get_agent_dir", lambda: "/tmp/agent")
    monkeypatch.setattr(main_mod, "has_project_trust_inputs", lambda _cwd: True)

    handled, code = await main_mod._handle_package_command(["list", "--no-approve"])
    out = capsys.readouterr().out

    assert handled is True
    assert code == 0
    assert created_options == [{"projectTrusted": False}]
    assert "User packages:" in out
    assert "Project packages:" not in out
    assert "./local-ext" not in out
