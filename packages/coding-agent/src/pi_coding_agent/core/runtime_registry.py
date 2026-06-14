"""Runtime process registry for tau sessions and loops."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from pi_coding_agent.config import get_agent_dir

_REGISTRY_NAME = "runtime_processes.json"
_TERM_GRACE_SECONDS = 5


def _registry_path() -> str:
    return os.path.join(get_agent_dir(), _REGISTRY_NAME)


def _registry_paths(include_nested: bool = False, root: str | None = None) -> list[str]:
    paths = [os.path.abspath(_registry_path())]
    if include_nested:
        base = Path(root or os.getcwd()).resolve()
        for candidate in (base / ".tau").rglob(_REGISTRY_NAME):
            resolved = str(candidate.resolve())
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _read(path: str | None = None) -> list[dict[str, Any]]:
    path = path or _registry_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _write(entries: list[dict[str, Any]], path: str | None = None) -> None:
    path = path or _registry_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pruned(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if _alive(int(entry.get("pid") or 0))]


def register_process(
    *,
    kind: str,
    session_id: str,
    cwd: str,
    agent_dir: str | None = None,
    goal: str | None = None,
) -> str:
    token = str(uuid.uuid4())
    entry = {
        "token": token,
        "kind": kind,
        "session_id": session_id,
        "pid": os.getpid(),
        "cwd": os.path.abspath(cwd),
        "agent_dir": os.path.abspath(os.path.expanduser(agent_dir or get_agent_dir())),
        "goal": goal,
        "started_at": int(time.time()),
    }
    entries = [item for item in _pruned(_read()) if item.get("token") != token]
    entries.append(entry)
    _write(entries)
    atexit.register(unregister_process, token)
    return token


def unregister_process(token: str) -> None:
    try:
        _write([entry for entry in _read() if entry.get("token") != token])
    except Exception:
        pass


def list_processes(*, include_nested: bool = False, root: str | None = None) -> list[dict[str, Any]]:
    all_entries: list[dict[str, Any]] = []
    for path in _registry_paths(include_nested=include_nested, root=root):
        entries = _pruned(_read(path))
        _write(entries, path)
        for entry in entries:
            entry["_registry_path"] = path
        all_entries.extend(entries)
    return all_entries


def _child_pids(pid: int) -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            child = int(line.strip())
        except ValueError:
            continue
        pids.append(child)
        pids.extend(_child_pids(child))
    return pids


def _terminate_pid_tree(pid: int) -> None:
    descendants = list(reversed(_child_pids(pid)))
    for target in descendants:
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + _TERM_GRACE_SECONDS
    targets = [pid, *descendants]
    while time.time() < deadline:
        if not any(_alive(target) for target in targets):
            return
        time.sleep(0.05)
    for target in descendants:
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def kill_processes(
    target: str | None = None,
    *,
    include_nested: bool = True,
    root: str | None = None,
) -> list[dict[str, Any]]:
    current_pid = os.getpid()
    entries = list_processes(include_nested=include_nested, root=root)
    selected: list[dict[str, Any]] = []
    if target:
        needle = target.strip()
        selected = [
            entry for entry in entries
            if str(entry.get("session_id") or "").startswith(needle)
            or str(entry.get("token") or "").startswith(needle)
            or str(entry.get("pid") or "") == needle
        ]
    else:
        selected = entries

    killed: list[dict[str, Any]] = []
    for entry in selected:
        pid = int(entry.get("pid") or 0)
        if pid <= 0 or pid == current_pid:
            continue
        _terminate_pid_tree(pid)
        killed.append(entry)

    killed_by_registry: dict[str, set[str]] = {}
    for entry in killed:
        registry_path = str(entry.get("_registry_path") or _registry_path())
        killed_by_registry.setdefault(registry_path, set()).add(str(entry.get("token")))
    for registry_path, tokens in killed_by_registry.items():
        _write(
            [
                entry
                for entry in _read(registry_path)
                if str(entry.get("token")) not in tokens
                and _alive(int(entry.get("pid") or 0))
            ],
            registry_path,
        )
    return killed
