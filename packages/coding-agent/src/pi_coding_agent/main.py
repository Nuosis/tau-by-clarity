"""Main entry point — mirrors packages/coding-agent/src/main.ts."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .cli_sub.args import parse_args, print_help
from .cli_sub.file_processor import process_file_arguments
from .cli_sub.list_models import list_models
from .cli_sub.session_picker import select_session
from pi_ai import get_model
from pi_ai.env_api_keys import PROVIDER_ENV_VARS

from .config import APP_NAME, get_agent_dir
from .core.auth_storage import AuthStorage
from .core.cli_debug_log import (
    attach_session_event_logging,
    configure_cli_debug_logging,
    log_event,
    log_exception,
    log_session_snapshot,
)
from .core.event_bus import create_event_bus
from .core.extensions.loader import load_extensions
from .core.model_registry import ModelRegistry
from .core.package_manager import DefaultPackageManager
from .core.resource_loader import (
    DefaultResourceLoader,
    DefaultResourceLoaderOptions,
    get_extension_discovery_paths,
)
from .core.agent_session_runtime import (
    CreateAgentSessionRuntimeResult,
    create_agent_session_runtime,
)
from .core.agent_session_services import AgentSessionServices
from .core.sdk import CreateAgentSessionOptions, create_agent_session
from .core.session_manager import SessionManager
from .core.settings_manager import SettingsManager
from .core.trust_manager import ProjectTrustStore, has_project_trust_inputs
from .migrations import run_migrations, show_deprecation_warnings
from .modes import run_interactive_mode, run_print_mode, run_rpc_mode

_LOCAL_DISPATCH_ENV = "PI_PY_LOCAL_DISPATCH"


def _find_local_project_root(cwd: str) -> str | None:
    """Return the nearest uv project that pins clarity-pi."""
    current = Path(cwd).resolve()
    for path in (current, *current.parents):
        pyproject = path / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if "clarity-pi" in text:
            return str(path)
    return None


def _dispatch_to_local_project(args: Sequence[str], cwd: str) -> None:
    """Re-exec into the project-pinned clarity-pi when launched globally.

    A CLI process cannot source ~/.zshrc into its parent shell. This makes the
    global pi-py wrapper prefer the initialized project's uv environment even
    when the user's current terminal has not sourced the zsh helper yet.
    """
    if os.environ.get(_LOCAL_DISPATCH_ENV):
        return
    if "--init" in args:
        return
    project_root = _find_local_project_root(cwd)
    if not project_root:
        return
    env = os.environ.copy()
    env[_LOCAL_DISPATCH_ENV] = "1"
    if args and args[0] == "update":
        os.execvpe(
            "uv",
            ["uv", "add", "--project", project_root, "--upgrade-package", "clarity-pi", "clarity-pi", *args[1:]],
            env,
        )
    os.execvpe(
        "uv",
        ["uv", "run", "--project", project_root, "python", "-m", "pi_coding_agent.main", *args],
        env,
    )


def _load_env_files(cwd: str) -> None:
    """Load .env from current workspace (best-effort)."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        load_dotenv(os.path.join(cwd, ".env"), override=False)
    except Exception:
        pass


async def _build_initial_prompt(parsed) -> tuple[str, list[dict] | None]:
    """Build initial prompt text/images from @file args + positional messages."""
    text = ""
    images = None
    if parsed.file_args:
        processed = await process_file_arguments(parsed.file_args)
        text = processed.text
        images = processed.images or None

    if parsed.messages:
        # Match TS behavior: first positional message is prompt, rest handled by mode
        text = f"{text}{parsed.messages[0]}"
        parsed.messages = parsed.messages[1:]
    return text, images


async def _read_piped_stdin() -> str | None:
    """Read piped stdin content; return None when stdin is TTY."""
    if sys.stdin.isatty():
        return None
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, sys.stdin.read)
    data = (data or "").strip()
    return data or None


async def _prompt_confirm(message: str) -> bool:
    print(f"{message} [y/N] ", end="", flush=True)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, sys.stdin.readline)
    answer = (answer or "").strip().lower()
    return answer in {"y", "yes"}


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.endswith(".jsonl")


def _resolve_model_for_session(parsed, model_registry, settings_manager):
    """Resolve CLI --model/--provider/--api-key into a Model object + thinking level.

    CreateAgentSessionOptions only accepts a Model instance, not strings. This
    helper also injects a CLI-supplied --api-key into the right env var so the
    provider modules pick it up.
    """
    # Inject --api-key into the provider's expected env var.
    if parsed.api_key and parsed.provider:
        env_var = PROVIDER_ENV_VARS.get(parsed.provider)
        if env_var:
            os.environ[env_var] = parsed.api_key

    # Resolve model string ("provider/id:thinking" or just "id") → Model.
    # CLI flags win; otherwise fall back to the saved defaults (deep-merged
    # global + project settings) so a flagless launch uses defaultProvider /
    # defaultModel instead of guessing from whatever env key happens to exist.
    model = None
    model_id = parsed.model or settings_manager.get_default_model()
    provider = parsed.provider or settings_manager.get_default_provider()
    if model_id or provider:
        model = model_registry.resolve_model(
            model_id=model_id,
            provider=provider,
        )

    thinking = parsed.thinking or (settings_manager.get_default_thinking_level() or "off")

    return model, thinking


async def _create_resource_loader(parsed, cwd: str, settings_manager: SettingsManager) -> DefaultResourceLoader:
    inherit = bool(getattr(parsed, "inherit", False))
    loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(
            cwd=cwd,
            agent_dir=get_agent_dir(),
            settings_manager=settings_manager,
            additional_extension_paths=parsed.extensions or [],
            additional_skill_paths=parsed.skills or [],
            additional_prompt_template_paths=parsed.prompt_templates or [],
            additional_theme_paths=parsed.themes or [],
            no_extensions=parsed.no_extensions,
            no_skills=parsed.no_skills,
            no_prompt_templates=parsed.no_prompt_templates,
            no_themes=parsed.no_themes,
            no_context_files=parsed.no_context_files,
            system_prompt=parsed.system_prompt,
            append_system_prompt=parsed.append_system_prompt,
            inherit_global=inherit,
        )
    )
    await loader.reload()
    return loader


async def _create_runtime_host(
    parsed: Any,
    *,
    cwd: str,
    session,
    auth_storage: AuthStorage,
    model_registry: ModelRegistry,
    settings_manager: SettingsManager,
    resource_loader: DefaultResourceLoader,
    resolved_model,
    thinking,
):
    """Create the runtime host used by interactive/RPC session replacement."""
    agent_dir = get_agent_dir()

    async def create_runtime(options: dict[str, Any]) -> CreateAgentSessionRuntimeResult:
        runtime_cwd = options.get("cwd") or cwd
        # Mirror the launch-time inherit choice: without --inherit, global
        # resource arrays (extensions/skills/prompts/themes) must stay out of the
        # runtime-built session — otherwise create_agent_session_runtime, which
        # rebuilds the session via this factory, would silently re-introduce the
        # global ~/.pi-py extensions the user opted out of.
        runtime_settings = SettingsManager.create(
            cwd=runtime_cwd,
            agent_dir=agent_dir,
            options={"keepAgentResources": bool(os.environ.get("PI_CODING_AGENT_DIR"))},
            inherit_global=bool(getattr(parsed, "inherit", False)),
        )
        session_vars = getattr(parsed, "session_vars", None)
        if session_vars:
            runtime_settings.apply_overrides({"session_vars": session_vars})
        runtime_loader = await _create_resource_loader(parsed, runtime_cwd, runtime_settings)
        runtime_services = AgentSessionServices(
            cwd=runtime_cwd,
            agent_dir=agent_dir,
            auth_storage=auth_storage,
            settings_manager=runtime_settings,
            model_registry=model_registry,
            resource_loader=runtime_loader,
        )
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=runtime_cwd,
                model=resolved_model,
                thinking_level=thinking,
                session_manager=options["session_manager"],
                auth_storage=auth_storage,
                model_registry=model_registry,
                settings_manager=runtime_settings,
                resource_loader=runtime_loader,
                tools=parsed.tools,
                exclude_tools=getattr(parsed, "exclude_tools", None),
                no_tools="all" if parsed.no_tools else "builtin" if getattr(parsed, "no_builtin_tools", False) else None,
                session_vars=session_vars,
            )
        )
        return CreateAgentSessionRuntimeResult(
            session=result.session,
            services=runtime_services,
            extensions_result=result.extensions_result,
            model_fallback_message=result.model_fallback_message,
        )

    return await create_agent_session_runtime(
        create_runtime,
        {
            "cwd": cwd,
            "session_manager": session.session_manager,
        },
    )


async def _resolve_session_path(session_arg: str, cwd: str, session_dir: str | None) -> dict[str, Any]:
    if _looks_like_path(session_arg):
        return {"type": "path", "path": session_arg}

    local_sessions = await SessionManager.list(cwd, session_dir)
    local_matches = [s for s in local_sessions if s.session_id.startswith(session_arg)]
    if local_matches:
        return {"type": "local", "path": local_matches[0].file_path}

    global_sessions = await SessionManager.list_all()
    global_matches = [s for s in global_sessions if s.session_id.startswith(session_arg)]
    if global_matches:
        match = global_matches[0]
        return {"type": "global", "path": match.file_path, "cwd": match.cwd}

    return {"type": "not_found", "arg": session_arg}


async def _create_session_manager(parsed: Any, cwd: str) -> SessionManager | None:
    if parsed.no_session:
        return SessionManager.in_memory(cwd)
    if parsed.session:
        resolved = await _resolve_session_path(parsed.session, cwd, parsed.session_dir)
        rtype = resolved["type"]
        if rtype in {"path", "local"}:
            return SessionManager.open(resolved["path"], parsed.session_dir)
        if rtype == "global":
            print(f"Session found in different project: {resolved['cwd']}", file=sys.stderr)
            if not await _prompt_confirm("Fork this session into current directory?"):
                print("Aborted.")
                return None
            return SessionManager.fork_from(resolved["path"], cwd, parsed.session_dir)
        print(f"No session found matching '{resolved['arg']}'", file=sys.stderr)
        return None
    if parsed.fork:
        resolved = await _resolve_session_path(parsed.fork, cwd, parsed.session_dir)
        if resolved["type"] in {"path", "local", "global"}:
            return SessionManager.fork_from(resolved["path"], cwd, parsed.session_dir)
        print(f"No session found matching '{resolved['arg']}'", file=sys.stderr)
        return None
    if parsed.continue_:
        return SessionManager.continue_recent(cwd, parsed.session_dir)
    if parsed.session_id:
        return SessionManager.create(cwd, parsed.session_dir, session_id=parsed.session_id)
    if parsed.session_dir:
        return SessionManager.create(cwd, parsed.session_dir)
    return None


def _report_settings_errors(settings_manager: SettingsManager, context: str) -> None:
    for item in settings_manager.drain_errors():
        scope = item.get("scope", "unknown")
        error = item.get("error")
        message = str(error) if error else "Unknown settings error"
        print(f"Warning ({context}, {scope} settings): {message}", file=sys.stderr)


def _resolve_project_trusted(cwd: str, agent_dir: str, trust_override: bool | None = None) -> bool:
    if trust_override is not None:
        return trust_override
    return not has_project_trust_inputs(cwd) or ProjectTrustStore(agent_dir).get(cwd) is True


def _parse_package_command(args: Sequence[str]) -> dict[str, Any] | None:
    if not args:
        return None
    raw_command = args[0]
    command = "remove" if raw_command == "uninstall" else raw_command
    if command not in {"install", "remove", "update", "list"}:
        return None

    source: str | None = None
    update_target: dict[str, str | None] | None = None
    local = False
    force = False
    project_trust_override: bool | None = None
    help_requested = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    missing_option_value: str | None = None
    conflicting_options: str | None = None
    self_flag = False
    extensions_flag = False
    extension_flag_source: str | None = None

    index = 1
    while index < len(args):
        arg = args[index]
        if arg in {"-h", "--help"}:
            help_requested = True
            index += 1
            continue
        if arg in {"-l", "--local"}:
            if command in {"install", "remove"}:
                local = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--self":
            if command == "update":
                self_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--extensions":
            if command == "update":
                extensions_flag = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg in {"--approve", "-a"}:
            project_trust_override = True
            index += 1
            continue
        if arg in {"--no-approve", "-na"}:
            project_trust_override = False
            index += 1
            continue
        if arg == "--force":
            if command == "update":
                force = True
            else:
                invalid_option = invalid_option or arg
            index += 1
            continue
        if arg == "--extension":
            if command != "update":
                invalid_option = invalid_option or arg
                index += 1
                continue
            value = args[index + 1] if index + 1 < len(args) else None
            if not value or value.startswith("-"):
                missing_option_value = missing_option_value or arg
            elif extension_flag_source:
                conflicting_options = conflicting_options or "--extension can only be provided once"
                index += 1
            else:
                extension_flag_source = value
                index += 1
            index += 1
            continue
        if arg.startswith("-"):
            invalid_option = invalid_option or arg
            index += 1
            continue
        if source is None:
            source = arg
        else:
            invalid_argument = invalid_argument or arg
        index += 1

    if command == "update":
        if extension_flag_source:
            if self_flag or extensions_flag:
                conflicting_options = conflicting_options or "--extension cannot be combined with --self or --extensions"
            if source:
                conflicting_options = conflicting_options or "--extension cannot be combined with a positional source"
            update_target = {"type": "extensions", "source": extension_flag_source}
        elif source:
            source_is_all = source == "all"
            source_is_self = source in {"self", "pi", "pi-py"}
            if source_is_all:
                update_target = {"type": "all", "source": None}
            if source_is_self:
                update_target = {"type": "all" if extensions_flag else "self", "source": None}
            elif not source_is_all:
                if extensions_flag or self_flag:
                    conflicting_options = (
                        conflicting_options
                        or "positional update targets cannot be combined with --self or --extensions"
                    )
                update_target = {"type": "extensions", "source": source}
        elif self_flag and extensions_flag:
            update_target = {"type": "all", "source": None}
        elif self_flag:
            update_target = {"type": "self", "source": None}
        elif extensions_flag:
            update_target = {"type": "extensions", "source": None}
        else:
            update_target = {"type": "self", "source": None}

    return {
        "command": command,
        "source": source,
        "update_target": update_target,
        "local": local,
        "force": force,
        "project_trust_override": project_trust_override,
        "help": help_requested,
        "invalid_option": invalid_option,
        "invalid_argument": invalid_argument,
        "missing_option_value": missing_option_value,
        "conflicting_options": conflicting_options,
    }


def _package_usage(command: str) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l] [--approve|--no-approve]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l] [--approve|--no-approve]"
    if command == "update":
        return (
            f"{APP_NAME} update [source|self|pi|pi-py|all] [--self] [--extensions] "
            "[--extension <source>] [--approve|--no-approve] [--force]"
        )
    return f"{APP_NAME} list [--approve|--no-approve]"


def _print_package_help(command: str) -> None:
    usage = _package_usage(command)
    if command == "install":
        print(f"Usage:\n  {usage}\n\nInstall a package and add it to settings.\n")
    elif command == "remove":
        print(f"Usage:\n  {usage}\n\nRemove a package and its source from settings.\n\nAlias: {APP_NAME} uninstall <source> [-l]\n")
    elif command == "update":
        print(
            f"Usage:\n  {usage}\n\n"
            "Update pi-py by default. Use --extensions, --extension <source>, or all to update installed packages.\n"
        )
    else:
        print(f"Usage:\n  {usage}\n\nList installed packages from user and project settings.\n")


async def _handle_package_command(args: Sequence[str]) -> tuple[bool, int]:
    parsed = _parse_package_command(args)
    if not parsed:
        return False, 0

    command = parsed["command"]
    source = parsed["source"]
    local = parsed["local"]
    force = parsed["force"]
    update_target = parsed["update_target"]

    if parsed["help"]:
        _print_package_help(command)
        return True, 0
    if parsed["invalid_option"]:
        print(f'Unknown option {parsed["invalid_option"]} for "{command}".', file=sys.stderr)
        print(f'Use "{APP_NAME} --help" or "{_package_usage(command)}".', file=sys.stderr)
        return True, 1
    if parsed["missing_option_value"]:
        print(f'Missing value for {parsed["missing_option_value"]}.', file=sys.stderr)
        print(f"Usage: {_package_usage(command)}", file=sys.stderr)
        return True, 1
    if parsed["invalid_argument"]:
        print(f'Unexpected argument {parsed["invalid_argument"]}.', file=sys.stderr)
        print(f"Usage: {_package_usage(command)}", file=sys.stderr)
        return True, 1
    if parsed["conflicting_options"]:
        print(parsed["conflicting_options"], file=sys.stderr)
        print(f"Usage: {_package_usage(command)}", file=sys.stderr)
        return True, 1
    if command in {"install", "remove"} and not source:
        print(f"Missing {command} source.", file=sys.stderr)
        print(f"Usage: {_package_usage(command)}", file=sys.stderr)
        return True, 1

    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    project_trusted = _resolve_project_trusted(cwd, agent_dir, parsed["project_trust_override"])
    if not project_trusted and command in {"install", "remove"} and local:
        print("Project is not trusted. Use --approve to modify local package config.", file=sys.stderr)
        return True, 1
    settings_manager = SettingsManager.create(cwd, agent_dir, {"projectTrusted": project_trusted})
    _report_settings_errors(settings_manager, "package command")
    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager)
    package_manager.set_progress_callback(
        lambda event: print(event.message, file=sys.stderr) if event.type == "start" and event.message else None
    )

    try:
        if command == "install":
            await package_manager.install_and_persist(source, {"local": local})
            print(f"Installed {source}")
            return True, 0
        if command == "remove":
            removed = await package_manager.remove_and_persist(source, {"local": local})
            if not removed:
                print(f"No matching package found for {source}", file=sys.stderr)
                return True, 1
            print(f"Removed {source}")
            return True, 0
        if command == "list":
            configured_packages = package_manager.list_configured_packages()
            user_packages = [pkg for pkg in configured_packages if pkg.scope == "user"]
            project_packages = [pkg for pkg in configured_packages if pkg.scope == "project"]
            if not configured_packages:
                print("No packages installed.")
                return True, 0

            def _format_pkg(pkg: Any, scope: str) -> None:
                display = f"{pkg.source} (filtered)" if pkg.filtered else pkg.source
                print(f"  {display}")
                if pkg.installed_path:
                    print(f"    {pkg.installed_path}")

            if user_packages:
                print("User packages:")
                for pkg in user_packages:
                    _format_pkg(pkg, "user")
            if project_packages:
                if user_packages:
                    print()
                print("Project packages:")
                for pkg in project_packages:
                    _format_pkg(pkg, "project")
            return True, 0

        if update_target and update_target["type"] == "all":
            await package_manager.update(None)
            print("Updated packages")
            await package_manager.self_update(force=force)
            print(f"Updated {APP_NAME}")
            return True, 0
        if update_target and update_target["type"] == "self":
            await package_manager.self_update(force=force)
            print(f"Updated {APP_NAME}")
            return True, 0
        target_source = update_target.get("source") if update_target else source
        await package_manager.update(target_source)
        if target_source:
            print(f"Updated {target_source}")
        else:
            print("Updated packages")
        return True, 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return True, 1


async def _handle_config_command(args: Sequence[str]) -> tuple[bool, int]:
    if not args or args[0] != "config":
        return False, 0
    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    trust_override = None
    for arg in args:
        if arg in {"--approve", "-a"}:
            trust_override = True
        elif arg in {"--no-approve", "-na"}:
            trust_override = False
    settings_manager = SettingsManager.create(
        cwd,
        agent_dir,
        {"projectTrusted": _resolve_project_trusted(cwd, agent_dir, trust_override)},
    )
    _report_settings_errors(settings_manager, "config command")
    package_manager = DefaultPackageManager(cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager)
    resolved = await package_manager.resolve()

    print("Resolved package resources:")
    for label, entries in (
        ("extensions", resolved.extensions),
        ("skills", resolved.skills),
        ("prompts", resolved.prompts),
        ("themes", resolved.themes),
    ):
        print(f"- {label}: {len(entries)}")
    return True, 0


async def _run(args: Sequence[str]) -> int:
    log_event("run_start", args=list(args), cwd=os.getcwd())
    # Load workspace environment variables early so model/api-key resolution
    # can see keys from .env (e.g. GEMINI_API_KEY).
    _load_env_files(os.getcwd())
    log_event("env_files_loaded")

    handled, exit_code = await _handle_package_command(args)
    if handled:
        log_event("package_command_handled", exit_code=exit_code)
        return exit_code
    handled, exit_code = await _handle_config_command(args)
    if handled:
        log_event("config_command_handled", exit_code=exit_code)
        return exit_code

    migration_result = run_migrations(os.getcwd())
    migrated_auth_providers = migration_result.get("migratedAuthProviders", [])
    deprecation_warnings = migration_result.get("deprecationWarnings", [])
    log_event(
        "migrations_checked",
        migrated_auth_providers=migrated_auth_providers,
        deprecation_warning_count=len(deprecation_warnings or []),
    )

    first_pass = parse_args(list(args))
    event_bus = create_event_bus()
    ext_paths = first_pass.extensions or []
    if not first_pass.no_extensions:
        ext_paths = (
            get_extension_discovery_paths(
                os.getcwd(),
                get_agent_dir(),
                inherit_global=bool(getattr(first_pass, "inherit", False)),
            )
            + ext_paths
        )
    log_event("extension_first_pass_start", extension_paths=ext_paths)
    extensions_result = await load_extensions(ext_paths, os.getcwd(), event_bus)
    log_event(
        "extension_first_pass_end",
        loaded_count=len(extensions_result.extensions),
        error_count=len(getattr(extensions_result, "errors", []) or []),
    )
    extension_flags: dict[str, str] = {}
    for ext in extensions_result.extensions:
        for name, flag in ext.flags.items():
            extension_flags[name] = "string" if flag.type == "string" else "boolean"

    parsed = parse_args(list(args), extension_flags=extension_flags)
    log_event(
        "args_parsed",
        mode=parsed.mode,
        print_mode=parsed.print_mode,
        init=getattr(parsed, "init", False),
        resume=parsed.resume,
        session=parsed.session,
        session_id=parsed.session_id,
        session_dir=parsed.session_dir,
        no_session=parsed.no_session,
        provider=parsed.provider,
        model=parsed.model,
        tools=parsed.tools,
        no_tools=parsed.no_tools,
        no_builtin_tools=parsed.no_builtin_tools,
        extension_flags=extension_flags,
    )

    if parsed.version:
        from .config import VERSION

        print(VERSION)
        log_event("version_printed", version=VERSION)
        return 0

    if parsed.help:
        print_help()
        log_event("help_printed")
        return 0

    # Read piped stdin for non-rpc mode
    if parsed.mode != "rpc":
        stdin_content = await _read_piped_stdin()
        if stdin_content is not None:
            parsed.print_mode = True
            parsed.messages.insert(0, stdin_content)
            log_event("stdin_prompt_loaded", chars=len(stdin_content))

    if parsed.export:
        from .core.export_html import export_from_file

        log_event("export_start", export=parsed.export)
        output_path = parsed.messages[0] if parsed.messages else None
        exported = await export_from_file(parsed.export, output_path=output_path)
        print(f"Exported to: {exported}")
        log_event("export_end", exported=exported)
        return 0

    if parsed.mode == "rpc" and parsed.file_args:
        print("Error: @file arguments are not supported in RPC mode", file=sys.stderr)
        log_event("rpc_file_args_rejected", file_args=parsed.file_args)
        return 1

    cwd = os.getcwd()
    log_event("cwd_resolved", cwd=cwd)
    # Seed the new global dir (~/.pi-py/agent) with auth/models from the legacy
    # Node dir on first run so existing API keys keep working.
    from .config import migrate_legacy_global_config

    migrate_legacy_global_config()
    log_event("legacy_config_migration_checked")
    # --init: scaffold the project-local .pi-py structure, then continue to launch.
    if getattr(parsed, "init", False):
        from .config import scaffold_project

        created = scaffold_project(cwd)
        log_event("project_scaffold_checked", created=[os.path.relpath(path, cwd) for path in created])
        if created:
            print(f"Initialized .pi-py agent project in {cwd}:", file=sys.stderr)
            for path in created:
                print(f"  + {os.path.relpath(path, cwd)}", file=sys.stderr)
        else:
            print(".pi-py project already initialized.", file=sys.stderr)
    # Sessions are contained per-project by default (unless --session-dir given).
    if not parsed.session_dir and not parsed.no_session:
        from .config import get_project_sessions_dir

        parsed.session_dir = get_project_sessions_dir(cwd)
        log_event("session_dir_defaulted", session_dir=parsed.session_dir)
    # Every launch (not just --init) leaves a populated, visible project config
    # so a normal `pi-py` run never produces a half-empty .pi-py (sessions dir
    # but no settings.json). Inherited global defaults are written in.
    if not parsed.inherit:
        from .config import ensure_project_settings

        ensure_project_settings(cwd)
        log_event("project_settings_ensured")
    settings_manager = SettingsManager.create(
        cwd,
        get_agent_dir(),
        {"keepAgentResources": bool(os.environ.get("PI_CODING_AGENT_DIR"))},
        inherit_global=parsed.inherit,
    )
    _report_settings_errors(settings_manager, "startup")
    if parsed.session_vars:
        settings_manager.apply_overrides({"session_vars": parsed.session_vars})
    log_event(
        "settings_loaded",
        inherit=parsed.inherit,
        default_provider=settings_manager.get_default_provider(),
        default_model=settings_manager.get_default_model(),
        default_thinking=settings_manager.get_default_thinking_level(),
        agent_name=settings_manager.get_agent_name(),
        tools=settings_manager.get_tools(),
        extensions=settings_manager.get_extensions(),
        memory_enabled=settings_manager.get().memory_enabled,
        session_vars=settings_manager.get().session_vars,
    )
    auth_storage = AuthStorage()
    # Pass auth_storage so /model and --list-models surface every provider you
    # have credentials for (via has_auth on the store), not just the handful in
    # the env-var fallback map.
    model_registry = ModelRegistry(auth_storage=auth_storage)
    session_manager = await _create_session_manager(parsed, cwd)
    log_event(
        "session_manager_created",
        has_session_manager=session_manager is not None,
        session_dir=parsed.session_dir,
    )
    if parsed.session and session_manager is None:
        log_event("session_manager_missing_for_requested_session")
        return 1

    resolved_model, thinking = _resolve_model_for_session(parsed, model_registry, settings_manager)
    log_event(
        "model_resolved",
        provider=getattr(resolved_model, "provider", None),
        model_id=getattr(resolved_model, "id", None),
        api=getattr(resolved_model, "api", None),
        context_window=getattr(resolved_model, "context_window", None),
        max_tokens=getattr(resolved_model, "max_tokens", None),
        thinking=thinking,
    )
    resource_loader = await _create_resource_loader(parsed, cwd, settings_manager)
    extensions_result = resource_loader.get_extensions()
    log_event(
        "resource_loader_ready",
        extensions_count=len(extensions_result.get("extensions") or []),
        extension_diagnostics=extensions_result.get("diagnostics") or [],
    )

    opts = CreateAgentSessionOptions(
        cwd=cwd,
        model=resolved_model,
        thinking_level=thinking,
        session_manager=session_manager,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        tools=parsed.tools,
        exclude_tools=parsed.exclude_tools,
        no_tools="all" if parsed.no_tools else "builtin" if parsed.no_builtin_tools else None,
        session_vars=parsed.session_vars,
    )
    result = await create_agent_session(opts)
    session = result.session
    event_unsub = attach_session_event_logging(session)
    log_session_snapshot("created", session)
    if parsed.name:
        session.session_manager.append_session_info(parsed.name)
        log_event("session_named", name=parsed.name)

    if parsed.list_models is not None:
        pattern = parsed.list_models if isinstance(parsed.list_models, str) else None
        log_event("list_models_start", pattern=pattern)
        await list_models(model_registry, pattern)
        log_event("list_models_end")
        return 0

    if parsed.mode == "rpc":
        log_event("mode_start", mode="rpc")
        runtime_host = await _create_runtime_host(
            parsed,
            cwd=cwd,
            session=session,
            auth_storage=auth_storage,
            model_registry=model_registry,
            settings_manager=settings_manager,
            resource_loader=resource_loader,
            resolved_model=resolved_model,
            thinking=thinking,
        )
        await run_rpc_mode(runtime_host)
        log_event("mode_end", mode="rpc")
        return 0

    initial_prompt, images = await _build_initial_prompt(parsed)

    # Print mode (explicit) or JSON mode
    if parsed.print_mode or parsed.mode in ("text", "json"):
        mode_name = "json" if parsed.mode == "json" else "text"
        log_event("mode_start", mode=mode_name)
        prompt = initial_prompt
        if not prompt and parsed.messages:
            prompt = parsed.messages[0]
        if not prompt:
            print("No prompt provided. Use --help for usage.", file=sys.stderr)
            log_event("print_mode_missing_prompt")
            return 1
        exit_code = await run_print_mode(
            session,
            prompt,
            show_thinking=bool(parsed.verbose),
            json_output=parsed.mode == "json",
        )
        log_event("mode_end", mode=mode_name, exit_code=exit_code)
        return exit_code

    # Default interactive mode
    # (Branded startup banner is rendered inside the TUI, after the
    # [Extension issues] section — see modes/interactive/tui.py.)
    if deprecation_warnings:
        await show_deprecation_warnings(deprecation_warnings)

    if migrated_auth_providers and parsed.verbose:
        print(f"Migrated auth providers: {', '.join(migrated_auth_providers)}", file=sys.stderr)

    # --resume: interactive session picker
    if parsed.resume:
        log_event("resume_picker_start")
        selected = await select_session(
            lambda: SessionManager.list(cwd, parsed.session_dir),
            SessionManager.list_all,
        )
        if not selected:
            print("No session selected")
            log_event("resume_picker_cancelled")
            return 0
        log_event("resume_picker_selected", selected=selected)
        sm = SessionManager.open(selected, parsed.session_dir)
        resolved_model, thinking = _resolve_model_for_session(parsed, model_registry, settings_manager)
        resource_loader = await _create_resource_loader(parsed, cwd, settings_manager)
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                model=resolved_model,
                thinking_level=thinking,
                session_manager=sm,
                auth_storage=auth_storage,
                model_registry=model_registry,
                settings_manager=settings_manager,
                resource_loader=resource_loader,
                tools=parsed.tools,
                no_tools="all" if parsed.no_tools else None,
            )
        )
        session = result.session
        if event_unsub:
            try:
                event_unsub()
            except Exception:
                pass
        event_unsub = attach_session_event_logging(session)
        log_session_snapshot("resumed", session)

    initial_messages = []
    if initial_prompt:
        initial_messages.append(initial_prompt)
    initial_messages.extend(parsed.messages[1:] if parsed.messages else [])
    runtime_host = await _create_runtime_host(
        parsed,
        cwd=session.session_manager.get_cwd(),
        session=session,
        auth_storage=auth_storage,
        model_registry=model_registry,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        resolved_model=resolved_model,
        thinking=thinking,
    )
    log_event("mode_start", mode="interactive", initial_message_count=len(initial_messages or []))
    await run_interactive_mode(runtime_host, initial_messages=initial_messages or None)
    log_event("mode_end", mode="interactive")
    if event_unsub:
        try:
            event_unsub()
        except Exception:
            pass
    return 0


def main(args: Sequence[str] | None = None) -> None:
    """CLI entrypoint used by project script."""
    run_args = args if args is not None else sys.argv[1:]
    _dispatch_to_local_project(run_args, os.getcwd())
    configure_cli_debug_logging(cwd=os.getcwd(), argv=run_args)
    try:
        exit_code = asyncio.run(_run(run_args))
    except BaseException as exc:
        log_exception("unhandled_exception", exc)
        raise
    log_event("process_exit", exit_code=exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
