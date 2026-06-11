"""
CLI argument parsing and help display.

Mirrors packages/coding-agent/src/cli/args.ts
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Literal

from pi_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR

Mode = Literal["text", "json", "rpc"]

VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


@dataclass
class Args:
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    session_vars: dict[str, str] = field(default_factory=dict)
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)
    diagnostics: list[dict[str, str]] = field(default_factory=list)

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: list[str] = field(default_factory=list)
    thinking: str | None = None
    continue_: bool = False
    resume: bool = False
    help: bool = False
    version: bool = False
    mode: Mode | None = None
    name: str | None = None
    no_session: bool = False
    session: str | None = None
    session_id: str | None = None
    fork: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: bool = False
    no_builtin_tools: bool = False
    extensions: list[str] | None = None
    no_extensions: bool = False
    print_mode: bool = False
    export: str | None = None
    no_skills: bool = False
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool = False
    themes: list[str] | None = None
    no_themes: bool = False
    no_context_files: bool = False
    list_models: str | bool | None = None
    verbose: bool = False
    offline: bool = False
    project_trust_override: bool | None = None
    # Pull in global (~/.pi-py/agent) skills/prompts/extensions/settings.
    # Default: project-local config only.
    inherit: bool = False
    # Scaffold the project-local .pi-py structure here, then launch.
    init: bool = False


VALID_TOOL_NAMES = {"read", "bash", "edit", "write", "grep", "find", "ls"}


def _session_var_name(name: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").upper()
    if not cleaned or not re.match(r"^[A-Z_][A-Z0-9_]*$", cleaned):
        return None
    return cleaned


def _parse_session_var_token(arg: str) -> tuple[str, str] | None:
    if "=" in arg:
        name, value = arg.split("=", 1)
    else:
        return None
    key = _session_var_name(name)
    if not key or value == "":
        return None
    return key, value


def parse_args(
    args: list[str],
    extension_flags: dict[str, Literal["boolean", "string"]] | None = None,
) -> Args:
    """Parse CLI arguments into an Args dataclass."""
    result = Args()
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(args):
            i += 1
            m = args[i]
            if m in ("text", "json", "rpc"):
                result.mode = m  # type: ignore[assignment]
        elif arg in ("--continue", "-c"):
            result.continue_ = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(args):
            i += 1
            result.provider = args[i]
        elif arg == "--model" and i + 1 < len(args):
            i += 1
            result.model = args[i]
        elif arg == "--api-key" and i + 1 < len(args):
            i += 1
            result.api_key = args[i]
        elif arg == "--system-prompt" and i + 1 < len(args):
            i += 1
            result.system_prompt = args[i]
        elif arg == "--append-system-prompt" and i + 1 < len(args):
            i += 1
            result.append_system_prompt.append(args[i])
        elif arg in ("--name", "-n"):
            if i + 1 < len(args):
                i += 1
                result.name = args[i]
            else:
                result.diagnostics.append({"type": "error", "message": "--name requires a value"})
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(args):
            i += 1
            result.session = args[i]
        elif arg == "--session-id" and i + 1 < len(args):
            i += 1
            result.session_id = args[i]
        elif arg == "--fork" and i + 1 < len(args):
            i += 1
            result.fork = args[i]
        elif arg == "--session-dir" and i + 1 < len(args):
            i += 1
            result.session_dir = args[i]
        elif arg == "--models" and i + 1 < len(args):
            i += 1
            result.models = [s.strip() for s in args[i].split(",")]
        elif arg in ("--no-tools", "-nt"):
            result.no_tools = True
        elif arg in ("--no-builtin-tools", "-nbt"):
            result.no_builtin_tools = True
        elif arg in ("--tools", "-t") and i + 1 < len(args):
            i += 1
            tool_names = [s.strip() for s in args[i].split(",")]
            valid: list[str] = []
            for name in tool_names:
                if name in VALID_TOOL_NAMES:
                    valid.append(name)
                else:
                    print(
                        f"Warning: Unknown tool \"{name}\". Valid tools: {', '.join(sorted(VALID_TOOL_NAMES))}",
                        file=sys.stderr,
                    )
            result.tools = valid
        elif arg in ("--exclude-tools", "-xt") and i + 1 < len(args):
            i += 1
            result.exclude_tools = [s.strip() for s in args[i].split(",") if s.strip()]
        elif arg == "--thinking" and i + 1 < len(args):
            i += 1
            level = args[i]
            if is_valid_thinking_level(level):
                result.thinking = level
            else:
                print(
                    f'Warning: Invalid thinking level "{level}". Valid values: {", ".join(VALID_THINKING_LEVELS)}',
                    file=sys.stderr,
                )
        elif arg in ("--print", "-p"):
            result.print_mode = True
            next_arg = args[i + 1] if i + 1 < len(args) else None
            if next_arg is not None and not next_arg.startswith("@") and (
                not next_arg.startswith("-") or next_arg.startswith("---")
            ):
                result.messages.append(next_arg)
                i += 1
        elif arg == "--export" and i + 1 < len(args):
            i += 1
            result.export = args[i]
        elif arg in ("--extension", "-e") and i + 1 < len(args):
            i += 1
            if result.extensions is None:
                result.extensions = []
            result.extensions.append(args[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(args):
            i += 1
            if result.skills is None:
                result.skills = []
            result.skills.append(args[i])
        elif arg == "--prompt-template" and i + 1 < len(args):
            i += 1
            if result.prompt_templates is None:
                result.prompt_templates = []
            result.prompt_templates.append(args[i])
        elif arg == "--theme" and i + 1 < len(args):
            i += 1
            if result.themes is None:
                result.themes = []
            result.themes.append(args[i])
        elif arg in ("--var", "--session-var") and i + 1 < len(args):
            i += 1
            parsed_var = _parse_session_var_token(args[i])
            if parsed_var is None:
                result.diagnostics.append({"type": "error", "message": "--var requires KEY=VALUE"})
            else:
                key, value = parsed_var
                result.session_vars[key] = value
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg in ("--no-context-files", "-nc"):
            result.no_context_files = True
        elif arg == "--list-models":
            if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@"):
                i += 1
                result.list_models = args[i]
            else:
                result.list_models = True
        elif arg == "--verbose":
            result.verbose = True
        elif arg in ("--approve", "-a"):
            result.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            result.project_trust_override = False
        elif arg == "--offline":
            result.offline = True
        elif arg == "--inherit":
            result.inherit = True
        elif arg == "--init":
            result.init = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--"):
            eq_index = arg.find("=")
            if eq_index != -1:
                flag_name = arg[2:eq_index]
                value = arg[eq_index + 1:]
                result.unknown_flags[flag_name] = value
                key = _session_var_name(flag_name)
                if key:
                    result.session_vars[key] = value
            else:
                flag_name = arg[2:]
                ext_flag = (extension_flags or {}).get(flag_name)
                next_arg = args[i + 1] if i + 1 < len(args) else None
                value: bool | str
                if ext_flag == "boolean":
                    value = True
                elif ext_flag == "string" and next_arg is not None:
                    i += 1
                    value = args[i]
                elif next_arg is not None and not next_arg.startswith("-") and not next_arg.startswith("@"):
                    i += 1
                    value = args[i]
                else:
                    value = True
                result.unknown_flags[flag_name] = value
                key = _session_var_name(flag_name)
                if key:
                    result.session_vars[key] = "true" if value is True else str(value)
        elif arg.startswith("-"):
            result.diagnostics.append({"type": "error", "message": f"Unknown option: {arg}"})
        elif not arg.startswith("-"):
            session_var = _parse_session_var_token(arg)
            if session_var is not None:
                key, value = session_var
                result.session_vars[key] = value
            else:
                result.messages.append(arg)

        i += 1

    return result


parseArgs = parse_args
isValidThinkingLevel = is_valid_thinking_level


def print_help() -> None:
    """Print CLI help text."""
    print(f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source|self|pi]   Update pi and installed extensions
  {APP_NAME} list [--approve|--no-approve]
                                  List installed extensions from settings
  {APP_NAME} config [--no-approve]
                                  Open TUI to enable/disable package resources
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --var KEY=VALUE                Set an arbitrary session variable for prompt templates
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --name, -n <name>              Set session display name
  --no-session                   Don't save session (ephemeral)
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names to enable
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names to disable
  --thinking <level>             Set thinking level: {', '.join(VALID_THINKING_LEVELS)}
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PI_OFFLINE=1)
  --inherit                      Also load global (~/.pi-py/agent) skills, prompts,
                                   extensions, and settings (default: project-local only)
  --init                         Scaffold a .pi-py project (settings, skills/, prompts/,
                                   extensions/, AGENTS.md) in the current dir, then launch
  --help, -h                     Show this help
  --version, -v                  Show version number

Environment Variables:
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  GEMINI_API_KEY                   - Google Gemini API key
  {ENV_AGENT_DIR:<32} - Agent config directory (default: ~/{CONFIG_DIR_NAME}/agent)
  {ENV_SESSION_DIR:<32} - Session storage directory override
""")
