from .read import ReadOperations, DefaultReadOperations, create_read_tool, read_tool
from .write import create_write_tool, write_tool
from .edit import create_edit_tool, edit_tool
from .bash import create_bash_tool, bash_tool
from .grep import create_grep_tool, grep_tool
from .find import create_find_tool, find_tool
from .ls import create_ls_tool, ls_tool
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    truncate_head,
    truncate_tail,
    truncate_line,
    format_size,
)
from .output_accumulator import OutputAccumulator, OutputSnapshot
from .file_mutation_queue import (
    active_file_mutation_queue_count,
    get_mutation_queue_key,
    with_file_mutation_queue,
)
from .render_utils import (
    get_text_output,
    invalid_arg_text,
    normalize_display_text,
    render_tool_path,
    replace_tabs,
    shorten_path,
    str_or_none,
)
from .tool_definition_wrapper import (
    create_tool_definition_from_agent_tool,
    wrap_tool_definition,
    wrap_tool_definitions,
)

TOOL_NAMES = ("read", "bash", "edit", "write", "grep", "find", "ls")
all_tool_names = set(TOOL_NAMES)


def create_tool(tool_name: str, cwd: str, options: dict | None = None):
    opts = options or {}
    if tool_name == "read":
        return create_read_tool(cwd, **opts.get("read", {}))
    if tool_name == "bash":
        return create_bash_tool(cwd, **opts.get("bash", {}))
    if tool_name == "edit":
        return create_edit_tool(cwd)
    if tool_name == "write":
        return create_write_tool(cwd)
    if tool_name == "grep":
        return create_grep_tool(cwd)
    if tool_name == "find":
        return create_find_tool(cwd)
    if tool_name == "ls":
        return create_ls_tool(cwd)
    raise ValueError(f"Unknown tool name: {tool_name}")


def create_tool_definition(tool_name: str, cwd: str, options: dict | None = None):
    return create_tool_definition_from_agent_tool(create_tool(tool_name, cwd, options))


def create_read_tool_definition(cwd: str, options: dict | None = None):
    opts = options or {}
    return create_tool_definition_from_agent_tool(create_read_tool(cwd, **opts))


def create_bash_tool_definition(cwd: str, options: dict | None = None):
    opts = options or {}
    return create_tool_definition_from_agent_tool(create_bash_tool(cwd, **opts))


def create_edit_tool_definition(cwd: str, options: dict | None = None):
    del options
    return create_tool_definition_from_agent_tool(create_edit_tool(cwd))


def create_write_tool_definition(cwd: str, options: dict | None = None):
    del options
    return create_tool_definition_from_agent_tool(create_write_tool(cwd))


def create_grep_tool_definition(cwd: str, options: dict | None = None):
    del options
    return create_tool_definition_from_agent_tool(create_grep_tool(cwd))


def create_find_tool_definition(cwd: str, options: dict | None = None):
    del options
    return create_tool_definition_from_agent_tool(create_find_tool(cwd))


def create_ls_tool_definition(cwd: str, options: dict | None = None):
    del options
    return create_tool_definition_from_agent_tool(create_ls_tool(cwd))


def create_coding_tools(cwd: str, options: dict | None = None):
    return [create_tool(name, cwd, options) for name in ("read", "bash", "edit", "write")]


def create_read_only_tools(cwd: str, options: dict | None = None):
    return [create_tool(name, cwd, options) for name in ("read", "grep", "find", "ls")]


def create_all_tools(cwd: str, options: dict | None = None):
    return {name: create_tool(name, cwd, options) for name in TOOL_NAMES}


def create_coding_tool_definitions(cwd: str, options: dict | None = None):
    return [create_tool_definition(name, cwd, options) for name in ("read", "bash", "edit", "write")]


def create_read_only_tool_definitions(cwd: str, options: dict | None = None):
    return [create_tool_definition(name, cwd, options) for name in ("read", "grep", "find", "ls")]


def create_all_tool_definitions(cwd: str, options: dict | None = None):
    return {name: create_tool_definition(name, cwd, options) for name in TOOL_NAMES}


createTool = create_tool
createToolDefinition = create_tool_definition
createReadToolDefinition = create_read_tool_definition
createBashToolDefinition = create_bash_tool_definition
createEditToolDefinition = create_edit_tool_definition
createWriteToolDefinition = create_write_tool_definition
createGrepToolDefinition = create_grep_tool_definition
createFindToolDefinition = create_find_tool_definition
createLsToolDefinition = create_ls_tool_definition
createCodingTools = create_coding_tools
createReadOnlyTools = create_read_only_tools
createAllTools = create_all_tools
createCodingToolDefinitions = create_coding_tool_definitions
createReadOnlyToolDefinitions = create_read_only_tool_definitions
createAllToolDefinitions = create_all_tool_definitions

__all__ = [
    "create_read_tool", "read_tool",
    "create_write_tool", "write_tool",
    "create_edit_tool", "edit_tool",
    "create_bash_tool", "bash_tool",
    "create_grep_tool", "grep_tool",
    "create_find_tool", "find_tool",
    "create_ls_tool", "ls_tool",
    "TOOL_NAMES", "all_tool_names",
    "create_tool", "create_tool_definition",
    "create_read_tool_definition", "create_bash_tool_definition",
    "create_edit_tool_definition", "create_write_tool_definition",
    "create_grep_tool_definition", "create_find_tool_definition",
    "create_ls_tool_definition",
    "create_coding_tools", "create_read_only_tools", "create_all_tools",
    "create_coding_tool_definitions", "create_read_only_tool_definitions",
    "create_all_tool_definitions",
    "createTool", "createToolDefinition",
    "createReadToolDefinition", "createBashToolDefinition",
    "createEditToolDefinition", "createWriteToolDefinition",
    "createGrepToolDefinition", "createFindToolDefinition",
    "createLsToolDefinition",
    "createCodingTools", "createReadOnlyTools", "createAllTools",
    "createCodingToolDefinitions", "createReadOnlyToolDefinitions",
    "createAllToolDefinitions",
    "DEFAULT_MAX_BYTES", "DEFAULT_MAX_LINES", "GREP_MAX_LINE_LENGTH",
    "TruncationResult", "truncate_head", "truncate_tail", "truncate_line", "format_size",
    "OutputAccumulator", "OutputSnapshot",
    "active_file_mutation_queue_count", "get_mutation_queue_key", "with_file_mutation_queue",
    "get_text_output", "invalid_arg_text", "normalize_display_text", "render_tool_path",
    "replace_tabs", "shorten_path", "str_or_none",
    "create_tool_definition_from_agent_tool", "wrap_tool_definition", "wrap_tool_definitions",
]
