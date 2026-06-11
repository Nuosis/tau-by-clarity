"""Modes subpackage — mirrors packages/coding-agent/src/modes/ in the TypeScript source."""
from .interactive import InteractiveMode, runInteractiveMode, run_interactive_mode
from .print_mode import PrintModeOptions, run_print_mode
from .rpc import (
    RpcClient,
    RpcClientOptions,
    RpcCommand,
    RpcEventListener,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcSessionState,
    run_rpc_mode,
)

runPrintMode = run_print_mode
runRpcMode = run_rpc_mode

__all__ = [
    "InteractiveMode",
    "PrintModeOptions",
    "RpcClient",
    "RpcClientOptions",
    "RpcCommand",
    "RpcEventListener",
    "RpcExtensionUIRequest",
    "RpcExtensionUIResponse",
    "RpcResponse",
    "RpcSessionState",
    "runInteractiveMode",
    "runPrintMode",
    "runRpcMode",
    "run_interactive_mode",
    "run_print_mode",
    "run_rpc_mode",
]
