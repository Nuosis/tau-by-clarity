"""CLI entrypoint: drive any tau agent on a goal until the judges stop it.

    python -m tau_loop --agent-dir /path/to/agent --goal "Build a parts catalog site"

There is intentionally no --max-iterations flag: the loop runs until
goal_accomplished or churn_detector fires.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

from .loop import run_loop
from .steering import SteeringInbox


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tau_loop",
        description="Drive a tau agent on a goal until the judge suite stops it (no max loops).",
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="The explicit user-owned goal (the WHAT) to drive to completion.",
    )
    parser.add_argument(
        "--agent-dir",
        default=os.environ.get("TAU_LOOP_AGENT_DIR") or os.getcwd(),
        help="Target agent root (dir containing .tau/). Defaults to $TAU_LOOP_AGENT_DIR or cwd.",
    )
    parser.add_argument(
        "--project-root", default=None, help="Project the work targets. Defaults to the agent dir."
    )
    parser.add_argument(
        "--iteration-timeout",
        type=int,
        default=None,
        help="Per-iteration timeout in seconds for an agent turn (default: none).",
    )
    args = parser.parse_args(argv)

    def emit(text: str) -> None:
        sys.stdout.write(text.rstrip() + "\n\n")
        sys.stdout.flush()

    # Steering: read lines the user types while the loop runs, stage them, and
    # acknowledge immediately so a typed message is never silently consumed.
    inbox = SteeringInbox()

    def _reader() -> None:
        try:
            for line in sys.stdin:
                ack = inbox.submit(line)
                if ack:
                    sys.stderr.write(ack + "\n")
                    sys.stderr.flush()
        except Exception:
            pass

    if sys.stdin and not sys.stdin.closed:
        threading.Thread(target=_reader, daemon=True).start()

    try:
        from pi_coding_agent.core.runtime_registry import register_process

        register_process(
            kind="tau_loop",
            session_id=f"tau-loop-{os.getpid()}",
            cwd=os.getcwd(),
            agent_dir=args.agent_dir,
            goal=args.goal,
        )
    except Exception:
        pass

    result = run_loop(
        args.goal,
        args.agent_dir,
        project_root=args.project_root,
        emit=emit,
        iteration_timeout=args.iteration_timeout,
        inbox=inbox,
    )
    return 0 if (result.terminal and result.terminal.condition == "goal_accomplished") else 1


if __name__ == "__main__":
    raise SystemExit(main())
