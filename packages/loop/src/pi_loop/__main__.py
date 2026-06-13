"""CLI entrypoint: drive any pi-py agent on a goal until the judges stop it.

    python -m pi_loop --agent-dir /path/to/agent "Build a parts catalog site"

There is intentionally no --max-iterations flag: the loop runs until
goal_accomplished or churn_detector fires.
"""

from __future__ import annotations

import argparse
import os
import sys

from .loop import run_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pi_loop",
        description="Drive a pi-py agent on a goal until the judge suite stops it (no max loops).",
    )
    parser.add_argument("goal", help="The user-owned goal (the WHAT) to drive to completion.")
    parser.add_argument(
        "--agent-dir",
        default=os.environ.get("PI_LOOP_AGENT_DIR") or os.getcwd(),
        help="Target agent root (dir containing .pi-py/). Defaults to $PI_LOOP_AGENT_DIR or cwd.",
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

    result = run_loop(
        args.goal,
        args.agent_dir,
        project_root=args.project_root,
        emit=emit,
        iteration_timeout=args.iteration_timeout,
    )
    return 0 if (result.terminal and result.terminal.condition == "goal_accomplished") else 1


if __name__ == "__main__":
    raise SystemExit(main())
