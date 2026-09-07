"""Run the fixed benchmark suite for one experiment."""

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

from runner.benchmark_workloads import STANDARD_BENCHMARK
from runner.experiments import EXPERIMENTS
from runner.reporting import output_paths, run_and_save

ROOT = Path(__file__).resolve().parents[1]


def experiment_name(value: str) -> str:
    if value in EXPERIMENTS:
        return value
    message = f"unknown experiment {value!r}"
    if suggestion := get_close_matches(value, EXPERIMENTS, n=1):
        message += f"; did you mean {suggestion[0]!r}?"
    raise argparse.ArgumentTypeError(message)


def git_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_commit": commit, "git_dirty_before_run": bool(status.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard inference benchmark.")
    parser.add_argument(
        "--experiment",
        type=experiment_name,
        default="e00_baseline",
        metavar=f"{{{','.join(EXPERIMENTS)}}}",
        help="experiment to benchmark (default: e00_baseline)",
    )
    args = parser.parse_args()
    started_at = datetime.now().astimezone()
    source = git_metadata()
    report_path, log_path = output_paths(
        f"{args.experiment}_benchmark_8b_modal",
        started_at,
    )
    command = shlex.join(["uv", "run", "tools/benchmark.py", *sys.argv[1:]])

    def run() -> dict:
        print(
            f"Benchmarking {args.experiment}: {STANDARD_BENCHMARK.model} "
            f"{STANDARD_BENCHMARK.dtype} on {STANDARD_BENCHMARK.gpu}"
        )
        from runner.modal import benchmark

        return {
            "started_at": started_at.isoformat(),
            "command": command,
            **source,
            **benchmark(args.experiment),
        }

    # Modal's live renderer emits every animation frame when output is teed.
    run_and_save(command, report_path, log_path, run, interactive=False)


if __name__ == "__main__":
    main()
