"""Run the fixed benchmark suite for one experiment."""

import argparse
import json
import shlex
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import TextIO

from runner.benchmark_workloads import STANDARD_BENCHMARK
from runner.experiments import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    """Write CLI output to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def experiment_name(value: str) -> str:
    if value in EXPERIMENTS:
        return value
    message = f"unknown experiment {value!r}"
    if suggestion := get_close_matches(value, EXPERIMENTS, n=1):
        message += f"; did you mean {suggestion[0]!r}?"
    raise argparse.ArgumentTypeError(message)


def output_paths(experiment: str, now: datetime) -> tuple[Path, Path]:
    directory = ROOT / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%H_%M_%S_%f}_{experiment}_benchmark_8b_modal"
    return directory / f"{stem}.json", directory / f"{stem}.log"


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
    report_path, log_path = output_paths(args.experiment, started_at)
    command = ["uv", "run", "tools/benchmark.py", *sys.argv[1:]]

    with log_path.open("x") as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"$ {shlex.join(command)}")
            print(
                f"Benchmarking {args.experiment}: {STANDARD_BENCHMARK.model} "
                f"{STANDARD_BENCHMARK.dtype} on {STANDARD_BENCHMARK.gpu}"
            )
            try:
                from runner.modal import benchmark

                report = {
                    "started_at": started_at.isoformat(),
                    "command": shlex.join(command),
                    **source,
                    **benchmark(args.experiment),
                }
                with report_path.open("x") as output:
                    json.dump(report, output, indent=2)
                    output.write("\n")
                print(json.dumps(report, indent=2))
                print(f"Saved report: {report_path}")
                print(f"Saved CLI output: {log_path}")
            except Exception:
                traceback.print_exc()
                print(f"Saved failed CLI output: {log_path}")
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
