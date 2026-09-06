"""Choose an experiment and location, then compare and save its logits."""

import argparse
import json
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

from harness.profiles import PROFILES
from runner.experiments import EXPERIMENTS
from runner.prompts import DEFAULT_PROMPT, PROMPT_SUITES


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


def output_paths(experiment: str, model: str, location: str) -> tuple[Path, Path]:
    now = datetime.now().astimezone()
    directory = (
        Path(__file__).resolve().parents[1] / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%H_%M_%S_%f}_{experiment}_{model}_{location}"
    return directory / f"{stem}.json", directory / f"{stem}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fp32 and bf16 model outputs.")
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        default="e00_baseline",
        help="experiment model definition (default: e00_baseline)",
    )
    parser.add_argument(
        "--model",
        choices=tuple(PROFILES),
        default="1b",
        help="model size (default: 1b)",
    )
    parser.add_argument(
        "--location",
        choices=("local", "modal"),
        default="local",
        help="local: CPU; modal: A100 40GB GPU (default: local)",
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt",
        help=f"run one custom prompt (default: {DEFAULT_PROMPT!r})",
    )
    prompts.add_argument(
        "--suite",
        choices=tuple(PROMPT_SUITES),
        help="run a fixed prompt suite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = (
        dict(PROMPT_SUITES[args.suite])
        if args.suite
        else {"prompt": args.prompt if args.prompt is not None else DEFAULT_PROMPT}
    )
    report_path, log_path = output_paths(args.experiment, args.model, args.location)

    with log_path.open("x") as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            command = ["uv", "run", "tools/compare_logits.py", *sys.argv[1:]]
            print(f"$ {shlex.join(command)}")
            try:
                if args.location == "local":
                    from runner.local import run
                else:
                    from runner.modal import run

                report = run(args.experiment, args.model, prompts)
                report["location"] = args.location
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
