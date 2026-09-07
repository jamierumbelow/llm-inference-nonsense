"""Shared JSON and CLI-log reporting for runner tools."""

import json
import sys
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[4]


class Tee:
    """Write CLI output to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log: TextIO, *, interactive: bool) -> None:
        self.terminal = terminal
        self.log = log
        self.interactive = interactive

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.interactive and self.terminal.isatty()


def output_paths(stem: str, now: datetime | None = None) -> tuple[Path, Path]:
    now = now or datetime.now().astimezone()
    directory = ROOT / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    directory.mkdir(parents=True, exist_ok=True)
    timestamped_stem = f"{now:%H_%M_%S_%f}_{stem}"
    return directory / f"{timestamped_stem}.json", directory / f"{timestamped_stem}.log"


def run_and_save(
    command: str,
    report_path: Path,
    log_path: Path,
    operation: Callable[[], dict],
    *,
    interactive: bool,
) -> None:
    """Run an operation while saving its report and complete CLI transcript."""
    with log_path.open("x") as log:
        stdout = Tee(sys.stdout, log, interactive=interactive)
        stderr = Tee(sys.stderr, log, interactive=interactive)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"$ {command}")
            try:
                report = operation()
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
