"""Choose an experiment and location, then compare and save its logits."""

import argparse
import shlex
import sys

from harness.profiles import PROFILES
from runner.experiments import EXPERIMENTS
from runner.prompts import DEFAULT_PROMPT, PROMPT_SUITES
from runner.reporting import output_paths, run_and_save


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
    report_path, log_path = output_paths(f"{args.experiment}_{args.model}_{args.location}")
    command = shlex.join(["uv", "run", "tools/compare_logits.py", *sys.argv[1:]])

    def run_comparison() -> dict:
        if args.location == "local":
            from runner.local import run
        else:
            from runner.modal import run

        report = run(args.experiment, args.model, prompts)
        report["location"] = args.location
        return report

    run_and_save(command, report_path, log_path, run_comparison, interactive=True)


if __name__ == "__main__":
    main()
