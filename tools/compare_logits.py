"""Choose where to compare models, then print and save the report."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from harness.profiles import PROFILES

DEFAULT_PROMPT = "The present King of France is"


def save_report(report: dict, model: str, location: str) -> Path:
    now = datetime.now().astimezone()
    directory = (
        Path(__file__).resolve().parents[1] / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%H_%M_%S_%f}_{model}_{location}.json"
    with path.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fp32 and bf16 model outputs.")
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
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="text to run through each model variant",
    )
    args = parser.parse_args()
    if args.location == "local":
        from runner.local import run
    else:
        from runner.modal import run

    report = run(args.model, args.prompt)
    report["model"] = args.model
    report["location"] = args.location
    path = save_report(report, args.model, args.location)
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
