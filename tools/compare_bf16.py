"""Choose where to compare models, then print and save the report."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from harness.comparison import compare

SHORT_PROMPT = "The present King of France is"
LONG_PROMPT = (
    "The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection "
    "of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text "
    "out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for "
    "multilingual dialogue use cases and outperform many of the available open source"
)

PROMPT = LONG_PROMPT


def save_report(report: dict, profile: str) -> Path:
    now = datetime.now().astimezone()
    directory = (
        Path(__file__).resolve().parents[1] / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%H_%M_%S_%f}_{profile}.json"
    with path.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fp32 and bf16 model outputs.")
    parser.add_argument(
        "--profile",
        choices=("local", "modal"),
        default="local",
        help="local: 1B on CPU; modal: 8B on an A100 40GB (default local)",
    )
    args = parser.parse_args()
    if args.profile == "modal":
        from modal_compare import run

        report = run(PROMPT)
    else:
        report = compare("dev", "cpu", PROMPT)
    path = save_report(report, args.profile)
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
