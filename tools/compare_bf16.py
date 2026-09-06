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
MODEL_PROFILES = {"1b": "dev", "8b": "target"}


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
        choices=tuple(MODEL_PROFILES),
        default="1b",
        help="model size (default: 1b)",
    )
    parser.add_argument(
        "--location",
        choices=("local", "modal"),
        default="local",
        help="local: CPU; modal: A100 40GB GPU (default: local)",
    )
    args = parser.parse_args()
    model_profile = MODEL_PROFILES[args.model]
    if args.location == "modal":
        from modal_compare import run

        report = run(model_profile, PROMPT)
    else:
        report = compare(model_profile, "cpu", PROMPT)
    report["model"] = args.model
    report["location"] = args.location
    path = save_report(report, args.model, args.location)
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
