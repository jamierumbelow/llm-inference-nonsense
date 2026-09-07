"""Generate one continuation with an experiment model."""

import argparse
import shlex
from difflib import get_close_matches

from harness.profiles import PROFILES
from runner.experiments import EXPERIMENTS
from runner.prompts import DEFAULT_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text greedily with one model.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--experiment",
        metavar=f"{{{','.join(EXPERIMENTS)}}}",
        default="e00_baseline",
        help="experiment model definition (default: e00_baseline)",
    )
    parser.add_argument(
        "--model",
        metavar=f"{{{','.join(PROFILES)}}}",
        default="1b",
        help="model size (default: 1b)",
    )
    parser.add_argument(
        "--location",
        metavar="{local,modal}",
        default="local",
        help="local: CPU; modal: A100 40GB GPU (default: local)",
    )
    parser.add_argument(
        "--dtype",
        metavar="{fp32,bf16}",
        default="bf16",
        help="model precision (default: bf16)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args, unknown = parser.parse_known_args()

    problems = []
    for option, value, choices in (
        ("--experiment", args.experiment, tuple(EXPERIMENTS)),
        ("--model", args.model, tuple(PROFILES)),
        ("--location", args.location, ("local", "modal")),
        ("--dtype", args.dtype, ("fp32", "bf16")),
    ):
        if value not in choices:
            problem = f"argument {option}: invalid value {value!r}; expected {', '.join(choices)}"
            if suggestion := get_close_matches(value, choices, n=1):
                problem += f"; did you mean {suggestion[0]!r}?"
            problems.append(problem)

    if unknown:
        problem = f"unrecognized arguments: {shlex.join(unknown)}"
        # argparse exposes its registered options through the parser actions.
        known_options = [option for action in parser._actions for option in action.option_strings]
        suggestions = {
            suggestion[0]
            for token in unknown
            if token.startswith("-")
            if (suggestion := get_close_matches(token, known_options, n=1))
        }
        if suggestions:
            problem += f"; did you mean {', '.join(sorted(suggestions))}?"
        problems.append(problem)

    if problems:
        parser.error("\n  ".join(problems))
    return args


def main() -> None:
    args = parse_args()

    if args.location == "local":
        from runner.local import generate
    else:
        from runner.modal import generate

    result = generate(
        args.experiment,
        args.model,
        args.dtype,
        args.prompt,
        args.max_new_tokens,
    )
    print(result["text"])


if __name__ == "__main__":
    main()
