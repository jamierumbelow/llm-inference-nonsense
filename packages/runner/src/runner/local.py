"""Run model comparisons on the local CPU."""

from collections.abc import Mapping

from runner.comparison import compare
from runner.generation import generate as generate_on_device


def run(experiment: str, model: str, prompts: Mapping[str, str]) -> dict:
    return compare(experiment, model, "cpu", prompts)


def generate(
    experiment: str,
    model: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    return generate_on_device(experiment, model, "cpu", dtype, prompt, max_new_tokens)
