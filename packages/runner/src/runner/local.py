"""Run model comparisons on the local CPU."""

from runner.comparison import compare


def run(model: str, prompt: str) -> dict:
    return compare(model, "cpu", prompt)
