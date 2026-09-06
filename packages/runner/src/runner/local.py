"""Run model comparisons on the local CPU."""

from collections.abc import Mapping

from runner.comparison import compare


def run(experiment: str, model: str, prompts: Mapping[str, str]) -> dict:
    return compare(experiment, model, "cpu", prompts)
