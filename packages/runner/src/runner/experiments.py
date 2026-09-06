"""Discover the model definition exported by an experiment package."""

from importlib import import_module
from typing import Protocol, cast

import torch
from torch import nn

from harness.profiles import ModelProfile

EXPERIMENTS = {"e00_baseline": "e00_baseline"}


class Experiment(Protocol):
    def load_model(
        self,
        profile: str | ModelProfile,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> nn.Module: ...


def get_experiment(name: str) -> Experiment:
    try:
        module_name = EXPERIMENTS[name]
    except KeyError:
        known = ", ".join(EXPERIMENTS)
        raise KeyError(f"unknown experiment {name!r}; known experiments: {known}") from None
    return cast(Experiment, import_module(module_name))
