"""Discover the model definition exported by an experiment package."""

from collections.abc import Callable, Collection
from importlib import import_module
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from harness.profiles import ModelProfile

EXPERIMENTS = {
    "e00_baseline": "e00_baseline",
    "e01_kv_cache": "e01_kv_cache",
}


class Experiment(Protocol):
    ATTENTION_BACKEND: str
    USES_CACHE: bool
    FULL_RECOMPUTATION: bool
    GENERATION_ALGORITHM: str

    def load_model(
        self,
        profile: str | ModelProfile,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> nn.Module: ...

    def prefill(self, model: nn.Module, input_ids: Tensor) -> Tensor: ...

    def generate(
        self,
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_ids: Collection[int] = (),
        on_token: Callable[[], None] | None = None,
    ) -> Tensor: ...


def get_experiment(name: str) -> Experiment:
    try:
        module_name = EXPERIMENTS[name]
    except KeyError:
        known = ", ".join(EXPERIMENTS)
        raise KeyError(f"unknown experiment {name!r}; known experiments: {known}") from None
    return cast(Experiment, import_module(module_name))
