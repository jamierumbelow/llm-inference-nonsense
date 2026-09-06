"""Fill an experiment's model definition from a cached checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from safetensors import safe_open
from torch import nn

from harness.config import LlamaConfig
from harness.profiles import ModelProfile, get_profile


def load_model[ModelT: nn.Module](
    build_model: Callable[[LlamaConfig], ModelT],
    profile: str | ModelProfile,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    tied_weights: Sequence[tuple[str, str]] = (),
) -> ModelT:
    """Load a profile's weights into an experiment model.

    The model is built on the meta device so no memory is spent on random
    initialisation. ``tied_weights`` maps checkpoint keys that may be omitted
    to the keys whose parameters they share.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    path = profile.snapshot_path()
    cfg = LlamaConfig.from_file(path / "config.json")

    with torch.device("meta"):
        model = build_model(cfg)

    # transfer one tensor at a time, avoiding a full fp32 checkpoint copy in CPU RAM.
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():  # noqa: SIM118 -- safe_open is not a dict/iterable
                state[key] = checkpoint.get_tensor(key).to(device=device, dtype=dtype)

    aliases = []
    for target, source in tied_weights:
        if target not in state:
            state[target] = state[source]
            aliases.append((target, source))

    model.load_state_dict(state, strict=True, assign=True)
    for target, source in aliases:
        _set_parameter(model, target, _get_parameter(model, source))

    return model.to(device).eval()


def _get_parameter(model: nn.Module, path: str) -> nn.Parameter:
    value: object = model
    for part in path.split("."):
        value = getattr(value, part)
    if not isinstance(value, nn.Parameter):
        raise TypeError(f"{path!r} is not a parameter")
    return value


def _set_parameter(model: nn.Module, path: str, parameter: nn.Parameter) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent: object = model
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    if not isinstance(parent, nn.Module):
        raise TypeError(f"parent of {path!r} is not a module")
    setattr(parent, name, parameter)
