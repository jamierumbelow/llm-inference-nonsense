"""Build a model and fill it from a cached checkpoint."""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from harness.config import LlamaConfig
from harness.model import LlamaForCausalLM, rope_inv_freq
from harness.profiles import ModelProfile, get_profile


def load_model(
    profile: str | ModelProfile,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaForCausalLM:
    """Load a profile's weights into our model on the given device and dtype.

    The model is built on the meta device so no memory is spent on random
    initialisation; parameters are assigned straight from the checkpoint.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    path = profile.snapshot_path()
    cfg = LlamaConfig.from_file(path / "config.json")

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    # Checkpoints may be sharded across several files; the union is the full state dict.
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        state.update(load_file(shard, device="cpu"))
    state = {k: v.to(dtype) for k, v in state.items()}

    if cfg.tie_word_embeddings:
        # The checkpoint omits lm_head; strict loading still wants the key present.
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight  # assign=True untied them

    # Non-persistent buffers are not in the checkpoint and were built on the meta
    # device, so they hold no data. Recompute them for real.
    model.model.inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)

    return model.to(device).eval()
