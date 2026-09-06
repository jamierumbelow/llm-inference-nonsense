"""The initial, deliberately simple Llama implementation."""

import torch

from e00_baseline.model import LlamaForCausalLM
from harness.loader import load_model as load_checkpoint
from harness.profiles import ModelProfile


def load_model(
    profile: str | ModelProfile,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaForCausalLM:
    """Build the baseline model and load a Hugging Face checkpoint into it."""
    return load_checkpoint(
        LlamaForCausalLM,
        profile,
        device,
        dtype,
        tied_weights=(("lm_head.weight", "model.embed_tokens.weight"),),
    )
