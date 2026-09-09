"""The baseline Llama implementation, ready for the KV-cache experiment."""

from collections.abc import Callable, Collection

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from e01_kv_cache.model import LlamaForCausalLM
from harness.generation import greedy_generate
from harness.loader import load_model as load_checkpoint
from harness.profiles import ModelProfile

ATTENTION_BACKEND = "SDPA math"
USES_CACHE = False
FULL_RECOMPUTATION = True
GENERATION_ALGORITHM = "greedy"


def load_model(
    profile: str | ModelProfile,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaForCausalLM:
    return load_checkpoint(
        LlamaForCausalLM,
        profile,
        device,
        dtype,
        tied_weights=(("lm_head.weight", "model.embed_tokens.weight"),),
    )


def prefill(model: LlamaForCausalLM, input_ids: Tensor) -> Tensor:
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        return model(input_ids)


def generate(
    model: LlamaForCausalLM,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_ids: Collection[int] = (),
    on_token: Callable[[], None] | None = None,
) -> Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return greedy_generate(
            model,
            input_ids,
            max_new_tokens,
            eos_token_ids,
            on_token,
        )
