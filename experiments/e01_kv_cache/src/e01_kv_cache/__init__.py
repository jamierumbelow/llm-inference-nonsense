"""Baseline with a KV cache"""

from collections.abc import Callable, Collection

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from e01_kv_cache.model import KVCache, LlamaForCausalLM
from harness.generation import greedy_generate
from harness.loader import load_model as load_checkpoint
from harness.profiles import ModelProfile

ATTENTION_BACKEND = "SDPA math"
USES_CACHE = True
FULL_RECOMPUTATION = False
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
    cache: KVCache | None = None

    def cached_forward(tokens: Tensor) -> Tensor:
        nonlocal cache
        if cache is None:
            cache = KVCache.allocate(
                model.config,
                batch_size=input_ids.shape[0],
                capacity=input_ids.shape[1] + max_new_tokens,
                device=input_ids.device,
                dtype=model.model.embed_tokens.weight.dtype,
            )
        current_input = tokens if cache.length == 0 else tokens[:, -1:]
        return model(current_input, cache=cache)

    with sdpa_kernel(SDPBackend.MATH):
        return greedy_generate(
            cached_forward,
            input_ids,
            max_new_tokens,
            eos_token_ids,
            on_token,
        )
