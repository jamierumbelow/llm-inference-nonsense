from typing import cast

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from e01_kv_cache import generate, load_model
from e01_kv_cache.model import DecoderLayer, KVCache, LlamaForCausalLM
from harness.config import LlamaConfig
from harness.profiles import get_profile


def tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rope_scaling=None,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_ids=frozenset({2}),
    )


def test_cached_decode_matches_full_recomputation() -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 5, 8, 13]])
    cache = KVCache.allocate(
        model.config,
        batch_size=1,
        capacity=input_ids.shape[1],
        device=input_ids.device,
        dtype=model.model.embed_tokens.weight.dtype,
    )

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected = model(input_ids)
        prefill_logits = model(input_ids[:, :3], cache=cache)
        decode_logits = model(input_ids[:, 3:], cache=cache)

    torch.testing.assert_close(prefill_logits, expected[:, :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decode_logits[:, -1], expected[:, -1], atol=1e-5, rtol=1e-5)
    assert cache.length == input_ids.shape[1]
    assert cache.storage.is_contiguous()


def test_generation_processes_the_prompt_once_then_one_token_at_a_time() -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config()).eval()
    sequence_lengths = []

    def record_input(_module, args) -> None:
        sequence_lengths.append(args[0].shape[1])

    first_layer = cast(DecoderLayer, model.model.layers[0])
    handle = first_layer.self_attn.q_proj.register_forward_pre_hook(record_input)
    try:
        output = generate(model, torch.tensor([[1, 5, 8]]), max_new_tokens=3)
    finally:
        handle.remove()

    assert output.shape == (1, 6)
    assert sequence_lengths == [3, 1, 1]


def test_cache_rejects_a_sequence_beyond_its_capacity() -> None:
    model = LlamaForCausalLM(tiny_config()).eval()
    cache = KVCache.allocate(
        model.config,
        batch_size=1,
        capacity=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        model(torch.tensor([[1, 5]]), cache=cache)
        try:
            model(torch.tensor([[8]]), cache=cache)
        except ValueError as error:
            assert str(error) == "KV cache capacity 2 exceeded by position 3"
        else:
            raise AssertionError("expected the cache capacity check to fail")


@pytest.mark.slow
def test_cached_checkpoint_logits_match_full_recomputation() -> None:
    from transformers import AutoTokenizer

    profile = get_profile("1b")
    tokenizer = AutoTokenizer.from_pretrained(profile.snapshot_path(), local_files_only=True)
    input_ids = tokenizer("The present King of France is", return_tensors="pt").input_ids
    model = load_model(profile, device="cpu", dtype=torch.float32)
    cache = KVCache.allocate(
        model.config,
        batch_size=1,
        capacity=input_ids.shape[1] + 1,
        device=input_ids.device,
        dtype=torch.float32,
    )

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected_prefill = model(input_ids)
        actual_prefill = model(input_ids, cache=cache)
        next_token = expected_prefill[:, -1].argmax(dim=-1, keepdim=True)
        expected_decode = model(torch.cat((input_ids, next_token), dim=1))[:, -1]
        actual_decode = model(next_token, cache=cache)[:, -1]

    torch.testing.assert_close(actual_prefill, expected_prefill, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_decode, expected_decode, atol=1e-3, rtol=1e-3)
