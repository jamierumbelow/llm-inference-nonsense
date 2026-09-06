import pytest

from harness.config import LlamaConfig
from harness.profiles import get_profile

MINIMAL = {
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def test_defaults_when_optional_fields_absent() -> None:
    cfg = LlamaConfig.from_dict(MINIMAL)
    assert cfg.num_key_value_heads == 4
    assert cfg.num_kv_groups == 1
    assert cfg.head_dim == 16
    assert cfg.rope_scaling is None
    assert cfg.tie_word_embeddings is False
    assert cfg.eos_token_ids == frozenset({2})


def test_rejects_non_llama() -> None:
    with pytest.raises(ValueError, match="model_type"):
        LlamaConfig.from_dict({**MINIMAL, "model_type": "mistral"})


def test_rejects_unknown_rope_type() -> None:
    bad = {**MINIMAL, "rope_scaling": {"rope_type": "yarn", "factor": 2.0}}
    with pytest.raises(ValueError, match="rope_type"):
        LlamaConfig.from_dict(bad)


@pytest.mark.slow
def test_1b_checkpoint_config() -> None:
    cfg = LlamaConfig.from_file(get_profile("1b").snapshot_path() / "config.json")
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.num_kv_groups == 4
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 8192
    assert cfg.vocab_size == 128256
    assert cfg.tie_word_embeddings is True
    assert cfg.rope_theta == 500000.0
    assert cfg.rope_scaling is not None
    assert cfg.rope_scaling.factor == 32.0
    assert cfg.rope_scaling.original_max_position_embeddings == 8192
    assert cfg.eos_token_ids == frozenset({128001, 128008, 128009})
