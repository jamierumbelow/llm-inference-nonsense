from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RopeScaling:
    """Llama 3 uses rotary position embeddings (RoPE) to extend the context window
    beyond the original 2048 tokens.
    The scaling factors are used to adjust the RoPE frequencies for longer sequences.
    original paper: https://arxiv.org/abs/2104.09864
    intro blog post: https://krasserm.github.io/2022/12/13/rotary-position-embedding/
    empirical work: https://arxiv.org/abs/2310.05209
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScaling | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_ids: frozenset[int]

    @property
    def num_kv_groups(self) -> int:
        """Query heads per key/value head. 1 means plain multi-head attention."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LlamaConfig:
        if d.get("model_type") != "llama":
            raise ValueError(f"expected a llama config, got model_type={d.get('model_type')!r}")

        heads = d["num_attention_heads"]
        kv_heads = d.get("num_key_value_heads", heads)
        if heads % kv_heads:
            raise ValueError(f"num_attention_heads={heads} not divisible by kv heads={kv_heads}")

        scaling = d.get("rope_scaling")
        if scaling is not None:
            if scaling.get("rope_type") != "llama3":
                raise ValueError(f"unsupported rope_type {scaling.get('rope_type')!r}")
            scaling = RopeScaling(
                factor=scaling["factor"],
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                original_max_position_embeddings=scaling["original_max_position_embeddings"],
            )

        eos = d["eos_token_id"]
        eos_ids = frozenset(eos if isinstance(eos, list) else [eos])

        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=d.get("head_dim", d["hidden_size"] // heads),
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=d["rope_theta"],
            rope_scaling=scaling,
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d["bos_token_id"],
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_file(cls, path: Path) -> LlamaConfig:
        with path.open() as f:
            return cls.from_dict(json.load(f))
