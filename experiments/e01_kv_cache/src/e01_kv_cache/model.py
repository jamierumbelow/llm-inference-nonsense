"""Experiment 01: the baseline Llama 3 implementation with a KV cache.

The prompt is processed once. Each later call projects only the newest token,
then attends over the keys and values retained from every earlier position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from harness.config import LlamaConfig, RopeScaling


def rope_inv_freq(head_dim: int, theta: float, scaling: RopeScaling | None) -> Tensor:
    """Per-pair rotation frequencies for rotary position embeddings.

    Base RoPE: pair j of the head rotates at theta ** (-2j / hd). Llama 3 then
    rescales these for long context: pairs with a wavelength longer than
    original_ctx / low_freq_factor are slowed down by `factor`, pairs shorter
    than original_ctx / high_freq_factor are left alone, and the band in between
    is linearly blended. Returns shape [hd / 2] in float32.
    """
    # Keep this buffer real when the loader constructs parameters on the meta device.
    j = torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu")
    inv_freq = 1.0 / (theta ** (j / head_dim))
    if scaling is None:
        return inv_freq

    original_ctx = scaling.original_max_position_embeddings
    low_wavelen = original_ctx / scaling.low_freq_factor
    high_wavelen = original_ctx / scaling.high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_wavelen, inv_freq / scaling.factor, inv_freq)
    blend = (original_ctx / wavelen - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    blended = (1 - blend) * scaled / scaling.factor + blend * scaled
    in_band = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
    return torch.where(in_band, blended, scaled)


def rope_cos_sin(inv_freq: Tensor, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """cos and sin tables for the given positions, each of shape [T, hd].

    The hd/2 frequencies are duplicated along the last axis so they line up with
    the rotate_half convention below (first half of the head paired with the second
    half), which is the layout the HF checkpoint's q/k weights are permuted for.
    """
    angles = positions.float()[:, None] * inv_freq[None, :]  # [T, hd/2]
    angles = torch.cat([angles, angles], dim=-1)  # [T, hd]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each head's (first half, second half) pairs by position-dependent angles.

    x: [B, heads, T, hd]. cos, sin: [T, hd], broadcast over batch and heads.
    """
    return x * cos + rotate_half(x) * sin


@dataclass(slots=True)
class KVCache:
    """Preallocated keys and values for one generation request."""

    storage: Tensor
    """[layers, 2, B, KV, capacity, hd]"""
    length: int = 0

    @classmethod
    def allocate(
        cls,
        cfg: LlamaConfig,
        batch_size: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        if batch_size < 1:
            raise ValueError("KV cache batch size must be positive")
        if capacity < 1:
            raise ValueError("KV cache capacity must be positive")
        if capacity > cfg.max_position_embeddings:
            raise ValueError(
                f"KV cache capacity {capacity} exceeds the model limit "
                f"of {cfg.max_position_embeddings}"
            )
        return cls(
            torch.empty(
                cfg.num_hidden_layers,
                2,
                batch_size,
                cfg.num_key_value_heads,
                capacity,
                cfg.head_dim,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def batch_size(self) -> int:
        return self.storage.shape[2]

    @property
    def capacity(self) -> int:
        return self.storage.shape[4]

    def write(
        self,
        layer_index: int,
        keys: Tensor,
        values: Tensor,
    ) -> int:
        end = self.length + keys.shape[2]
        if end > self.capacity:
            raise ValueError(f"KV cache capacity {self.capacity} exceeded by position {end}")
        self.storage[layer_index, 0, :, :, self.length : end].copy_(keys)
        self.storage[layer_index, 1, :, :, self.length : end].copy_(values)
        return end

    def prefix(self, layer_index: int, end: int) -> tuple[Tensor, Tensor]:
        return (
            self.storage[layer_index, 0, :, :, :end],
            self.storage[layer_index, 1, :, :, :end],
        )


class Attention(nn.Module):
    """Grouped-query attention: H query heads share KV heads in groups of H / KV."""

    def __init__(self, cfg: LlamaConfig, layer_index: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_index = layer_index
        d, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(d, cfg.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * hd, d, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        cfg = self.cfg

        # Project and split into heads: [B, T, heads * hd] -> [B, heads, T, hd].
        q = self.q_proj(x).view(bsz, seqlen, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            end = cache.write(self.layer_index, k, v)
            k, v = cache.prefix(self.layer_index, end)

        # Each KV head serves num_kv_groups consecutive query heads, so repeat
        # along the head axis: [B, KV, T, hd] -> [B, H, T, hd].
        k = k.repeat_interleave(cfg.num_kv_groups, dim=1)
        v = v.repeat_interleave(cfg.num_kv_groups, dim=1)

        # softmax(q k^T / sqrt(hd)) v with a causal mask. [B, H, T, hd]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seqlen > 1)

        # Merge heads back: [B, H, T, hd] -> [B, T, H * hd].
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x)). D -> F -> D."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        d, f = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: LlamaConfig, layer_index: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_index)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    """Embeddings, the decoder stack, and the final norm. Returns hidden states, not logits."""

    inv_freq: Tensor

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, layer_index) for layer_index in range(cfg.num_hidden_layers)
        )
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.register_buffer(
            "inv_freq",
            rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
    ) -> Tensor:
        x = self.embed_tokens(input_ids)  # [B, T, D]
        cos, sin = rope_cos_sin(self.inv_freq, positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, cache)
        if cache is not None:
            cache.length += input_ids.shape[1]
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    """The full model: hidden states projected to next-token logits over the vocabulary."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:  # the 1B shares its output projection with the embedding
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """input_ids: [B, T] token ids. Returns logits [B, T, V] in the model dtype."""
        if cache is not None:
            if input_ids.shape[0] != cache.batch_size:
                raise ValueError("input batch size does not match the KV cache")
            if cache.length and input_ids.shape[1] != 1:
                raise ValueError("cached decode accepts exactly one new token per call")
        start = cache.length if cache is not None else 0
        if positions is None:
            positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
        hidden = self.model(input_ids, positions, cache)
        return self.lm_head(hidden)
