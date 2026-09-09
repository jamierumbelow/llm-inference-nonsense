"""Shared token-generation algorithms."""

from collections.abc import Callable, Collection

import torch
from torch import Tensor

type Forward = Callable[[Tensor], Tensor]


def greedy_generate(
    forward: Forward,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_ids: Collection[int] = (),
    on_token: Callable[[], None] | None = None,
) -> Tensor:
    """Generate tokens by repeatedly choosing the highest next-token logit."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("greedy generation currently requires input_ids shaped [1, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("greedy generation requires at least one input token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    tokens = input_ids
    stop_tokens = set(eos_token_ids)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = forward(tokens)
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
                raise ValueError("forward must return logits shaped [1, sequence, vocabulary]")
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if on_token is not None:
                on_token()
            if stop_tokens and int(next_token.item()) in stop_tokens:
                break

    return tokens
