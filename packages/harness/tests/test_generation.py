import torch

from harness.generation import greedy_generate


def test_greedy_generation_feeds_each_token_back_into_the_model() -> None:
    inputs = []
    next_tokens = iter((4, 5, 6))

    def forward(token_ids: torch.Tensor) -> torch.Tensor:
        inputs.append(token_ids.tolist())
        logits = torch.zeros(1, token_ids.shape[1], 8)
        logits[0, -1, next(next_tokens)] = 1
        return logits

    output = greedy_generate(forward, torch.tensor([[1, 2, 3]]), max_new_tokens=3)

    assert output.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert inputs == [
        [[1, 2, 3]],
        [[1, 2, 3, 4]],
        [[1, 2, 3, 4, 5]],
    ]


def test_greedy_generation_stops_after_eos() -> None:
    next_tokens = iter((4, 5, 6))

    def forward(token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(1, token_ids.shape[1], 8)
        logits[0, -1, next(next_tokens)] = 1
        return logits

    output = greedy_generate(
        forward,
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=3,
        eos_token_ids={5},
    )

    assert output.tolist() == [[1, 2, 3, 4, 5]]
