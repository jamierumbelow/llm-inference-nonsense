import gc

import pytest
import torch

from e00_baseline import load_model
from harness.profiles import get_profile

PROMPT = "The present King of France is"


@pytest.fixture(scope="module")
def prompt_ids() -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(get_profile("1b").snapshot_path())
    return tok(PROMPT, return_tensors="pt").input_ids


@pytest.mark.slow
def test_logits_match_transformers(prompt_ids: torch.Tensor) -> None:
    """Our forward pass must reproduce the reference logits in float32 on CPU.

    fp32 makes the tolerance meaningful: a wrong RoPE table or head grouping
    shows up as a large discrepancy, not something hiding inside fp16 noise.
    The two models are loaded one after the other to stay within laptop memory.
    """
    from transformers import AutoModelForCausalLM

    path = get_profile("1b").snapshot_path()
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        expected = ref(prompt_ids).logits
    del ref
    gc.collect()

    model = load_model("1b", device="cpu", dtype=torch.float32)
    with torch.no_grad():
        actual = model(prompt_ids)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert actual[0, -1].argmax() == expected[0, -1].argmax()


@pytest.mark.slow
def test_tied_head_shares_embedding() -> None:
    model = load_model("1b", device="cpu", dtype=torch.float32)
    assert model.lm_head.weight is model.model.embed_tokens.weight
