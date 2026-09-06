"""Compare our model with Transformers in fp32 and bf16.

Read compare() for the sweep, run_model() for one forward pass, and
compare_outputs() for the logit differences. The runner owns the Transformers
dependency; the model and loader do not depend on it.
"""

import gc
from collections.abc import Sequence
from itertools import combinations, product
from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.loader import load_model
from harness.profiles import ModelProfile, get_profile

type Implementation = Literal["transformers", "custom"]
type Precision = Literal["fp32", "bf16"]

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def compare(
    model_profile: str,
    device: str,
    prompt: str,
    *,
    implementations: Sequence[Implementation] = ("transformers", "custom"),
    precisions: Sequence[Precision] = ("fp32", "bf16"),
) -> dict:
    """Run every chosen implementation at every chosen precision, then compare pairs."""
    if not implementations or not precisions:
        raise ValueError("choose at least one implementation and precision")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the modal profile")
        torch.set_float32_matmul_precision("highest")

    profile = get_profile(model_profile)
    path = profile.snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)
    outputs = {}
    runs = {}

    # Only one model is alive at a time; saved logits stay on CPU.
    for implementation, precision in product(implementations, precisions):
        name = f"{implementation}_{precision}"
        print(f"Running {implementation} {precision}...", flush=True)
        outputs[name], runs[name] = run_model(
            implementation, profile, device, DTYPES[precision], ids, tokenizer
        )

    return {
        "prompt": prompt,
        "token_ids": ids.cpu().tolist(),
        "model_profile": model_profile,
        "checkpoint": str(path),
        "torch_version": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "attention_backend": "SDPA math",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "use_cache": False,
        "runs": runs,
        "comparisons": compare_outputs(outputs, tokenizer),
    }


def run_model(
    implementation: Implementation,
    profile: ModelProfile,
    device: str,
    dtype: torch.dtype,
    ids: Tensor,
    tokenizer,
) -> tuple[Tensor, dict]:
    """Load one model, run the prompt, summarise its logits, then release it."""
    if implementation == "transformers":
        model = (
            cast(
                torch.nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    profile.snapshot_path(),
                    dtype=dtype,
                    attn_implementation="sdpa",
                    local_files_only=True,
                ),
            )
            .to(device=device)
            .eval()
        )
    else:
        model = load_model(profile, device=device, dtype=dtype)

    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            if implementation == "transformers":
                logits = model(ids, use_cache=False).logits
            else:
                logits = model(ids)
        logits = logits.cpu()
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    values, indices = logits[0, -1].float().topk(5)
    summary = {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }
    return logits.float(), summary


def compare_outputs(outputs: dict, tokenizer) -> dict:
    comparisons = {}
    for reference, candidate in combinations(outputs, 2):
        expected, actual = outputs[reference], outputs[candidate]
        error = (actual - expected).abs()
        disagreements = []
        for position in range(expected.shape[1]):
            ref_token = int(expected[0, position].argmax())
            actual_token = int(actual[0, position].argmax())
            if ref_token != actual_token:
                top = expected[0, position].topk(2).values
                disagreements.append(
                    {
                        "position": position,
                        "reference_token": tokenizer.decode([ref_token]),
                        "candidate_token": tokenizer.decode([actual_token]),
                        "reference_top_2_gap": float(top[0] - top[1]),
                    }
                )
        comparisons[f"{candidate}_vs_{reference}"] = {
            "mean_absolute_error": float(error.mean()),
            "max_absolute_error": float(error.max()),
            "last_position_mean_absolute_error": float(error[0, -1].mean()),
            "last_position_max_absolute_error": float(error[0, -1].max()),
            "argmax_agreement": f"{expected.shape[1] - len(disagreements)}/{expected.shape[1]}",
            "disagreements": disagreements,
        }
    return comparisons
