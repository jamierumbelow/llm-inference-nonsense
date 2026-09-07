"""Compare an experiment's model with Transformers in fp32 and bf16."""

import gc
from collections.abc import Mapping, Sequence
from itertools import combinations, product
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.profiles import ModelProfile, get_profile
from runner.config import DTYPES
from runner.experiments import Experiment, get_experiment

type Implementation = Literal["transformers", "custom"]
type Precision = Literal["fp32", "bf16"]


def compare(
    experiment_name: str,
    model_name: str,
    device: str,
    prompts: Mapping[str, str],
    *,
    implementations: Sequence[Implementation] = ("transformers", "custom"),
    precisions: Sequence[Precision] = ("fp32", "bf16"),
) -> dict:
    """Run each model variant over the same named prompts, then compare its logits."""
    if not prompts:
        raise ValueError("choose at least one prompt")
    if not implementations or not precisions:
        raise ValueError("choose at least one implementation and precision")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Modal runner")
        torch.set_float32_matmul_precision("highest")

    experiment = get_experiment(experiment_name)
    profile = get_profile(model_name)
    path = profile.snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    token_ids = {
        name: tokenizer(prompt, return_tensors="pt").input_ids for name, prompt in prompts.items()
    }
    outputs: dict[str, dict[str, Tensor]] = {name: {} for name in prompts}
    runs: dict[str, dict[str, dict]] = {name: {} for name in prompts}

    # Load each variant once, run every prompt, then release it before loading the next.
    for implementation, precision in product(implementations, precisions):
        variant = f"{implementation}_{precision}"
        print(f"Running {variant}...", flush=True)
        model = load_variant(implementation, experiment, profile, device, DTYPES[precision])
        try:
            for prompt_name, ids in token_ids.items():
                logits = run_prompt(model, implementation, experiment, ids.to(device=device))
                outputs[prompt_name][variant] = logits.float()
                runs[prompt_name][variant] = summarize(logits, tokenizer)
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return {
        "experiment": experiment_name,
        "model": model_name,
        "checkpoint": str(path),
        "torch_version": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "attention_backend": experiment.ATTENTION_BACKEND,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "use_cache": experiment.USES_CACHE,
        "prompts": {
            name: {
                "text": prompts[name],
                "token_ids": token_ids[name].tolist(),
                "runs": runs[name],
                "comparisons": compare_outputs(outputs[name], tokenizer),
            }
            for name in prompts
        },
    }


def load_variant(
    implementation: Implementation,
    experiment: Experiment,
    profile: ModelProfile,
    device: str,
    dtype: torch.dtype,
) -> nn.Module:
    if implementation == "custom":
        return experiment.load_model(profile, device, dtype)
    return (
        cast(
            nn.Module,
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


def run_prompt(
    model: nn.Module,
    implementation: Implementation,
    experiment: Experiment,
    ids: Tensor,
) -> Tensor:
    if implementation == "transformers":
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            logits = model(ids, use_cache=False).logits
    else:
        logits = experiment.prefill(model, ids)
    return cast(Tensor, logits).cpu()


def summarize(logits: Tensor, tokenizer) -> dict:
    values, indices = logits[0, -1].float().topk(5)
    return {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }


def compare_outputs(outputs: Mapping[str, Tensor], tokenizer) -> dict:
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
