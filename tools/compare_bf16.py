import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.loader import load_model
from harness.profiles import get_profile

SHORT_PROMPT = "The present King of France is"
LONG_PROMPT = (
    "The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection "
    "of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text "
    "out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for "
    "multilingual dialogue use cases and outperform many of the available open source"
)

PROMPT = LONG_PROMPT


def compare(model_profile: str, device: str, prompt: str = PROMPT) -> dict:
    """Run all four models sequentially, keeping comparison logits on CPU."""
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the modal profile")
        torch.set_float32_matmul_precision("highest")

    path = get_profile(model_profile).snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)
    outputs = {}
    runs = {}

    # transformers fp32
    print("Running transformers fp32...", flush=True)
    model = (
        cast(
            torch.nn.Module,
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
            ),
        )
        .to(device=device)
        .eval()
    )
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids, use_cache=False).logits
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = logits.cpu()
    values, indices = logits[0, -1].float().topk(5)
    runs["transformers_fp32"] = {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "forward_seconds": elapsed,
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }
    outputs["transformers_fp32"] = logits.float().clone()
    del model, logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # transformers bf16
    print("Running transformers bf16...", flush=True)
    model = (
        cast(
            torch.nn.Module,
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
            ),
        )
        .to(device=device)
        .eval()
    )
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids, use_cache=False).logits
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = logits.cpu()
    values, indices = logits[0, -1].float().topk(5)
    runs["transformers_bf16"] = {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "forward_seconds": elapsed,
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }
    outputs["transformers_bf16"] = logits.float().clone()
    del model, logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # custom fp32
    print("Running custom fp32...", flush=True)
    model = load_model(model_profile, device=device, dtype=torch.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = logits.cpu()
    values, indices = logits[0, -1].float().topk(5)
    runs["custom_fp32"] = {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "forward_seconds": elapsed,
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }
    outputs["custom_fp32"] = logits.float().clone()
    del model, logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # custom bf16
    print("Running custom bf16...", flush=True)
    model = load_model(model_profile, device=device, dtype=torch.bfloat16)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = logits.cpu()
    values, indices = logits[0, -1].float().topk(5)
    runs["custom_bf16"] = {
        "logit_dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "forward_seconds": elapsed,
        "top_5_next_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(indices.tolist(), values.tolist(), strict=True)
        ],
    }
    outputs["custom_bf16"] = logits.float().clone()
    del model, logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    comparisons = compare_outputs(outputs, tokenizer)
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
        "comparisons": comparisons,
    }


def compare_outputs(outputs: dict, tokenizer) -> dict:
    comparisons = {}
    for reference, candidate in (
        ("transformers_fp32", "transformers_bf16"),
        ("transformers_fp32", "custom_fp32"),
        ("transformers_fp32", "custom_bf16"),
        ("transformers_bf16", "custom_fp32"),
        ("transformers_bf16", "custom_bf16"),
        ("custom_fp32", "custom_bf16"),
    ):
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


def save_report(report: dict, profile: str) -> Path:
    now = datetime.now().astimezone()
    directory = (
        Path(__file__).resolve().parents[1] / "output_logs" / f"{now.month}_{now.day}_{now.year}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now:%H_%M_%S_%f}_{profile}.json"
    with path.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fp32 and bf16 model outputs.")
    parser.add_argument(
        "--profile",
        choices=("local", "modal"),
        default="local",
        help="local: 1B on CPU; modal: 8B on an A100 40GB (default local)",
    )
    args = parser.parse_args()
    if args.profile == "modal":
        from modal_compare import run

        report = run(PROMPT)
    else:
        report = compare("dev", "cpu")
    path = save_report(report, args.profile)
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
