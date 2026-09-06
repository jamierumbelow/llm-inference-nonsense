import argparse
import gc
import json
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.loader import load_model
from harness.profiles import get_profile

SHORT_PROMPT = "The present King of France is"
LONG_PROMPT = "The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source"

PROMPT = LONG_PROMPT

def main() -> None:
    # parse flags
    parser = argparse.ArgumentParser(description="Compare fp32 and bf16 model outputs.")
    parser.add_argument(
        "--profile",
        choices=("local", "modal"),
        default="local",
        help="execution profile (default local)",
    )
    args = parser.parse_args()
    if args.profile == "modal":
        parser.error("the modal profile is not configured yet; use --profile local")

    path = get_profile("dev").snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    outputs = {}
    runs = {}

    # transformers fp32
    print("Running transformers fp32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids, use_cache=False).logits
    elapsed = time.perf_counter() - start
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

    # transformers bf16
    print("Running transformers bf16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids, use_cache=False).logits
    elapsed = time.perf_counter() - start
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

    # custom fp32
    print("Running custom fp32...", flush=True)
    model = load_model("dev", device="cpu", dtype=torch.float32)
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids)
    elapsed = time.perf_counter() - start
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

    # custom bf16
    print("Running custom bf16...", flush=True)
    model = load_model("dev", device="cpu", dtype=torch.bfloat16)
    start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits = model(ids)
    elapsed = time.perf_counter() - start
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

    # compare them all
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
        for position in range(ids.shape[1]):
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
            "argmax_agreement": f"{ids.shape[1] - len(disagreements)}/{ids.shape[1]}",
            "disagreements": disagreements,
        }
    print(
        json.dumps(
            {
                "prompt": PROMPT,
                "token_ids": ids.tolist(),
                "checkpoint": str(path),
                "torch_version": torch.__version__,
                "device": "cpu",
                "attention_backend": "SDPA math",
                "use_cache": False,
                "runs": runs,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
