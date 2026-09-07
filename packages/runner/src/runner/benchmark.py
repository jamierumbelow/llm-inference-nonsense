"""Measure the fixed benchmark suite for one experiment."""

import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

from harness.generation import greedy_generate
from harness.profiles import get_profile
from runner.benchmark_workloads import STANDARD_BENCHMARK, BenchmarkWorkload
from runner.experiments import get_experiment
from runner.generation import DTYPES

WARMUP_RUNS = 3
MEASURED_RUNS = 10


def benchmark(experiment_name: str, device: str = "cuda") -> dict:
    """Run every standard workload against one experiment model."""
    suite = STANDARD_BENCHMARK
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the standard benchmark")

    experiment = get_experiment(experiment_name)
    profile = get_profile(suite.model)
    path = profile.snapshot_path()
    tokenizer, tokenizer_loading_ms = measure_once(
        lambda: AutoTokenizer.from_pretrained(path, local_files_only=True)
    )
    model, model_loading_ms = measure_once(
        lambda: experiment.load_model(profile, device, DTYPES[suite.dtype]), device
    )

    def forward(input_ids: Tensor) -> Tensor:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            return model(input_ids)

    results = {}
    tokenization_ms = {}
    for workload in suite.workloads:
        print(f"Benchmarking {workload.name}...", flush=True)
        source_ids, tokenization_ms[workload.name] = measure_once(
            lambda prompt=workload.prompt: tokenizer(prompt, return_tensors="pt").input_ids
        )
        input_ids = workload_input(source_ids, workload, suite.batch_size, device)
        if workload.kind == "prefill":
            measurements, next_token_id = benchmark_prefill(forward, input_ids, device)
            output = {
                "next_token_id": next_token_id,
                "next_token_text": tokenizer.decode([next_token_id]),
            }
        else:
            measurements, generated = benchmark_decode(
                forward, input_ids, workload.output_tokens, device
            )
            generated_ids = generated[:, input_ids.shape[1] :]
            output = {
                "text": decode(tokenizer, generated_ids[0]),
                "token_ids": generated_ids[0].tolist(),
            }

        results[workload.name] = {
            "kind": workload.kind,
            "input_tokens": workload.input_tokens,
            "output_tokens": workload.output_tokens,
            "input": {
                "text": decode(tokenizer, input_ids[0]),
                "token_ids": input_ids[0].tolist(),
            },
            "output": output,
            **measurements,
        }

    return {
        "experiment": experiment_name,
        "suite": suite.name,
        "model": suite.model,
        "location": suite.location,
        "checkpoint": str(path),
        "checkpoint_repo": profile.repo_id,
        "dtype": suite.dtype,
        "device": device,
        "gpu_requested": suite.gpu,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device == "cuda" else None,
        "runtime": {
            "batch_size": suite.batch_size,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "attention_backend": "SDPA math",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "use_cache": False,
            "generation": "greedy",
            "stop_on_eos": False,
            "full_recomputation": True,
        },
        "setup": {
            "tokenizer_loading_ms": tokenizer_loading_ms,
            "model_loading_ms": model_loading_ms,
            "tokenization_ms": tokenization_ms,
            "total_tokenization_ms": sum(tokenization_ms.values()),
        },
        "results": results,
    }


def workload_input(
    source_ids: Tensor,
    workload: BenchmarkWorkload,
    batch_size: int,
    device: str,
) -> Tensor:
    if source_ids.shape[1] < workload.input_tokens:
        raise ValueError(
            f"{workload.name} needs {workload.input_tokens} input tokens, "
            f"but its prompt only produced {source_ids.shape[1]}"
        )
    return source_ids[:, : workload.input_tokens].repeat(batch_size, 1).to(device=device)


def benchmark_prefill(
    forward: Callable[[Tensor], Tensor],
    input_ids: Tensor,
    device: str,
) -> tuple[dict, int]:
    next_token_id = 0

    def capture(logits: Tensor) -> None:
        nonlocal next_token_id
        next_token_id = int(logits[0, -1].argmax().item())

    latency, peak_memory = measure(lambda: forward(input_ids), device, capture_last=capture)
    return (
        {
            "prefill_latency_ms": summarize(latency),
            "peak_gpu_memory_bytes": maximum_memory(peak_memory),
            "raw_measurements": [
                {"prefill_latency_ms": elapsed, "peak_gpu_memory_bytes": peak}
                for elapsed, peak in zip(latency, peak_memory, strict=True)
            ],
        },
        next_token_id,
    )


def benchmark_decode(
    forward: Callable[[Tensor], Tensor],
    input_ids: Tensor,
    output_tokens: int,
    device: str,
) -> tuple[dict, Tensor]:
    first_token, _ = measure(lambda: greedy_generate(forward, input_ids, 1), device)
    generated = input_ids.cpu()

    def capture(output_ids: Tensor) -> None:
        nonlocal generated
        generated = output_ids.cpu()

    total, peak_memory = measure(
        lambda: greedy_generate(forward, input_ids, output_tokens),
        device,
        capture_last=capture,
    )
    per_token = [elapsed / output_tokens for elapsed in total]
    return (
        {
            "time_to_first_token_ms": summarize(first_token),
            "total_generation_latency_ms": summarize(total),
            "average_time_per_token_ms": summarize(per_token),
            "peak_gpu_memory_bytes": maximum_memory(peak_memory),
            "raw_measurements": [
                {
                    "time_to_first_token_ms": ttft,
                    "total_generation_latency_ms": generation,
                    "average_time_per_token_ms": average,
                    "peak_gpu_memory_bytes": peak,
                }
                for ttft, generation, average, peak in zip(
                    first_token, total, per_token, peak_memory, strict=True
                )
            ],
        },
        generated,
    )


def decode(tokenizer: Any, token_ids: Tensor) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def measure(
    operation: Callable[[], Any],
    device: str,
    *,
    capture_last: Callable[[Any], None] | None = None,
) -> tuple[list[float], list[int | None]]:
    """Warm up an operation, then return wall latency and peak allocation per run."""
    for _ in range(WARMUP_RUNS):
        result = operation()
        synchronize(device)
        del result

    latencies = []
    peak_memory = []
    for run in range(MEASURED_RUNS):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        start = time.perf_counter()
        result = operation()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1_000)
        peak_memory.append(torch.cuda.max_memory_allocated() if device == "cuda" else None)
        if capture_last is not None and run == MEASURED_RUNS - 1:
            capture_last(result)
        del result
    return latencies, peak_memory


def measure_once(operation: Callable[[], Any], device: str = "cpu") -> tuple[Any, float]:
    """Return an operation's result and synchronized wall latency in milliseconds."""
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1_000


def maximum_memory(values: Sequence[int | None]) -> int | None:
    return max((value for value in values if value is not None), default=None)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def summarize(values: Sequence[float]) -> dict:
    """Summarize repeated timings without discarding the raw observations."""
    if not values:
        raise ValueError("cannot summarize an empty measurement set")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
