"""Measure the fixed benchmark suite for one experiment."""

import platform
import statistics
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from harness.profiles import get_profile
from runner.benchmark_workloads import STANDARD_BENCHMARK, BenchmarkWorkload
from runner.config import DTYPES
from runner.experiments import get_experiment
from runner.gpu_specs import (
    GpuSpec,
    actual_gpu_hardware,
    get_gpu_spec,
    validate_gpu_hardware,
)
from runner.measurements import (
    MEASURED_RUNS,
    STABILIZATION_RUNS,
    WARMUP_RUNS,
    MemorySample,
    TokenTimer,
    current_gpu_memory,
    intervals,
    measure,
    measure_once,
    raw_memory,
    stabilize,
    summarize,
    summarize_memory,
)


def benchmark(experiment_name: str, device: str = "cuda") -> dict:
    """Run every standard workload against one experiment model."""
    suite = STANDARD_BENCHMARK
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the standard benchmark")

    gpu_spec = get_gpu_spec(suite.gpu)
    actual_hardware = actual_gpu_hardware(device)
    validate_gpu_hardware(gpu_spec, actual_hardware)
    experiment = get_experiment(experiment_name)
    profile = get_profile(suite.model)
    path = profile.snapshot_path()
    tokenizer, tokenizer_loading_ms = measure_once(
        lambda: AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            clean_up_tokenization_spaces=False,
        )
    )
    model, model_loading_ms = measure_once(
        lambda: experiment.load_model(profile, device, DTYPES[suite.dtype]), device
    )
    model_memory = current_gpu_memory(device)

    inputs = {}
    tokenization_ms = {}
    for workload in suite.workloads:
        source_ids, tokenization_ms[workload.name] = measure_once(
            lambda prompt=workload.prompt: tokenizer(prompt, return_tensors="pt").input_ids
        )
        inputs[workload.name] = workload_input(source_ids, workload, suite.batch_size, device)

    stabilization_workload = max(
        (workload for workload in suite.workloads if workload.kind == "prefill"),
        key=lambda workload: workload.input_tokens,
    )
    _, stabilization_ms = measure_once(
        lambda: stabilize(
            lambda: experiment.prefill(model, inputs[stabilization_workload.name]),
            device,
        ),
        device,
    )
    stabilized_memory = current_gpu_memory(device)

    results = {}
    for workload in suite.workloads:
        print(f"Benchmarking {workload.name}...", flush=True)
        input_ids = inputs[workload.name]
        if workload.kind == "prefill":
            measurements, next_token_id = benchmark_prefill(
                lambda current_input=input_ids: experiment.prefill(model, current_input),
                input_ids,
                device,
                model_memory,
            )
            output = {
                "next_token_id": next_token_id,
                "next_token_text": tokenizer.decode([next_token_id]),
            }
        else:

            def run_generate(
                on_token: Callable[[], None],
                current_input: Tensor = input_ids,
                output_tokens: int = workload.output_tokens,
            ) -> Tensor:
                return experiment.generate(
                    model,
                    current_input,
                    output_tokens,
                    on_token=on_token,
                )

            measurements, generated = benchmark_decode(
                run_generate,
                input_ids,
                workload.output_tokens,
                device,
                model_memory,
                gpu_spec,
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
        "model_parameters_billions": profile.params_b,
        "dtype": suite.dtype,
        "device": device,
        "gpu_requested": suite.gpu,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device == "cuda" else None,
        "hardware": {
            "published": gpu_spec.report(),
            "actual": actual_hardware,
        },
        "runtime": {
            "batch_size": suite.batch_size,
            "stabilization_runs": STABILIZATION_RUNS,
            "stabilization_workload": stabilization_workload.name,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "attention_backend": experiment.ATTENTION_BACKEND,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "use_cache": experiment.USES_CACHE,
            "full_recomputation": experiment.FULL_RECOMPUTATION,
            "generation": experiment.GENERATION_ALGORITHM,
            "stop_on_eos": False,
        },
        "setup": {
            "tokenizer_loading_ms": tokenizer_loading_ms,
            "model_loading_ms": model_loading_ms,
            "tokenization_ms": tokenization_ms,
            "total_tokenization_ms": sum(tokenization_ms.values()),
            "stabilization_ms": stabilization_ms,
            "gpu_memory_after_model_load": model_memory,
            "gpu_memory_after_stabilization": stabilized_memory,
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
    operation: Callable[[], Tensor],
    input_ids: Tensor,
    device: str,
    model_memory: MemorySample,
) -> tuple[dict, int]:
    next_token_id = 0

    def capture(logits: Tensor) -> None:
        nonlocal next_token_id
        next_token_id = int(logits[0, -1].argmax().item())

    latency, memory = measure(operation, device, capture_last=capture)
    raw = [
        {
            "prefill_latency_ms": elapsed,
            "input_tokens_per_second": input_ids.numel() / (elapsed / 1_000),
            **raw_memory(sample, model_memory),
        }
        for elapsed, sample in zip(latency, memory, strict=True)
    ]
    return (
        {
            **summarize_runs(raw, ("prefill_latency_ms", "input_tokens_per_second")),
            **summarize_memory(memory, model_memory),
            "raw_measurements": raw,
        },
        next_token_id,
    )


def benchmark_decode(
    generate: Callable[[Callable[[], None]], Tensor],
    input_ids: Tensor,
    output_tokens: int,
    device: str,
    model_memory: MemorySample,
    gpu_spec: GpuSpec,
) -> tuple[dict, Tensor]:
    token_timestamps: list[list[float]] = []
    generated = input_ids.cpu()
    timer = TokenTimer(input_ids.device, output_tokens)

    def operation() -> Tensor:
        timer.start()
        return generate(timer.record_token)

    def capture_timing(_: Tensor) -> None:
        token_timestamps.append(timer.timestamps_ms())

    def capture_output(token_ids: Tensor) -> None:
        nonlocal generated
        generated = token_ids.cpu()

    total, memory = measure(
        operation,
        device,
        capture_each=capture_timing,
        capture_last=capture_output,
    )
    if any(len(timestamps) != output_tokens for timestamps in token_timestamps):
        raise RuntimeError("generation did not record the requested number of tokens")

    raw = []
    for elapsed, timestamps, sample in zip(total, token_timestamps, memory, strict=True):
        latencies = intervals(timestamps)
        per_token = statistics.mean(latencies[1:])
        rate = output_tokens / (elapsed / 1_000)
        per_dollar = output_tokens / ((elapsed / 1_000) * gpu_spec.modal_gpu_usd_per_second)
        raw.append(
            {
                "time_to_first_token_ms": timestamps[0],
                "time_per_output_token_ms": per_token,
                "total_generation_latency_ms": elapsed,
                "generation_tokens_per_second": rate,
                "generation_tokens_per_gpu_dollar": per_dollar,
                "generation_gpu_cost_per_million_tokens_usd": 1_000_000 / per_dollar,
                "token_timestamps_ms": timestamps,
                "token_latencies_ms": latencies,
                **raw_memory(sample, model_memory),
            }
        )

    return (
        {
            **summarize_runs(
                raw,
                (
                    "time_to_first_token_ms",
                    "time_per_output_token_ms",
                    "total_generation_latency_ms",
                    "generation_tokens_per_second",
                    "generation_tokens_per_gpu_dollar",
                    "generation_gpu_cost_per_million_tokens_usd",
                ),
            ),
            **summarize_memory(memory, model_memory),
            "raw_measurements": raw,
        },
        generated,
    )


def summarize_runs(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    """Summarize report fields directly from their raw per-run rows."""
    return {metric: summarize([row[metric] for row in rows]) for metric in metrics}


def decode(tokenizer: Any, token_ids: Tensor) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
