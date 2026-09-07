"""Reusable timing, statistics, and GPU-memory measurements."""

import math
import statistics
import time
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

import torch

STABILIZATION_RUNS = 5
WARMUP_RUNS = 3
MEASURED_RUNS = 10

type MemorySample = dict[str, int] | None


class TokenTimer:
    """Record token boundaries without synchronizing the GPU between tokens."""

    def __init__(self, device: torch.device, max_tokens: int) -> None:
        self.device = device
        self.started_at: float | None = None
        self.timestamps: list[float] = []
        self.recorded_tokens = 0
        self.start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        self.token_events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(max_tokens)]
            if device.type == "cuda"
            else []
        )

    def start(self) -> None:
        self.timestamps.clear()
        self.recorded_tokens = 0
        if self.device.type == "cuda":
            assert self.start_event is not None
            self.start_event.record()
        else:
            self.started_at = time.perf_counter()

    def record_token(self) -> None:
        if self.device.type == "cuda":
            event = self.token_events[self.recorded_tokens]
            event.record()
        else:
            assert self.started_at is not None
            self.timestamps.append((time.perf_counter() - self.started_at) * 1_000)
        self.recorded_tokens += 1

    def timestamps_ms(self) -> list[float]:
        if self.device.type != "cuda":
            return list(self.timestamps)
        if self.start_event is None:
            raise RuntimeError("token timer was not started")
        return [
            self.start_event.elapsed_time(event)
            for event in self.token_events[: self.recorded_tokens]
        ]


def stabilize(operation: Callable[[], Any], device: str) -> None:
    """Bring kernels, allocators, and GPU clocks into a repeatable warm state."""
    for _ in range(STABILIZATION_RUNS):
        result = operation()
        synchronize(device)
        del result


def measure(
    operation: Callable[[], Any],
    device: str,
    *,
    capture_each: Callable[[Any], None] | None = None,
    capture_last: Callable[[Any], None] | None = None,
) -> tuple[list[float], list[MemorySample]]:
    """Warm up an operation, then return wall latency and peak memory per run."""
    for _ in range(WARMUP_RUNS):
        result = operation()
        synchronize(device)
        del result

    latencies = []
    memory = []
    for run in range(MEASURED_RUNS):
        reset_peak_gpu_memory(device)
        synchronize(device)
        start = time.perf_counter()
        result = operation()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1_000)
        memory.append(peak_gpu_memory(device))
        if capture_each is not None:
            capture_each(result)
        if capture_last is not None and run == MEASURED_RUNS - 1:
            capture_last(result)
        del result
    return latencies, memory


def measure_once(operation: Callable[[], Any], device: str = "cpu") -> tuple[Any, float]:
    """Return an operation's result and synchronized wall latency in milliseconds."""
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1_000


def current_gpu_memory(device: str) -> MemorySample:
    if device != "cuda":
        return None
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def peak_gpu_memory(device: str) -> MemorySample:
    if device != "cuda":
        return None
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def reset_peak_gpu_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def summarize_memory(samples: Sequence[MemorySample], model_memory: MemorySample) -> dict:
    allocated = maximum_memory(samples, "peak_allocated_bytes")
    reserved = maximum_memory(samples, "peak_reserved_bytes")
    baseline = model_memory["allocated_bytes"] if model_memory is not None else None
    return {
        "peak_gpu_memory_allocated_bytes": allocated,
        "peak_gpu_memory_reserved_bytes": reserved,
        "peak_gpu_memory_above_model_bytes": (
            allocated - baseline if allocated is not None and baseline is not None else None
        ),
    }


def raw_memory(sample: MemorySample, model_memory: MemorySample) -> dict:
    if sample is None:
        return {
            "peak_gpu_memory_allocated_bytes": None,
            "peak_gpu_memory_reserved_bytes": None,
            "peak_gpu_memory_above_model_bytes": None,
        }
    baseline = model_memory["allocated_bytes"] if model_memory is not None else 0
    return {
        **sample,
        "peak_gpu_memory_above_model_bytes": sample["peak_allocated_bytes"] - baseline,
    }


def maximum_memory(samples: Sequence[MemorySample], key: str) -> int | None:
    return max(
        (sample[key] for sample in samples if sample is not None),
        default=None,
    )


def intervals(cumulative: Sequence[float]) -> list[float]:
    if not cumulative:
        raise ValueError("cannot derive token latencies from an empty timeline")
    return [cumulative[0], *[b - a for a, b in pairwise(cumulative)]]


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def summarize(values: Sequence[float]) -> dict:
    """Summarize repeated measurements without discarding raw observations."""
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
