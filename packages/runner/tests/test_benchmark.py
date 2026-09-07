from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import runner.benchmark as benchmark_module
from runner.benchmark import (
    MEASURED_RUNS,
    WARMUP_RUNS,
    measure,
    measure_once,
    summarize,
)
from runner.benchmark_workloads import BenchmarkSuite, BenchmarkWorkload


class FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([[0, 1, 2, 3]]))

    def decode(self, token_ids, **kwargs) -> str:
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return " ".join(str(value) for value in values)


class FakeModel:
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., 2] = 1
        return logits


def test_benchmark_records_inputs_outputs_and_raw_measurements(monkeypatch) -> None:
    suite = BenchmarkSuite(
        name="test",
        model="test-model",
        location="local",
        gpu="none",
        dtype="bf16",
        batch_size=1,
        workloads=(
            BenchmarkWorkload("prefill", "prefill", "prompt", 3, 0),
            BenchmarkWorkload("decode", "decode", "prompt", 2, 2),
        ),
    )
    profile = SimpleNamespace(
        repo_id="example/checkpoint",
        snapshot_path=lambda: Path("/tmp/checkpoint"),
    )
    experiment = SimpleNamespace(load_model=lambda profile, device, dtype: FakeModel())
    monkeypatch.setattr(benchmark_module, "STANDARD_BENCHMARK", suite)
    monkeypatch.setattr(benchmark_module, "get_profile", lambda name: profile)
    monkeypatch.setattr(benchmark_module, "get_experiment", lambda name: experiment)
    monkeypatch.setattr(
        benchmark_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(benchmark_module, "WARMUP_RUNS", 0)
    monkeypatch.setattr(benchmark_module, "MEASURED_RUNS", 1)

    report = benchmark_module.benchmark("e00", device="cpu")

    assert report["checkpoint"] == "/tmp/checkpoint"
    assert report["runtime"]["batch_size"] == 1
    assert report["results"]["prefill"]["input"]["token_ids"] == [0, 1, 2]
    assert report["results"]["prefill"]["output"]["next_token_id"] == 2
    assert report["results"]["decode"]["output"]["token_ids"] == [2, 2]
    assert len(report["results"]["decode"]["raw_measurements"]) == 1


def test_measure_excludes_warmups_and_supports_cpu() -> None:
    calls = 0
    captured = []

    def operation() -> int:
        nonlocal calls
        calls += 1
        return calls

    latencies, peak_memory = measure(operation, "cpu", capture_last=captured.append)

    assert calls == WARMUP_RUNS + MEASURED_RUNS
    assert len(latencies) == MEASURED_RUNS
    assert all(latency >= 0 for latency in latencies)
    assert peak_memory == [None] * MEASURED_RUNS
    assert captured == [WARMUP_RUNS + MEASURED_RUNS]


def test_measure_once_returns_the_result_and_latency() -> None:
    result, latency = measure_once(lambda: "loaded")

    assert result == "loaded"
    assert latency >= 0


def test_summarize_reports_variation() -> None:
    summary = summarize([1, 2, 3, 4, 5])

    assert summary["mean"] == 3
    assert summary["median"] == 3
    assert summary["minimum"] == 1
    assert summary["maximum"] == 5
    assert summary["standard_deviation"] == pytest.approx(1.5811, rel=1e-4)
    assert summary["p10"] == pytest.approx(1.4)
    assert summary["p90"] == pytest.approx(4.6)
