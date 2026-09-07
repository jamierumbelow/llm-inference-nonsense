from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import runner.benchmark as benchmark_module
from harness.generation import greedy_generate
from runner.benchmark_workloads import BenchmarkSuite, BenchmarkWorkload
from runner.measurements import (
    MEASURED_RUNS,
    WARMUP_RUNS,
    intervals,
    measure,
    measure_once,
    summarize,
)


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
        gpu="A100-40GB",
        dtype="bf16",
        batch_size=1,
        workloads=(
            BenchmarkWorkload("prefill", "prefill", "prompt", 3, 0),
            BenchmarkWorkload("decode", "decode", "prompt", 2, 2),
        ),
    )
    profile = SimpleNamespace(
        repo_id="example/checkpoint",
        params_b=1.0,
        snapshot_path=lambda: Path("/tmp/checkpoint"),
    )
    experiment = SimpleNamespace(
        ATTENTION_BACKEND="test",
        USES_CACHE=False,
        FULL_RECOMPUTATION=True,
        GENERATION_ALGORITHM="greedy",
        load_model=lambda profile, device, dtype: FakeModel(),
        prefill=lambda model, input_ids: model(input_ids),
        generate=lambda model, input_ids, max_new_tokens, on_token=None: greedy_generate(
            model, input_ids, max_new_tokens, on_token=on_token
        ),
    )
    monkeypatch.setattr(benchmark_module, "STANDARD_BENCHMARK", suite)
    monkeypatch.setattr(benchmark_module, "get_profile", lambda name: profile)
    monkeypatch.setattr(benchmark_module, "get_experiment", lambda name: experiment)
    monkeypatch.setattr(
        benchmark_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    report = benchmark_module.benchmark("e00", device="cpu")

    assert report["checkpoint"] == "/tmp/checkpoint"
    assert report["runtime"]["batch_size"] == 1
    assert report["results"]["prefill"]["input"]["token_ids"] == [0, 1, 2]
    assert report["results"]["prefill"]["output"]["next_token_id"] == 2
    assert report["results"]["decode"]["output"]["token_ids"] == [2, 2]
    assert len(report["results"]["decode"]["raw_measurements"]) == MEASURED_RUNS
    assert len(report["results"]["decode"]["raw_measurements"][0]["token_latencies_ms"]) == 2
    assert report["results"]["decode"]["time_per_output_token_ms"]["mean"] >= 0
    assert report["results"]["decode"]["generation_tokens_per_gpu_dollar"]["mean"] > 0
    assert report["hardware"]["published"]["modal_gpu_usd_per_hour"] == pytest.approx(2.0988)
    assert report["setup"]["gpu_memory_after_model_load"] is None


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


def test_intervals_turns_cumulative_token_times_into_durations() -> None:
    assert intervals([2.0, 5.5, 9.0]) == [2.0, 3.5, 3.5]


def test_summarize_reports_variation() -> None:
    summary = summarize([1, 2, 3, 4, 5])

    assert summary["mean"] == 3
    assert summary["median"] == 3
    assert summary["minimum"] == 1
    assert summary["maximum"] == 5
    assert summary["standard_deviation"] == pytest.approx(1.5811, rel=1e-4)
    assert summary["p10"] == pytest.approx(1.4)
    assert summary["p90"] == pytest.approx(4.6)
