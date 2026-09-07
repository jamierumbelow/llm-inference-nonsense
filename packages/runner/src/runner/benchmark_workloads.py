"""The fixed benchmark suite shared by every experiment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type WorkloadKind = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    name: str
    kind: WorkloadKind
    prompt: str
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("benchmark prompts cannot be empty")
        if self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")
        if self.kind == "prefill" and self.output_tokens != 0:
            raise ValueError("prefill workloads cannot generate output tokens")
        if self.kind == "decode" and self.output_tokens < 1:
            raise ValueError("decode workloads must generate at least one token")


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    name: str
    model: str
    location: str
    gpu: str
    dtype: str
    batch_size: int
    workloads: tuple[BenchmarkWorkload, ...]


BENCHMARK_PROMPT = Path(__file__).with_name("benchmark_prompt.txt").read_text().strip()

STANDARD_BENCHMARK = BenchmarkSuite(
    name="standard",
    model="8b",
    location="modal",
    gpu="A100-40GB",
    dtype="bf16",
    batch_size=1,
    workloads=(
        BenchmarkWorkload(
            "short_prefill", "prefill", BENCHMARK_PROMPT, input_tokens=128, output_tokens=0
        ),
        BenchmarkWorkload(
            "medium_prefill", "prefill", BENCHMARK_PROMPT, input_tokens=512, output_tokens=0
        ),
        BenchmarkWorkload(
            "long_prefill", "prefill", BENCHMARK_PROMPT, input_tokens=1024, output_tokens=0
        ),
        BenchmarkWorkload(
            "short_decode", "decode", BENCHMARK_PROMPT, input_tokens=128, output_tokens=32
        ),
        BenchmarkWorkload(
            "long_decode", "decode", BENCHMARK_PROMPT, input_tokens=128, output_tokens=128
        ),
    ),
)
