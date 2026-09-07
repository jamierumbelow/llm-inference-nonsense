"""Run model comparisons on Modal."""

import time
from collections.abc import Mapping
from pathlib import Path

import modal
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from harness.profiles import CHECKPOINT_FILES, get_profile
from runner.benchmark import benchmark as benchmark_on_device
from runner.benchmark_workloads import STANDARD_BENCHMARK
from runner.comparison import compare
from runner.config import MODAL_SECRET, MODAL_VOLUME
from runner.generation import generate as generate_on_device

ROOT = Path(__file__).resolve().parents[4]
CACHE_PATH = "/checkpoints"
EXPERIMENT_DIRS = sorted(
    path for path in (ROOT / "experiments").iterdir() if (path / "pyproject.toml").exists()
)
PYTHONPATH = ["/app/packages/harness/src", "/app/packages/runner/src"] + [
    f"/app/experiments/{path.name}/src" for path in EXPERIMENT_DIRS
]

# Install the locked workspace dependencies, then mount source for quick iteration.
# Modal's Image.uv_sync helper does not currently support uv workspaces.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("uv==0.12.9")
    .workdir("/app")
    .env(
        {
            "HF_HOME": CACHE_PATH,
            "UV_PROJECT_ENVIRONMENT": "/opt/venv",
            "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": ":".join(PYTHONPATH),
        }
    )
    .add_local_file(ROOT / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/app/uv.lock", copy=True)
    .add_local_file(
        ROOT / "packages/harness/pyproject.toml",
        "/app/packages/harness/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        ROOT / "packages/runner/pyproject.toml",
        "/app/packages/runner/pyproject.toml",
        copy=True,
    )
)
for experiment_dir in EXPERIMENT_DIRS:
    image = image.add_local_file(
        experiment_dir / "pyproject.toml",
        f"/app/experiments/{experiment_dir.name}/pyproject.toml",
        copy=True,
    )
image = (
    image.run_commands("uv sync --frozen --no-install-workspace")
    .add_local_dir(
        ROOT / "packages/harness/src", "/app/packages/harness/src", ignore=["__pycache__"]
    )
    .add_local_dir(ROOT / "packages/runner/src", "/app/packages/runner/src", ignore=["__pycache__"])
)
for experiment_dir in EXPERIMENT_DIRS:
    image = image.add_local_dir(
        experiment_dir / "src",
        f"/app/experiments/{experiment_dir.name}/src",
        ignore=["__pycache__"],
    )

app = modal.App("llm-inference-compare", image=image)
cache = modal.Volume.from_name(MODAL_VOLUME)


def gpu_function(timeout: int):
    """Apply the resource configuration shared by every GPU operation."""
    return app.function(
        gpu=STANDARD_BENCHMARK.gpu,
        volumes={CACHE_PATH: cache},
        memory=49152,
        timeout=timeout,
        max_containers=1,
        scaledown_window=2,
    )


@app.function(
    secrets=[modal.Secret.from_name(MODAL_SECRET, required_keys=["HF_TOKEN"])],
    volumes={CACHE_PATH: cache},
    memory=8192,
    timeout=1800,
    max_containers=1,
)
def prepare_checkpoint(model: str) -> dict:
    profile = get_profile(model)
    try:
        path = snapshot_download(
            profile.repo_id,
            allow_patterns=CHECKPOINT_FILES,
            local_files_only=True,
        )
        cache_hit = True
    except LocalEntryNotFoundError:
        path = snapshot_download(profile.repo_id, allow_patterns=CHECKPOINT_FILES)
        cache_hit = False
    cache.commit()
    return {"cache_hit": cache_hit, "checkpoint": path}


@gpu_function(timeout=600)
def compare_on_gpu(experiment: str, model: str, prompts: dict[str, str]) -> dict:
    cache.reload()
    return compare(experiment, model, "cuda", prompts)


@gpu_function(timeout=600)
def generate_on_gpu(
    experiment: str,
    model: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    cache.reload()
    return generate_on_device(experiment, model, "cuda", dtype, prompt, max_new_tokens)


@gpu_function(timeout=1800)
def benchmark_on_gpu(experiment: str) -> dict:
    cache.reload()
    return benchmark_on_device(experiment)


def run(experiment: str, model: str, prompts: Mapping[str, str]) -> dict:
    with modal.enable_output(), app.run():
        # Download on CPU so GPU time is only used for the comparison.
        prepare_checkpoint.remote(model)
        return compare_on_gpu.remote(experiment, model, dict(prompts))


def generate(
    experiment: str,
    model: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    with modal.enable_output(), app.run():
        prepare_checkpoint.remote(model)
        return generate_on_gpu.remote(experiment, model, dtype, prompt, max_new_tokens)


def benchmark(experiment: str) -> dict:
    with modal.enable_output(), app.run():
        # Treat an already-running Modal app as the boundary of the end-to-end run.
        end_to_end_start = time.perf_counter()
        checkpoint_start = time.perf_counter()
        checkpoint = prepare_checkpoint.remote(STANDARD_BENCHMARK.model)
        checkpoint_preparation_ms = (time.perf_counter() - checkpoint_start) * 1_000

        gpu_start = time.perf_counter()
        report = benchmark_on_gpu.remote(experiment)
        gpu_benchmark_ms = (time.perf_counter() - gpu_start) * 1_000

        report["setup"] = {
            "checkpoint_preparation_ms": checkpoint_preparation_ms,
            "checkpoint_cache_hit": checkpoint["cache_hit"],
            **report["setup"],
        }
        report["modal_gpu_benchmark_ms"] = gpu_benchmark_ms
        gpu_rate = report["hardware"]["published"]["modal_gpu_usd_per_second"]
        report["estimated_modal_gpu_cost_usd"] = gpu_benchmark_ms / 1_000 * gpu_rate
        report["end_to_end_latency_ms"] = (time.perf_counter() - end_to_end_start) * 1_000
        return report
