"""Modal resources used by `compare_bf16.py --location modal`."""

from pathlib import Path

import modal
from setup_modal import SECRET_NAME, VOLUME_NAME

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = "/checkpoints"

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
            "PYTHONPATH": "/app/packages/harness/src:/app/tools",
        }
    )
    .add_local_file(ROOT / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/app/uv.lock", copy=True)
    .add_local_file(
        ROOT / "packages/harness/pyproject.toml",
        "/app/packages/harness/pyproject.toml",
        copy=True,
    )
    .run_commands("uv sync --frozen --no-install-workspace")
    .add_local_dir(
        ROOT / "packages/harness/src", "/app/packages/harness/src", ignore=["__pycache__"]
    )
    .add_local_dir(ROOT / "tools", "/app/tools", ignore=["__pycache__"])
)

app = modal.App("llm-inference-compare", image=image)
cache = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])],
    volumes={CACHE_PATH: cache},
    memory=8192,
    timeout=1800,
    max_containers=1,
)
def prepare_checkpoint(model_profile: str) -> None:
    from download_weights import download

    download(model_profile)
    cache.commit()


@app.function(
    gpu="A100-40GB",
    volumes={CACHE_PATH: cache},
    memory=49152,  # CPU RAM for the Transformers fp32 checkpoint before GPU transfer.
    timeout=600,
    max_containers=1,
    scaledown_window=2,
)
def compare_on_gpu(model_profile: str, prompt: str) -> dict:
    from harness.comparison import compare

    cache.reload()
    return compare(model_profile, "cuda", prompt)


def run(model_profile: str, prompt: str) -> dict:
    with modal.enable_output(), app.run():
        # Download on CPU so GPU time is only used for the comparison.
        prepare_checkpoint.remote(model_profile)
        return compare_on_gpu.remote(model_profile, prompt)
