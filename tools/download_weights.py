"""Download model weights into the Hugging Face cache.

Usage:
    uv run tools/download_weights.py dev       # Llama 3.2 1B, runs on the laptop
    uv run tools/download_weights.py target    # Llama 3.1 8B, needs a GPU box
    uv run tools/download_weights.py --list

Weights land in $HF_HOME (set in mise.toml), never in the repo. Both models
are gated: accept the licence on the model page, then `uv run hf auth login`
or export HF_TOKEN.

Only the safetensors weights and tokenizer/config files are fetched. The
`original/` directory in each repo holds the same weights as consolidated
PyTorch checkpoints and would double the download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import get_token, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from harness.profiles import PROFILES, get_profile

ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def dir_size_gb(path: Path) -> float:
    # Cache snapshots are symlinks into a blob store; resolve so we count bytes once.
    return sum(f.resolve().stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def download(profile: str) -> Path:
    repo_id = get_profile(profile).repo_id
    print(f"{profile}: {repo_id}")
    try:
        path = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS))
    except GatedRepoError:
        sys.exit(
            f"{repo_id} is gated. Accept the licence at https://huggingface.co/{repo_id} "
            "with the account your token belongs to, then retry."
        )
    except RepositoryNotFoundError:
        sys.exit(f"{repo_id} not found, or the token has no access to it.")
    print(f"  {path}  ({dir_size_gb(path):.2f} GB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights into the HF cache.")
    parser.add_argument("profiles", nargs="*", choices=list(PROFILES), metavar="PROFILE")
    parser.add_argument("--list", action="store_true", help="show profiles and exit")
    args = parser.parse_args()

    if args.list or not args.profiles:
        for name, p in PROFILES.items():
            print(f"{name:8} {p.repo_id}")
        return

    if get_token() is None:
        sys.exit("No Hugging Face token. Run `uv run hf auth login` or export HF_TOKEN.")

    for profile in args.profiles:
        download(profile)


if __name__ == "__main__":
    main()
