import sys

import modal
from huggingface_hub import get_token
from runner.config import MODAL_SECRET, MODAL_VOLUME


def main() -> None:
    token = get_token()
    if not token:
        sys.exit("No Hugging Face token. Run `uv run hf auth login` or export HF_TOKEN.")

    modal.Secret.objects.create(MODAL_SECRET, {"HF_TOKEN": token}, allow_existing=True)
    modal.Secret.from_name(MODAL_SECRET, required_keys=["HF_TOKEN"]).hydrate()
    print(f"Secret ready: {MODAL_SECRET} (existing values preserved)")

    modal.Volume.objects.create(MODAL_VOLUME, allow_existing=True)
    print(f"Checkpoint cache ready: {MODAL_VOLUME}")


if __name__ == "__main__":
    main()
