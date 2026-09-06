import sys

import modal
from huggingface_hub import get_token

SECRET_NAME = "huggingface"
VOLUME_NAME = "llm-inference-checkpoints"


def main() -> None:
    token = get_token()
    if not token:
        sys.exit("No Hugging Face token. Run `uv run hf auth login` or export HF_TOKEN.")

    modal.Secret.objects.create(SECRET_NAME, {"HF_TOKEN": token}, allow_existing=True)
    modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"]).hydrate()
    print(f"Secret ready: {SECRET_NAME} (existing values preserved)")

    modal.Volume.objects.create(VOLUME_NAME, allow_existing=True)
    print(f"Checkpoint cache ready: {VOLUME_NAME}")


if __name__ == "__main__":
    main()
