"""Configuration shared by runner modules and setup tools."""

import torch

MODAL_SECRET = "huggingface"
MODAL_VOLUME = "llm-inference-checkpoints"

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
