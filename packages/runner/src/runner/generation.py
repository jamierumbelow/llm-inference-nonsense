"""Load one experiment model and generate one continuation."""

import torch
from transformers import AutoTokenizer

from harness.config import LlamaConfig
from harness.profiles import get_profile
from runner.config import DTYPES
from runner.experiments import get_experiment


def generate(
    experiment_name: str,
    model_name: str,
    device: str,
    dtype_name: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Generate a continuation from one custom model."""
    experiment = get_experiment(experiment_name)
    profile = get_profile(model_name)
    path = profile.snapshot_path()
    config = LlamaConfig.from_file(path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)
    model = experiment.load_model(profile, device, DTYPES[dtype_name])

    output_ids = experiment.generate(
        model,
        input_ids,
        max_new_tokens,
        eos_token_ids=config.eos_token_ids,
    )
    prompt_tokens = input_ids.shape[1]
    return {
        "experiment": experiment_name,
        "model": model_name,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "dtype": dtype_name,
        "prompt": prompt,
        "input_token_ids": input_ids[0].tolist(),
        "generated_token_ids": output_ids[0, prompt_tokens:].tolist(),
        "text": tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "generated_text": tokenizer.decode(
            output_ids[0, prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
    }
