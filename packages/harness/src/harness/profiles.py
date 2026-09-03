"""Named model profiles.

A profile is the unit that a run, a result file, and a download command refer
to. Code never mentions a Hugging Face repo id directly; it asks for a profile
by name so the mapping lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    repo_id: str
    params_b: float
    """Parameter count in billions, for FLOPs and bandwidth estimates."""
    notes: str = ""


PROFILES: dict[str, ModelProfile] = {
    p.name: p
    for p in (
        ModelProfile(
            name="dev",
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            params_b=1.24,
            notes=(
                "Runs on the laptop (MPS/CPU). Same architecture as the target apart from "
                "tied input/output embeddings. Doubles as the speculative-decoding draft."
            ),
        ),
        ModelProfile(
            name="target",
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            params_b=8.03,
            notes="The model every experiment is measured on. Needs a GPU box.",
        ),
    )
}


def get_profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(PROFILES)
        raise KeyError(f"unknown model profile {name!r}; known profiles: {known}") from None
