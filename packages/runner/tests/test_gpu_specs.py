import pytest

from runner.gpu_specs import A100_40GB, validate_gpu_hardware


def test_gpu_guard_accepts_the_requested_hardware() -> None:
    validate_gpu_hardware(
        A100_40GB,
        {
            "name": "NVIDIA A100-SXM4-40GB",
            "total_memory_bytes": 42_405_855_232,
        },
    )


def test_gpu_guard_rejects_a_modal_upgrade() -> None:
    with pytest.raises(RuntimeError, match=r"requested A100-40GB.*A100-SXM4-80GB"):
        validate_gpu_hardware(
            A100_40GB,
            {
                "name": "NVIDIA A100-SXM4-80GB",
                "total_memory_bytes": 85_094_825_984,
            },
        )
