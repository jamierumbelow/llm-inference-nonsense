"""Published hardware and pricing data used to contextualize benchmark results."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class GpuSpec:
    modal_name: str
    memory_capacity_bytes: int
    memory_bandwidth_bytes_per_second: int
    peak_dense_bf16_flops_per_second: int
    peak_sparse_bf16_flops_per_second: int
    modal_gpu_usd_per_second: float
    pricing_as_of: str
    pricing_source: str
    hardware_source: str

    def report(self) -> dict:
        values = asdict(self)
        values["modal_gpu_usd_per_hour"] = self.modal_gpu_usd_per_second * 3_600
        return values


A100_40GB = GpuSpec(
    modal_name="A100-40GB",
    memory_capacity_bytes=40_000_000_000,
    memory_bandwidth_bytes_per_second=1_555_000_000_000,
    peak_dense_bf16_flops_per_second=312_000_000_000_000,
    peak_sparse_bf16_flops_per_second=624_000_000_000_000,
    modal_gpu_usd_per_second=0.000583,
    pricing_as_of="2026-09-06",
    pricing_source="https://modal.com/pricing",
    hardware_source=(
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
        "a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
    ),
)

GPU_SPECS = {A100_40GB.modal_name: A100_40GB}


def get_gpu_spec(name: str) -> GpuSpec:
    try:
        return GPU_SPECS[name]
    except KeyError:
        known = ", ".join(GPU_SPECS)
        raise KeyError(f"unknown GPU spec {name!r}; known specs: {known}") from None


def actual_gpu_hardware(device: str) -> dict | None:
    if device != "cuda":
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": properties.multi_processor_count,
    }
