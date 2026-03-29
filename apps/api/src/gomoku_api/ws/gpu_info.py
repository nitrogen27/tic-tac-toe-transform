"""GPU detection via PyTorch CUDA."""

from __future__ import annotations


def get_gpu_info() -> dict:
    """Return GPU info matching the legacy server protocol."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "backend": "cuda",
                "name": props.name,
                "vram": {
                    "total": props.total_mem,
                    "totalMB": round(props.total_mem / 1024 / 1024),
                },
            }
    except ImportError:
        pass
    return {"available": False, "backend": "cpu", "name": "CPU", "vram": {}}
