"""Device selection migrated unchanged from the previous local dependency."""

from __future__ import annotations

import re


_CUDA_DEVICE = re.compile(r"^cuda(?::(?P<index>\d+))?$")


def resolve_learning_device(
    requested: str | None,
    *,
    cuda_available: bool,
    cuda_device_count: int,
) -> str:
    """Resolve a device without importing Torch or silently hiding GPU errors."""
    device = (requested or "auto").strip().lower()
    if device == "auto":
        return "cuda" if cuda_available and cuda_device_count > 0 else "cpu"
    if device == "cpu":
        return "cpu"
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None:
        raise ValueError(
            f"Unsupported learning device '{requested}'. "
            "Use auto, cpu, cuda, or cuda:N."
        )
    if not cuda_available or cuda_device_count <= 0:
        raise RuntimeError(
            f"Learning device '{device}' requested, but PyTorch cannot access a CUDA GPU. "
            "Start Docker with --gpus all and verify the NVIDIA Container Toolkit."
        )
    index = int(match.group("index") or 0)
    if index >= cuda_device_count:
        raise RuntimeError(
            f"Learning device '{device}' requested, but only {cuda_device_count} "
            "CUDA device(s) are visible inside the container."
        )
    return device
