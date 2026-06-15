"""Device selection helpers for training and inference."""

from __future__ import annotations

import torch


def get_best_device() -> str:
    """
    Detect and return the best available device.

    Priority:
        1. cuda
        2. mps
        3. cpu
    """
    if torch.cuda.is_available():
        return "cuda"

    # MPS is available on Apple Silicon-enabled PyTorch builds.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_device() -> torch.device:
    """Return the selected device as a torch.device object."""
    return torch.device(get_best_device())
