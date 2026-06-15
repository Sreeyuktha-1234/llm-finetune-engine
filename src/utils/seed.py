"""Utilities for setting reproducible random seeds."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seeds for random number generators used in training.

    Args:
        seed: Seed value used across libraries.
        deterministic: Whether to force deterministic CUDA behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enforce deterministic algorithms where supported.
        torch.use_deterministic_algorithms(True, warn_only=True)
