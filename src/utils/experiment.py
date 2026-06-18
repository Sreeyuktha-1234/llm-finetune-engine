"""Utilities for managing experiment output directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


_EXPERIMENT_DIR_PATTERN = re.compile(r"^experiment_(\d{3})$")


def _next_experiment_name(outputs_root: Path) -> str:
    """Compute the next experiment folder name under outputs root."""
    highest_index = 0

    for child in outputs_root.iterdir():
        if not child.is_dir():
            continue

        match = _EXPERIMENT_DIR_PATTERN.match(child.name)
        if match:
            highest_index = max(highest_index, int(match.group(1)))

    return f"experiment_{highest_index + 1:03d}"


def create_experiment_dirs(
    outputs_root: str | Path = "outputs",
    experiment_name: Optional[str] = None,
) -> dict[str, str | Path]:
    """
    Create an experiment folder with standard subdirectories.

    The generated structure is:
        outputs/<experiment_name>/checkpoints
        outputs/<experiment_name>/logs
        outputs/<experiment_name>/metrics
    """
    root_path = Path(outputs_root)
    root_path.mkdir(parents=True, exist_ok=True)

    resolved_name = experiment_name or _next_experiment_name(root_path)
    experiment_dir = root_path / resolved_name

    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    metrics_dir = experiment_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiment_name": resolved_name,
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "metrics_dir": metrics_dir,
    }