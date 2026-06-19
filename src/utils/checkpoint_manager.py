"""Utilities for saving, loading, and resuming training checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class CheckpointManager:
    """Handle checkpoint persistence for model and training state."""

    STATE_FILE_NAME = "training_state.pt"
    METADATA_FILE_NAME = "metadata.json"

    @staticmethod
    def save_checkpoint(
        output_dir: PathLike,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a full training checkpoint.

        Saved artifacts:
        - ``training_state.pt`` with model/optimizer/scheduler/scaler state.
        - ``metadata.json`` for human-readable summary.
        - HuggingFace model/tokenizer via ``save_pretrained`` if available.

        Returns:
            Path to the created checkpoint directory.
        """
        base_dir = Path(output_dir)
        checkpoint_dir = base_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": metrics or {},
        }

        if hasattr(model, "state_dict"):
            state["model_state_dict"] = model.state_dict()
        else:
            raise ValueError("Model must provide state_dict() for checkpoint saving.")

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None and hasattr(scaler, "state_dict"):
            state["scaler_state_dict"] = scaler.state_dict()

        state_path = checkpoint_dir / CheckpointManager.STATE_FILE_NAME
        torch.save(state, state_path)

        summary: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "state_file": CheckpointManager.STATE_FILE_NAME,
            "has_optimizer_state": optimizer is not None,
            "has_scheduler_state": scheduler is not None,
            "has_scaler_state": scaler is not None,
            "metrics": metrics or {},
        }
        if metadata:
            summary.update(metadata)

        metadata_path = checkpoint_dir / CheckpointManager.METADATA_FILE_NAME
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(checkpoint_dir))

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(checkpoint_dir))

        logger.info("Checkpoint saved at %s", checkpoint_dir)
        return checkpoint_dir

    @staticmethod
    def load_checkpoint(
        checkpoint_path: PathLike,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        scaler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load training state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or state file.
            model: Model instance to restore.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            map_location: Device mapping for torch.load.
            scaler: Optional gradient scaler to restore.
            strict: Passed to model.load_state_dict.

        Returns:
            A dictionary with at least ``epoch``, ``global_step``, and ``metrics``.
        """
        state_file = CheckpointManager._resolve_state_file(checkpoint_path)
        if not state_file.exists():
            raise FileNotFoundError(f"Checkpoint state file not found: {state_file}")

        state = torch.load(state_file, map_location=map_location)

        model_state = state.get("model_state_dict")
        if model_state is None:
            raise KeyError("Checkpoint does not contain model_state_dict.")
        model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            opt_state = state.get("optimizer_state_dict")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

        if scheduler is not None:
            sched_state = state.get("scheduler_state_dict")
            if sched_state is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(sched_state)

        if scaler is not None:
            scaler_state = state.get("scaler_state_dict")
            if scaler_state is not None and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(scaler_state)

        loaded = {
            "epoch": int(state.get("epoch", 0)),
            "global_step": int(state.get("global_step", 0)),
            "metrics": state.get("metrics", {}),
            "checkpoint_path": str(state_file.parent),
        }

        logger.info(
            "Checkpoint loaded from %s (epoch=%d, global_step=%d)",
            state_file,
            loaded["epoch"],
            loaded["global_step"],
        )
        return loaded

    @staticmethod
    def resume_training(
        output_dir: PathLike,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[PathLike] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        scaler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Resume training from a specific or latest checkpoint.

        Args:
            output_dir: Directory that contains checkpoints.
            model: Model instance to restore.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            checkpoint_path: Explicit checkpoint path. If omitted, latest is used.
            map_location: Device mapping for torch.load.
            scaler: Optional gradient scaler to restore.
            strict: Passed to model.load_state_dict.

        Returns:
            Dictionary containing loaded state and resume indices:
            - epoch
            - global_step
            - start_epoch
            - checkpoint_path
            - metrics
        """
        if checkpoint_path is None:
            latest = CheckpointManager.find_latest_checkpoint(output_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"No checkpoints found in output directory: {output_dir}"
                )
            checkpoint_path = latest

        loaded = CheckpointManager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            scaler=scaler,
            strict=strict,
        )
        loaded["start_epoch"] = loaded["epoch"] + 1
        return loaded

    @staticmethod
    def find_latest_checkpoint(output_dir: PathLike) -> Optional[Path]:
        """Find the most recent checkpoint directory in the output folder."""
        base_dir = Path(output_dir)
        if not base_dir.exists():
            return None

        candidates = []
        for checkpoint_dir in base_dir.glob("checkpoint-epoch-*"):
            if not checkpoint_dir.is_dir():
                continue

            epoch = CheckpointManager._extract_epoch(checkpoint_dir.name)
            state_file = checkpoint_dir / CheckpointManager.STATE_FILE_NAME
            if epoch is not None and state_file.exists():
                candidates.append((epoch, checkpoint_dir))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

        # Fallback: choose newest folder that contains a state file.
        fallback_dirs = [
            p
            for p in base_dir.iterdir()
            if p.is_dir() and (p / CheckpointManager.STATE_FILE_NAME).exists()
        ]
        if not fallback_dirs:
            return None

        return max(fallback_dirs, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _resolve_state_file(checkpoint_path: PathLike) -> Path:
        """Resolve state file from a checkpoint directory or direct file path."""
        path = Path(checkpoint_path)
        if path.is_dir():
            return path / CheckpointManager.STATE_FILE_NAME
        return path

    @staticmethod
    def _extract_epoch(name: str) -> Optional[int]:
        """Extract epoch integer from names like checkpoint-epoch-3."""
        prefix = "checkpoint-epoch-"
        if not name.startswith(prefix):
            return None
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)


def save_checkpoint(*args, **kwargs) -> Path:
    """Module-level alias for CheckpointManager.save_checkpoint."""
    return CheckpointManager.save_checkpoint(*args, **kwargs)


def load_checkpoint(*args, **kwargs) -> Dict[str, Any]:
    """Module-level alias for CheckpointManager.load_checkpoint."""
    return CheckpointManager.load_checkpoint(*args, **kwargs)


def resume_training(*args, **kwargs) -> Dict[str, Any]:
    """Module-level alias for CheckpointManager.resume_training."""
    return CheckpointManager.resume_training(*args, **kwargs)
