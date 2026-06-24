"""
Weights & Biases experiment tracker for LLM fine-tuning runs.

Tracks per-step and per-epoch metrics:
  - train loss
  - eval loss
  - learning-rate curves
  - GPU utilisation (memory used / total, utilisation %)

Usage
-----
    tracker = WandbTracker(project="gpt2-finetune")
    tracker.start_run(run_name="run-001", config=training_config.to_dict())

    for epoch in range(num_epochs):
        tracker.log_epoch_start(epoch)

        for step, batch_loss in enumerate(train_loop(epoch)):
            tracker.log_step(
                step=global_step,
                loss=batch_loss,
                learning_rate=scheduler.get_last_lr()[0],
            )

        tracker.log_epoch_end(epoch, avg_loss=epoch_avg_loss, eval_loss=val_loss)
        tracker.log_checkpoint(epoch=epoch, checkpoint_path=ckpt_dir)

    tracker.end_run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False
    logger.warning(
        "wandb is not installed. WandbTracker will operate in no-op mode. "
        "Install it with:  pip install wandb"
    )

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover – no NVIDIA driver / pynvml absent
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_metrics() -> Dict[str, float]:
    """Return current GPU utilisation metrics for all visible NVIDIA devices.

    Keys follow the pattern ``gpu/{i}/util_pct`` and ``gpu/{i}/mem_used_mb``.
    Returns an empty dict when pynvml or CUDA is unavailable.
    """
    if not _NVML_AVAILABLE:
        return {}

    metrics: Dict[str, float] = {}
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            prefix = f"gpu/{i}"
            metrics[f"{prefix}/util_pct"] = float(util.gpu)
            metrics[f"{prefix}/mem_used_mb"] = mem.used / (1024 ** 2)
            metrics[f"{prefix}/mem_total_mb"] = mem.total / (1024 ** 2)
            metrics[f"{prefix}/mem_used_pct"] = 100.0 * mem.used / mem.total
    except pynvml.NVMLError as exc:
        logger.debug("GPU metric collection failed: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class WandbTracker:
    """Thin wrapper around the W&B Python client for fine-tuning runs.

    Parameters
    ----------
    project:
        W&B project name (created automatically if it does not exist).
    entity:
        W&B entity (username or team).  Uses the default logged-in entity
        when *None*.
    tags:
        List of string tags attached to the run.
    log_gpu:
        Whether to append GPU utilisation metrics on every ``log_step``
        call.  Requires ``pynvml`` (``pip install pynvml``).
    """

    def __init__(
        self,
        project: str = "llm-finetune",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_gpu: bool = True,
    ) -> None:
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.log_gpu = log_gpu and _NVML_AVAILABLE
        self._run: Optional[Any] = None  # wandb.sdk.wandb_run.Run

        if not _WANDB_AVAILABLE:
            logger.warning("WandbTracker: wandb not available, all calls are no-ops.")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise a W&B run and upload the training config.

        Parameters
        ----------
        run_name:
            Human-readable label shown in the W&B UI.
        config:
            Flat or nested dictionary of hyperparameters logged as the run
            config (e.g. from ``TrainingConfig.to_dict()``).
        """
        if not _WANDB_AVAILABLE:
            return

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=self.tags,
            config=config or {},
            reinit=True,
        )
        logger.info(
            "W&B run started: %s  (project=%s  url=%s)",
            run_name,
            self.project,
            self._run.url if self._run else "n/a",
        )

    def end_run(self) -> None:
        """Finish the active W&B run."""
        if not _WANDB_AVAILABLE or self._run is None:
            return
        self._run.finish()
        self._run = None
        logger.info("W&B run ended.")

    # ------------------------------------------------------------------
    # Step-level logging  (train loss + learning rate + GPU)
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log training metrics for a single optimiser step.

        Metrics logged
        ~~~~~~~~~~~~~~
        * ``train/loss``
        * ``train/learning_rate``  (when provided)
        * ``gpu/{i}/util_pct``, ``gpu/{i}/mem_used_mb``, etc.  (when enabled)

        Parameters
        ----------
        step:
            Global training step index.
        loss:
            Scalar training loss for this step.
        learning_rate:
            Current learning rate from the scheduler.
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return

        metrics: Dict[str, float] = {"train/loss": loss}

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        if self.log_gpu:
            metrics.update(_gpu_metrics())

        wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Epoch-level logging  (train + eval loss, learning curves)
    # ------------------------------------------------------------------

    def log_epoch_start(self, epoch: int) -> None:
        """Record the beginning of an epoch.

        Parameters
        ----------
        epoch:
            Zero-based epoch index.
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return
        wandb.log({"epoch": epoch}, step=epoch)

    def log_epoch_end(
        self,
        epoch: int,
        avg_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log summary metrics at the end of an epoch.

        This method drives the *learning curves* panel in W&B by logging
        ``epoch/train_loss`` and ``epoch/eval_loss`` against the epoch index.

        Metrics logged
        ~~~~~~~~~~~~~~
        * ``epoch/train_loss``  (average training loss over the epoch)
        * ``epoch/eval_loss``   (validation loss after evaluation)
        * Any key/value pairs in *extra_metrics*

        Parameters
        ----------
        epoch:
            Zero-based epoch index (used as the W&B x-axis step).
        avg_loss:
            Mean training loss over all steps in this epoch.
        eval_loss:
            Validation / evaluation loss.
        extra_metrics:
            Additional scalars to log (e.g. perplexity, BLEU score).
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return

        metrics: Dict[str, float] = {"epoch": float(epoch)}

        if avg_loss is not None:
            metrics["epoch/train_loss"] = avg_loss
        if eval_loss is not None:
            metrics["epoch/eval_loss"] = eval_loss
        if extra_metrics:
            metrics.update(extra_metrics)

        wandb.log(metrics, step=epoch)

    # ------------------------------------------------------------------
    # Checkpoint logging
    # ------------------------------------------------------------------

    def log_checkpoint(
        self,
        epoch: int,
        checkpoint_path: Union[str, Path],
        artifact_name: Optional[str] = None,
    ) -> None:
        """Upload a checkpoint directory or file as a W&B Artifact.

        Parameters
        ----------
        epoch:
            Epoch at which the checkpoint was saved.
        checkpoint_path:
            Local path to the checkpoint directory or file.
        artifact_name:
            Name for the W&B Artifact.  Defaults to
            ``checkpoint-epoch-{epoch}``.
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return

        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("log_checkpoint: path does not exist: %s", path)
            return

        name = artifact_name or f"checkpoint-epoch-{epoch}"
        artifact = wandb.Artifact(name=name, type="model")

        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))

        self._run.log_artifact(artifact)
        logger.info("Checkpoint uploaded to W&B: epoch=%d  path=%s", epoch, path)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "WandbTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
