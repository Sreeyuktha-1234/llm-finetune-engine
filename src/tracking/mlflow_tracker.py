"""
MLflow experiment tracker for LLM fine-tuning runs.

Tracks per-step and per-epoch metrics (loss, learning rate),
epoch boundaries, and checkpoint paths.

Usage
-----
    tracker = MLflowTracker(experiment_name="gpt2-finetune")
    tracker.start_run(run_name="run-001", config=training_config.to_dict())

    for epoch in range(num_epochs):
        tracker.log_epoch_start(epoch)

        for step, batch_loss in enumerate(train_loop(epoch)):
            tracker.log_step(
                step=global_step,
                loss=batch_loss,
                learning_rate=scheduler.get_last_lr()[0],
            )

        tracker.log_epoch_end(epoch, avg_loss=epoch_avg_loss)
        tracker.log_checkpoint(epoch=epoch, checkpoint_path=ckpt_dir)

    tracker.end_run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "mlflow is not installed. MLflowTracker will operate in no-op mode. "
        "Install it with:  pip install mlflow"
    )


class MLflowTracker:
    """Thin wrapper around the MLflow Python client for fine-tuning runs.

    Parameters
    ----------
    experiment_name:
        Name of the MLflow experiment (created if it does not exist).
    tracking_uri:
        MLflow tracking server URI.  Defaults to the local ``mlruns/``
        directory (``mlflow.get_tracking_uri()``).
    tags:
        Optional key-value tags attached to every run in this experiment.
    """

    def __init__(
        self,
        experiment_name: str = "llm-finetune",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self._active = False

        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflowTracker: mlflow not available, all calls are no-ops.")
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start an MLflow run and log the training config as parameters.

        Parameters
        ----------
        run_name:
            Human-readable label shown in the MLflow UI.
        config:
            Flat dictionary of hyperparameters (e.g. from
            ``TrainingConfig.to_dict()``).  Each key is logged as an
            MLflow param.
        """
        if not _MLFLOW_AVAILABLE:
            return

        mlflow.start_run(run_name=run_name, tags=self.tags)
        self._active = True
        logger.info("MLflow run started: %s (experiment=%s)", run_name, self.experiment_name)

        if config:
            # MLflow params must be strings and ≤ 500 chars
            safe_params = {k: str(v)[:500] for k, v in config.items()}
            mlflow.log_params(safe_params)

    def end_run(self) -> None:
        """Finish the active MLflow run."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        mlflow.end_run()
        self._active = False
        logger.info("MLflow run ended.")

    # ------------------------------------------------------------------
    # Step-level logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log training metrics for a single optimiser step.

        Parameters
        ----------
        step:
            Global training step index (used as the MLflow ``step``).
        loss:
            Scalar training loss for this step.
        learning_rate:
            Current learning rate from the scheduler.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        metrics: Dict[str, float] = {"train/loss": loss}
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        mlflow.log_metrics(metrics, step=step)

    # ------------------------------------------------------------------
    # Epoch-level logging
    # ------------------------------------------------------------------

    def log_epoch_start(self, epoch: int) -> None:
        """Record the beginning of an epoch (logged as a metric flag).

        Parameters
        ----------
        epoch:
            Zero-based epoch index.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        mlflow.log_metric("epoch", float(epoch), step=epoch)

    def log_epoch_end(
        self,
        epoch: int,
        avg_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log summary metrics at the end of an epoch.

        Parameters
        ----------
        epoch:
            Zero-based epoch index.
        avg_loss:
            Mean training loss over the epoch.
        eval_loss:
            Validation loss, if evaluation was performed.
        extra_metrics:
            Any additional scalar metrics to log (e.g. perplexity).
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        metrics: Dict[str, float] = {}
        if avg_loss is not None:
            metrics["epoch/train_loss"] = avg_loss
        if eval_loss is not None:
            metrics["epoch/eval_loss"] = eval_loss
        if extra_metrics:
            metrics.update(extra_metrics)

        if metrics:
            mlflow.log_metrics(metrics, step=epoch)

    # ------------------------------------------------------------------
    # Checkpoint logging
    # ------------------------------------------------------------------

    def log_checkpoint(
        self,
        epoch: int,
        checkpoint_path: Union[str, Path],
    ) -> None:
        """Record a saved checkpoint as an MLflow artifact.

        The checkpoint directory (or file) is uploaded to the artifact
        store under ``checkpoints/checkpoint-epoch-{epoch}/``.

        Parameters
        ----------
        epoch:
            Epoch at which the checkpoint was saved.
        checkpoint_path:
            Local path to the checkpoint directory or file produced by
            :class:`~src.utils.checkpoint_manager.CheckpointManager`.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        path = Path(checkpoint_path)
        artifact_subdir = f"checkpoints/checkpoint-epoch-{epoch}"

        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_subdir)
        elif path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
        else:
            logger.warning("log_checkpoint: path does not exist: %s", path)
            return

        logger.info("Checkpoint logged to MLflow: epoch=%d  path=%s", epoch, path)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
