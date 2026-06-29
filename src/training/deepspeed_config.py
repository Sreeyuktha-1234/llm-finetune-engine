"""
DeepSpeed configuration module for LLM fine-tuning.
Provides ZeRO Stage 2 and ZeRO Stage 3 configuration builders.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Base parameters shared across ZeRO stages
# ---------------------------------------------------------------------------

@dataclass
class DeepSpeedBaseParams:
    """Common parameters shared by all ZeRO stages."""

    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    gradient_accumulation_steps: str = "auto"
    gradient_clipping: float = 1.0

    # Mixed precision
    fp16_enabled: bool = True
    fp16_loss_scale: int = 0                  # 0 = dynamic loss scaling
    fp16_loss_scale_window: int = 1000
    fp16_initial_scale_power: int = 16
    fp16_hysteresis: int = 2
    fp16_min_loss_scale: int = 1

    bf16_enabled: bool = False                # mutually exclusive with fp16

    # Optimizer (offload-aware; set by stage builder when needed)
    optimizer_type: str = "AdamW"
    optimizer_lr: str = "auto"
    optimizer_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2

    # Scheduler
    scheduler_type: str = "WarmupDecayLR"
    scheduler_warmup_min_lr: float = 0.0
    scheduler_warmup_max_lr: str = "auto"
    scheduler_warmup_num_steps: str = "auto"
    scheduler_total_num_steps: str = "auto"


# ---------------------------------------------------------------------------
# ZeRO Stage 2
# ---------------------------------------------------------------------------

def get_zero_stage2_config(
    base: Optional[DeepSpeedBaseParams] = None,
    *,
    allgather_partitions: bool = True,
    allgather_bucket_size: int = 5_000_000_00,
    overlap_comm: bool = True,
    reduce_scatter: bool = True,
    reduce_bucket_size: int = 5_000_000_00,
    contiguous_gradients: bool = True,
    # CPU offload
    offload_optimizer: bool = False,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """
    Build a DeepSpeed ZeRO Stage 2 configuration dictionary.

    ZeRO Stage 2 partitions **gradients** and **optimizer states** across
    data-parallel ranks, while keeping model parameters replicated.

    Args:
        base:                   Shared base parameters. Defaults to
                                :class:`DeepSpeedBaseParams` with default values.
        allgather_partitions:   Gather all-reduce partitions instead of one
                                partition at a time.
        allgather_bucket_size:  Byte-size of each all-gather bucket.
        overlap_comm:           Overlap backward-pass gradient communication
                                with computation.
        reduce_scatter:         Use reduce-scatter instead of all-reduce.
        reduce_bucket_size:     Byte-size of each reduce bucket.
        contiguous_gradients:   Copy gradients to a contiguous buffer before
                                reduction (reduces memory fragmentation).
        offload_optimizer:      Offload optimizer state and computation to CPU.
        pin_memory:             Pin CPU memory when ``offload_optimizer=True``.

    Returns:
        A ``dict`` ready to be serialised to ``ds_config.json`` or passed
        directly to a HuggingFace ``TrainingArguments``.
    """
    if base is None:
        base = DeepSpeedBaseParams()

    optimizer_offload: Dict[str, Any] = {}
    if offload_optimizer:
        optimizer_offload = {
            "device": "cpu",
            "pin_memory": pin_memory,
        }

    config: Dict[str, Any] = {
        "train_batch_size": base.train_batch_size,
        "train_micro_batch_size_per_gpu": base.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": base.gradient_accumulation_steps,
        "gradient_clipping": base.gradient_clipping,

        "fp16": {
            "enabled": base.fp16_enabled and not base.bf16_enabled,
            "loss_scale": base.fp16_loss_scale,
            "loss_scale_window": base.fp16_loss_scale_window,
            "initial_scale_power": base.fp16_initial_scale_power,
            "hysteresis": base.fp16_hysteresis,
            "min_loss_scale": base.fp16_min_loss_scale,
        },

        "bf16": {
            "enabled": base.bf16_enabled,
        },

        "optimizer": {
            "type": base.optimizer_type,
            "params": {
                "lr": base.optimizer_lr,
                "betas": list(base.optimizer_betas),
                "eps": base.optimizer_eps,
                "weight_decay": base.optimizer_weight_decay,
            },
        },

        "scheduler": {
            "type": base.scheduler_type,
            "params": {
                "warmup_min_lr": base.scheduler_warmup_min_lr,
                "warmup_max_lr": base.scheduler_warmup_max_lr,
                "warmup_num_steps": base.scheduler_warmup_num_steps,
                "total_num_steps": base.scheduler_total_num_steps,
            },
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": allgather_partitions,
            "allgather_bucket_size": allgather_bucket_size,
            "overlap_comm": overlap_comm,
            "reduce_scatter": reduce_scatter,
            "reduce_bucket_size": reduce_bucket_size,
            "contiguous_gradients": contiguous_gradients,
            **({"offload_optimizer": optimizer_offload} if offload_optimizer else {}),
        },

        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    return config


# ---------------------------------------------------------------------------
# ZeRO Stage 3
# ---------------------------------------------------------------------------

def get_zero_stage3_config(
    base: Optional[DeepSpeedBaseParams] = None,
    *,
    # Parameter sharding
    stage3_max_live_parameters: int = 1_000_000_000,
    stage3_max_reuse_distance: int = 1_000_000_000,
    stage3_prefetch_bucket_size: int = 500_000_000,
    stage3_param_persistence_threshold: int = 1_000_000,
    stage3_gather_16bit_weights_on_model_save: bool = True,
    # Gradient buckets
    reduce_bucket_size: int = 500_000_000,
    contiguous_gradients: bool = True,
    overlap_comm: bool = True,
    # CPU / NVMe offload
    offload_optimizer: bool = False,
    offload_param: bool = False,
    pin_memory: bool = True,
    nvme_path: str = "/local_nvme",
) -> Dict[str, Any]:
    """
    Build a DeepSpeed ZeRO Stage 3 configuration dictionary.

    ZeRO Stage 3 partitions **model parameters**, **gradients**, and
    **optimizer states** across all data-parallel ranks, enabling training
    of models that exceed a single GPU's memory.

    Args:
        base:                                      Shared base parameters.
        stage3_max_live_parameters:                Max number of parameters
                                                   resident in GPU memory at
                                                   any time.
        stage3_max_reuse_distance:                 Parameters are kept in
                                                   memory if they will be
                                                   reused within this many
                                                   elements.
        stage3_prefetch_bucket_size:               Byte-size of each
                                                   prefetch bucket for
                                                   forward-pass parameter
                                                   gathering.
        stage3_param_persistence_threshold:        Parameters smaller than
                                                   this are kept permanently
                                                   on GPU.
        stage3_gather_16bit_weights_on_model_save: Gather full fp16 weights
                                                   before ``save_pretrained``.
        reduce_bucket_size:                        Byte-size of each gradient
                                                   reduce bucket.
        contiguous_gradients:                      Use contiguous gradient
                                                   buffer.
        overlap_comm:                              Overlap communication with
                                                   computation.
        offload_optimizer:                         Offload optimizer state to
                                                   CPU (or NVMe).
        offload_param:                             Offload model parameters to
                                                   CPU (or NVMe).
        pin_memory:                                Pin CPU memory for offload
                                                   buffers.
        nvme_path:                                 Local NVMe path used when
                                                   ``offload_*`` targets
                                                   ``"nvme"``.

    Returns:
        A ``dict`` ready to be serialised to ``ds_config.json`` or passed
        directly to a HuggingFace ``TrainingArguments``.
    """
    if base is None:
        base = DeepSpeedBaseParams()

    # Build optional offload sub-sections only when requested so that the
    # resulting JSON stays clean when offloading is disabled.
    optimizer_offload: Dict[str, Any] = {}
    if offload_optimizer:
        optimizer_offload = {
            "device": "cpu",
            "pin_memory": pin_memory,
        }

    param_offload: Dict[str, Any] = {}
    if offload_param:
        param_offload = {
            "device": "cpu",
            "pin_memory": pin_memory,
        }

    config: Dict[str, Any] = {
        "train_batch_size": base.train_batch_size,
        "train_micro_batch_size_per_gpu": base.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": base.gradient_accumulation_steps,
        "gradient_clipping": base.gradient_clipping,

        "fp16": {
            "enabled": base.fp16_enabled and not base.bf16_enabled,
            "loss_scale": base.fp16_loss_scale,
            "loss_scale_window": base.fp16_loss_scale_window,
            "initial_scale_power": base.fp16_initial_scale_power,
            "hysteresis": base.fp16_hysteresis,
            "min_loss_scale": base.fp16_min_loss_scale,
        },

        "bf16": {
            "enabled": base.bf16_enabled,
        },

        "optimizer": {
            "type": base.optimizer_type,
            "params": {
                "lr": base.optimizer_lr,
                "betas": list(base.optimizer_betas),
                "eps": base.optimizer_eps,
                "weight_decay": base.optimizer_weight_decay,
            },
        },

        "scheduler": {
            "type": base.scheduler_type,
            "params": {
                "warmup_min_lr": base.scheduler_warmup_min_lr,
                "warmup_max_lr": base.scheduler_warmup_max_lr,
                "warmup_num_steps": base.scheduler_warmup_num_steps,
                "total_num_steps": base.scheduler_total_num_steps,
            },
        },

        "zero_optimization": {
            "stage": 3,
            "overlap_comm": overlap_comm,
            "contiguous_gradients": contiguous_gradients,
            "reduce_bucket_size": reduce_bucket_size,
            "stage3_prefetch_bucket_size": stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
            "stage3_max_live_parameters": stage3_max_live_parameters,
            "stage3_max_reuse_distance": stage3_max_reuse_distance,
            "stage3_gather_16bit_weights_on_model_save": stage3_gather_16bit_weights_on_model_save,
            **({"offload_optimizer": optimizer_offload} if offload_optimizer else {}),
            **({"offload_param": param_offload} if offload_param else {}),
        },

        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    return config


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_deepspeed_config(
    stage: int,
    base: Optional[DeepSpeedBaseParams] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Return a ZeRO configuration dict for the requested *stage*.

    Args:
        stage:   ZeRO stage to use. Supported values: ``2``, ``3``.
        base:    Optional :class:`DeepSpeedBaseParams` instance.
        **kwargs: Extra keyword arguments forwarded verbatim to the
                 stage-specific builder function.

    Raises:
        ValueError: If *stage* is not ``2`` or ``3``.

    Returns:
        DeepSpeed configuration dictionary.
    """
    if stage == 2:
        return get_zero_stage2_config(base, **kwargs)
    if stage == 3:
        return get_zero_stage3_config(base, **kwargs)
    raise ValueError(
        f"Unsupported ZeRO stage: {stage!r}. Choose 2 or 3."
    )
