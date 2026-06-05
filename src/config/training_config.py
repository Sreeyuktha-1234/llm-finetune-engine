"""Training configuration for full fine-tuning, LoRA, and QLoRA."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    """Shared training hyperparameters and output paths."""

    # Core optimization settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    seed: int = 42

    # Save locations for each training strategy
    output_dir_full: Path = Path("outputs/checkpoints")
    output_dir_lora: Path = Path("outputs/lora_checkpoints")
    output_dir_qlora: Path = Path("outputs/qlora_checkpoints")

    # LoRA-specific defaults
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn"])

    # QLoRA-specific defaults
    qlora_r: int = 64
    qlora_alpha: int = 16
    qlora_dropout: float = 0.05
    qlora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    # Quantization defaults
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    def to_dict(self) -> dict:
        """Serialize config for logging/checkpoint metadata."""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "adam_epsilon": self.adam_epsilon,
            "seed": self.seed,
            "output_dir_full": str(self.output_dir_full),
            "output_dir_lora": str(self.output_dir_lora),
            "output_dir_qlora": str(self.output_dir_qlora),
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "qlora_r": self.qlora_r,
            "qlora_alpha": self.qlora_alpha,
            "qlora_dropout": self.qlora_dropout,
            "qlora_target_modules": self.qlora_target_modules,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }
