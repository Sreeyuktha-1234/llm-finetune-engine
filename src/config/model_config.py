"""Model configuration for fine-tuning workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Central model settings shared by training and inference."""

    # Hugging Face model ID or local path
    model_name: str = "gpt2"

    # Optional path to a local checkpoint to resume from
    checkpoint_path: Optional[Path] = None

    # Auto-detect device unless explicitly set ("cpu" or "cuda")
    device: Optional[str] = None

    # Tokenization defaults
    max_length: int = 128

    def to_dict(self) -> dict:
        """Serialize config for logging/checkpoint metadata."""
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": self.device,
            "max_length": self.max_length,
        }
