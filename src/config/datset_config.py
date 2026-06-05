"""Dataset configuration for data paths and split names."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Central dataset path and split configuration."""

    # Data locations
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    dataset_file: str = "dataset.json"

    # Split names
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

    @property
    def dataset_path(self) -> Path:
        """Return absolute-ish dataset JSON path relative to repo root."""
        return self.raw_data_dir / self.dataset_file

    def to_dict(self) -> dict:
        """Serialize config for logging/checkpoint metadata."""
        return {
            "raw_data_dir": str(self.raw_data_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "dataset_file": self.dataset_file,
            "dataset_path": str(self.dataset_path),
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
        }
