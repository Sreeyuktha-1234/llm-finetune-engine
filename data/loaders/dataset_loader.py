"""
Dataset loader module for loading and managing datasets for LLM fine-tuning.
Supports loading from raw JSON files and split-aware dataset access.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from data.loaders.formatter import format_instruction_sample, is_instruction_sample

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and manage datasets for LLM fine-tuning."""

    def __init__(
        self,
        data_path: str = "data/raw/dataset.json",
        split: Optional[str] = None,
    ):
        """
        Initialize the dataset loader.

        Args:
            data_path: Path to the JSON dataset file.
            split: Optional split to load ('train', 'validation', 'test').
                   If None, all samples are loaded.
        """
        self.data_path = Path(data_path)
        self.split = split
        self._raw: Optional[Dict] = None
        self._samples: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[Dict]:
        """
        Load dataset from disk and return samples for the requested split.

        Returns:
            List of sample dicts, each containing at minimum 'id' and 'text'.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the requested split is not found in the dataset.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        self._raw = self._read_json()
        all_samples: List[Dict] = self._raw.get("data", [])

        if self.split is not None:
            split_ids = self._get_split_ids(self.split)
            id_set = set(split_ids)
            samples = [s for s in all_samples if s.get("id") in id_set]
            logger.info(
                "Loaded %d samples for split '%s' from %s",
                len(samples),
                self.split,
                self.data_path,
            )
        else:
            samples = all_samples
            logger.info(
                "Loaded %d samples (all splits) from %s",
                len(samples),
                self.data_path,
            )

        samples = self._normalize_samples(samples)
        self._samples = samples
        return samples

    def get_texts(self) -> List[str]:
        """
        Return only the text field from each loaded sample.

        Returns:
            List of text strings.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        self._ensure_loaded()
        return [s["text"] for s in self._samples]  # type: ignore[index]

    def get_categories(self) -> List[str]:
        """
        Return the category labels for each loaded sample.

        Returns:
            List of category strings.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        self._ensure_loaded()
        return [s.get("category", "") for s in self._samples]  # type: ignore[index]

    @property
    def metadata(self) -> Dict:
        """
        Return the dataset metadata block (version, description, etc.).

        Returns:
            Metadata dict, or empty dict if not present.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        self._ensure_loaded()
        return self._raw.get("metadata", {})  # type: ignore[union-attr]

    @property
    def available_splits(self) -> List[str]:
        """
        Return the list of split names defined in the dataset.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        self._ensure_loaded()
        return list(self._raw.get("splits", {}).keys())  # type: ignore[union-attr]

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Dict:
        self._ensure_loaded()
        return self._samples[index]  # type: ignore[index]

    def __iter__(self):
        self._ensure_loaded()
        return iter(self._samples)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_json(self) -> Dict:
        """Read and parse the JSON file from disk."""
        if not self.data_path.exists():
            logger.error("Dataset file not found: %s", self.data_path)
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in dataset file: %s", self.data_path)
            raise

    def _get_split_ids(self, split: str) -> List[int]:
        """Resolve sample IDs for the given split name."""
        splits: Dict = self._raw.get("splits", {})  # type: ignore[union-attr]
        if split not in splits:
            raise ValueError(
                f"Split '{split}' not found. Available splits: {list(splits.keys())}"
            )
        return splits[split]

    def _ensure_loaded(self) -> None:
        """Raise RuntimeError if load() has not been called."""
        if self._samples is None:
            raise RuntimeError("Call load() before accessing dataset contents.")

    def _normalize_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Ensure each sample has a text field for downstream tokenization/training.

        If a sample uses instruction-format keys (instruction/input/output), build
        the SFT text template and store it in sample['text'].
        """
        normalized: List[Dict] = []
        for sample in samples:
            sample_copy = dict(sample)
            if "text" not in sample_copy and is_instruction_sample(sample_copy):
                sample_copy["text"] = format_instruction_sample(sample_copy)
            normalized.append(sample_copy)
        return normalized
