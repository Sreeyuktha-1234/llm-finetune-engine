"""
Tokenizer manager for loading Hugging Face tokenizers and preparing tensors.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TokenizerManager:
    """Load and apply Hugging Face tokenizers for dataset preprocessing."""

    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize the tokenizer manager.

        Args:
            model_name: Hugging Face model/tokenizer identifier.
            max_length: Maximum token sequence length.
            padding: Padding strategy used by tokenizer calls.
            truncation: Whether to truncate sequences to max_length.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Load and return the Hugging Face tokenizer.

        Returns:
            Loaded tokenizer instance.
        """
        if self.tokenizer is not None:
            return self.tokenizer

        logger.info("Loading tokenizer: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # GPT-style tokenizers may not define a pad token by default.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        return tokenizer

    def tokenize_texts(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a sequence of texts and return PyTorch tensors.

        Args:
            texts: Sequence of text strings.
            max_length: Optional per-call max length override.

        Returns:
            Dict with tensor keys such as input_ids and attention_mask.
        """
        tokenizer = self.load_tokenizer()
        effective_max_length = max_length or self.max_length

        encodings = tokenizer(
            list(texts),
            padding=self.padding,
            truncation=self.truncation,
            max_length=effective_max_length,
            return_tensors="pt",
        )
        return encodings

    def tokenize_dataset(
        self,
        samples: Sequence[Dict],
        text_key: str = "text",
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize dataset samples by extracting text values.

        Args:
            samples: Sequence of dataset sample dictionaries.
            text_key: Dictionary key containing sample text.
            max_length: Optional per-call max length override.

        Returns:
            Dict of PyTorch tensors.
        """
        texts = [str(sample.get(text_key, "")) for sample in samples]
        return self.tokenize_texts(texts, max_length=max_length)

    def save_tensors(self, tensors: Dict[str, torch.Tensor], output_path: Path) -> None:
        """
        Save tokenized tensor dictionary to disk.

        Args:
            tensors: Tokenized tensor outputs.
            output_path: Destination path for torch serialized file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, output_path)
        logger.info("Saved tokenized tensors to: %s", output_path)

    def get_config(self) -> Dict[str, object]:
        """Return the active tokenizer configuration."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
        }
