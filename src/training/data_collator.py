"""
Dynamic batch collation for causal language model training.

This collator handles:
- Dynamic padding to the longest sequence in each batch.
- Attention mask generation via tokenizer padding.
- Label construction for causal LM training.
- Ignore index masking for padded label positions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class CausalLMDataCollator:
    """Data collator for causal LM training with dynamic padding."""

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self) -> None:
        # Ensure we have a valid padding token for dynamic padding.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has no pad_token_id and no eos_token to reuse as pad token."
                )

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate tokenized samples into a padded batch for causal LM.

        Expected each feature to contain at least `input_ids`.
        Optional `labels` are supported. If labels are absent, labels are copied
        from `input_ids` and then padding positions are masked with `ignore_index`.
        """
        if not features:
            raise ValueError("Cannot collate an empty batch.")

        mutable_features: List[Dict[str, Any]] = []
        raw_labels: List[Optional[List[int]]] = []

        for feature in features:
            item = dict(feature)
            label_value = item.pop("labels", None)
            raw_labels.append(self._to_list(label_value))
            mutable_features.append(item)

        batch = self.tokenizer.pad(
            mutable_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        if any(label is not None for label in raw_labels):
            labels = self._build_labels_from_features(batch, raw_labels)
        else:
            labels = batch["input_ids"].clone()

        labels = labels.masked_fill(batch["attention_mask"] == 0, self.ignore_index)
        batch["labels"] = labels
        return batch

    def _build_labels_from_features(
        self,
        batch: Dict[str, torch.Tensor],
        raw_labels: Sequence[Optional[List[int]]],
    ) -> torch.Tensor:
        """Pad optional per-sample labels to batch sequence length."""
        seq_len = int(batch["input_ids"].shape[1])
        padding_side = self.tokenizer.padding_side

        padded_labels: List[List[int]] = []
        for index, label in enumerate(raw_labels):
            values = label
            if values is None:
                values = batch["input_ids"][index].tolist()

            if len(values) > seq_len:
                values = values[:seq_len] if padding_side == "right" else values[-seq_len:]

            pad_len = seq_len - len(values)
            if padding_side == "right":
                values = values + [self.ignore_index] * pad_len
            else:
                values = [self.ignore_index] * pad_len + values

            padded_labels.append(values)

        return torch.tensor(padded_labels, dtype=torch.long)

    @staticmethod
    def _to_list(value: Any) -> Optional[List[int]]:
        """Normalize tensor/list labels to a plain Python list of ints."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return list(value)
