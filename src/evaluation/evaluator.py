"""Evaluation helpers for language model validation."""

from __future__ import annotations

from typing import Dict

import torch

from src.evaluation.metrics import (
    causal_lm_loss_and_accuracy,
    perplexity_from_loss,
)


class Evaluator:
    """Evaluate a causal language model on a tokenized dataloader."""

    def __init__(self, device: str | None = None, ignore_index: int = -100):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ignore_index = ignore_index

    def evaluate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Run evaluation and return aggregate loss, perplexity, and accuracy.

        Expects each batch to contain:
        - input_ids
        - labels (optional; falls back to input_ids)
        - attention_mask (optional)
        """
        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_accuracy = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels", input_ids)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                loss, acc = causal_lm_loss_and_accuracy(
                    logits=logits,
                    labels=labels,
                    attention_mask=attention_mask,
                    ignore_index=self.ignore_index,
                )

                total_loss += loss
                total_accuracy += acc
                steps += 1

        if steps == 0:
            return {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}

        mean_loss = total_loss / steps
        mean_accuracy = total_accuracy / steps

        return {
            "loss": float(mean_loss),
            "perplexity": float(perplexity_from_loss(mean_loss)),
            "accuracy": float(mean_accuracy),
        }
