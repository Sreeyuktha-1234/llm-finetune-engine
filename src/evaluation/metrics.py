"""Evaluation metrics for language-model fine-tuning."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from mean cross-entropy loss."""
    # Clamp to avoid overflow for very large losses.
    return float(math.exp(min(loss, 100.0)))


def token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute token-level accuracy for logits and labels.

    Args:
        logits: Tensor shaped (batch, seq_len, vocab_size).
        labels: Tensor shaped (batch, seq_len).
        ignore_index: Label value excluded from accuracy.
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits shape (batch, seq_len, vocab_size)")
    if labels.ndim != 2:
        raise ValueError("Expected labels shape (batch, seq_len)")

    preds = torch.argmax(logits, dim=-1)
    valid_mask = labels != ignore_index

    valid_tokens = valid_mask.sum().item()
    if valid_tokens == 0:
        return 0.0

    correct = ((preds == labels) & valid_mask).sum().item()
    return float(correct / valid_tokens)


def causal_lm_loss_and_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> tuple[float, float]:
    """Compute causal-LM cross-entropy loss and token accuracy.

    The labels are shifted left by one token to match next-token prediction.
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits shape (batch, seq_len, vocab_size)")
    if labels.ndim != 2:
        raise ValueError("Expected labels shape (batch, seq_len)")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, ignore_index)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )

    acc = token_accuracy(shift_logits, shift_labels, ignore_index=ignore_index)
    return float(loss.item()), float(acc)
