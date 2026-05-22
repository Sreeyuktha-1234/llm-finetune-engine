"""
Trainer module for fine-tuning LLMs on custom datasets.
Handles training loop, evaluation, checkpointing, and logging.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from src.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Tokenize and wrap a list of text samples for use with DataLoader."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        """
        Args:
            texts:      Raw text strings.
            tokenizer:  HuggingFace tokenizer (already loaded).
            max_length: Maximum token length per sample.
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Fine-tune a causal language model on a text dataset."""

    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "outputs/checkpoints",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        max_length: int = 128,
        warmup_steps: int = 0,
        save_every_n_epochs: int = 1,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_name:          HuggingFace model identifier.
            output_dir:          Directory to save checkpoints and final model.
            num_epochs:          Number of training epochs.
            batch_size:          Training batch size.
            learning_rate:       Optimizer learning rate.
            weight_decay:        L2 weight decay applied to non-bias parameters.
            adam_epsilon:        Epsilon for numerical stability in AdamW.
            max_length:          Max token length passed to the tokenizer.
            warmup_steps:        Linear warmup steps for the LR scheduler.
            save_every_n_epochs: Save a checkpoint every N completed epochs.
            device:              'cuda', 'cpu', or None for auto-detect.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components initialised in setup()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None

        # Metrics history
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []

        logger.info(
            "Trainer created | model=%s device=%s epochs=%d lr=%g wd=%g",
            model_name, self.device, num_epochs, learning_rate, weight_decay,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load model and tokenizer, build optimizer and LR scheduler stub."""
        logger.info("Loading model and tokenizer …")
        loader = ModelLoader(self.model_name, device=self.device)
        self.model, self.tokenizer = loader.load_model_and_tokenizer()
        self.model.train()

        self.optimizer = self._build_optimizer()
        logger.info(
            "Optimizer: AdamW | lr=%g  weight_decay=%g  eps=%g",
            self.learning_rate, self.weight_decay, self.adam_epsilon,
        )

    def _build_optimizer(self) -> AdamW:
        """
        Build an AdamW optimizer with separate parameter groups:

        * Parameters whose names contain 'bias' or belong to LayerNorm layers
          are excluded from weight decay (standard practice for transformers).
        * All other parameters receive the configured weight decay.

        Returns:
            Configured AdamW optimizer.
        """
        assert self.model is not None, "Call setup() before _build_optimizer()."
        # Names of modules whose parameters should NOT be weight-decayed
        no_decay_patterns = ("bias", "LayerNorm.weight", "layer_norm.weight")

        decay_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and not any(pat in name for pat in no_decay_patterns)
        ]
        no_decay_params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad and any(pat in name for pat in no_decay_patterns)
        ]

        logger.info(
            "Parameter groups | with_decay=%d  no_decay=%d",
            len(decay_params), len(no_decay_params),
        )

        param_groups = [
            {"params": decay_params,    "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for a causal LM, ignoring padding tokens.

        The standard causal LM objective shifts labels left by one position so
        that each token predicts the *next* token.  Positions that correspond to
        padding (``attention_mask == 0``) are excluded from the average by
        setting their label to ``-100`` (PyTorch's ``ignore_index`` default).

        Args:
            input_ids:      ``(batch, seq_len)``  — token IDs (used as labels).
            attention_mask: ``(batch, seq_len)``  — 1 for real tokens, 0 for pad.
            logits:         ``(batch, seq_len, vocab_size)`` — raw model output.

        Returns:
            Scalar mean cross-entropy loss over non-padding positions.
        """
        # Shift so token i predicts token i+1
        shift_logits = logits[:, :-1, :].contiguous()          # (B, L-1, V)
        shift_labels = input_ids[:, 1:].contiguous()           # (B, L-1)
        shift_mask   = attention_mask[:, 1:].contiguous()      # (B, L-1)

        # Mask out padding positions
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def _build_dataloader(self, texts: List[str], shuffle: bool = True) -> DataLoader:
        """Tokenize texts and return a DataLoader."""
        assert self.tokenizer is not None, "Call setup() before building a dataloader."
        dataset = TextDataset(texts, self.tokenizer, max_length=self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _init_scheduler(self, total_steps: int) -> None:
        """Create a linear warmup scheduler once total steps are known."""
        assert self.optimizer is not None, "Call setup() before _init_scheduler()."
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

    # ------------------------------------------------------------------
    # Core training / evaluation steps
    # ------------------------------------------------------------------

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch and return mean loss."""
        assert self.model is not None and self.optimizer is not None, \
            "Call setup() before training."
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = self.compute_loss(input_ids, attention_mask, outputs.logits)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def _eval_epoch(self, dataloader: DataLoader) -> float:
        """Run one evaluation pass and return mean loss."""
        assert self.model is not None, "Call setup() before evaluation."
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                total_loss += self.compute_loss(
                    input_ids, attention_mask, outputs.logits
                ).item()

        return total_loss / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> None:
        """Save model + tokenizer checkpoint for the given epoch."""
        assert self.model is not None and self.tokenizer is not None, \
            "Call setup() before saving a checkpoint."
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))

        # Persist metrics alongside weights
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"train_losses": self.train_losses, "eval_losses": self.eval_losses},
                f,
                indent=2,
            )

        logger.info("Checkpoint saved → %s", checkpoint_dir)

    def save_final_model(self) -> None:
        """Save the final fine-tuned model and tokenizer."""
        assert self.model is not None and self.tokenizer is not None, \
            "Call setup() before saving the final model."
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info("Final model saved → %s", final_dir)

    # ------------------------------------------------------------------
    # Public training entry-point
    # ------------------------------------------------------------------

    def train(
        self,
        train_texts: List[str],
        eval_texts: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full fine-tuning loop.

        Args:
            train_texts: List of training text strings.
            eval_texts:  Optional list of validation text strings.

        Returns:
            Dict with 'train_losses' and 'eval_losses' per epoch.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call setup() before train().")

        train_loader = self._build_dataloader(train_texts, shuffle=True)
        eval_loader = (
            self._build_dataloader(eval_texts, shuffle=False)
            if eval_texts
            else None
        )

        total_steps = len(train_loader) * self.num_epochs
        self._init_scheduler(total_steps)

        logger.info(
            "Starting training | samples=%d steps/epoch=%d total_steps=%d",
            len(train_texts), len(train_loader), total_steps,
        )

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            eval_loss: Optional[float] = None
            if eval_loader is not None:
                eval_loss = self._eval_epoch(eval_loader)
                self.eval_losses.append(eval_loss)
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f eval_loss=%.4f",
                    epoch, self.num_epochs, train_loss, eval_loss,
                )
            else:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f",
                    epoch, self.num_epochs, train_loss,
                )

            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

        self.save_final_model()

        return {"train_losses": self.train_losses, "eval_losses": self.eval_losses}
