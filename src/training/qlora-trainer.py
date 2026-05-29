"""
QLoRA Fine-tuning trainer using PEFT + bitsandbytes.
Implements Quantized Low-Rank Adaptation (QLoRA) for memory-efficient LLM fine-tuning.
Uses 4-bit NF4 quantization via bitsandbytes to drastically reduce GPU memory usage
while preserving model quality.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization Config Helper
# ---------------------------------------------------------------------------

def build_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
) -> BitsAndBytesConfig:
    """
    Build a bitsandbytes quantization config for 4-bit loading.

    Args:
        load_in_4bit:              Load weights in 4-bit precision.
        bnb_4bit_quant_type:       Quantization type — "nf4" (NormalFloat4, recommended)
                                   or "fp4" (FP4).
        bnb_4bit_use_double_quant: Nest a second quantization on the quantization
                                   constants to save an additional ~0.4 bits per param.
        bnb_4bit_compute_dtype:    Dtype used for computation during the forward pass
                                   (bfloat16 recommended for modern GPUs).

    Returns:
        BitsAndBytesConfig instance.
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )


# ---------------------------------------------------------------------------
# Training Arguments
# ---------------------------------------------------------------------------

@dataclass
class QLoRATrainingArguments:
    """Arguments for QLoRA training configuration."""

    # Model and data
    model_name: str = "gpt2"
    output_dir: str = "outputs/qlora_checkpoints"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    max_length: int = 128
    gradient_accumulation_steps: int = 4

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # 4-bit quantization settings (bitsandbytes)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True     # double quantization for extra savings
    bnb_4bit_compute_dtype: str = "bfloat16"  # compute dtype as string for serialization

    # Memory optimization
    gradient_checkpointing: bool = True

    # Checkpointing
    save_every_n_epochs: int = 1
    save_total_limit: int = 3

    # Other
    seed: int = 42
    device: Optional[str] = None

    def compute_dtype_as_torch(self) -> torch.dtype:
        """Convert the string compute dtype to a torch.dtype."""
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = mapping.get(self.bnb_4bit_compute_dtype)
        if dtype is None:
            raise ValueError(
                f"Unsupported bnb_4bit_compute_dtype '{self.bnb_4bit_compute_dtype}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return dtype


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
# QLoRA Trainer
# ---------------------------------------------------------------------------

class QLoRATrainer:
    """
    Fine-tune a causal language model using QLoRA.

    Key differences from plain LoRA:
    - Base model weights are loaded in 4-bit via bitsandbytes (NF4 quantization).
    - Double quantization further compresses quantization constants.
    - `prepare_model_for_kbit_training` casts layer-norms / embeddings to fp32
      and enables gradient checkpointing for activation-memory savings.
    - Only LoRA adapter parameters are trained in fp32 (compute dtype = bfloat16).
    """

    def __init__(self, args: QLoRATrainingArguments):
        """
        Initialize the QLoRA trainer.

        Args:
            args: QLoRATrainingArguments containing all configuration.
        """
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Set seed
        torch.manual_seed(args.seed)

        # Build bitsandbytes 4-bit quantization config
        bnb_config = build_bnb_config(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=args.compute_dtype_as_torch(),
        )

        # GPU-efficient model loading: map to CUDA when available, else CPU
        device_map = "auto" if torch.cuda.is_available() else None

        logger.info(
            f"Loading model '{args.model_name}' with 4-bit quantization "
            f"(quant_type={args.bnb_4bit_quant_type}, "
            f"double_quant={args.bnb_4bit_use_double_quant}, "
            f"compute_dtype={args.bnb_4bit_compute_dtype}, "
            f"device_map={device_map})"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare base model for k-bit (4-bit) training:
        # - Casts layer-norm and embedding weights to fp32
        # - Enables gradient checkpointing to trade compute for memory
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

        # Apply LoRA adapters on top of the quantized base model
        self._setup_lora()

        # Training state
        self.optimizer: Optional[AdamW] = None
        self.scheduler = None
        self.global_step = 0

        logger.info("QLoRA Trainer initialized successfully")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_lora(self):
        """Configure and apply LoRA adapters using PEFT."""
        logger.info("Setting up LoRA adapters on quantized model...")

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules,
            lora_dropout=self.args.lora_dropout,
            bias=self.args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info(
            f"QLoRA adapters configured: r={self.args.lora_r}, "
            f"alpha={self.args.lora_alpha}, "
            f"dropout={self.args.lora_dropout}"
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def prepare_dataset(self, texts: List[str]) -> DataLoader:
        """
        Prepare a DataLoader from raw text samples.

        Args:
            texts: List of text strings.

        Returns:
            DataLoader ready for training.
        """
        dataset = TextDataset(texts, self.tokenizer, self.args.max_length)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    def setup_optimizer_and_scheduler(self, train_dataloader: DataLoader):
        """
        Setup AdamW optimizer (LoRA params only) and linear warmup scheduler.

        Args:
            train_dataloader: Training DataLoader (used to compute total steps).
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        total_steps = (
            len(train_dataloader)
            // self.args.gradient_accumulation_steps
            * self.args.num_epochs
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Optimizer ready. Total training steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        """
        Run the QLoRA training loop with gradient accumulation.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader:   Optional DataLoader for validation.
        """
        self.setup_optimizer_and_scheduler(train_dataloader)

        # Accelerator handles mixed precision, device placement, and grad accumulation
        (
            self.model,
            self.optimizer,
            train_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )

        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)

        logger.info("Starting QLoRA training loop...")
        self.model.train()

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["input_ids"],
                    )
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.detach().float().item()
                self.global_step += 1

                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    logger.info(
                        f"Epoch {epoch + 1}/{self.args.num_epochs} | "
                        f"Batch {batch_idx + 1} | "
                        f"Loss: {avg_loss:.4f}"
                    )

            avg_epoch_loss = total_loss / len(train_dataloader)
            logger.info(
                f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}"
            )

            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")

            if (epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model(epoch + 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, val_dataloader: DataLoader) -> float:
        """
        Compute average loss over a validation DataLoader.

        Args:
            val_dataloader: Validation DataLoader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["input_ids"],
                )
                total_loss += outputs.loss.detach().float().item()

        self.model.train()
        return total_loss / len(val_dataloader)

    # ------------------------------------------------------------------
    # Saving / Loading
    # ------------------------------------------------------------------

    def save_model(self, epoch: Optional[int] = None):
        """
        Save LoRA adapters, tokenizer, and training config.

        Args:
            epoch: Epoch number to include in the checkpoint directory name.
        """
        save_dir = (
            self.output_dir / f"checkpoint-epoch-{epoch}"
            if epoch is not None
            else self.output_dir / "final_model"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        config_path = save_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.args.__dict__, f, indent=2, default=str)

        logger.info(f"Model checkpoint saved to {save_dir}")

    def load_adapter(self, adapter_path: str):
        """
        Load a previously saved LoRA adapter onto the quantized base model.

        Args:
            adapter_path: Path to the saved PEFT adapter directory.
        """
        logger.info(f"Loading adapter from {adapter_path}")
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info("Adapter loaded successfully")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate text using the fine-tuned QLoRA model.

        Args:
            prompt:         Input prompt for text generation.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Generated text string (excluding the input prompt).
        """
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
