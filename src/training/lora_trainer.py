"""
LoRA Fine-tuning trainer using PEFT (Parameter-Efficient Fine-Tuning).
Implements Low-Rank Adaptation for efficient LLM fine-tuning.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Arguments
# ---------------------------------------------------------------------------

@dataclass
class LoRATrainingArguments:
    """Arguments for LoRA training configuration."""
    
    # Model and data
    model_name: str = "gpt2"
    output_dir: str = "outputs/lora_checkpoints"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    max_length: int = 128
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    
    # Checkpointing
    save_every_n_epochs: int = 1
    save_total_limit: int = 3
    
    # Other
    seed: int = 42
    device: Optional[str] = None


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
# LoRA Trainer
# ---------------------------------------------------------------------------

class LoRATrainer:
    """Fine-tune a causal language model using LoRA adapters."""

    def __init__(self, args: LoRATrainingArguments):
        """
        Initialize the LoRA trainer.

        Args:
            args: LoRATrainingArguments containing all configuration.
        """
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set seed
        torch.manual_seed(args.seed)
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup PEFT and LoRA adapters
        self._setup_lora()
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        
        logger.info("LoRA Trainer initialized successfully")

    def _setup_lora(self):
        """Configure and apply LoRA adapters using PEFT."""
        logger.info("Setting up LoRA adapters...")
        
        # Define LoRA config
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules,
            lora_dropout=self.args.lora_dropout,
            bias=self.args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info(f"LoRA adapters configured: r={self.args.lora_r}, "
                   f"alpha={self.args.lora_alpha}, "
                   f"dropout={self.args.lora_dropout}")

    def prepare_dataset(self, texts: List[str]) -> DataLoader:
        """
        Prepare a dataset for training.

        Args:
            texts: List of text samples.

        Returns:
            DataLoader for the dataset.
        """
        dataset = TextDataset(texts, self.tokenizer, self.args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        return dataloader

    def setup_optimizer_and_scheduler(self, train_dataloader: DataLoader):
        """
        Setup optimizer and learning rate scheduler.

        Args:
            train_dataloader: DataLoader for training data.
        """
        # Only optimize trainable parameters (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.args.num_epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )
        
        logger.info(f"Optimizer and scheduler setup: total_steps={total_steps}")

    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """
        Train the model with LoRA adapters.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: Optional DataLoader for validation data.
        """
        # Setup optimizer
        self.setup_optimizer_and_scheduler(train_dataloader)
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            train_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
        )
        
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        
        logger.info("Starting training loop...")
        self.model.train()
        
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["input_ids"],
                )
                loss = outputs.loss
                
                # Backward pass
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.detach().item()
                self.global_step += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    logger.info(
                        f"Epoch {epoch + 1}/{self.args.num_epochs} | "
                        f"Batch {batch_idx + 1} | "
                        f"Loss: {avg_loss:.4f}"
                    )
            
            # End of epoch
            avg_epoch_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model(epoch + 1)

    def _validate(self, val_dataloader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_dataloader: DataLoader for validation data.

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
                loss = outputs.loss
                total_loss += loss.detach().item()
        
        self.model.train()
        return total_loss / len(val_dataloader)

    def save_model(self, epoch: Optional[int] = None):
        """
        Save the LoRA adapters and model configuration.

        Args:
            epoch: Optional epoch number to include in checkpoint name.
        """
        if epoch is not None:
            save_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        else:
            save_dir = self.output_dir / "final_model"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using accelerator to handle distributed training
        self.accelerator.wait_for_everyone()
        
        # Save LoRA adapters
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training config
        config_path = save_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.args.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")

    def load_adapter(self, adapter_path: str):
        """
        Load a saved LoRA adapter.

        Args:
            adapter_path: Path to the saved adapter directory.
        """
        logger.info(f"Loading adapter from {adapter_path}")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info("Adapter loaded successfully")

    def inference(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the fine-tuned model.

        Args:
            prompt: Input prompt for generation.
            max_length: Maximum length of generated text.

        Returns:
            Generated text.
        """
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if self.args.device:
            inputs = inputs.to(self.args.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example training configuration
    args = LoRATrainingArguments(
        model_name="gpt2",
        output_dir="outputs/lora_checkpoints",
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
    )
    
    # Initialize trainer
    trainer = LoRATrainer(args)
    
    # Example texts (in practice, load from your dataset)
    sample_texts = [
        "This is a sample text for fine-tuning.",
        "LoRA adapters enable efficient model training.",
        "Parameter-efficient fine-tuning reduces memory usage.",
    ]
    
    # Prepare dataset
    train_dataloader = trainer.prepare_dataset(sample_texts)
    
    # Train
    trainer.train(train_dataloader)
    
    # Save final model
    trainer.save_model()
    
    # Example inference
    prompt = "The future of AI is"
    generated = trainer.inference(prompt)
    print(f"Generated text: {generated}")
