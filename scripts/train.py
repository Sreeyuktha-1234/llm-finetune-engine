"""
Training entrypoint for full fine-tuning, LoRA, and QLoRA workflows.

Examples:
    python scripts/train.py --mode full --model-name gpt2
    python scripts/train.py --mode lora --model-name gpt2 --num-epochs 1
    python scripts/train.py --mode qlora --model-name gpt2 --num-epochs 1
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from data.loaders.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an LLM fine-tuning workflow")

    parser.add_argument(
        "--mode",
        choices=["full", "lora", "qlora"],
        default="full",
        help="Training mode to run",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/dataset.json",
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split used for training",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Split used for evaluation/validation",
    )

    parser.add_argument("--model-name", default="gpt2", help="Model identifier")
    parser.add_argument("--output-dir", default=None, help="Output directory override")

    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Initialize root logging config."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_texts_for_split(data_path: str, split: Optional[str]) -> List[str]:
    """Load text samples from a dataset split."""
    loader = DatasetLoader(data_path=data_path, split=split)
    loader.load()
    return loader.get_texts()


def safe_eval_texts(data_path: str, split: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Load eval texts if split exists; otherwise return None with a warning message."""
    try:
        texts = load_texts_for_split(data_path, split)
        return texts, None
    except ValueError as exc:
        return None, str(exc)


def run_full_training(args: argparse.Namespace, train_texts: List[str], eval_texts: Optional[List[str]]) -> None:
    """Run standard full fine-tuning."""
    from src.training.trainer import Trainer

    output_dir = args.output_dir or "outputs/checkpoints"

    trainer = Trainer(
        model_name=args.model_name,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
    )
    trainer.setup()
    metrics = trainer.train(train_texts=train_texts, eval_texts=eval_texts)

    logger.info("Training finished. Metrics: %s", metrics)


def run_lora_training(args: argparse.Namespace, train_texts: List[str], eval_texts: Optional[List[str]]) -> None:
    """Run LoRA fine-tuning."""
    from src.training.lora_trainer import LoRATrainer, LoRATrainingArguments

    output_dir = args.output_dir or "outputs/lora_checkpoints"

    lora_args = LoRATrainingArguments(
        model_name=args.model_name,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        seed=args.seed,
    )

    trainer = LoRATrainer(lora_args)
    train_loader = trainer.prepare_dataset(train_texts)
    eval_loader = trainer.prepare_dataset(eval_texts) if eval_texts else None
    trainer.train(train_loader, eval_loader)
    trainer.save_model()


def _load_qlora_module():
    """Load qlora module from hyphenated filename."""
    module_path = ROOT_DIR / "src" / "training" / "qlora-trainer.py"
    module_name = "src.training.qlora_trainer_dynamic"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_qlora_training(args: argparse.Namespace, train_texts: List[str], eval_texts: Optional[List[str]]) -> None:
    """Run QLoRA fine-tuning."""
    qlora_module = _load_qlora_module()
    QLoRATrainer = qlora_module.QLoRATrainer
    QLoRATrainingArguments = qlora_module.QLoRATrainingArguments

    output_dir = args.output_dir or "outputs/qlora_checkpoints"

    qlora_args = QLoRATrainingArguments(
        model_name=args.model_name,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        seed=args.seed,
    )

    trainer = QLoRATrainer(qlora_args)
    train_loader = trainer.prepare_dataset(train_texts)
    eval_loader = trainer.prepare_dataset(eval_texts) if eval_texts else None
    trainer.train(train_loader, eval_loader)
    trainer.save_model()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Loading train split '%s' from %s", args.train_split, args.data_path)
    train_texts = load_texts_for_split(args.data_path, args.train_split)
    logger.info("Loaded %d training samples", len(train_texts))

    eval_texts, eval_warning = safe_eval_texts(args.data_path, args.eval_split)
    if eval_warning:
        logger.warning("Evaluation split unavailable. Proceeding without eval. %s", eval_warning)
    elif eval_texts is not None:
        logger.info("Loaded %d eval samples from split '%s'", len(eval_texts), args.eval_split)

    if args.mode == "full":
        run_full_training(args, train_texts, eval_texts)
    elif args.mode == "lora":
        run_lora_training(args, train_texts, eval_texts)
    elif args.mode == "qlora":
        run_qlora_training(args, train_texts, eval_texts)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
