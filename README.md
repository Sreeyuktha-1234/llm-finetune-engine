# llm-finetune-engine

## Training

Use the unified training entrypoint:

```bash
python scripts/train.py --mode full --model-name gpt2
```

Supported modes:

- `full`: Standard full-parameter fine-tuning.
- `lora`: Parameter-efficient LoRA fine-tuning.
- `qlora`: 4-bit quantized QLoRA fine-tuning.

Examples:

```bash
# Full fine-tuning
python scripts/train.py --mode full --model-name gpt2

# LoRA fine-tuning
python scripts/train.py --mode lora --model-name gpt2 --num-epochs 1

# QLoRA fine-tuning
python scripts/train.py --mode qlora --model-name gpt2 --num-epochs 1
```

### Common options

```bash
python scripts/train.py \
	--mode full \
	--data-path data/raw/dataset.json \
	--train-split train \
	--eval-split validation \
	--model-name gpt2 \
	--num-epochs 3 \
	--batch-size 4 \
	--learning-rate 5e-5 \
	--max-length 128 \
	--warmup-steps 0
```

Key arguments:

- `--mode`: `full`, `lora`, or `qlora`.
- `--data-path`: Path to dataset JSON.
- `--train-split`: Dataset split for training.
- `--eval-split`: Dataset split for validation.
- `--output-dir`: Optional override for checkpoint output directory.
- `--log-level`: Logging level (for example `INFO` or `DEBUG`).
