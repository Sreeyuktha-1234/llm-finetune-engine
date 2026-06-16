"""Utilities for formatting instruction datasets for supervised fine-tuning."""

from typing import Dict


def format_instruction_sample(sample: Dict) -> str:
    """
    Convert a single instruction-style sample into an SFT prompt-completion string.

    Expected input keys:
        - instruction
        - input
        - output

    Returns:
        Formatted string with Instruction/Input/Response sections.
    """
    instruction = str(sample.get("instruction", "")).strip()
    user_input = str(sample.get("input", "")).strip()
    output = str(sample.get("output", "")).strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Response:\n"
        f"{output}"
    )


def is_instruction_sample(sample: Dict) -> bool:
    """Return True when the sample contains instruction-format fields."""
    required_keys = {"instruction", "input", "output"}
    return required_keys.issubset(sample.keys())
