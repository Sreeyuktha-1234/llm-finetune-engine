"""
Export pipeline for LoRA and QLoRA adapter weights.

Supports three export modes:
  - adapter-only  : save only the PEFT adapter files (default)
  - merged        : merge adapter weights into the base model and save the
                    full model in HuggingFace format
  - gguf-ready    : merge + save in fp16 safetensors layout, ready for
                    downstream GGUF conversion tools
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Export configuration dataclass
# ---------------------------------------------------------------------------

ExportMode = Literal["adapter-only", "merged", "gguf-ready"]


@dataclass
class ExportConfig:
    """
    Configuration for an adapter export run.

    Attributes:
        base_model_name:    HuggingFace model ID or local path of the *base*
                            model that was fine-tuned.
        adapter_path:       Path to the directory that contains the saved PEFT
                            adapter (``adapter_config.json`` +
                            ``adapter_model.safetensors`` / ``adapter_model.bin``).
        output_dir:         Directory where exported artefacts will be written.
        export_mode:        One of ``"adapter-only"``, ``"merged"``, or
                            ``"gguf-ready"``.
        adapter_type:       ``"lora"`` or ``"qlora"`` — controls whether 4-bit
                            dequantisation is applied before merging.
        device:             Device used for the merge forward pass (``"cpu"``
                            is safe; use ``"cuda"`` for speed when VRAM allows).
        safe_serialization: Write ``.safetensors`` files instead of ``.bin``.
        push_to_hub:        If ``True``, push the exported artefacts to the
                            HuggingFace Hub after saving locally.
        hub_repo_id:        Target Hub repository, e.g. ``"username/my-model"``.
                            Required when ``push_to_hub=True``.
        hub_private:        Create a private Hub repository.
        torch_dtype:        Override the dtype used when loading the base model
                            for merging. Defaults to ``torch.float16``.
        bnb_4bit_quant_type: NF4 quantisation type used by QLoRA. Only relevant
                            when ``adapter_type="qlora"``.
    """

    base_model_name: str
    adapter_path: str
    output_dir: str = "outputs/exported"
    export_mode: ExportMode = "adapter-only"
    adapter_type: Literal["lora", "qlora"] = "lora"
    device: str = "cpu"
    safe_serialization: bool = True
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
    hub_private: bool = False
    torch_dtype: torch.dtype = field(default=torch.float16)
    bnb_4bit_quant_type: str = "nf4"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_base_model_for_merge(cfg: ExportConfig) -> AutoModelForCausalLM:
    """
    Load the base model suitable for weight-merging.

    For QLoRA, the base model must be reloaded in 4-bit precision so that
    PEFT can dequantise the adapter deltas before merging them into fp16
    weights.

    Args:
        cfg: Active :class:`ExportConfig`.

    Returns:
        Loaded ``AutoModelForCausalLM`` instance.
    """
    if cfg.adapter_type == "qlora":
        logger.info(
            "QLoRA adapter detected — loading base model with 4-bit quantisation "
            "for dequantisation-aware merge."
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            quantization_config=bnb_config,
            device_map=cfg.device,
            trust_remote_code=True,
        )
    else:
        logger.info("LoRA adapter detected — loading base model in fp16.")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device,
            trust_remote_code=True,
        )
    return model


def _attach_and_merge(
    base_model: AutoModelForCausalLM,
    adapter_path: str,
) -> AutoModelForCausalLM:
    """
    Wrap *base_model* with the PEFT adapter at *adapter_path*, then merge
    the adapter weights into the base model parameters and unload PEFT.

    Args:
        base_model:   Pre-loaded base ``AutoModelForCausalLM``.
        adapter_path: Path to the directory containing PEFT adapter files.

    Returns:
        Plain ``AutoModelForCausalLM`` with adapter weights merged in.
    """
    logger.info("Attaching PEFT adapter from '%s'.", adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("Merging adapter weights into base model …")
    merged_model = peft_model.merge_and_unload()
    logger.info("Merge complete.")
    return merged_model


# ---------------------------------------------------------------------------
# Public export functions
# ---------------------------------------------------------------------------

def export_adapter_only(cfg: ExportConfig) -> Path:
    """
    Copy PEFT adapter files to *output_dir* without touching the base model.

    This is the lightest export mode: only the low-rank weight matrices
    (typically a few MB) are written to disk.  The base model must be
    available at inference time.

    Args:
        cfg: Active :class:`ExportConfig`.

    Returns:
        Absolute :class:`~pathlib.Path` to the output directory.
    """
    src = Path(cfg.adapter_path).resolve()
    dst = Path(cfg.output_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting adapter files from '%s' → '%s'.", src, dst)

    adapter_files = list(src.glob("adapter_config.json")) + list(
        src.glob("adapter_model.*")
    )
    if not adapter_files:
        raise FileNotFoundError(
            f"No adapter files found in '{src}'. "
            "Expected 'adapter_config.json' and 'adapter_model.*'."
        )

    for f in adapter_files:
        shutil.copy2(f, dst / f.name)
        logger.info("  Copied: %s", f.name)

    # Write a small export manifest for traceability
    manifest = {
        "export_mode": "adapter-only",
        "adapter_type": cfg.adapter_type,
        "base_model": cfg.base_model_name,
        "adapter_source": str(src),
        "output_dir": str(dst),
    }
    (dst / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("Adapter-only export complete: '%s'.", dst)
    return dst


def export_merged(cfg: ExportConfig) -> Path:
    """
    Merge the adapter into the base model and save the full HuggingFace
    model (weights + tokenizer + config) to *output_dir*.

    The merged checkpoint is fully self-contained and can be loaded with
    ``AutoModelForCausalLM.from_pretrained(output_dir)`` without PEFT.

    Args:
        cfg: Active :class:`ExportConfig`.

    Returns:
        Absolute :class:`~pathlib.Path` to the output directory.
    """
    output_path = Path(cfg.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    base_model = _load_base_model_for_merge(cfg)
    merged_model = _attach_and_merge(base_model, cfg.adapter_path)

    logger.info(
        "Saving merged model to '%s' (safe_serialization=%s).",
        output_path,
        cfg.safe_serialization,
    )
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=cfg.safe_serialization,
    )

    # Save tokenizer alongside the model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_path))
    logger.info("Tokenizer saved.")

    manifest = {
        "export_mode": "merged",
        "adapter_type": cfg.adapter_type,
        "base_model": cfg.base_model_name,
        "adapter_source": str(Path(cfg.adapter_path).resolve()),
        "output_dir": str(output_path),
        "safe_serialization": cfg.safe_serialization,
    }
    (output_path / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("Merged export complete: '%s'.", output_path)
    return output_path


def export_gguf_ready(cfg: ExportConfig) -> Path:
    """
    Merge the adapter into the base model, cast to fp16, and save in a
    layout compatible with downstream GGUF conversion tools
    (e.g. ``llama.cpp``'s ``convert.py``).

    This is identical to :func:`export_merged` but forces ``torch.float16``
    and always uses safetensors serialisation.

    Args:
        cfg: Active :class:`ExportConfig`.

    Returns:
        Absolute :class:`~pathlib.Path` to the output directory.
    """
    # Force fp16 + safetensors for GGUF pipeline
    cfg.torch_dtype = torch.float16
    cfg.safe_serialization = True

    output_path = Path(cfg.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    base_model = _load_base_model_for_merge(cfg)

    # Cast to fp16 before merging to ensure clean weight dtype
    base_model = base_model.to(torch.float16)

    merged_model = _attach_and_merge(base_model, cfg.adapter_path)
    merged_model = merged_model.to(torch.float16)

    logger.info("Saving GGUF-ready model to '%s'.", output_path)
    merged_model.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_path))

    manifest = {
        "export_mode": "gguf-ready",
        "adapter_type": cfg.adapter_type,
        "base_model": cfg.base_model_name,
        "adapter_source": str(Path(cfg.adapter_path).resolve()),
        "output_dir": str(output_path),
        "dtype": "float16",
        "safe_serialization": True,
    }
    (output_path / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("GGUF-ready export complete: '%s'.", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

class ExportPipeline:
    """
    Unified pipeline for exporting LoRA and QLoRA adapter weights.

    Usage::

        from src.pipelines.export_pipeline import ExportPipeline, ExportConfig

        # ── LoRA adapter-only export ──────────────────────────────────────
        cfg = ExportConfig(
            base_model_name="meta-llama/Llama-2-7b-hf",
            adapter_path="outputs/lora_checkpoints/epoch-3",
            output_dir="outputs/exported/lora-adapter",
            export_mode="adapter-only",
            adapter_type="lora",
        )
        pipeline = ExportPipeline(cfg)
        pipeline.run()

        # ── QLoRA merged export ───────────────────────────────────────────
        cfg = ExportConfig(
            base_model_name="meta-llama/Llama-2-7b-hf",
            adapter_path="outputs/qlora_checkpoints/epoch-3",
            output_dir="outputs/exported/qlora-merged",
            export_mode="merged",
            adapter_type="qlora",
        )
        pipeline = ExportPipeline(cfg)
        pipeline.run()
    """

    _DISPATCH: Dict[str, Any] = {
        "adapter-only": export_adapter_only,
        "merged": export_merged,
        "gguf-ready": export_gguf_ready,
    }

    def __init__(self, config: ExportConfig) -> None:
        """
        Args:
            config: :class:`ExportConfig` describing the export to perform.
        """
        self.config = config
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise informative errors for obviously invalid configurations."""
        adapter_path = Path(self.config.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"adapter_path does not exist: '{adapter_path}'"
            )
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No 'adapter_config.json' found in '{adapter_path}'. "
                "Ensure the adapter was saved with PeftModel.save_pretrained()."
            )
        if self.config.push_to_hub and not self.config.hub_repo_id:
            raise ValueError(
                "'hub_repo_id' must be set when 'push_to_hub=True'."
            )
        if self.config.export_mode not in self._DISPATCH:
            raise ValueError(
                f"Unknown export_mode '{self.config.export_mode}'. "
                f"Choose one of: {list(self._DISPATCH)}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Execute the export according to :attr:`config`.

        Returns:
            Absolute path to the directory containing the exported artefacts.
        """
        logger.info(
            "Starting %s export for %s adapter '%s'.",
            self.config.export_mode,
            self.config.adapter_type.upper(),
            self.config.adapter_path,
        )

        export_fn = self._DISPATCH[self.config.export_mode]
        output_path: Path = export_fn(self.config)

        if self.config.push_to_hub:
            self._push_to_hub(output_path)

        return output_path

    # ------------------------------------------------------------------
    # Hub upload
    # ------------------------------------------------------------------

    def _push_to_hub(self, output_path: Path) -> None:
        """
        Push exported artefacts at *output_path* to the HuggingFace Hub.

        Requires ``huggingface_hub`` to be installed and the user to be
        authenticated (``huggingface-cli login``).

        Args:
            output_path: Local directory produced by the export step.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for Hub upload. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        api = HfApi()
        repo_id = self.config.hub_repo_id

        logger.info(
            "Pushing exported artefacts to Hub repository '%s' …", repo_id
        )
        api.create_repo(
            repo_id=repo_id,
            private=self.config.hub_private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=repo_id,
            commit_message=(
                f"Export: {self.config.export_mode} / "
                f"{self.config.adapter_type.upper()}"
            ),
        )
        logger.info("Hub upload complete: https://huggingface.co/%s", repo_id)
