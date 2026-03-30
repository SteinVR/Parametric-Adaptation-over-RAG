"""Doc-to-LoRA checkpoint loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import BitsAndBytesConfig

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "pytorch_model.bin"


def resolve_checkpoint_file(checkpoint_root: Path | str) -> Path:
    """Resolve the concrete D2L checkpoint file from a directory or file path."""
    path = Path(checkpoint_root)
    checkpoint_file = path if path.suffix == ".bin" else path / CHECKPOINT_FILENAME
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Doc-to-LoRA checkpoint not found: {checkpoint_file}. "
            "The experiment expects a local SakanaAI doc-to-lora checkpoint and does not download it."
        )
    return checkpoint_file


def load_d2l_model(
    checkpoint_root: Path | str,
    *,
    base_model_kwargs: dict[str, Any] | None = None,
    use_flash_attn: bool = False,
    train: bool = False,
) -> ModulatedPretrainedModel:
    """Load the frozen Doc-to-LoRA hypernetwork checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Doc-to-LoRA packaging requires a CUDA-visible runtime."
        )

    checkpoint_file = resolve_checkpoint_file(checkpoint_root)
    logger.info("Loading Doc-to-LoRA checkpoint from %s", checkpoint_file)
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    model_kwargs = dict(base_model_kwargs or {})
    model_kwargs.setdefault(
        "quantization_config",
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model_kwargs.setdefault("device_map", "auto")
    model_kwargs.setdefault("local_files_only", True)
    model_kwargs.setdefault("attn_implementation", "eager")

    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=train,
        base_model_kwargs=model_kwargs,
        use_flash_attn=use_flash_attn,
        use_sequence_packing=False,
    )
    model.reset()
    return model

