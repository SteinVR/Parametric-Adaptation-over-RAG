"""Adapter-aware generation model loaders."""

from __future__ import annotations

import logging
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

logger = logging.getLogger(__name__)


def load_backbone_with_adapter(
    *,
    model_name: str,
    adapter_dir: Path,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the quantized backbone and attach a PEFT adapter."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit adapter inference requires a CUDA-visible runtime with bitsandbytes support."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loaded adapter from %s", adapter_dir)
    return model, tokenizer
