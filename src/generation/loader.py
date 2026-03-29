"""Backbone model loader: Gemma-2-2b-it in 4-bit NF4."""

from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def load_backbone(
    model_name: str = "google/gemma-2-2b-it",
    quantization: str = "nf4",
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load backbone LLM in quantized mode.

    Returns (model, tokenizer). Model is on CUDA in eval mode.
    """
    logger.info("Loading backbone %s (quantization=%s)", model_name, quantization)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Backbone loaded: %s (device=%s)", model_name, model.device)
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM) -> None:
    """Explicitly delete model and free VRAM."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, VRAM freed")
