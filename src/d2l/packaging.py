"""Generate and export per-document Doc-to-LoRA adapters."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ctx_to_lora.utils import generated_lora_to_state_dict, get_lora_module_names

from src.d2l.adapter_io import ExportedAdapterArtifact, save_peft_lora_adapter
from src.d2l.corpus import CorpusDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocAdapterGenerationResult:
    """Metrics for one per-document Doc-to-LoRA packaging run."""

    doc_index: int
    doc_name: str
    doc_id: str
    page_count: int
    word_count: int
    char_count: int
    generation_seconds: float
    peak_vram_mb: float | None
    adapter: ExportedAdapterArtifact


def generate_document_adapter(
    *,
    model,
    document: CorpusDocument,
    output_dir: Path,
) -> DocAdapterGenerationResult:
    """Internalize one document and export its generated LoRA adapter."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        "Generating Doc-to-LoRA adapter for %s (%d pages, %d words)",
        document.doc_name,
        document.page_count,
        document.word_count,
    )
    start = time.perf_counter()
    try:
        model.internalize(document.full_text)
        generated_state_dict = extract_generated_lora_state_dict(model)
        peft_config = model.peft_config["default"]
        export_artifact = save_peft_lora_adapter(
            adapter_dir=output_dir,
            state_dict=generated_state_dict,
            peft_config=peft_config,
            base_model_name=model.base_model.name_or_path,
        )
    finally:
        model.reset()
    generation_seconds = time.perf_counter() - start
    peak_vram_mb = _peak_vram_mb()
    return DocAdapterGenerationResult(
        doc_index=document.doc_index,
        doc_name=document.doc_name,
        doc_id=document.doc_id,
        page_count=document.page_count,
        word_count=document.word_count,
        char_count=document.char_count,
        generation_seconds=generation_seconds,
        peak_vram_mb=peak_vram_mb,
        adapter=export_artifact,
    )


def extract_generated_lora_state_dict(model) -> dict[str, torch.Tensor]:
    """Convert a D2L internalized LoRA bundle into a PEFT state dict."""
    generated_loras = model.generated_loras
    if generated_loras is None:
        raise RuntimeError("No generated LoRAs found on the Doc-to-LoRA model")

    peft_config = model.peft_config["default"]
    target_modules = sorted(
        str(module) for module in (peft_config.target_modules or [])
    )
    layer_indices = sorted(int(idx) for idx in model.hypernet.layer_indices)
    module_names = get_lora_module_names(
        model.base_model,
        target_modules=target_modules,
        layer_indices=layer_indices,
    )
    return generated_lora_to_state_dict(
        generated_loras,
        module_names,
        target_modules,
        layer_indices,
    )


def average_lora_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average a list of PEFT LoRA state dicts tensor-by-tensor."""
    if not state_dicts:
        raise ValueError("Expected at least one LoRA state dict to merge")

    reference_keys = set(state_dicts[0].keys())
    for idx, state_dict in enumerate(state_dicts[1:], start=2):
        if set(state_dict.keys()) != reference_keys:
            raise ValueError(f"LoRA state dict {idx} has mismatched keys")

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        tensors = [
            state_dict[key].detach().to(dtype=torch.float32, device="cpu")
            for state_dict in state_dicts
        ]
        first_shape = tensors[0].shape
        if any(t.shape != first_shape for t in tensors[1:]):
            raise ValueError(f"LoRA tensor shape mismatch for key: {key}")
        merged[key] = (
            torch.stack(tensors, dim=0)
            .mean(dim=0)
            .to(dtype=state_dicts[0][key].dtype)
        )
    return merged


def _peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024
