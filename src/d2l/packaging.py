"""Generate and export per-document Doc-to-LoRA adapters."""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.utils import generated_lora_to_state_dict, get_lora_module_names

from src.d2l.adapter_io import ExportedAdapterArtifact, save_peft_lora_adapter
from src.d2l.corpus import CorpusDocument

logger = logging.getLogger(__name__)
DEFAULT_CTX_CHUNK_TOKENS = int(os.environ.get("D2L_CTX_CHUNK_TOKENS", "1024"))


@dataclass(frozen=True, slots=True)
class DocAdapterGenerationResult:
    """Metrics for one per-document Doc-to-LoRA packaging run."""

    doc_index: int
    doc_name: str
    doc_id: str
    page_count: int
    word_count: int
    char_count: int
    context_token_count: int
    generation_seconds: float
    peak_vram_mb: float | None
    peak_rss_mb: float | None
    chunk_count: int
    used_chunk_fallback: bool
    chunk_merge_mean_explained_variance: float | None
    adapter: ExportedAdapterArtifact


def generate_document_adapter(
    *,
    model,
    document: CorpusDocument,
    output_dir: Path,
    progress_path: Path | None = None,
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
    tokenizer = get_tokenizer(
        model.ctx_encoder.base_model.name_or_path,
        tokenizer_kwargs={"local_files_only": True},
    )
    tokenized_ctx = tokenize_ctx_text(
        {"context": [document.full_text]},
        tokenizer,
    )["ctx_ids"][0]
    context_token_count = len(tokenized_ctx)
    chunk_plan = split_too_long_ctx(
        {"ctx_ids": tokenized_ctx},
        model.base_model.name_or_path,
        num_chunk_probs=None,
        max_chunk_len=DEFAULT_CTX_CHUNK_TOKENS,
        min_chunk_len=-1,
        max_num_split=None,
        is_train=False,
    )
    ctx_chunks: list[list[int]] = chunk_plan["ctx_ids"]
    used_chunk_fallback = len(ctx_chunks) > 1
    chunk_token_lengths = [len(chunk) for chunk in ctx_chunks]
    logger.info(
        "Doc %s tokenized to %d ctx tokens; using %d chunk(s) (max_chunk_len=%d, rss=%.1f MB)",
        document.doc_name,
        context_token_count,
        len(ctx_chunks),
        DEFAULT_CTX_CHUNK_TOKENS,
        _process_rss_mb() or -1.0,
    )
    _write_progress(
        progress_path,
        {
            "status": "tokenized",
            "doc_index": document.doc_index,
            "doc_name": document.doc_name,
            "context_token_count": context_token_count,
            "chunk_count": len(ctx_chunks),
            "chunk_token_lengths": chunk_token_lengths,
            "rss_mb": _process_rss_mb(),
            "used_chunk_fallback": used_chunk_fallback,
        },
    )

    start = time.perf_counter()
    chunk_merge_mean_explained_variance: float | None = None
    try:
        generated_state_dict: dict[str, torch.Tensor]
        if used_chunk_fallback:
            generated_state_dict, chunk_merge_mean_explained_variance = (
                _generate_chunked_adapter_state_dict(
                    model=model,
                    ctx_chunks=ctx_chunks,
                    progress_path=progress_path,
                )
            )
        else:
            logger.info(
                "Doc %s uses single-pass internalization (rss=%.1f MB)",
                document.doc_name,
                _process_rss_mb() or -1.0,
            )
            model.internalize(document.full_text)
            generated_state_dict = extract_generated_lora_state_dict(model)

        peft_config = model.peft_config
        export_artifact = save_peft_lora_adapter(
            adapter_dir=output_dir,
            state_dict=generated_state_dict,
            peft_config=peft_config,
            base_model_name=model.base_model.name_or_path,
        )
        _write_progress(
            progress_path,
            {
                "status": "saved",
                "doc_index": document.doc_index,
                "doc_name": document.doc_name,
                "context_token_count": context_token_count,
                "chunk_count": len(ctx_chunks),
                "chunk_token_lengths": chunk_token_lengths,
                "rss_mb": _process_rss_mb(),
                "used_chunk_fallback": used_chunk_fallback,
                "adapter_dir": str(output_dir),
                "adapter_file_size_bytes": export_artifact.byte_size,
                "chunk_merge_mean_explained_variance": (
                    chunk_merge_mean_explained_variance
                ),
            },
        )
    finally:
        model.reset()
    generation_seconds = time.perf_counter() - start
    peak_vram_mb = _peak_vram_mb()
    peak_rss_mb = _process_rss_mb()
    return DocAdapterGenerationResult(
        doc_index=document.doc_index,
        doc_name=document.doc_name,
        doc_id=document.doc_id,
        page_count=document.page_count,
        word_count=document.word_count,
        char_count=document.char_count,
        context_token_count=context_token_count,
        generation_seconds=generation_seconds,
        peak_vram_mb=peak_vram_mb,
        peak_rss_mb=peak_rss_mb,
        chunk_count=len(ctx_chunks),
        used_chunk_fallback=used_chunk_fallback,
        chunk_merge_mean_explained_variance=chunk_merge_mean_explained_variance,
        adapter=export_artifact,
    )


def extract_generated_lora_state_dict(
    model,
    generated_loras: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a D2L internalized LoRA bundle into a PEFT state dict."""
    generated_loras = model.generated_loras if generated_loras is None else generated_loras
    if generated_loras is None:
        raise RuntimeError("No generated LoRAs found on the Doc-to-LoRA model")
    generated_loras = _normalize_generated_loras(generated_loras)

    peft_config = model.peft_config
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
    """Average PEFT LoRA tensors using the frozen EXP-004 merge contract.

    EXP-004 specifies a simple arithmetic mean over matching adapter matrices.
    That means `lora_A` and `lora_B` tensors are averaged independently, without
    changing adapter rank or PEFT config semantics.
    """
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


@dataclass(frozen=True, slots=True)
class DeltaWMergeResult:
    """Result of ΔW-average LoRA merge with SVD re-decomposition."""

    state_dict: dict[str, torch.Tensor]
    explained_variance: dict[str, float]
    mean_explained_variance: float
    merge_seconds: float


def delta_w_average_lora_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    rank: int,
    weights: list[float] | None = None,
) -> DeltaWMergeResult:
    """Merge LoRA adapters by averaging weight updates with SVD re-decomposition.

    Unlike factor-wise averaging (avg A, avg B separately), this computes the
    true mean of per-adapter updates ΔWᵢ = Bᵢ @ Aᵢ, then decomposes back to
    rank-r via truncated SVD to remain PEFT-compatible.

    Returns a ``DeltaWMergeResult`` with the merged state dict and per-layer
    explained variance ratios (fraction of singular-value energy preserved by
    the rank-r truncation).
    """
    if not state_dicts:
        raise ValueError("Expected at least one LoRA state dict to merge")
    if weights is not None and len(weights) != len(state_dicts):
        raise ValueError("Expected one weight per LoRA state dict")

    reference_keys = set(state_dicts[0].keys())
    for idx, sd in enumerate(state_dicts[1:], start=2):
        if set(sd.keys()) != reference_keys:
            raise ValueError(f"LoRA state dict {idx} has mismatched keys")
    merge_weights = (
        torch.tensor(weights, dtype=torch.float32)
        if weights is not None
        else torch.full((len(state_dicts),), 1.0 / len(state_dicts), dtype=torch.float32)
    )
    merge_weights = merge_weights / merge_weights.sum()

    # Group keys into LoRA A/B pairs and non-LoRA keys.
    lora_prefixes: dict[str, dict[str, str]] = {}
    non_lora_keys: list[str] = []

    for key in sorted(reference_keys):
        if ".lora_A." in key:
            prefix = key.split(".lora_A.")[0]
            lora_prefixes.setdefault(prefix, {})["A"] = key
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.")[0]
            lora_prefixes.setdefault(prefix, {})["B"] = key
        else:
            non_lora_keys.append(key)

    merged: dict[str, torch.Tensor] = {}
    explained_variances: dict[str, float] = {}

    start = time.perf_counter()

    for prefix, ab_keys in sorted(lora_prefixes.items()):
        if "A" not in ab_keys or "B" not in ab_keys:
            raise ValueError(f"Incomplete LoRA pair for prefix: {prefix}")

        key_a, key_b = ab_keys["A"], ab_keys["B"]
        orig_dtype = state_dicts[0][key_a].dtype

        # ΔW_avg = Σ wᵢ (Bᵢ @ Aᵢ)
        delta_w_sum: torch.Tensor | None = None
        for weight, sd in zip(merge_weights, state_dicts, strict=True):
            a = sd[key_a].detach().to(dtype=torch.float32)  # (r, d_in)
            b = sd[key_b].detach().to(dtype=torch.float32)
            if b.ndim != 2:
                raise ValueError(f"Unexpected LoRA B shape for {key_b}: {b.shape}")
            if b.shape[0] == a.shape[0]:
                b_for_dw = b.transpose(0, 1)  # internal D2L layout: (r, d_out)
                save_b_transposed = True
            elif b.shape[1] == a.shape[0]:
                b_for_dw = b  # PEFT layout: (d_out, r)
                save_b_transposed = False
            else:
                raise ValueError(
                    f"Incompatible LoRA pair shapes for {prefix}: A={a.shape}, B={b.shape}"
                )
            dw = (b_for_dw @ a) * weight.item()  # (d_out, d_in)
            delta_w_sum = dw if delta_w_sum is None else delta_w_sum + dw
        assert delta_w_sum is not None
        delta_w_avg = delta_w_sum

        # Truncated SVD → rank-r factors.
        U, S, Vh = torch.linalg.svd(delta_w_avg, full_matrices=False)

        total_energy = (S**2).sum()
        truncated_energy = (S[:rank] ** 2).sum()
        ev = (truncated_energy / total_energy).item() if total_energy > 0 else 1.0
        explained_variances[prefix] = ev

        sqrt_s = S[:rank].sqrt()
        new_b = U[:, :rank] * sqrt_s.unsqueeze(0)  # (d_out, r)
        new_a = sqrt_s.unsqueeze(1) * Vh[:rank, :]  # (r, d_in)

        merged[key_b] = (
            new_b.transpose(0, 1).to(dtype=orig_dtype)
            if save_b_transposed
            else new_b.to(dtype=orig_dtype)
        )
        merged[key_a] = new_a.to(dtype=orig_dtype)

    # Average any non-LoRA keys (e.g. biases) directly.
    for key in non_lora_keys:
        tensors = [sd[key].detach().to(torch.float32) for sd in state_dicts]
        weighted = torch.zeros_like(tensors[0])
        for weight, tensor in zip(merge_weights, tensors, strict=True):
            weighted = weighted + (tensor * weight.item())
        merged[key] = weighted.to(dtype=state_dicts[0][key].dtype)

    merge_seconds = time.perf_counter() - start

    mean_ev = (
        sum(explained_variances.values()) / len(explained_variances)
        if explained_variances
        else 1.0
    )

    logger.info(
        "ΔW-avg merge: %d LoRA pairs, rank=%d, mean explained variance=%.4f (%.1fs)",
        len(lora_prefixes),
        rank,
        mean_ev,
        merge_seconds,
    )

    return DeltaWMergeResult(
        state_dict=merged,
        explained_variance=explained_variances,
        mean_explained_variance=mean_ev,
        merge_seconds=merge_seconds,
    )


def _generate_chunked_adapter_state_dict(
    *,
    model,
    ctx_chunks: list[list[int]],
    progress_path: Path | None,
) -> tuple[dict[str, torch.Tensor], float | None]:
    """Generate one rank-compatible adapter from multiple ctx chunks."""
    chunk_state_paths: list[Path] = []
    chunk_state_dicts_in_memory: list[dict[str, torch.Tensor]] = []
    peft_rank = int(model.peft_config.r)
    chunk_lengths: list[float] = []
    for chunk_index, chunk_ids in enumerate(ctx_chunks, start=1):
        logger.info(
            "Generating chunk %d/%d (%d tokens, rss=%.1f MB)",
            chunk_index,
            len(ctx_chunks),
            len(chunk_ids),
            _process_rss_mb() or -1.0,
        )
        chunk_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=model.device)
        chunk_attn_mask = torch.ones_like(chunk_tensor)
        generated_loras, _ = model.generate_weights(chunk_tensor, chunk_attn_mask)
        chunk_state_dict = extract_generated_lora_state_dict(model, generated_loras)
        chunk_state_path = (
            progress_path.with_name(f"{progress_path.stem}_chunk{chunk_index}.pt")
            if progress_path is not None
            else None
        )
        if chunk_state_path is not None:
            torch.save(chunk_state_dict, chunk_state_path)
            chunk_state_paths.append(chunk_state_path)
        else:
            chunk_state_dicts_in_memory.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in chunk_state_dict.items()
                }
            )
        chunk_lengths.append(float(len(chunk_ids)))
        _write_progress(
            progress_path,
            {
                "status": "chunk_generated",
                "chunk_index": chunk_index,
                "chunk_count": len(ctx_chunks),
                "chunk_tokens": len(chunk_ids),
                "rss_mb": _process_rss_mb(),
                "chunk_state_path": (
                    str(chunk_state_path) if chunk_state_path is not None else None
                ),
            },
        )
        del chunk_tensor
        del chunk_attn_mask
        del generated_loras
        del chunk_state_dict
        _release_host_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if chunk_state_paths:
        chunk_state_dicts = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in chunk_state_paths
        ]
    elif chunk_state_dicts_in_memory:
        chunk_state_dicts = chunk_state_dicts_in_memory
    else:
        raise RuntimeError("No chunk adapter artifacts were produced")

    if len(chunk_state_dicts) == 1:
        return chunk_state_dicts[0], None

    merge_result = delta_w_average_lora_state_dicts(
        chunk_state_dicts,
        rank=peft_rank,
        weights=chunk_lengths,
    )
    logger.info(
        "Collapsed %d chunk adapters back to rank %d (mean EV=%.4f, rss=%.1f MB)",
        len(ctx_chunks),
        peft_rank,
        merge_result.mean_explained_variance,
        _process_rss_mb() or -1.0,
    )
    return merge_result.state_dict, merge_result.mean_explained_variance


def _normalize_generated_loras(
    generated_loras: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize raw D2L LoRA tensors to [n_layers, r, dim] for export."""
    normalized: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, matrices in generated_loras.items():
        normalized[module_name] = {}
        for matrix_name, tensor in matrices.items():
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError(
                        f"Expected batch size 1 for export, got {tensor.shape} "
                        f"for {module_name}.{matrix_name}"
                    )
                normalized[module_name][matrix_name] = tensor.squeeze(0)
            elif tensor.ndim == 3:
                normalized[module_name][matrix_name] = tensor
            else:
                raise ValueError(
                    f"Unexpected generated LoRA shape {tensor.shape} "
                    f"for {module_name}.{matrix_name}"
                )
    return normalized


def _write_progress(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _process_rss_mb() -> float | None:
    """Read the current process RSS from /proc for lightweight memory logs."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return None


def _release_host_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        return


def _peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024
