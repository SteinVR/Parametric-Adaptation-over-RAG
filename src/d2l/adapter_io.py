"""Export helpers for Doc-to-LoRA generated adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
BIN_WEIGHTS_NAME = "adapter_model.bin"


@dataclass(frozen=True, slots=True)
class ExportedAdapterArtifact:
    """PEFT-compatible adapter artifact written to disk."""

    adapter_dir: Path
    config_path: Path
    weight_path: Path
    tensor_count: int
    byte_size: int


def save_peft_lora_adapter(
    *,
    adapter_dir: Path,
    state_dict: dict[str, torch.Tensor],
    peft_config: LoraConfig,
    base_model_name: str,
) -> ExportedAdapterArtifact:
    """Persist a generated LoRA state dict in PEFT's adapter directory layout."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    export_config = deepcopy(peft_config)
    export_config.base_model_name_or_path = base_model_name
    export_config.inference_mode = True
    export_config.save_pretrained(str(adapter_dir))

    weight_path = adapter_dir / SAFETENSORS_WEIGHTS_NAME
    export_state_dict = _normalize_state_dict_for_peft_save(state_dict)
    save_safetensors_file(
        {key: value.detach().cpu().contiguous() for key, value in export_state_dict.items()},
        str(weight_path),
    )
    config_path = adapter_dir / "adapter_config.json"
    byte_size = sum(
        path.stat().st_size for path in adapter_dir.iterdir() if path.is_file()
    )
    return ExportedAdapterArtifact(
        adapter_dir=adapter_dir,
        config_path=config_path,
        weight_path=weight_path,
        tensor_count=len(export_state_dict),
        byte_size=byte_size,
    )


def load_peft_lora_state_dict(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Load a PEFT adapter state dict from disk."""
    safetensors_path = adapter_dir / SAFETENSORS_WEIGHTS_NAME
    if safetensors_path.exists():
        return load_safetensors_file(str(safetensors_path), device="cpu")

    bin_path = adapter_dir / BIN_WEIGHTS_NAME
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"Missing adapter weights in {adapter_dir}: expected "
        f"{SAFETENSORS_WEIGHTS_NAME} or {BIN_WEIGHTS_NAME}"
    )


def _normalize_state_dict_for_peft_save(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert internal D2L LoRA orientation to PEFT's on-disk layout when needed."""
    normalized = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
    }
    for key, value in list(normalized.items()):
        if ".lora_B." not in key or value.ndim != 2:
            continue
        prefix = key.split(".lora_B.")[0]
        key_a = f"{prefix}.lora_A.weight"
        if key_a not in normalized:
            continue
        rank = normalized[key_a].shape[0]
        if value.shape[0] == rank:
            normalized[key] = value.transpose(0, 1).contiguous()
    return normalized
