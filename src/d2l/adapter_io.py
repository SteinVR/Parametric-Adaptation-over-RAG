"""Export helpers for Doc-to-LoRA generated adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig


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

    weight_path = adapter_dir / "adapter_model.bin"
    torch.save(state_dict, weight_path)
    config_path = adapter_dir / "adapter_config.json"
    byte_size = sum(
        path.stat().st_size for path in adapter_dir.iterdir() if path.is_file()
    )
    return ExportedAdapterArtifact(
        adapter_dir=adapter_dir,
        config_path=config_path,
        weight_path=weight_path,
        tensor_count=len(state_dict),
        byte_size=byte_size,
    )


def load_peft_lora_state_dict(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Load a PEFT adapter state dict from disk."""
    weight_path = adapter_dir / "adapter_model.bin"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {weight_path}")
    return torch.load(weight_path, map_location="cpu", weights_only=False)
