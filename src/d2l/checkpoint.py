"""Doc-to-LoRA checkpoint loading and validation."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import Tensor
from transformers import BitsAndBytesConfig

from ctx_to_lora.modeling import hypernet as hypernet_module
from ctx_to_lora.modeling import idefics2 as idefics2_module
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "pytorch_model.bin"

# Pretrained hypernetwork checkpoint published by SakanaAI on the Hugging Face Hub.
# https://huggingface.co/SakanaAI/doc-to-lora/tree/main/gemma_demo/checkpoint-80000
D2L_HF_REPO_ID = "SakanaAI/doc-to-lora"
D2L_HF_CHECKPOINT_SUBFOLDER = "gemma_demo/checkpoint-80000"


def resolve_checkpoint_file(
    checkpoint_root: Path | str,
    *,
    download_if_missing: bool = True,
) -> Path:
    """Resolve the concrete D2L checkpoint file from a directory or file path.

    Looks for the checkpoint locally first. When it is absent and
    ``download_if_missing`` is set, the ~1.3 GB ``pytorch_model.bin`` is fetched
    from the Hugging Face Hub (``SakanaAI/doc-to-lora``) into the local Hub cache.
    """
    path = Path(checkpoint_root)
    checkpoint_file = path if path.suffix == ".bin" else path / CHECKPOINT_FILENAME
    if checkpoint_file.exists():
        return checkpoint_file

    if not download_if_missing:
        raise FileNotFoundError(
            f"Doc-to-LoRA checkpoint not found: {checkpoint_file}. "
            "Pass download_if_missing=True or fetch it from "
            f"https://huggingface.co/{D2L_HF_REPO_ID}."
        )

    logger.info(
        "Local Doc-to-LoRA checkpoint missing (%s); downloading from Hugging Face Hub %s/%s",
        checkpoint_file,
        D2L_HF_REPO_ID,
        D2L_HF_CHECKPOINT_SUBFOLDER,
    )
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=D2L_HF_REPO_ID,
        filename=CHECKPOINT_FILENAME,
        subfolder=D2L_HF_CHECKPOINT_SUBFOLDER,
    )
    return Path(downloaded)


def load_d2l_model(
    checkpoint_root: Path | str,
    *,
    base_model_kwargs: dict[str, Any] | None = None,
    use_flash_attn: bool = False,
    train: bool = False,
    offload_hypernet: bool = True,
) -> ModulatedPretrainedModel:
    """Load the frozen Doc-to-LoRA hypernetwork checkpoint.

    Applies a local lean-loader shim to make D2L fit on 8 GB VRAM / 30 GB RAM:

    1. **Skip redundant ``_init_model``** — D2L's ``from_state_dict`` calls
       ``_init_model()`` twice (once in ``__init__``, once in ``load_state_dict``),
       creating two full copies of the ctx_encoder and hypernetwork.

    2. **Load ctx/hypernet directly on CPU** — the upstream `_init_model()` loads
       a second Gemma copy as ``ctx_encoder`` plus the hypernetwork on GPU, then
       the current local code offloads them afterwards. This still spikes VRAM and
       RAM during construction, so we patch `_init_model()` during loading.

    3. **Keep generated ctx features off GPU** — upstream `generate_weights()`
       always moves `ctx_features` to `self.device` (the base model GPU), which
       recreates the OOM even if the hypernetwork was offloaded to CPU. We bind a
       lean `generate_weights()` implementation that keeps the ctx path on CPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Doc-to-LoRA packaging requires a CUDA-visible runtime.")

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
    model_kwargs.setdefault("low_cpu_mem_usage", True)

    with _patched_d2l_loader_methods():
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=train,
            base_model_kwargs=model_kwargs,
            use_flash_attn=use_flash_attn,
            use_sequence_packing=False,
        )
    del state_dict
    gc.collect()
    _empty_cuda_cache()

    model.reset()

    if offload_hypernet:
        _configure_cpu_runtime(model)

    return model


@contextmanager
def _patched_d2l_loader_methods():
    """Temporarily patch D2L to use the local low-memory load path."""
    original_load_state_dict = ModulatedPretrainedModel.load_state_dict
    original_init_model = ModulatedPretrainedModel._init_model
    ModulatedPretrainedModel.load_state_dict = _lean_load_state_dict
    ModulatedPretrainedModel._init_model = _lean_init_model
    try:
        yield
    finally:
        ModulatedPretrainedModel.load_state_dict = original_load_state_dict
        ModulatedPretrainedModel._init_model = original_init_model


def _lean_load_state_dict(
    self: ModulatedPretrainedModel,
    state_dict: dict[str, Any],
    *_args: Any,
    **_kwargs: Any,
) -> Any:
    """Replacement for ``ModulatedPretrainedModel.load_state_dict``.

    Skips the redundant ``_init_model()`` call — the model was already
    initialised by ``__init__`` moments ago with the same configs.
    """
    self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
    self.hypernet_config = state_dict.pop("hypernet_config")
    self.ctx_encoder_args = state_dict.pop("ctx_encoder_args")

    if self.base_model_name_or_path != self.base_model.name_or_path:
        raise ValueError(
            f"Base model name mismatch: model={self.base_model.name_or_path}, "
            f"checkpoint={self.base_model_name_or_path}"
        )

    cleaned_state_dict = _remove_compile_prefix(state_dict)
    load_result = self.hypernet.load_state_dict(cleaned_state_dict, strict=True)
    cleaned_state_dict.clear()
    gc.collect()
    logger.info("load result: %s", load_result)
    return load_result


def _lean_init_model(self: ModulatedPretrainedModel) -> None:
    """Replacement for D2L's `_init_model()` with CPU-first ctx/hypernet loading."""
    self.base_model.disable_adapter_layers()
    self.hypernet = hypernet_module.HyperLoRA(self.hypernet_config).to("cpu")
    self.hypernet = self.hypernet.to(torch.float32)
    self.patch_lora_forward()
    self.ctx_encoder = _load_ctx_encoder_on_cpu(self)


def _load_ctx_encoder_on_cpu(self: ModulatedPretrainedModel):
    """Build the context encoder directly on CPU to avoid GPU load spikes."""
    ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
    if ctx_model_name is None:
        ctx_model_name = self.base_model.config.name_or_path

    ctx_model_kwargs = {
        "device_map": "cpu",
        "local_files_only": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    quantize_ctx_encoder = bool(self.ctx_encoder_args.quantize_ctx_encoder)

    try:
        encoder_model = hypernet_module.get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=False,
            model_kwargs=ctx_model_kwargs,
            use_q_lora=quantize_ctx_encoder,
            device="cpu",
            dtype=torch.bfloat16,
        )
    except Exception as exc:
        if not quantize_ctx_encoder:
            raise
        logger.warning(
            "CPU 4-bit ctx_encoder init failed (%s). Falling back to CPU bf16.",
            exc,
        )
        gc.collect()
        _empty_cuda_cache()
        encoder_model = hypernet_module.get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=False,
            model_kwargs=ctx_model_kwargs,
            use_q_lora=False,
            device="cpu",
            dtype=torch.bfloat16,
        )

    return hypernet_module.CTX_ENCODER_CLS[self.ctx_encoder_args.ctx_encoder_type](
        encoder_model, self.ctx_encoder_args
    )


def _remove_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop the `torch.compile` key prefix used by some saved checkpoints."""
    compiled_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(compiled_prefix):
            state_dict[key[len(compiled_prefix) :]] = state_dict.pop(key)
    return state_dict


def _configure_cpu_runtime(model: ModulatedPretrainedModel) -> None:
    """Bind a CPU-first ctx/hypernet runtime to the loaded D2L model.

    `_lean_init_model()` already builds the ctx encoder and hypernetwork on CPU.
    Here we enforce those placements again, then replace `generate_weights()` so
    the path no longer copies multi-GB ctx features to the base-model GPU.
    """
    model.hypernet.to("cpu")
    model.ctx_encoder.to("cpu")
    model.enable_iterative_mode(True)
    _patch_model_perceiver_runtime(model)
    model.generate_weights = MethodType(_lean_generate_weights, model)
    gc.collect()
    _empty_cuda_cache()
    logger.info("Configured CPU runtime for hypernetwork + ctx_encoder")


def _patch_model_perceiver_runtime(model: ModulatedPretrainedModel) -> None:
    """Patch only this model's eager perceiver stack for CPU iterative inference."""
    perceiver = getattr(getattr(model.hypernet, "aggregator", None), "perceiver", None)
    if perceiver is None:
        return

    for resampler in (perceiver.encoder, perceiver.decoder):
        if getattr(resampler, "_d2l_patched", False):
            continue

        resampler.forward = MethodType(_patched_resampler_forward, resampler)
        resampler._d2l_patched = True

        for layer in resampler.layers:
            self_attn = layer.self_attn
            if getattr(self_attn, "_d2l_patched", False):
                continue
            self_attn.forward = MethodType(_patched_attention_forward, self_attn)
            self_attn._d2l_patched = True

    logger.info("Patched model-local Idefics2 perceiver runtime for eager iterative inference")


def _patched_attention_forward(
    self,
    latents: Tensor,
    context: Tensor | None = None,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    past_key_value: tuple[Tensor] | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    is_cross_attn: bool = False,
    **_kwargs: Any,
) -> tuple[Tensor, Tensor | None, tuple[Tensor] | None]:
    """Instance-scoped eager attention compatible with D2L's `is_cross_attn` call."""
    del position_ids, use_cache

    if latents.ndim != 3:
        raise ValueError(f"Expected 3D latents for patched perceiver attention, got {latents.shape}")
    kv_inp = context if is_cross_attn else latents
    if kv_inp is None or kv_inp.ndim != 3:
        raise ValueError(f"Expected 3D context for patched perceiver attention, got {None if kv_inp is None else kv_inp.shape}")

    bsz, q_len, _ = latents.size()
    kv_seq_len = kv_inp.size(1)

    query_states = self.q_proj(latents)
    key_states = self.k_proj(kv_inp)
    value_states = self.v_proj(kv_inp)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx
        )

    key_states = idefics2_module.repeat_kv(key_states, self.num_key_value_groups)
    value_states = idefics2_module.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / (self.head_dim**0.5)

    if attention_mask is not None:
        expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
        if attention_mask.size() != expected_mask_shape:
            raise ValueError(
                f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _patched_resampler_forward(
    self,
    context: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> Tensor:
    """Instance-scoped eager resampler that avoids flash-only `unpad_input` paths."""
    del position_ids

    if context.ndim != 3:
        raise ValueError(f"Expected 3D context for patched perceiver resampler, got {context.shape}")

    bsz = context.shape[0]
    latents = self.latents_q.unsqueeze(0).expand((bsz, *self.latents_q.size()))
    compressed_context = latents

    if attention_mask is not None:
        attention_mask = idefics2_module._prepare_4d_attention_mask(
            attention_mask,
            latents.dtype,
            tgt_len=self.n_latents,
        )

    for layer in self.layers:
        layer_outputs = layer(
            latents=compressed_context,
            context=context,
            attention_mask=attention_mask if layer.is_cross_attn else None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        compressed_context = layer_outputs[0]

    return self.layernorm(compressed_context)


def _lean_generate_weights(
    self: ModulatedPretrainedModel,
    ctx_ids: Tensor,
    ctx_attn_mask: Tensor | None = None,
    ctx_position_ids: Tensor | None = None,
    **kwargs: Any,
):
    """Run ctx encoding and hypernet generation without a GPU round-trip."""
    with torch.no_grad():
        ctx_input_device = self.ctx_encoder.base_model.get_input_embeddings().weight.device
        ctx_ids = ctx_ids.to(ctx_input_device)
        if ctx_attn_mask is not None:
            ctx_attn_mask = ctx_attn_mask.to(ctx_input_device)
        if ctx_position_ids is not None:
            ctx_position_ids = ctx_position_ids.to(ctx_input_device)

        ctx_encoder_kwargs = {
            "input_ids": ctx_ids,
            "attention_mask": ctx_attn_mask,
            "position_ids": ctx_position_ids,
        }
        if isinstance(self.ctx_encoder.base_model, hypernet_module.ModernBertModel):
            assert ctx_position_ids is not None
            position_ids = ctx_position_ids.flatten()
            indices = torch.arange(
                position_ids.size(0), device=position_ids.device, dtype=torch.int32
            )
            cu_seqlens = torch.cat(
                (
                    indices[position_ids == 0],
                    torch.tensor(
                        position_ids.size(),
                        device=position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            )
            ctx_encoder_kwargs = {
                "input_ids": ctx_ids.squeeze(0),
                "cu_seqlens": cu_seqlens,
                "max_seqlen": position_ids.max() + 1,
                "attention_mask": -1,
                "seq_len": -1,
                "batch_size": -1,
            }

        ctx_features = self.ctx_encoder(**ctx_encoder_kwargs, **kwargs)
        if isinstance(self.ctx_encoder.base_model, hypernet_module.ModernBertModel):
            ctx_features = ctx_features.unsqueeze(0)

        hypernet_device = next(self.hypernet.parameters()).device
        hypernet_dtype = next(self.hypernet.parameters()).dtype
        ctx_features = ctx_features.to(device=hypernet_device, dtype=hypernet_dtype)
        if ctx_attn_mask is not None:
            ctx_attn_mask = ctx_attn_mask.to(hypernet_device)
        if ctx_position_ids is not None:
            ctx_position_ids = ctx_position_ids.to(hypernet_device)

    if self.user_defined_scaling == 1:
        return self.hypernet.generate_weights(
            ctx_features, ctx_attn_mask, ctx_position_ids
        )

    lora_dict, _ = self.hypernet.generate_weights(
        ctx_features, ctx_attn_mask, ctx_position_ids
    )
    for module_name in lora_dict:
        lora_dict[module_name]["A"] = (
            lora_dict[module_name]["A"] * self.user_defined_scaling
        )
        lora_dict[module_name]["B"] = (
            lora_dict[module_name]["B"] * self.user_defined_scaling
        )
    return lora_dict, None


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
