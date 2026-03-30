"""No-retrieval inference runner for Doc-to-LoRA adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch

from src.d2l.prompt import format_d2l_no_retrieval_prompt
from src.evaluation.schemas import Prediction
from src.generation.adapters import load_backbone_with_adapter
from src.generation.loader import unload_model
from src.generation.pipeline import GenerationPipeline

logger = logging.getLogger(__name__)


PromptBuilder = Callable[[str, str], str]


def run_d2l_no_retrieval_generation(
    *,
    model_name: str,
    adapter_dir: Path,
    eval_refs: list[dict[str, Any]],
    max_new_tokens: int = 256,
    prompt_builder: PromptBuilder = format_d2l_no_retrieval_prompt,
) -> tuple[list[Prediction], float | None]:
    """Generate predictions with a PEFT adapter and the frozen EXP-004 prompt."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_backbone_with_adapter(
        model_name=model_name,
        adapter_dir=adapter_dir,
    )
    pipeline = GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        max_retries=1,
    )

    predictions: list[Prediction] = []
    peak_vram_mb = 0.0
    try:
        for ref in eval_refs:
            prompt_text = prompt_builder(str(ref["question"]), str(ref["answer_type"]))
            prediction = pipeline.generate_answer(
                question=str(ref["question"]),
                answer_type=str(ref["answer_type"]),
                question_id=str(ref["question_id"]),
                prompt_text=prompt_text,
            )
            predictions.append(prediction)
            if torch.cuda.is_available():
                peak_vram_mb = max(
                    peak_vram_mb,
                    torch.cuda.max_memory_allocated() / 1024 / 1024,
                )
    finally:
        unload_model(model)

    return predictions, (peak_vram_mb if torch.cuda.is_available() else None)
