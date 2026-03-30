"""S2 closed-book inference runner: adapter-only generation, no retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.evaluation.schemas import Prediction
from src.generation.adapters import load_backbone_with_adapter
from src.generation.loader import unload_model
from src.generation.pipeline import GenerationPipeline
from src.generation.prompt import format_closed_book_prompt

logger = logging.getLogger(__name__)


def run_s2_closed_generation(
    *,
    model_name: str,
    adapter_dir: Path,
    eval_refs: list[dict],
    max_new_tokens: int = 256,
) -> tuple[list[Prediction], float | None]:
    """Generate closed-book predictions for one seed. No retrieval."""

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
    for ref in eval_refs:
        prompt_text = format_closed_book_prompt(
            question=str(ref["question"]),
            answer_type=str(ref["answer_type"]),
        )
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

    unload_model(model)
    return predictions, (peak_vram_mb if torch.cuda.is_available() else None)
