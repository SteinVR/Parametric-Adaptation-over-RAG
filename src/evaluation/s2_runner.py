"""S2 inference runner: S1 retrieval plus adapter-backed generation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import torch

from src.rag_pipeline import PipelineConfig
from src.evaluation.schemas import PageRef, Prediction
from src.generation.adapters import load_backbone_with_adapter
from src.generation.loader import unload_model
from src.generation.pipeline import GenerationPipeline
from src.generation.prompt import format_context_from_chunks
from src.retrieval.staged import staged_retrieve_all

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedEvalSample:
    """Question plus frozen retrieval context used across all seeds."""

    question_id: str
    question: str
    answer_type: str
    context: str
    predicted_pages: list[PageRef]


def prepare_s2_eval_samples(
    *,
    refs: list[dict],
    corpus_dir: Path,
    index_output_dir: Path,
) -> list[RetrievedEvalSample]:
    """Run the S1 retrieval pipeline once and cache eval contexts."""

    pipeline_config = PipelineConfig(
        documents_dir=corpus_dir,
        output_dir=index_output_dir,
    )
    retrieval_results = staged_retrieve_all(
        questions=[str(ref["question"]) for ref in refs],
        pipeline_config=pipeline_config,
        candidate_budget=pipeline_config.candidate_budget,
    )
    samples: list[RetrievedEvalSample] = []
    for ref, result in zip(refs, retrieval_results):
        samples.append(
            RetrievedEvalSample(
                question_id=str(ref["question_id"]),
                question=str(ref["question"]),
                answer_type=str(ref["answer_type"]),
                context=format_context_from_chunks(result.evidence_chunks),
                predicted_pages=[
                    PageRef(doc_id=page_ref.doc_id, page_number=page_number)
                    for page_ref in result.page_references
                    for page_number in page_ref.page_numbers
                ],
            )
        )
    logger.info("Prepared %d retrieval-backed eval samples", len(samples))
    return samples


def run_s2_generation(
    *,
    model_name: str,
    adapter_dir: Path,
    eval_samples: list[RetrievedEvalSample],
    max_new_tokens: int = 256,
) -> tuple[list[Prediction], float | None]:
    """Generate predictions for one seed using the frozen retrieval cache."""

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
    for sample in eval_samples:
        prediction = pipeline.generate_answer(
            question=sample.question,
            answer_type=sample.answer_type,
            context=sample.context,
            question_id=sample.question_id,
        )
        prediction.predicted_pages = sample.predicted_pages
        predictions.append(prediction)
        if torch.cuda.is_available():
            peak_vram_mb = max(
                peak_vram_mb,
                torch.cuda.max_memory_allocated() / 1024 / 1024,
            )

    unload_model(model)
    return predictions, (peak_vram_mb if torch.cuda.is_available() else None)
