"""EXP-002: S1 Classical RAG Baseline.

Builds the hybrid RAG index over 8-doc corpus, runs retrieval + generation
on 50 eval questions, and produces evaluation metrics.

Usage (from repo root):
    python experiments/EXP-002_s1_rag_baseline/run.py [--skip-index] [--subset N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (for OPENAI_API_KEY etc.)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import torch

import config as cfg
from external.pdf_rag_pipeline import PipelineConfig
from src.data.io import load_goldset, load_json, save_json
from src.evaluation.runner import EvalRunner
from src.evaluation.schemas import PageRef, Prediction
from src.generation.loader import load_backbone, unload_model
from src.generation.pipeline import GenerationPipeline
from src.generation.prompt import format_context_from_chunks
from src.retrieval.indexer import build_rag_index
from src.retrieval.staged import staged_retrieve_all

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

INDEX_OUTPUT_DIR = cfg.RESULTS_DIR / "EXP-002" / "index"
RESULTS_OUTPUT_DIR = cfg.RESULTS_DIR / "EXP-002"


def _free_gpu():
    """Force-free all GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main(skip_index: bool = False, subset: int | None = None) -> None:
    logger.info("=== EXP-002: S1 Classical RAG Baseline ===")

    # ── 1. Load goldset + split ──────────────────────────────────────
    refs = load_goldset(cfg.GOLDSET_PATH)
    refs_by_id = {r["question_id"]: r for r in refs}
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    eval_ids = split["eval"]
    eval_refs = [refs_by_id[qid] for qid in eval_ids if qid in refs_by_id]

    if subset:
        eval_refs = eval_refs[:subset]
        logger.info("Running on subset: %d questions", len(eval_refs))
    else:
        logger.info("Running on full eval set: %d questions", len(eval_refs))

    # ── 2. Build or load RAG index ───────────────────────────────────
    pipeline_config = PipelineConfig(
        documents_dir=cfg.CORPUS_DIR,
        output_dir=INDEX_OUTPUT_DIR,
    )

    if skip_index and pipeline_config.qdrant_dir.exists():
        logger.info("Skipping index build (--skip-index), loading existing index")
    else:
        logger.info("Building RAG index...")
        t0 = time.perf_counter()
        pipeline_config = build_rag_index(
            corpus_dir=cfg.CORPUS_DIR,
            output_dir=INDEX_OUTPUT_DIR,
            goldset_path=cfg.GOLDSET_PATH,
            pipeline_config=pipeline_config,
        )
        index_time = time.perf_counter() - t0
        logger.info("Index built in %.1f seconds", index_time)
        # Free embedding model VRAM after indexing
        _free_gpu()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Staged retrieval (sequential model loading for 8GB VRAM)
    # ══════════════════════════════════════════════════════════════════
    logger.info("PHASE 1: Staged retrieval for %d questions...", len(eval_refs))
    questions = [ref["question"] for ref in eval_refs]
    retrieval_results = staged_retrieve_all(
        questions=questions,
        pipeline_config=pipeline_config,
        candidate_budget=pipeline_config.candidate_budget,
    )

    # Build retrieval cache: qid -> (context_str, page_refs)
    retrieval_cache: dict[str, tuple] = {}
    for ref, result in zip(eval_refs, retrieval_results):
        qid = ref["question_id"]
        context = format_context_from_chunks(result.evidence_chunks)
        page_refs = [
            PageRef(doc_id=pr.doc_id, page_number=pn)
            for pr in result.page_references
            for pn in pr.page_numbers
        ]
        retrieval_cache[qid] = (context, page_refs)
        logger.info("  %s: %d evidence, %d pages", qid[:12],
                     len(result.evidence_chunks), len(page_refs))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Generate all answers (backbone on GPU)
    # ══════════════════════════════════════════════════════════════════
    logger.info("PHASE 2: Generation for %d questions...", len(eval_refs))
    model, tokenizer = load_backbone(
        model_name=cfg.BACKBONE_MODEL,
        quantization=cfg.BACKBONE_QUANTIZATION,
    )
    gen_pipeline = GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        max_retries=1,
    )

    predictions: list[Prediction] = []
    peak_vram_mb = 0.0

    for i, ref in enumerate(eval_refs):
        qid = ref["question_id"]
        answer_type = ref["answer_type"]
        context, page_refs = retrieval_cache[qid]

        logger.info("[gen %d/%d] %s (%s)", i + 1, len(eval_refs), qid[:12], answer_type)

        pred = gen_pipeline.generate_answer(
            question=ref["question"],
            answer_type=answer_type,
            context=context,
            question_id=qid,
        )
        pred.predicted_pages = page_refs
        predictions.append(pred)

        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / 1024 / 1024
            peak_vram_mb = max(peak_vram_mb, vram)

        logger.info(
            "  → output=%s | malformed=%s | ttft=%.0fms | total=%.0fms",
            str(pred.parsed_answer)[:60],
            pred.is_malformed,
            pred.ttft_ms or 0,
            pred.latency_ms or 0,
        )

    unload_model(model)

    # ── 7. Save raw predictions ──────────────────────────────────────
    save_json(
        [p.model_dump() for p in predictions],
        RESULTS_OUTPUT_DIR / "predictions.json",
    )

    # ── 8. Run evaluation ────────────────────────────────────────────
    logger.info("Running evaluation...")
    eval_runner = EvalRunner(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
        judge_model=cfg.JUDGE_MODEL,
        judge_reasoning=cfg.JUDGE_REASONING,
        grounding_beta=cfg.GROUNDING_BETA,
        q_main_weights=cfg.Q_MAIN_WEIGHTS,
    )

    report = eval_runner.evaluate(
        predictions=predictions,
        system_id="S1",
        experiment_id="EXP-002",
        split="eval",
        compute_grounding_flag=True,
    )

    # ── 9. Save results ──────────────────────────────────────────────
    eval_runner.save_report(report, RESULTS_OUTPUT_DIR)

    # Systems metrics
    systems_metrics = {
        "peak_vram_mb": peak_vram_mb,
        "ttft_median_ms": report.ttft_median_ms,
        "ttft_p95_ms": report.ttft_p95_ms,
        "latency_median_ms": report.latency_median_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "malformed_rate": report.malformed_rate,
    }
    save_json(systems_metrics, RESULTS_OUTPUT_DIR / "systems_metrics.json")

    # ── 10. Print summary ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXP-002 RESULTS (S1 Classical RAG)")
    logger.info("=" * 60)
    logger.info("Q_main:  %.4f", report.q_main)
    logger.info("S_det:   %.4f", report.s_det)
    logger.info("S_asst:  %.4f", report.s_asst)
    if report.grounding_f_beta is not None:
        logger.info("G(F_β):  %.4f", report.grounding_f_beta)
    logger.info("Malformed: %.1f%%", report.malformed_rate * 100)
    logger.info("Peak VRAM: %.0f MB", peak_vram_mb)
    logger.info("-" * 60)
    for atype, metrics in report.breakdown_by_type.items():
        logger.info("  %s: %s", atype, metrics)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-002: S1 Classical RAG Baseline")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building if it exists")
    parser.add_argument("--subset", type=int, default=None, help="Run on first N eval questions only")
    args = parser.parse_args()
    main(skip_index=args.skip_index, subset=args.subset)
