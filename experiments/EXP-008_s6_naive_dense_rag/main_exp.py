"""EXP-008: S6 Naive Dense RAG ablation.

Self-contained experiment: FAISS microchunk-only index → dense top-5 retrieval →
base Gemma-2-2b-it generation (no adapter, no reranker, no evidence compression).
All S6-specific code lives in this file — no pollution of the S1 pipeline.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import config as cfg
import torch
from src.data.io import load_goldset, load_json, save_json
from src.evaluation.runner import EvalRunner
from src.evaluation.schemas import PageRef, Prediction
from src.generation.loader import load_backbone, unload_model
from src.generation.pipeline import GenerationPipeline
from src.generation.prompt import format_context_from_chunks
from src.retrieval.indexer import build_doc_id_map

from external.pdf_rag_pipeline import (
    build_corpus,
    build_dense_embedder,
    build_index_chunks,
    build_query_embedder,
    parse_pdf,
    serialize_document_tables,
)

logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)

_EXP_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "exp008_config",
    Path(__file__).with_name("config.py"),
)
if _EXP_CONFIG_SPEC is None or _EXP_CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load experiment config")
exp_cfg = importlib.util.module_from_spec(_EXP_CONFIG_SPEC)
_EXP_CONFIG_SPEC.loader.exec_module(exp_cfg)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FaissChunkMeta:
    """Metadata for a chunk stored in FAISS (keyed by row index)."""
    chunk_id: str
    doc_id: str
    page_span: list[int]
    text: str
    chunk_type: str


@dataclass
class RetrievedChunk:
    """Chunk returned by FAISS retrieval (duck-types for format_context_from_chunks)."""
    doc_id: str
    page_span: list[int]
    text: str
    score: float


# ── FAISS index building ─────────────────────────────────────────────────────

def build_s6_index(
    corpus_dir: Path,
    goldset_path: Path,
    output_dir: Path,
) -> tuple[object, list[FaissChunkMeta]]:
    """Build FAISS IndexFlatIP from microchunk-only chunks. Returns (index, metadata)."""
    import faiss

    stem_to_sha = build_doc_id_map(goldset_path)
    logger.info("Doc ID map: %d documents", len(stem_to_sha))

    # Parse PDFs → pages
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    logger.info("Parsing %d PDFs", len(pdf_files))
    all_pages = []
    for pdf_path in pdf_files:
        doc = parse_pdf(pdf_path)
        table_blocks = serialize_document_tables(doc)
        pages = build_corpus(doc, table_blocks)
        sha256_id = stem_to_sha.get(doc.doc_id, doc.doc_id)
        for page in pages:
            page.doc_id = sha256_id
        all_pages.extend(pages)
        logger.info("  %s → %d pages", pdf_path.name, len(pages))

    # Chunk (microchunk only)
    chunks = build_index_chunks(
        all_pages,
        enabled_chunk_families=exp_cfg.ENABLED_CHUNK_FAMILIES,
        token_chunk_size=exp_cfg.TOKEN_CHUNK_SIZE,
        token_chunk_overlap=exp_cfg.TOKEN_CHUNK_OVERLAP,
    )
    # Remap any lingering stem doc_ids
    for chunk in chunks:
        if chunk.doc_id not in stem_to_sha.values():
            remapped = stem_to_sha.get(chunk.doc_id)
            if remapped:
                chunk.doc_id = remapped
    logger.info("Generated %d microchunks", len(chunks))

    # Embed (dense only, no BM25)
    logger.info("Embedding %d chunks with dense embedder...", len(chunks))
    embedder = build_dense_embedder()
    chunk_texts = [c.text for c in chunks]
    dense_vectors = embedder.encode(chunk_texts)
    _free_gpu()

    # Build FAISS index
    vectors = np.array(dense_vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)  # cosine via IP on normalized vectors
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("FAISS index: %d vectors, dim=%d", index.ntotal, dim)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = [
        FaissChunkMeta(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            page_span=list(c.page_span),
            text=c.text,
            chunk_type=c.chunk_type,
        )
        for c in chunks
    ]
    save_json(
        [{"chunk_id": m.chunk_id, "doc_id": m.doc_id, "page_span": m.page_span,
          "text": m.text, "chunk_type": m.chunk_type} for m in metadata],
        output_dir / "chunk_metadata.json",
    )
    logger.info("FAISS index saved to %s", output_dir)
    return index, metadata


def load_s6_index(index_dir: Path) -> tuple[object, list[FaissChunkMeta]]:
    """Load pre-built FAISS index + metadata."""
    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    raw_meta = load_json(index_dir / "chunk_metadata.json")
    metadata = [
        FaissChunkMeta(
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            page_span=m["page_span"],
            text=m["text"],
            chunk_type=m["chunk_type"],
        )
        for m in raw_meta
    ]
    logger.info("Loaded FAISS index: %d vectors, %d metadata entries", index.ntotal, len(metadata))
    return index, metadata


# ── FAISS retrieval ──────────────────────────────────────────────────────────

def faiss_retrieve_all(
    questions: list[str],
    index: object,
    metadata: list[FaissChunkMeta],
    top_k: int = 5,
) -> list[list[RetrievedChunk]]:
    """Dense-only retrieval: embed queries → FAISS search → top-k chunks."""
    import faiss as faiss_lib

    logger.info("Embedding %d queries for FAISS retrieval...", len(questions))
    query_embedder = build_query_embedder()
    query_vectors = np.array(query_embedder.encode(questions), dtype=np.float32)
    faiss_lib.normalize_L2(query_vectors)
    _free_gpu()

    logger.info("Searching FAISS index (top_k=%d)...", top_k)
    scores, indices = index.search(query_vectors, top_k)

    results: list[list[RetrievedChunk]] = []
    for q_idx in range(len(questions)):
        chunks = []
        for rank in range(top_k):
            faiss_idx = int(indices[q_idx, rank])
            if faiss_idx < 0:
                continue
            meta = metadata[faiss_idx]
            chunks.append(RetrievedChunk(
                doc_id=meta.doc_id,
                page_span=meta.page_span,
                text=meta.text,
                score=float(scores[q_idx, rank]),
            ))
        results.append(chunks)
    logger.info("Retrieved chunks for %d questions", len(results))
    return results


def chunks_to_page_refs(chunks: list[RetrievedChunk]) -> list[PageRef]:
    """Deduplicate (doc_id, page_number) pairs from retrieved chunks."""
    seen: set[tuple[str, int]] = set()
    refs: list[PageRef] = []
    for chunk in chunks:
        for page in chunk.page_span:
            key = (chunk.doc_id, page)
            if key not in seen:
                seen.add(key)
                refs.append(PageRef(doc_id=chunk.doc_id, page_number=page))
    return refs


# ── Generation ───────────────────────────────────────────────────────────────

def run_s6_generation(
    eval_refs: list[dict],
    retrieval_results: list[list[RetrievedChunk]],
) -> tuple[list[Prediction], float | None]:
    """Generate S6 predictions: base model (no adapter) + naive retrieved context."""

    model, tokenizer = load_backbone(
        model_name=cfg.BACKBONE_MODEL,
    )
    pipeline = GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=exp_cfg.MAX_NEW_TOKENS,
        temperature=0.0,
        do_sample=False,
        max_retries=1,
    )

    predictions: list[Prediction] = []
    peak_vram_mb = 0.0
    for ref, chunks in zip(eval_refs, retrieval_results):
        context = format_context_from_chunks(chunks)
        prediction = pipeline.generate_answer(
            question=str(ref["question"]),
            answer_type=str(ref["answer_type"]),
            question_id=str(ref["question_id"]),
            context=context,
        )
        prediction.predicted_pages = chunks_to_page_refs(chunks)
        predictions.append(prediction)
        if torch.cuda.is_available():
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

    unload_model(model)
    return predictions, (peak_vram_mb if torch.cuda.is_available() else None)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-008: S6 Naive Dense RAG")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with 2 questions")
    parser.add_argument("--skip-index", action="store_true", help="Skip index build (reuse existing)")
    args = parser.parse_args()

    logger.info("=== EXP-008 S6 Naive Dense RAG start === mode=%s", "smoke" if args.smoke else "full")

    # ── Phase 1: Build FAISS index ──
    if args.skip_index and exp_cfg.FAISS_INDEX_DIR.exists():
        logger.info("── Phase 1: Loading existing FAISS index ──")
        index, metadata = load_s6_index(exp_cfg.FAISS_INDEX_DIR)
    else:
        logger.info("── Phase 1: Building FAISS microchunk index ──")
        index, metadata = build_s6_index(
            corpus_dir=cfg.CORPUS_DIR,
            goldset_path=cfg.GOLDSET_PATH,
            output_dir=exp_cfg.FAISS_INDEX_DIR,
        )

    # ── Phase 2: Dense retrieval ──
    logger.info("── Phase 2: Dense-only retrieval ──")
    eval_refs = _load_eval_refs(limit=exp_cfg.SMOKE_EVAL_QUESTIONS if args.smoke else None)
    logger.info("Eval refs: %d questions", len(eval_refs))

    questions = [str(ref["question"]) for ref in eval_refs]
    retrieval_results = faiss_retrieve_all(
        questions=questions,
        index=index,
        metadata=metadata,
        top_k=exp_cfg.TOP_K,
    )

    # ── Phase 3: Generation ──
    logger.info("── Phase 3: Generation (base model, no adapter) ──")
    _free_gpu()
    predictions, peak_infer_vram_mb = run_s6_generation(eval_refs, retrieval_results)
    logger.info("Generation done: %d predictions, peak VRAM %.1f MB", len(predictions), peak_infer_vram_mb or 0)

    # ── Phase 4: Evaluation ──
    logger.info("── Phase 4: Scoring ──")
    result_dir = exp_cfg.RESULTS_DIR / ("smoke" if args.smoke else "main")
    eval_runner = EvalRunner(
        goldset_path=cfg.GOLDSET_PATH,
        split_path=cfg.DATA_SPLITS / "split_v1.json",
        judge_model=cfg.JUDGE_MODEL,
        judge_reasoning=cfg.JUDGE_REASONING,
        grounding_beta=cfg.GROUNDING_BETA,
        q_main_weights=cfg.Q_MAIN_WEIGHTS,
    )

    save_json([p.model_dump() for p in predictions], result_dir / "predictions.json")
    report = eval_runner.evaluate(
        predictions=predictions,
        system_id="S6",
        experiment_id=exp_cfg.EXPERIMENT_ID,
        split="eval",
        compute_grounding_flag=True,
    )
    eval_runner.save_report(report, result_dir)
    save_json({"peak_infer_vram_mb": peak_infer_vram_mb}, result_dir / "systems_metrics.json")

    logger.info(
        "S6 scored: Q_main=%.4f, S_det=%.4f, S_asst=%.4f, G=%.4f",
        report.q_main, report.s_det, report.s_asst, report.grounding_f_beta or 0,
    )

    # ── Delta vs S1 ──
    if exp_cfg.EXP002_REPORT_PATH.exists():
        s1_report = load_json(exp_cfg.EXP002_REPORT_PATH)
        delta = {
            "Q_main": report.q_main - s1_report["q_main"],
            "S_det": report.s_det - s1_report["s_det"],
            "S_asst": report.s_asst - s1_report["s_asst"],
            "G": (report.grounding_f_beta or 0) - (s1_report.get("grounding_f_beta") or 0),
        }
        save_json(delta, result_dir / "delta_vs_s1.json")
        logger.info("Delta vs S1: Q_main=%+.4f, G=%+.4f", delta["Q_main"], delta["G"])

    # ── Retrieval overlap with S1 ──
    _compute_retrieval_overlap(eval_refs, retrieval_results, result_dir)

    if not args.smoke:
        _write_report(report, result_dir)

    logger.info("=== EXP-008 complete ===")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_eval_refs(limit: int | None) -> list[dict]:
    refs = load_goldset(cfg.GOLDSET_PATH)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(cfg.DATA_SPLITS / "split_v1.json")
    eval_refs = [refs_by_id[qid] for qid in split["eval"]]
    return eval_refs[:limit] if limit is not None else eval_refs


def _compute_retrieval_overlap(
    eval_refs: list[dict],
    s6_results: list[list[RetrievedChunk]],
    output_dir: Path,
) -> None:
    """Compute Jaccard similarity between S6 and S1 retrieved page sets per question."""
    s1_predictions_path = cfg.RESULTS_DIR / "EXP-002" / "predictions.json"
    if not s1_predictions_path.exists():
        logger.warning("S1 predictions not found at %s, skipping overlap", s1_predictions_path)
        return

    s1_preds = load_json(s1_predictions_path)
    s1_pages_by_qid: dict[str, set[tuple[str, int]]] = {}
    for pred in s1_preds:
        qid = pred["question_id"]
        pages = {(p["doc_id"], p["page_number"]) for p in pred.get("predicted_pages", [])}
        s1_pages_by_qid[qid] = pages

    overlaps = []
    for ref, s6_chunks in zip(eval_refs, s6_results):
        qid = str(ref["question_id"])
        s6_pages = {(c.doc_id, p) for c in s6_chunks for p in c.page_span}
        s1_pages = s1_pages_by_qid.get(qid, set())
        if not s6_pages and not s1_pages:
            jaccard = 1.0
        elif not s6_pages or not s1_pages:
            jaccard = 0.0
        else:
            jaccard = len(s6_pages & s1_pages) / len(s6_pages | s1_pages)
        overlaps.append({"question_id": qid, "jaccard": jaccard, "s1_pages": len(s1_pages), "s6_pages": len(s6_pages)})

    mean_jaccard = sum(o["jaccard"] for o in overlaps) / len(overlaps) if overlaps else 0
    save_json({"mean_jaccard": mean_jaccard, "per_question": overlaps}, output_dir / "retrieval_overlap.json")
    logger.info("Retrieval overlap vs S1: mean Jaccard=%.4f", mean_jaccard)


def _write_report(report: object, result_dir: Path) -> None:
    delta_path = result_dir / "delta_vs_s1.json"
    delta = load_json(delta_path) if delta_path.exists() else {}
    overlap_path = result_dir / "retrieval_overlap.json"
    overlap = load_json(overlap_path) if overlap_path.exists() else {}

    lines = [
        "# Experiment Report: EXP-008 - S6 Naive Dense RAG Ablation",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Status:** Completed",
        "",
        "## 1. Goal",
        "",
        "- Quantify the combined contribution of S1's retrieval engineering",
        "  (hybrid search, RRF, reranker, evidence compression) AND chunk topology.",
        "- Delta(S1, S6) = value of full pipeline over naive dense RAG.",
        "",
        "## 2. Setup",
        "",
        f"- Backbone: `{cfg.BACKBONE_MODEL}` (no adapter)",
        f"- Embedding: `{exp_cfg.EMBEDDING_MODEL}` (dense only, no BM25)",
        f"- Index: FAISS IndexFlatIP (cosine via normalized IP)",
        f"- Retrieval: top-{exp_cfg.TOP_K}, no reranker, no compression",
        f"- Chunking: microchunk only ({exp_cfg.TOKEN_CHUNK_SIZE} tokens / {exp_cfg.TOKEN_CHUNK_OVERLAP} overlap)",
        "",
        "## 3. Results",
        "",
        f"- Q_main: {report.q_main:.4f}",
        f"- S_det: {report.s_det:.4f}",
        f"- S_asst: {report.s_asst:.4f}",
        f"- G (F_β=2.5): {report.grounding_f_beta:.4f}"
        if report.grounding_f_beta is not None
        else "- G: N/A",
        "",
        "## 4. Delta vs S1 (full hybrid RAG)",
        "",
    ]
    if delta:
        lines.extend([
            f"- Q_main: {delta['Q_main']:+.4f}",
            f"- S_det: {delta['S_det']:+.4f}",
            f"- S_asst: {delta['S_asst']:+.4f}",
            f"- G: {delta['G']:+.4f}",
            "",
            "Negative delta = S6 worse than S1 → full pipeline adds value.",
        ])

    lines.extend([
        "",
        "## 5. Retrieval Overlap with S1",
        "",
        f"- Mean Jaccard similarity: {overlap.get('mean_jaccard', 0):.4f}",
        "",
    ])

    lines.extend([
        "## 6. Artifacts",
        "",
        f"- FAISS index: `{exp_cfg.FAISS_INDEX_DIR}`",
        f"- Results: `{result_dir}`",
    ])
    exp_cfg.REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", exp_cfg.REPORT_PATH)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
