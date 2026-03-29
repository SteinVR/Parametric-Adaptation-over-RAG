"""Staged retrieval: sequential model loading to fit 8 GB VRAM.

Phase A: embed all queries (dense embedder on GPU) → free GPU
Phase B: search Qdrant (CPU + sparse BM25)
Phase C: rerank all candidates (reranker on GPU) → free GPU
Phase D: compress + page-lift (CPU only)
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external.pdf_rag_pipeline import (
    BM25SparseEncoder,
    EvidenceCompressor,
    PageLifter,
    PipelineConfig,
    Reranker,
    RetrievalResult,
    RetrievedChunk,
    build_query_embedder,
)

logger = logging.getLogger(__name__)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def staged_retrieve_all(
    questions: list[str],
    pipeline_config: PipelineConfig,
    candidate_budget: int = 10,
    candidate_multiplier: int = 3,
    rerank_budget: int = 5,
    evidence_budget: int = 3,
    min_rerank_score: float = 0.0,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Retrieve evidence for all questions with sequential model loading.

    Returns list[RetrievalResult] aligned with input questions.
    """
    n = len(questions)
    prefetch = candidate_budget * candidate_multiplier

    # ── Phase A: batch-embed all queries (GPU) ───────────────────────
    logger.info("Phase A: embedding %d queries...", n)
    embedder = build_query_embedder()
    query_vectors = embedder.encode(questions)
    del embedder
    _free_gpu()
    logger.info("Phase A done, embedder freed")

    # ── Phase B: search Qdrant with pre-computed vectors (CPU) ───────
    logger.info("Phase B: searching Qdrant...")
    # Load sparse encoder
    with open(pipeline_config.sparse_encoder_state_path) as f:
        sparse_state = json.load(f)
    sparse_encoder = BM25SparseEncoder.from_state(sparse_state)

    client = QdrantClient(path=str(pipeline_config.qdrant_dir))
    collection = pipeline_config.qdrant_collection
    all_candidates: list[list[dict]] = []

    try:
        for i, (question, dense_vec) in enumerate(zip(questions, query_vectors)):
            dense_vector = [float(v) for v in dense_vec]

            # Dense search
            dense_hits = client.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                limit=prefetch,
                with_payload=True,
                with_vectors=False,
            ).points

            # Sparse search
            sparse_vector = sparse_encoder.encode(question)
            sparse_hits = client.query_points(
                collection_name=collection,
                query=sparse_vector,
                using="sparse",
                limit=prefetch,
                with_payload=True,
                with_vectors=False,
            ).points

            # RRF fusion
            fused = _rrf_fuse(dense_hits, sparse_hits, dense_weight, sparse_weight, rrf_k)
            # Take top candidate_budget
            candidates = []
            for chunk_id, entry in fused[:candidate_budget]:
                c = dict(entry["payload"])
                c["chunk_id"] = chunk_id
                c["retrieval_score"] = entry["score"]
                candidates.append(c)
            all_candidates.append(candidates)
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()

    logger.info("Phase B done, %d candidate sets", len(all_candidates))

    # ── Phase C: rerank all candidates (GPU, bfloat16) ─────────────
    logger.info("Phase C: reranking (bfloat16)...")
    reranker = Reranker()
    # Force-load backend and convert to bfloat16 to halve VRAM
    reranker.backend._ensure_loaded()
    reranker.backend._model = reranker.backend._model.to(torch.bfloat16)
    logger.info("Reranker converted to bfloat16, VRAM: %.0f MB",
                torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0)

    all_reranked: list[list[dict]] = []
    for i, (question, candidates) in enumerate(zip(questions, all_candidates)):
        if not candidates:
            all_reranked.append([])
            continue
        reranked = reranker.rerank(question, candidates, top_k=rerank_budget)
        if min_rerank_score > 0:
            reranked = [c for c in reranked if float(c.get("rerank_score", 0)) >= min_rerank_score]
        all_reranked.append(reranked)

    del reranker
    _free_gpu()
    logger.info("Phase C done, reranker freed")

    # ── Phase D: compress + page-lift (CPU) ──────────────────────────
    logger.info("Phase D: evidence compression...")
    compressor = EvidenceCompressor()
    lifter = PageLifter()

    results: list[RetrievalResult] = []
    for question, candidates, reranked in zip(questions, all_candidates, all_reranked):
        evidence_input = reranked if reranked else candidates[:rerank_budget]
        selected = compressor.compress(evidence_input, max_evidence=evidence_budget)

        # Materialize typed chunks
        evidence_chunks = []
        for payload in selected:
            try:
                evidence_chunks.append(RetrievedChunk.from_payload(payload))
            except (KeyError, TypeError, ValueError):
                continue

        page_references = lifter.to_page_references(evidence_chunks) if evidence_chunks else []

        results.append(RetrievalResult(
            query=question,
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            evidence_chunks=evidence_chunks,
            page_references=page_references,
            is_unanswerable=not bool(page_references),
        ))

    logger.info("Phase D done, %d retrieval results", len(results))
    return results


def _rrf_fuse(
    dense_hits: list[models.ScoredPoint],
    sparse_hits: list[models.ScoredPoint],
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> list[tuple[str, dict]]:
    """Reciprocal rank fusion of dense and sparse hit lists."""
    accumulated: dict[str, dict] = {}

    def _apply(hits: list[models.ScoredPoint], weight: float) -> None:
        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            chunk_id = str(payload.get("chunk_id") or hit.id).strip()
            rrf_score = weight / (rrf_k + rank)
            if chunk_id not in accumulated:
                accumulated[chunk_id] = {"payload": payload, "score": 0.0}
            accumulated[chunk_id]["score"] += rrf_score

    _apply(dense_hits, dense_weight)
    _apply(sparse_hits, sparse_weight)

    return sorted(accumulated.items(), key=lambda x: (-x[1]["score"], x[0]))
