"""Factory: build RetrievalService from a PipelineConfig and existing index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag_pipeline import (
    BM25SparseEncoder,
    EvidenceCompressor,
    HybridSearchEngine,
    PageLifter,
    PipelineConfig,
    Reranker,
    RetrievalService,
    build_query_embedder,
)

logger = logging.getLogger(__name__)


def build_retrieval_service(
    pipeline_config: PipelineConfig,
    candidate_budget: int = 10,
    rerank_budget: int = 5,
    evidence_budget: int = 3,
    min_rerank_score: float = 0.0,
) -> RetrievalService:
    """Build a RetrievalService from an existing Qdrant index.

    Requires the index and sparse encoder state to exist at paths in pipeline_config.
    """
    # Load sparse encoder from saved state
    state_path = pipeline_config.sparse_encoder_state_path
    if not state_path.exists():
        raise FileNotFoundError(f"Sparse encoder state not found: {state_path}")
    with open(state_path) as f:
        sparse_state = json.load(f)
    sparse_encoder = BM25SparseEncoder.from_state(sparse_state)

    # Dense query embedder
    query_embedder = build_query_embedder()

    # Hybrid search engine
    search_engine = HybridSearchEngine(
        qdrant_dir=pipeline_config.qdrant_dir,
        dense_embedder=query_embedder,
        sparse_encoder=sparse_encoder,
        collection_name=pipeline_config.qdrant_collection,
        candidate_multiplier=pipeline_config.candidate_multiplier,
        dense_weight=pipeline_config.dense_weight,
        sparse_weight=pipeline_config.sparse_weight,
        rrf_k=pipeline_config.rrf_k,
    )

    # Reranker (cross-encoder with lexical fallback)
    reranker = Reranker()

    # Assemble service
    service = RetrievalService(
        search_backend=search_engine,
        reranker=reranker,
        evidence_compressor=EvidenceCompressor(),
        page_lifter=PageLifter(),
        candidate_budget=candidate_budget,
        rerank_budget=rerank_budget,
        evidence_budget=evidence_budget,
        min_rerank_score=min_rerank_score,
    )

    logger.info(
        "RetrievalService built (candidates=%d, rerank=%d, evidence=%d)",
        candidate_budget, rerank_budget, evidence_budget,
    )
    return service
