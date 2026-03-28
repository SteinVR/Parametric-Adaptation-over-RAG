"""Retrieval service — orchestrates search, rerank, compress, and page lifting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .evidence_compressor import EvidenceCompressor
from .page_lifter import PageLifter, PageReference, extract_physical_pages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievedChunk:
    """Typed evidence chunk returned by the retrieval service."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    chunk_type: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RetrievedChunk":
        page_span = extract_physical_pages(payload)
        if not page_span:
            raise ValueError("Retrieved chunks must reference at least one positive physical page.")

        chunk_id = str(payload["chunk_id"]).strip()
        doc_id = str(payload["doc_id"]).strip()
        if not chunk_id:
            raise ValueError("Retrieved chunks must include a non-empty chunk_id.")
        if not doc_id:
            raise ValueError("Retrieved chunks must include a non-empty doc_id.")

        rerank_score_raw = payload.get("rerank_score")
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_span=page_span,
            chunk_type=str(payload.get("chunk_type") or "unknown"),
            text=str(payload.get("text") or ""),
            retrieval_score=float(payload.get("retrieval_score", 0)),
            rerank_score=None if rerank_score_raw is None else float(rerank_score_raw),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class RetrievalResult:
    """Final retrieval output consumed by downstream answering."""

    query: str
    candidate_count: int
    reranked_count: int
    evidence_chunks: list[RetrievedChunk] = field(default_factory=list)
    page_references: list[PageReference] = field(default_factory=list)
    is_unanswerable: bool = False


# ---------------------------------------------------------------------------
# Protocols (structural typing for pluggable components)
# ---------------------------------------------------------------------------


class _SearchBackend:
    """Minimal interface — anything with .search(query, top_k) -> list[SearchResult]."""

    def search(self, query: str, top_k: int = 10) -> list[Any]:
        raise NotImplementedError


class _Reranker:
    """Minimal interface — anything with .rerank(query, candidates, top_k)."""

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievalService:
    """Coordinates search -> rerank -> compress -> page-lift into one call.

    Parameters
    ----------
    search_backend:
        Anything with ``.search(query, top_k) -> list[SearchResult]``
        (e.g. :class:`HybridSearchEngine`).
    reranker:
        Anything with ``.rerank(query, candidates, top_k) -> list[dict]``
        (e.g. :class:`Reranker` or :class:`LexicalFallbackReranker`).
        Pass ``None`` to skip reranking.
    evidence_compressor:
        Selects a compact, page-diverse evidence subset.
    candidate_budget:
        How many candidates to fetch from hybrid search.
    rerank_budget:
        How many of those to pass to the reranker.
    evidence_budget:
        Final number of evidence chunks to return.
    min_rerank_score:
        Drop reranked candidates below this threshold (0 = keep all).
    """

    search_backend: Any
    reranker: Any | None = None
    evidence_compressor: EvidenceCompressor = field(default_factory=EvidenceCompressor)
    page_lifter: PageLifter = field(default_factory=PageLifter)
    candidate_budget: int = 10
    rerank_budget: int = 5
    evidence_budget: int = 3
    min_rerank_score: float = 0.0

    def retrieve(self, query: str) -> RetrievalResult:
        """Run the full retrieval pipeline for a single query."""

        # 1. Search
        raw_results = self.search_backend.search(query, top_k=self.candidate_budget)
        candidates = self._results_to_payloads(raw_results)
        candidate_count = len(candidates)

        if candidate_count == 0:
            return self._empty_result(query, candidate_count=0, reranked_count=0)

        # 2. Rerank (optional)
        if self.reranker is not None:
            reranked = self.reranker.rerank(
                query,
                [dict(c) for c in candidates],
                top_k=self.rerank_budget,
            )
            reranked = [dict(c) for c in reranked[: self.rerank_budget]]
        else:
            reranked = [dict(c) for c in candidates[: self.rerank_budget]]

        reranked_count = len(reranked)
        if reranked_count == 0:
            return self._empty_result(query, candidate_count=candidate_count, reranked_count=0)

        # 3. Apply rerank score threshold
        surviving = self._apply_rerank_threshold(reranked)
        if not surviving:
            return self._empty_result(query, candidate_count=candidate_count, reranked_count=reranked_count)

        # 4. Evidence compression (page-diverse selection)
        selected = self.evidence_compressor.compress(surviving, max_evidence=self.evidence_budget)

        # 5. Materialize typed evidence chunks
        evidence_chunks = self._materialize_evidence(selected)
        if not evidence_chunks:
            return self._empty_result(query, candidate_count=candidate_count, reranked_count=reranked_count)

        # 6. Page references
        page_references = self.page_lifter.to_page_references(evidence_chunks)

        return RetrievalResult(
            query=query,
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            evidence_chunks=evidence_chunks,
            page_references=page_references,
            is_unanswerable=not bool(page_references),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _results_to_payloads(results: list[Any]) -> list[dict[str, Any]]:
        """Convert SearchResult objects (or dicts) into flat payload dicts."""
        payloads: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                payloads.append(dict(r))
            else:
                payload: dict[str, Any] = {
                    "chunk_id": getattr(r, "chunk_id", ""),
                    "doc_id": getattr(r, "doc_id", ""),
                    "page_span": list(getattr(r, "page_span", [])),
                    "chunk_type": getattr(r, "chunk_type", "unknown"),
                    "text": getattr(r, "text", ""),
                    "retrieval_score": float(getattr(r, "score", 0)),
                }
                metadata = getattr(r, "metadata", None)
                if isinstance(metadata, dict):
                    payload["metadata"] = dict(metadata)
                    # Promote rerank_score from metadata if present
                    if "rerank_score" in metadata:
                        payload["rerank_score"] = metadata["rerank_score"]
                payloads.append(payload)
        return payloads

    def _apply_rerank_threshold(self, reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.min_rerank_score <= 0:
            return [dict(item) for item in reranked]

        surviving: list[dict[str, Any]] = []
        for item in reranked:
            rerank_score = item.get("rerank_score")
            if rerank_score is None:
                continue
            if float(rerank_score) >= self.min_rerank_score:
                surviving.append(dict(item))
        return surviving

    @staticmethod
    def _materialize_evidence(payloads: list[dict[str, Any]]) -> list[RetrievedChunk]:
        evidence: list[RetrievedChunk] = []
        for payload in payloads:
            try:
                evidence.append(RetrievedChunk.from_payload(payload))
            except (KeyError, TypeError, ValueError):
                continue
        return evidence

    @staticmethod
    def _empty_result(
        query: str,
        *,
        candidate_count: int,
        reranked_count: int,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            evidence_chunks=[],
            page_references=[],
            is_unanswerable=True,
        )
