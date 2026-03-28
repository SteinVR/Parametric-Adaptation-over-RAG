"""Self-contained hybrid dense+sparse search with RRF fusion over a Qdrant index."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from qdrant_client import QdrantClient, models

from .reranker import LexicalFallbackReranker, Reranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class DenseEmbedder(Protocol):
    """Anything with a batch-encode method that returns dense float vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of text strings as dense vectors."""


@runtime_checkable
class SparseEncoder(Protocol):
    """Anything that encodes a single query string into a Qdrant SparseVector."""

    def encode(self, text: str) -> models.SparseVector:
        """Encode a single query string as a sparse vector."""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearchResult:
    """A single retrieved chunk with its fused relevance score."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    chunk_type: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class HybridSearchEngine:
    """Combined dense+sparse Qdrant retrieval with reciprocal rank fusion.

    The Qdrant client is lazy-loaded on the first :meth:`search` call so that
    construction is cheap and the directory path is validated only when needed.
    """

    def __init__(
        self,
        qdrant_dir: Path,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseEncoder,
        collection_name: str = "document_index",
        candidate_multiplier: int = 3,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
        reranker: Reranker | LexicalFallbackReranker | None = None,
        rerank_budget: int = 0,
    ) -> None:
        self._qdrant_dir = Path(qdrant_dir)
        self._dense_embedder = dense_embedder
        self._sparse_encoder = sparse_encoder
        self._collection_name = collection_name
        self._candidate_multiplier = max(candidate_multiplier, 1)
        self._dense_weight = float(dense_weight)
        self._sparse_weight = float(sparse_weight)
        self._rrf_k = int(rrf_k)
        self._reranker = reranker
        self._rerank_budget = max(int(rerank_budget), 0)
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Run hybrid dense+sparse search with RRF fusion.

        Parameters
        ----------
        query:
            Plain-text query string.
        top_k:
            Maximum number of results to return.
        """
        result_limit = max(int(top_k), 0)
        if result_limit == 0:
            return []

        self._ensure_client()
        client = self._client
        assert client is not None  # guaranteed by _ensure_client

        prefetch = result_limit * self._candidate_multiplier

        # --- Dense retrieval ---
        query_vectors = self._dense_embedder.encode([query])
        if not query_vectors:
            raise ValueError("Dense embedder returned an empty list for the query.")
        dense_vector = [float(v) for v in query_vectors[0]]

        dense_hits: list[models.ScoredPoint] = client.query_points(
            collection_name=self._collection_name,
            query=dense_vector,
            using="dense",
            limit=prefetch,
            with_payload=True,
            with_vectors=False,
        ).points

        # --- Sparse retrieval ---
        sparse_vector = self._sparse_encoder.encode(query)
        sparse_hits: list[models.ScoredPoint] = client.query_points(
            collection_name=self._collection_name,
            query=sparse_vector,
            using="sparse",
            limit=prefetch,
            with_payload=True,
            with_vectors=False,
        ).points

        logger.debug(
            "hybrid search: dense_hits=%d sparse_hits=%d query=%r",
            len(dense_hits),
            len(sparse_hits),
            query[:120],
        )

        # --- RRF fusion ---
        fused = self._rrf_fuse(dense_hits, sparse_hits)

        # --- Optional reranking ---
        if self._reranker is not None and self._rerank_budget > 0:
            rerank_window = fused[: self._rerank_budget]
            rerank_candidates = [
                {
                    **entry["payload"],
                    "retrieval_score": entry["score"],
                    "chunk_id": chunk_id,
                }
                for chunk_id, entry in rerank_window
            ]
            reranked = self._reranker.rerank(query, rerank_candidates, result_limit)
            logger.debug(
                "reranked %d -> %d candidates", len(rerank_candidates), len(reranked),
            )
            return [self._reranked_to_search_result(c) for c in reranked]

        fused = fused[:result_limit]
        return [self._to_search_result(chunk_id, entry) for chunk_id, entry in fused]

    def close(self) -> None:
        """Release the Qdrant client connection."""
        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()
        self._client = None
        logger.debug("HybridSearchEngine closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._qdrant_dir.exists():
            raise ValueError(f"Qdrant index directory does not exist: {self._qdrant_dir}")
        self._client = QdrantClient(path=str(self._qdrant_dir))
        logger.info(
            "HybridSearchEngine: opened Qdrant client at %s (collection=%r)",
            self._qdrant_dir,
            self._collection_name,
        )

    def _rrf_fuse(
        self,
        dense_hits: list[models.ScoredPoint],
        sparse_hits: list[models.ScoredPoint],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Fuse two ranked lists with reciprocal rank fusion.

        Each hit in *dense_hits* is ranked 1..N and each hit in *sparse_hits*
        is ranked 1..M. The combined score for a chunk is::

            dense_weight / (rrf_k + dense_rank)
          + sparse_weight / (rrf_k + sparse_rank)

        Chunks that appear in only one list receive a score contribution only
        from that list (the missing term contributes 0).
        """
        # Maps chunk_id -> (payload, combined_score)
        accumulated: dict[str, dict[str, Any]] = {}

        def _apply_hits(
            hits: list[models.ScoredPoint],
            weight: float,
        ) -> None:
            for rank, hit in enumerate(hits, start=1):
                payload = dict(hit.payload or {})
                chunk_id = _extract_chunk_id(payload, hit.id)
                rrf_score = weight / (self._rrf_k + rank)

                if chunk_id not in accumulated:
                    accumulated[chunk_id] = {
                        "payload": payload,
                        "score": 0.0,
                    }
                accumulated[chunk_id]["score"] += rrf_score

        _apply_hits(dense_hits, self._dense_weight)
        _apply_hits(sparse_hits, self._sparse_weight)

        sorted_chunks = sorted(
            accumulated.items(),
            key=lambda item: (-item[1]["score"], item[0]),
        )
        return sorted_chunks

    @staticmethod
    def _reranked_to_search_result(candidate: dict[str, Any]) -> SearchResult:
        """Build a SearchResult from a reranker output dict."""
        chunk_id = str(candidate.get("chunk_id") or "")
        page_span = _normalized_page_span(
            candidate.get("page_span") or candidate.get("page_numbers")
        )
        rerank_score = candidate.get("rerank_score")
        retrieval_score = candidate.get("retrieval_score")
        score = float(rerank_score) if rerank_score is not None else float(retrieval_score or 0)

        metadata: dict[str, Any] = {}
        if retrieval_score is not None:
            metadata["retrieval_score"] = float(retrieval_score)
        if rerank_score is not None:
            metadata["rerank_score"] = float(rerank_score)
        for extra_key in (
            "section", "clause", "neighboring_headings", "entities", "dates",
            "document_title", "document_family", "parent_page_numbers",
        ):
            value = candidate.get(extra_key)
            if value is not None:
                metadata[extra_key] = value

        return SearchResult(
            chunk_id=chunk_id,
            doc_id=str(candidate.get("doc_id") or ""),
            page_span=page_span,
            chunk_type=str(candidate.get("chunk_type") or "unknown"),
            text=str(candidate.get("text") or ""),
            score=score,
            metadata=metadata,
        )

    @staticmethod
    def _to_search_result(chunk_id: str, entry: dict[str, Any]) -> SearchResult:
        payload = entry["payload"]
        score = float(entry["score"])

        page_span = _normalized_page_span(
            payload.get("page_span") or payload.get("page_numbers")
        )

        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        # Surface optional rich payload fields into metadata for callers that
        # want them, without polluting the primary SearchResult fields.
        for extra_key in (
            "section",
            "clause",
            "neighboring_headings",
            "entities",
            "dates",
            "document_title",
            "document_family",
            "parent_page_numbers",
        ):
            value = payload.get(extra_key)
            if value is not None:
                metadata.setdefault(extra_key, value)

        return SearchResult(
            chunk_id=chunk_id,
            doc_id=str(payload.get("doc_id") or ""),
            page_span=page_span,
            chunk_type=str(payload.get("chunk_type") or "unknown"),
            text=str(payload.get("text") or ""),
            score=score,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_chunk_id(payload: dict[str, Any], point_id: Any) -> str:
    """Return the chunk_id from payload, falling back to the Qdrant point id."""
    chunk_id = str(payload.get("chunk_id") or "").strip()
    return chunk_id if chunk_id else str(point_id)


def _normalized_page_span(raw: Any) -> list[int]:
    """Coerce a raw page-span value to a list of positive integers."""
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            result.append(parsed)
    return result
