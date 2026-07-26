"""Evidence compression — select a compact page-grounded evidence set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .page_lifter import PageLifter, extract_physical_pages


@dataclass(slots=True)
class EvidenceCompressor:
    """Select a compact evidence set while preserving page-grounded support.

    The algorithm iterates over pages (ranked by best rerank/retrieval score)
    and picks up to *max_chunks_per_page* chunks from each page until the
    evidence budget is filled.  This ensures page diversity in the final set.
    """

    page_lifter: PageLifter = field(default_factory=PageLifter)
    max_chunks_per_page: int = 1

    def compress(
        self,
        reranked_chunks: list[dict[str, Any]],
        max_evidence: int,
    ) -> list[dict[str, Any]]:
        evidence_budget = max(int(max_evidence), 0)
        if evidence_budget == 0:
            return []

        valid_chunks: list[dict[str, Any]] = []
        valid_chunk_pages: list[set[int]] = []
        for chunk in self._sort_chunks(reranked_chunks):
            page_numbers = extract_physical_pages(chunk)
            if not page_numbers:
                continue
            valid_chunks.append(dict(chunk))
            valid_chunk_pages.append(set(page_numbers))

        if not valid_chunks:
            return []

        selected: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        lifted_pages = self.page_lifter.lift(valid_chunks)

        # Phase 1: pick chunks page-by-page for diversity
        for lifted_page in lifted_pages:
            page_picks = 0
            for chunk, page_numbers in zip(valid_chunks, valid_chunk_pages):
                if chunk.get("doc_id") != lifted_page.doc_id:
                    continue
                if lifted_page.page_number not in page_numbers:
                    continue

                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue

                selected.append(dict(chunk))
                seen_chunk_ids.add(chunk_id)
                page_picks += 1
                if len(selected) >= evidence_budget or page_picks >= self.max_chunks_per_page:
                    break

            if len(selected) >= evidence_budget:
                return selected[:evidence_budget]

        # Phase 2: fill remaining budget with best unseen chunks
        for chunk in valid_chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            selected.append(dict(chunk))
            seen_chunk_ids.add(chunk_id)
            if len(selected) >= evidence_budget:
                break

        return selected[:evidence_budget]

    def _sort_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            (dict(chunk) for chunk in chunks),
            key=lambda chunk: (
                -_optional_score(chunk.get("rerank_score")),
                -_score(chunk.get("retrieval_score")),
                str(chunk.get("chunk_id") or ""),
            ),
        )


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_score(value: Any) -> float:
    if value is None:
        return float("-inf")
    return _score(value)
