"""Qdrant persistence layer for the hybrid indexing pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .chunking import IndexChunk
from .embeddings import BM25SparseEncoder

LOGGER = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_and_persist_index(
    chunks: list[IndexChunk],
    dense_vectors: list[list[float]],
    sparse_encoder: BM25SparseEncoder,
    qdrant_dir: Path,
    collection_name: str,
) -> dict[str, dict[str, Any]]:
    """Create a Qdrant collection with dense + sparse vectors and upsert all points.

    Returns a page_parent_map dict keyed by chunk_id.
    """

    if not chunks:
        raise ValueError("No index chunks were generated from the canonical corpus.")
    if len(dense_vectors) != len(chunks):
        raise ValueError("Dense vector count does not match chunk count.")

    client = QdrantClient(path=str(qdrant_dir))
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=len(dense_vectors[0]),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        points: list[models.PointStruct] = []
        page_parent_map: dict[str, dict[str, Any]] = {}
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_encoder.encode_tokens(_tokenize(chunk.text)),
                    },
                    payload=chunk.payload(),
                )
            )
            page_parent_map[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page_span": list(chunk.page_span),
                "page_numbers": list(chunk.page_span),
                "parent_page_numbers": list(chunk.parent_page_numbers or chunk.page_span),
                "chunk_type": chunk.chunk_type,
                "section": chunk.section,
                "clause": chunk.clause,
                "neighboring_headings": list(chunk.neighboring_headings),
                "parser_provenance": chunk.parser_provenance,
                "heading_path": list(chunk.heading_path),
                "source_block_ids": list(chunk.source_block_ids),
            }

        client.upsert(collection_name=collection_name, points=points, wait=True)
        return page_parent_map
    finally:
        _close_client(client)


def load_qdrant_client(qdrant_dir: Path) -> QdrantClient:
    """Open an existing Qdrant database for search."""

    return QdrantClient(path=str(qdrant_dir))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _close_client(client: QdrantClient) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
