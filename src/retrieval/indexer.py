"""Corpus indexing: ingest PDFs → Qdrant hybrid index with doc_id remapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag_pipeline import (
    BM25SparseEncoder,
    PipelineConfig,
    build_and_persist_index,
    build_corpus,
    build_dense_embedder,
    build_index_chunks,
    parse_pdf,
    serialize_document_tables,
)
from src.data.io import load_json

logger = logging.getLogger(__name__)


def build_doc_id_map(goldset_path: Path) -> dict[str, str]:
    """Build filename_stem → sha256 mapping from goldset corpus_documents.

    Goldset has {sha256: stem}, we need the reverse {stem: sha256}.
    """
    gs = load_json(goldset_path)
    corpus_docs = gs.get("corpus_documents", {})
    return {stem: sha256 for sha256, stem in corpus_docs.items()}


def build_rag_index(
    corpus_dir: Path,
    output_dir: Path,
    goldset_path: Path,
    pipeline_config: PipelineConfig | None = None,
) -> PipelineConfig:
    """Full pipeline: parse PDFs → chunk → embed → persist to Qdrant.

    Remaps doc_id from filename stem to sha256 hash so grounding evaluation works.
    Returns the PipelineConfig used (for later RetrievalService construction).
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            documents_dir=corpus_dir,
            output_dir=output_dir,
        )
    pipeline_config.ensure_directories()

    # Doc ID remapping
    stem_to_sha = build_doc_id_map(goldset_path)
    logger.info("Doc ID map: %d documents", len(stem_to_sha))

    # 1. Parse all PDFs
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    logger.info("Parsing %d PDFs from %s", len(pdf_files), corpus_dir)

    all_pages = []
    for pdf_path in pdf_files:
        doc = parse_pdf(pdf_path)
        table_blocks = serialize_document_tables(doc)
        pages = build_corpus(doc, table_blocks)

        # Remap doc_id from stem to sha256
        sha256_id = stem_to_sha.get(doc.doc_id)
        if sha256_id is None:
            logger.warning("No sha256 mapping for doc_id=%s, keeping stem", doc.doc_id)
            sha256_id = doc.doc_id

        for page in pages:
            page.doc_id = sha256_id

        all_pages.extend(pages)
        logger.info("  %s → %d pages (doc_id=%s)", pdf_path.name, len(pages), sha256_id[:12])

    # 2. Chunk
    chunks = build_index_chunks(
        all_pages,
        enabled_chunk_families=set(pipeline_config.enabled_chunk_families),
        token_chunk_size=pipeline_config.token_chunk_size,
        token_chunk_overlap=pipeline_config.token_chunk_overlap,
    )
    logger.info("Generated %d chunks", len(chunks))

    # Remap chunk doc_ids too (chunking uses the page's doc_id)
    for chunk in chunks:
        if chunk.doc_id not in stem_to_sha.values():
            remapped = stem_to_sha.get(chunk.doc_id)
            if remapped:
                chunk.doc_id = remapped

    # 3. Embed
    logger.info("Building dense embeddings...")
    embedder = build_dense_embedder()
    chunk_texts = [c.text for c in chunks]
    dense_vectors = embedder.encode(chunk_texts)

    # 4. Sparse encoder
    sparse_encoder = BM25SparseEncoder(chunk_texts)

    # 5. Persist to Qdrant
    logger.info("Persisting index to %s", pipeline_config.qdrant_dir)
    page_parent_map = build_and_persist_index(
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_encoder=sparse_encoder,
        qdrant_dir=pipeline_config.qdrant_dir,
        collection_name=pipeline_config.qdrant_collection,
    )

    # Save sparse encoder state for query-time reconstruction
    state = sparse_encoder.export_state()
    with open(pipeline_config.sparse_encoder_state_path, "w") as f:
        json.dump(state, f)

    # Save index manifest
    manifest = {
        "chunk_count": len(chunks),
        "document_count": len(pdf_files),
        "collection_name": pipeline_config.qdrant_collection,
        "config": {
            "token_chunk_size": pipeline_config.token_chunk_size,
            "token_chunk_overlap": pipeline_config.token_chunk_overlap,
            "enabled_chunk_families": pipeline_config.enabled_chunk_families,
        },
    }
    with open(pipeline_config.index_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save page parent map
    with open(pipeline_config.output_dir / "page_parent_map.json", "w") as f:
        json.dump(page_parent_map, f, indent=2, default=str)

    logger.info("Index built: %d chunks from %d documents", len(chunks), len(pdf_files))
    return pipeline_config
