"""Pipeline configuration — pure dataclass with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """End-to-end PDF RAG pipeline configuration.

    All paths are resolved lazily from *documents_dir* and *output_dir*.
    """

    documents_dir: Path
    output_dir: Path

    # ── Chunking ──
    token_chunk_size: int = 300
    token_chunk_overlap: int = 50
    enabled_chunk_families: list[str] = field(
        default_factory=lambda: ["page", "section", "clause", "microchunk", "table"],
    )

    # ── Embeddings ──
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 16
    local_model_only: bool = True

    # ── Qdrant ──
    qdrant_collection: str = "document_index"

    # ── Retrieval ──
    candidate_budget: int = 10
    candidate_multiplier: int = 3
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60

    # ── Derived paths ──

    @property
    def corpus_dir(self) -> Path:
        return self.output_dir / "corpus"

    @property
    def corpus_path(self) -> Path:
        return self.corpus_dir / "corpus.jsonl"

    @property
    def page_map_path(self) -> Path:
        return self.corpus_dir / "page_map.json"

    @property
    def index_dir(self) -> Path:
        return self.output_dir / "index"

    @property
    def qdrant_dir(self) -> Path:
        return self.index_dir / "qdrant"

    @property
    def sparse_encoder_state_path(self) -> Path:
        return self.index_dir / "sparse_encoder_state.json"

    @property
    def index_manifest_path(self) -> Path:
        return self.index_dir / "index_manifest.json"

    def ensure_directories(self) -> None:
        """Create all output directories."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
