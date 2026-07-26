"""PDF RAG Pipeline — self-contained ingestion, chunking, indexing, and retrieval."""

from .config import PipelineConfig
from .ingestion.pdf_parser import parse_pdf
from .ingestion.table_serializer import serialize_document_tables
from .ingestion.corpus_builder import build_corpus
from .indexing.chunking import build_index_chunks, IndexChunk
from .indexing.embeddings import Qwen3DenseEmbedder, BM25SparseEncoder, build_dense_embedder, build_query_embedder
from .indexing.qdrant_store import build_and_persist_index
from .retrieval.hybrid_search import HybridSearchEngine, SearchResult
from .retrieval.reranker import Reranker, LexicalFallbackReranker
from .retrieval.page_lifter import PageLifter, PageReference
from .retrieval.evidence_compressor import EvidenceCompressor
from .retrieval.service import RetrievalService, RetrievalResult, RetrievedChunk

__all__ = [
    "PipelineConfig",
    "parse_pdf",
    "serialize_document_tables",
    "build_corpus",
    "build_index_chunks",
    "IndexChunk",
    "Qwen3DenseEmbedder",
    "BM25SparseEncoder",
    "build_dense_embedder",
    "build_query_embedder",
    "build_and_persist_index",
    "HybridSearchEngine",
    "SearchResult",
    "Reranker",
    "LexicalFallbackReranker",
    "PageLifter",
    "PageReference",
    "EvidenceCompressor",
    "RetrievalService",
    "RetrievalResult",
    "RetrievedChunk",
]
