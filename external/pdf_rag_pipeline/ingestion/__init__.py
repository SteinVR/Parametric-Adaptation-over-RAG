"""PDF ingestion package — parse, serialize, and build the canonical corpus."""

from .pdf_parser import parse_pdf
from .table_serializer import serialize_document_tables
from .corpus_builder import build_corpus

__all__ = [
    "parse_pdf",
    "serialize_document_tables",
    "build_corpus",
]
