"""Corpus extraction helpers for Doc-to-LoRA packaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from src.retrieval.indexer import build_doc_id_map


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One frozen PDF document with extracted text."""

    doc_index: int
    doc_name: str
    doc_id: str
    pdf_path: Path
    page_texts: list[str]
    full_text: str
    char_count: int
    word_count: int

    @property
    def page_count(self) -> int:
        return len(self.page_texts)


def load_frozen_corpus_documents(
    *,
    corpus_dir: Path,
    goldset_path: Path,
) -> list[CorpusDocument]:
    """Load all PDF documents in the frozen corpus and extract full text."""
    stem_to_doc_id = build_doc_id_map(goldset_path)
    pdf_paths = sorted(corpus_dir.glob("*.pdf"))
    documents: list[CorpusDocument] = []
    for doc_index, pdf_path in enumerate(pdf_paths, start=1):
        doc_name = pdf_path.stem
        doc_id = stem_to_doc_id.get(doc_name, doc_name)
        page_texts = extract_pdf_page_texts(pdf_path)
        full_text = render_document_text(page_texts)
        documents.append(
            CorpusDocument(
                doc_index=doc_index,
                doc_name=doc_name,
                doc_id=doc_id,
                pdf_path=pdf_path,
                page_texts=page_texts,
                full_text=full_text,
                char_count=len(full_text),
                word_count=_count_words(full_text),
            )
        )
    return documents


def extract_pdf_page_texts(pdf_path: Path) -> list[str]:
    """Extract page-level text from a PDF, preserving page order."""
    page_texts: list[str] = []
    document = fitz.open(pdf_path)
    try:
        for page in document:
            text = page.get_text("text").strip()
            page_texts.append(text)
    finally:
        document.close()
    return page_texts


def render_document_text(page_texts: Iterable[str]) -> str:
    """Render page texts into one D2L input string."""
    return "\n\n".join(text.strip() for text in page_texts if text.strip()).strip()


def _count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0
