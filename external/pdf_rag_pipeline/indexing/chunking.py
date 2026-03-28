"""Structure-aware chunking for canonical legal corpora."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable

from ..schemas import CanonicalPageRecord, ContentBlockType

LOGGER = logging.getLogger(__name__)

DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_BM25_TERMS_METADATA = 32

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"(?im)^(section\s+\d+[a-zA-Z0-9.-]*\s+[^\n]+)$")
_PART_RE = re.compile(r"(?im)^(part\s+\d+[a-zA-Z0-9.-]*\s+[^\n]+)$")
_ARTICLE_RE = re.compile(r"(?im)^(article\s+\d+[a-zA-Z0-9.-]*\s+[^\n]+)$")
_PARAGRAPH_CLAUSE_RE = re.compile(
    r"^(?P<label>(?:\([A-Za-z0-9]+\)|\d+[A-Za-z]?(?:[.)])?|[A-Za-z](?:[.)])))\s+(?P<body>.+)$",
    re.DOTALL,
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&/-]{2,}\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BODY_START_CUES = {
    "a",
    "an",
    "the",
    "this",
    "these",
    "those",
    "any",
    "each",
    "every",
    "no",
    "such",
    "additional",
    "further",
    "supplemental",
    "all",
    "if",
    "when",
    "where",
    "unless",
}
_BODY_VERBS = {"is", "are", "must", "shall", "may", "can", "will", "remains", "means", "includes"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeadingContext:
    """Rolling heading context preserved across continuation pages."""

    part_label: str | None = None
    article_label: str | None = None
    section_label_value: str | None = None

    def merged_with(self, text: str) -> "HeadingContext":
        explicit_part = _extract_part_heading(text)
        explicit_article = _extract_article_heading(text)
        explicit_section = _extract_section_heading(text)
        if explicit_part and explicit_part != self.part_label and explicit_article is None and explicit_section is None:
            return HeadingContext(part_label=explicit_part, article_label=None, section_label_value=None)
        if explicit_article and explicit_article != self.article_label and explicit_section is None:
            return HeadingContext(
                part_label=explicit_part or self.part_label,
                article_label=explicit_article,
                section_label_value=None,
            )
        return HeadingContext(
            part_label=explicit_part or self.part_label,
            article_label=explicit_article or self.article_label,
            section_label_value=explicit_section or self.section_label_value,
        )

    def neighboring_headings(self) -> list[str]:
        headings: list[str] = []
        if self.part_label:
            headings.append(self.part_label)
        if self.article_label:
            headings.append(self.article_label)
        if self.section_label_value:
            headings.append(self.section_label_value)
        return headings

    def section_label(self) -> str:
        if self.section_label_value:
            return self.section_label_value
        if self.article_label:
            return self.article_label
        if self.part_label:
            return self.part_label
        return ""


@dataclass(slots=True)
class PageChunkContext:
    """Per-page heading context used to derive truthful higher-level chunks."""

    page: CanonicalPageRecord
    headings: HeadingContext


@dataclass(slots=True)
class IndexChunk:
    """Intermediate indexing unit persisted into Qdrant."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    chunk_type: str
    text: str
    section: str
    clause: str | None
    neighboring_headings: list[str]
    entities: list[str]
    dates: list[str]
    token_count: int
    bm25_terms: list[str]
    document_title: str | None = None
    document_family: str | None = None
    parser_provenance: str | None = None
    heading_path: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    parent_page_numbers: list[int] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_span": list(self.page_span),
            "parent_page_numbers": list(self.parent_page_numbers or self.page_span),
            "chunk_type": self.chunk_type,
            "text": self.text,
            "section": self.section,
            "clause": self.clause,
            "neighboring_headings": list(self.neighboring_headings),
            "entities": list(self.entities),
            "dates": list(self.dates),
            "token_count": self.token_count,
            "bm25_terms": list(self.bm25_terms),
            "document_title": self.document_title,
            "document_family": self.document_family,
            "parser_provenance": self.parser_provenance,
            "heading_path": list(self.heading_path),
            "source_block_ids": list(self.source_block_ids),
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_index_chunks(
    pages: Iterable[CanonicalPageRecord],
    *,
    enabled_chunk_families: set[str] | None = None,
    token_chunk_size: int = 300,
    token_chunk_overlap: int = 50,
) -> list[IndexChunk]:
    """Derive structure-aware hybrid indexing chunks from canonical pages."""

    families = {
        str(name).strip()
        for name in (enabled_chunk_families or {"page", "section", "clause", "microchunk", "table"})
        if str(name).strip()
    }
    if not families:
        return []

    chunks: list[IndexChunk] = []
    pages_by_doc: dict[str, list[CanonicalPageRecord]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page.doc_id].append(page)

    for doc_id in sorted(pages_by_doc):
        page_contexts: list[PageChunkContext] = []
        last_headings = HeadingContext()
        for page in sorted(pages_by_doc[doc_id], key=lambda item: item.page_number):
            current_headings = last_headings.merged_with(page.text)
            last_headings = current_headings
            page_contexts.append(PageChunkContext(page=page, headings=current_headings))

        parent_page_numbers_by_page = _parent_page_numbers_by_page(page_contexts)
        for page_context in page_contexts:
            page = page_context.page
            neighboring_headings = page_context.headings.neighboring_headings()
            section_title = page_context.headings.section_label()
            page_prefix = f"{page.doc_id}-p{page.page_number}"
            parent_page_numbers = parent_page_numbers_by_page.get(page.page_number, [page.page_number])

            if "page" in families:
                chunks.append(
                    _make_chunk(
                        chunk_id=f"{page_prefix}-page",
                        doc_id=page.doc_id,
                        page_span=[page.page_number],
                        parent_page_numbers=parent_page_numbers,
                        chunk_type="page",
                        text=page.text,
                        section=section_title,
                        clause=None,
                        neighboring_headings=neighboring_headings,
                        document_title=page.document_title,
                        document_family=page.document_family,
                        parser_provenance=page.parser_provenance,
                        heading_path=page.heading_path,
                        source_block_ids=page.source_block_ids,
                    )
                )

            if "clause" in families or "microchunk" in families:
                chunks.extend(
                    _clause_chunks_for_page(
                        page,
                        page_context.headings,
                        parent_page_numbers=parent_page_numbers,
                        include_clause_chunks="clause" in families,
                        include_microchunks="microchunk" in families,
                        token_chunk_size=token_chunk_size,
                        token_chunk_overlap=token_chunk_overlap,
                    )
                )

            if "table" in families:
                for table_block in page.blocks:
                    if table_block.type is not ContentBlockType.TABLE:
                        continue
                    chunks.append(
                        _make_chunk(
                            chunk_id=table_block.block_id,
                            doc_id=page.doc_id,
                            page_span=[page.page_number],
                            parent_page_numbers=parent_page_numbers,
                            chunk_type="table",
                            text=table_block.text,
                            section=section_title,
                            clause=table_block.metadata.get("row_anchor"),
                            neighboring_headings=neighboring_headings,
                            document_title=page.document_title,
                            document_family=page.document_family,
                            parser_provenance=page.parser_provenance,
                            heading_path=_heading_path_for_block(page, table_block),
                            source_block_ids=_source_block_ids_for_block(page, table_block),
                        )
                    )

        if "section" in families:
            chunks.extend(_section_chunks_for_doc(page_contexts))

    return chunks


# ---------------------------------------------------------------------------
# Clause chunking
# ---------------------------------------------------------------------------


def _clause_chunks_for_page(
    page: CanonicalPageRecord,
    headings: HeadingContext,
    *,
    parent_page_numbers: list[int],
    include_clause_chunks: bool,
    include_microchunks: bool,
    token_chunk_size: int,
    token_chunk_overlap: int,
) -> list[IndexChunk]:
    clause_sources = _clause_sources_for_text(page.text)
    chunks: list[IndexChunk] = []
    neighboring_headings = headings.neighboring_headings() or [f"Page {page.page_number}"]
    section_title = headings.section_label()

    for clause_index, (clause_label, clause_text) in enumerate(clause_sources, start=1):
        clause_name = clause_label
        clause_chunk_id = f"{page.doc_id}-p{page.page_number}-clause-{clause_index}"
        if include_clause_chunks:
            chunks.append(
                _make_chunk(
                    chunk_id=clause_chunk_id,
                    doc_id=page.doc_id,
                    page_span=[page.page_number],
                    parent_page_numbers=parent_page_numbers,
                    chunk_type="clause",
                    text=clause_text,
                    section=section_title,
                    clause=clause_name,
                    neighboring_headings=neighboring_headings,
                    document_title=page.document_title,
                    document_family=page.document_family,
                    parser_provenance=page.parser_provenance,
                    heading_path=page.heading_path,
                    source_block_ids=page.source_block_ids,
                )
            )

        if include_microchunks:
            for micro_index, micro_text in enumerate(
                _split_microchunks(
                    clause_text,
                    token_chunk_size=token_chunk_size,
                    token_chunk_overlap=token_chunk_overlap,
                ),
                start=1,
            ):
                chunks.append(
                    _make_chunk(
                        chunk_id=f"{clause_chunk_id}-micro-{micro_index}",
                        doc_id=page.doc_id,
                        page_span=[page.page_number],
                        parent_page_numbers=parent_page_numbers,
                        chunk_type="microchunk",
                        text=micro_text,
                        section=section_title,
                        clause=clause_name,
                        neighboring_headings=neighboring_headings,
                        document_title=page.document_title,
                        document_family=page.document_family,
                        parser_provenance=page.parser_provenance,
                        heading_path=page.heading_path,
                        source_block_ids=page.source_block_ids,
                    )
                )
    return chunks


def _clause_sources_for_text(text: str) -> list[tuple[str, str]]:
    paragraphs = _content_paragraphs(text)
    lead_in_parts: list[str] = []
    numbered_clauses: list[tuple[str, str]] = []
    current_label: str | None = None
    current_parts: list[str] = []

    for paragraph in paragraphs:
        match = _PARAGRAPH_CLAUSE_RE.match(paragraph)
        if match:
            if current_label is None and lead_in_parts:
                numbered_clauses.append(("lead-in", "\n\n".join(lead_in_parts).strip()))
                lead_in_parts = []
            if current_label is not None and current_parts:
                numbered_clauses.append((current_label, "\n\n".join(current_parts).strip()))
            current_label = match.group("label").strip()
            current_parts = [paragraph.strip()]
            continue

        if current_label is not None:
            current_parts.append(paragraph.strip())
        else:
            lead_in_parts.append(paragraph.strip())

    if current_label is not None and current_parts:
        numbered_clauses.append((current_label, "\n\n".join(current_parts).strip()))

    if numbered_clauses:
        return numbered_clauses

    return [
        (str(index), paragraph)
        for index, paragraph in enumerate(paragraphs, start=1)
        if paragraph.strip()
    ]


def _content_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    content_paragraphs: list[str] = []
    for paragraph in paragraphs:
        split_heading = _split_heading_paragraph(paragraph)
        if split_heading is None:
            content_paragraphs.append(paragraph)
            continue
        if split_heading.body_text:
            content_paragraphs.append(split_heading.body_text)
    return content_paragraphs or paragraphs


# ---------------------------------------------------------------------------
# Microchunk splitting
# ---------------------------------------------------------------------------


def _split_microchunks(
    text: str,
    *,
    token_chunk_size: int = 300,
    token_chunk_overlap: int = 50,
    token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return [text.strip()]

    counter = token_counter or _count_tokens
    chunk_size = max(int(token_chunk_size), 1)
    overlap_target = max(int(token_chunk_overlap), 0)

    microchunks: list[str] = []
    start_index = 0
    while start_index < len(sentences):
        window: list[str] = []
        window_tokens = 0
        index = start_index
        while index < len(sentences):
            sentence = sentences[index]
            sentence_tokens = max(counter(sentence), 1)
            if window and (window_tokens + sentence_tokens) > chunk_size:
                break
            window.append(sentence)
            window_tokens += sentence_tokens
            index += 1
            if window_tokens >= chunk_size:
                break

        if not window:
            window = [sentences[start_index]]
            index = start_index + 1

        chunk = " ".join(window).strip()
        if chunk:
            microchunks.append(chunk)
        if index >= len(sentences):
            break

        overlap_tokens = 0
        overlap_count = 0
        for sentence in reversed(window):
            overlap_tokens += max(counter(sentence), 1)
            overlap_count += 1
            if overlap_tokens >= overlap_target:
                break

        next_start = index - overlap_count
        if next_start <= start_index:
            next_start = start_index + 1
        start_index = next_start

    return microchunks or [text.strip()]


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return _resolve_qwen_token_counter()(normalized)


@lru_cache(maxsize=1)
def _resolve_qwen_token_counter() -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_QWEN3_EMBEDDING_MODEL,
            local_files_only=True,
        )
    except Exception:
        return _fallback_token_count

    def _count_with_tokenizer(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _count_with_tokenizer


def _fallback_token_count(text: str) -> int:
    return len(_tokenize(text))


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------


def _extract_part_heading(text: str) -> str | None:
    return _extract_heading_by_kind(text, "part")


def _extract_section_heading(text: str) -> str | None:
    return _extract_heading_by_kind(text, "section")


def _extract_article_heading(text: str) -> str | None:
    return _extract_heading_by_kind(text, "article")


def _normalize_heading_value(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    tokens = normalized.split(" ", 1)
    if not tokens:
        return normalized
    if len(tokens) == 1:
        return tokens[0].title()
    return f"{tokens[0].title()} {tokens[1]}"


@dataclass(slots=True)
class SplitHeadingParagraph:
    """Heading label and optional body remainder extracted from one paragraph."""

    kind: str
    label: str
    body_text: str | None = None


def _extract_heading_by_kind(text: str, kind: str) -> str | None:
    for paragraph in [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]:
        split_heading = _split_heading_paragraph(paragraph)
        if split_heading is None or split_heading.kind != kind:
            continue
        return split_heading.label
    return None


def _split_heading_paragraph(paragraph: str) -> SplitHeadingParagraph | None:
    normalized = " ".join(paragraph.split()).strip()
    for kind, keyword in (("part", "part"), ("article", "article"), ("section", "section")):
        split_heading = _split_heading_keyword_paragraph(normalized, kind=kind, keyword=keyword)
        if split_heading is not None:
            return split_heading
    return None


def _split_heading_keyword_paragraph(
    paragraph: str,
    *,
    kind: str,
    keyword: str,
) -> SplitHeadingParagraph | None:
    match = re.match(rf"^(?P<prefix>{keyword}\s+\d+[A-Za-z0-9.-]*)\s+(?P<rest>.+)$", paragraph, flags=re.IGNORECASE)
    if not match:
        return None

    prefix = _normalize_heading_value(match.group("prefix"))
    rest_tokens = match.group("rest").split()
    if not rest_tokens:
        return SplitHeadingParagraph(kind=kind, label=prefix)

    heading_tokens: list[str] = []
    body_tokens: list[str] = []
    for index, token in enumerate(rest_tokens):
        if index > 0 and _looks_like_body_start(rest_tokens[index:]):
            body_tokens = rest_tokens[index:]
            break
        heading_tokens.append(token)

    label = prefix if not heading_tokens else _normalize_heading_value(f"{prefix} {' '.join(heading_tokens)}")
    body_text = " ".join(body_tokens).strip() or None
    return SplitHeadingParagraph(kind=kind, label=label, body_text=body_text)


def _looks_like_body_start(tokens: list[str]) -> bool:
    if not tokens:
        return False
    first = tokens[0].strip("()[]{}.,;:")
    if not first:
        return False

    lower_first = first.lower()
    if lower_first in _BODY_START_CUES:
        return True

    if len(tokens) >= 2:
        second = tokens[1].strip("()[]{}.,;:").lower()
        if second in _BODY_VERBS and first[:1].isupper() and not first.isupper():
            return True
    return False


# ---------------------------------------------------------------------------
# Section chunking
# ---------------------------------------------------------------------------


def _section_chunks_for_doc(page_contexts: list[PageChunkContext]) -> list[IndexChunk]:
    section_chunks: list[IndexChunk] = []
    current_group: list[PageChunkContext] = []
    current_key: tuple[str, tuple[str, ...]] | None = None

    for page_context in page_contexts:
        section_label = page_context.headings.section_label()
        neighboring_headings = tuple(page_context.headings.neighboring_headings())
        if not section_label or not neighboring_headings:
            if current_group:
                section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))
                current_group = []
                current_key = None
            continue

        group_key = (section_label, neighboring_headings)
        if current_key is None or group_key == current_key:
            current_group.append(page_context)
            current_key = group_key
            continue

        section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))
        current_group = [page_context]
        current_key = group_key

    if current_group:
        section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))

    return section_chunks


def _build_section_chunk(page_contexts: list[PageChunkContext], section_index: int) -> IndexChunk:
    first_page = page_contexts[0].page
    headings = page_contexts[0].headings
    section_text = "\n\n".join(_section_body_for_page(page_context.page) for page_context in page_contexts).strip()
    page_span = [page_context.page.page_number for page_context in page_contexts]
    return _make_chunk(
        chunk_id=f"{first_page.doc_id}-section-{section_index}",
        doc_id=first_page.doc_id,
        page_span=page_span,
        parent_page_numbers=page_span,
        chunk_type="section",
        text=section_text or "\n\n".join(page_context.page.text.strip() for page_context in page_contexts).strip(),
        section=headings.section_label(),
        clause=None,
        neighboring_headings=headings.neighboring_headings(),
        document_title=first_page.document_title,
        document_family=first_page.document_family,
        parser_provenance=first_page.parser_provenance,
        heading_path=_combined_heading_path(page_contexts),
        source_block_ids=_combined_source_block_ids(page_contexts),
    )


def _section_body_for_page(page: CanonicalPageRecord) -> str:
    return "\n\n".join(_content_paragraphs(page.text)).strip()


# ---------------------------------------------------------------------------
# Parent page number mapping
# ---------------------------------------------------------------------------


def _parent_page_numbers_by_page(page_contexts: list[PageChunkContext]) -> dict[int, list[int]]:
    parent_pages_by_page: dict[int, list[int]] = {}
    current_group: list[PageChunkContext] = []
    current_key: tuple[str, tuple[str, ...]] | None = None

    def flush_group() -> None:
        if not current_group:
            return
        group_pages = [page_context.page.page_number for page_context in current_group]
        for page_context in current_group:
            parent_pages_by_page[page_context.page.page_number] = list(group_pages)

    for page_context in page_contexts:
        section_label = page_context.headings.section_label()
        neighboring_headings = tuple(page_context.headings.neighboring_headings())
        if not section_label or not neighboring_headings:
            flush_group()
            current_group = []
            current_key = None
            parent_pages_by_page[page_context.page.page_number] = [page_context.page.page_number]
            continue

        group_key = (section_label, neighboring_headings)
        if current_key is None or group_key == current_key:
            current_group.append(page_context)
            current_key = group_key
            continue

        flush_group()
        current_group = [page_context]
        current_key = group_key

    flush_group()
    return parent_pages_by_page


# ---------------------------------------------------------------------------
# Chunk factory and helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    page_span: list[int],
    parent_page_numbers: list[int] | None = None,
    chunk_type: str,
    text: str,
    section: str,
    clause: str | None,
    neighboring_headings: list[str],
    document_title: str | None,
    document_family: str | None,
    parser_provenance: str | None = None,
    heading_path: list[str] | None = None,
    source_block_ids: list[str] | None = None,
) -> IndexChunk:
    normalized_text = text.strip()
    tokens = _tokenize(normalized_text)
    return IndexChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page_span=list(page_span),
        parent_page_numbers=list(parent_page_numbers or page_span),
        chunk_type=chunk_type,
        text=normalized_text,
        section=section,
        clause=clause,
        neighboring_headings=_ordered_unique(neighboring_headings),
        entities=_extract_entities(normalized_text),
        dates=_extract_dates(normalized_text),
        token_count=len(tokens),
        bm25_terms=_ordered_unique(tokens)[:MAX_BM25_TERMS_METADATA],
        document_title=document_title,
        document_family=document_family,
        parser_provenance=parser_provenance,
        heading_path=_ordered_unique(heading_path or []),
        source_block_ids=_ordered_unique(source_block_ids or []),
    )


def _combined_heading_path(page_contexts: list[PageChunkContext]) -> list[str]:
    combined: list[str] = []
    for page_context in page_contexts:
        combined.extend(page_context.page.heading_path)
    return _ordered_unique(combined)


def _combined_source_block_ids(page_contexts: list[PageChunkContext]) -> list[str]:
    combined: list[str] = []
    for page_context in page_contexts:
        combined.extend(page_context.page.source_block_ids)
    return _ordered_unique(combined)


def _heading_path_for_block(page: CanonicalPageRecord, block: Any) -> list[str]:
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, dict):
        heading_path = metadata.get("heading_path")
        if isinstance(heading_path, list):
            return _ordered_unique(str(item).strip() for item in heading_path if str(item).strip())
    return list(page.heading_path)


def _source_block_ids_for_block(page: CanonicalPageRecord, block: Any) -> list[str]:
    metadata = getattr(block, "metadata", None)
    candidate_ids: list[str] = []
    if isinstance(metadata, dict):
        source_block_ids = metadata.get("source_block_ids")
        if isinstance(source_block_ids, list):
            candidate_ids.extend(str(item).strip() for item in source_block_ids if str(item).strip())
        source_block_id = metadata.get("source_block_id")
        if source_block_id:
            candidate_ids.append(str(source_block_id).strip())
    block_id = getattr(block, "block_id", "")
    if block_id:
        candidate_ids.append(str(block_id).strip())
    candidate_ids.extend(page.source_block_ids)
    return _ordered_unique(candidate_ids)


def _extract_entities(text: str, limit: int = 8) -> list[str]:
    entities = [match.group(0) for match in _ENTITY_RE.finditer(text)]
    return _ordered_unique(entities)[:limit]


def _extract_dates(text: str, limit: int = 6) -> list[str]:
    dates = [match.group(0) for match in _DATE_RE.finditer(text)]
    return _ordered_unique(dates)[:limit]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
