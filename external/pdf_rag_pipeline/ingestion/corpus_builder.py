"""Canonical corpus assembly from parsed documents and serialized tables."""

from __future__ import annotations

import logging

from ..schemas import (
    CanonicalPageRecord,
    ContentBlock,
    ContentBlockType,
    PageMapRecord,
    ParsedDocument,
    TableBlock,
)

LOGGER = logging.getLogger(__name__)


def build_corpus(
    document: ParsedDocument,
    table_blocks_by_page: dict[int, list[TableBlock]],
) -> list[CanonicalPageRecord]:
    """Build canonical page-level records from a parsed document.

    Each page produces one CanonicalPageRecord containing text ContentBlocks
    (split on double-newlines) and TABLE ContentBlocks derived from the
    serialized table rows.
    """

    records: list[CanonicalPageRecord] = []
    for page in document.pages:
        text_blocks = _text_blocks_for_page(
            document.doc_id,
            page.page_number,
            page.text,
        )
        page_tables = table_blocks_by_page.get(page.page_number, [])
        table_content_blocks = _table_content_blocks_for_page(page.page_number, page_tables)
        table_ids = list(dict.fromkeys(block.table_id for block in table_content_blocks if block.table_id))

        records.append(
            CanonicalPageRecord(
                doc_id=document.doc_id,
                page_number=page.page_number,
                document_title=page.document_title,
                document_family=page.document_family,
                source_mode=page.source_mode,
                parse_status=page.parse_status,
                parser_provenance="pymupdf:native",
                text=page.text,
                blocks=[*text_blocks, *table_content_blocks],
                table_ids=table_ids,
            )
        )
    return records


def build_page_map_record(document: ParsedDocument) -> PageMapRecord:
    """Build the page map summary for one document."""

    return PageMapRecord(
        doc_id=document.doc_id,
        filename=document.filename,
        document_title=document.document_title,
        document_family=document.document_family,
        page_count=document.page_count,
        page_numbers=[page.page_number for page in document.pages],
    )


# ── Internal helpers ──


def _text_blocks_for_page(
    doc_id: str,
    page_number: int,
    text: str,
) -> list[ContentBlock]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    blocks: list[ContentBlock] = []
    for block_index, paragraph in enumerate(paragraphs, start=1):
        blocks.append(
            ContentBlock(
                block_id=f"{doc_id}-page-{page_number}-text-{block_index}",
                type=ContentBlockType.TEXT,
                text=paragraph,
                page_number=page_number,
            )
        )
    return blocks


def _table_content_blocks_for_page(
    page_number: int,
    table_blocks: list[TableBlock],
) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for table_block in table_blocks:
        for serialized_block in table_block.serialized_blocks:
            if serialized_block.source_page_number != page_number:
                continue
            blocks.append(
                ContentBlock(
                    block_id=serialized_block.block_id,
                    type=ContentBlockType.TABLE,
                    text=serialized_block.text,
                    page_number=serialized_block.source_page_number,
                    table_id=table_block.table_id,
                    metadata={
                        "row_anchor": serialized_block.row_anchor,
                        "column_context": serialized_block.column_context,
                        "table_caption": serialized_block.table_caption,
                        "source_page_number": serialized_block.source_page_number,
                    },
                )
            )
    return blocks
