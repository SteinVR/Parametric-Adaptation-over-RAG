"""Core data models for the PDF RAG pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceMode(StrEnum):
    """Source used for the final page text."""

    NATIVE = "native"


class ParseStatus(StrEnum):
    """Parse completion state for a page or document."""

    PARSED = "parsed"
    DEGRADED = "degraded"
    FAILED = "failed"


class ContentBlockType(StrEnum):
    """Retrieval-facing block type inside a canonical page record."""

    TEXT = "text"
    TABLE = "table"


# ── Table models ──


class RawTableRecord(BaseModel):
    """Raw table extracted from a page before serialization."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_number: int = Field(ge=1)
    rows: list[list[str]] = Field(default_factory=list)
    row_page_numbers: list[int] = Field(default_factory=list)
    has_header: bool = False
    raw_markdown: str | None = None
    raw_html: str | None = None
    header_signature: list[str] = Field(default_factory=list)
    caption: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    preceding_context: list[str] = Field(default_factory=list)
    following_context: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "rows" not in payload and "cells" in payload:
            payload["rows"] = payload.pop("cells")
        if "preceding_context" not in payload and "context_before" in payload:
            payload["preceding_context"] = payload.pop("context_before")
        if "following_context" not in payload and "context_after" in payload:
            payload["following_context"] = payload.pop("context_after")
        return payload


class SerializedTableBlock(BaseModel):
    """Self-contained retrieval block derived from a raw table."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    text: str
    source_page_number: int = Field(ge=1)
    row_anchor: str | None = None
    column_context: list[str] = Field(default_factory=list)
    table_caption: str | None = None
    footnotes: list[str] = Field(default_factory=list)


class TableBlock(BaseModel):
    """Structured table representation preserved in the canonical corpus."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_span: list[int] = Field(default_factory=list)
    raw_markdown: str | None = None
    raw_html: str | None = None
    header_signature: list[str] = Field(default_factory=list)
    serialized_blocks: list[SerializedTableBlock] = Field(default_factory=list)


# ── Page / Document models ──


class ParsedPage(BaseModel):
    """Intermediate parsed-page representation."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_number: int = Field(ge=1)
    document_title: str | None = None
    document_family: str | None = None
    source_mode: SourceMode = SourceMode.NATIVE
    parse_status: ParseStatus = ParseStatus.PARSED
    text: str
    text_blocks: list[str] = Field(default_factory=list)
    tables: list[RawTableRecord] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "tables" not in payload and "table_candidates" in payload:
            payload["tables"] = payload.pop("table_candidates")
        return payload


class ParsedDocument(BaseModel):
    """Intermediate parsed-document representation."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=0)
    document_title: str | None = None
    document_family: str | None = None
    parse_status: ParseStatus = ParseStatus.PARSED
    pages: list[ParsedPage] = Field(default_factory=list)


# ── Corpus models ──


class ContentBlock(BaseModel):
    """Retrieval-facing block stored on a page."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    type: ContentBlockType
    text: str
    page_number: int = Field(ge=1)
    table_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalPageRecord(BaseModel):
    """Canonical page-level record emitted by the ingestion stage."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_number: int = Field(ge=1)
    document_title: str | None = None
    document_family: str | None = None
    source_mode: SourceMode = SourceMode.NATIVE
    parse_status: ParseStatus = ParseStatus.PARSED
    parser_provenance: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)
    text: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class PageMapRecord(BaseModel):
    """Document-level page map summary for downstream lookups."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    document_title: str | None = None
    document_family: str | None = None
    page_count: int = Field(ge=0)
    page_numbers: list[int] = Field(default_factory=list)
