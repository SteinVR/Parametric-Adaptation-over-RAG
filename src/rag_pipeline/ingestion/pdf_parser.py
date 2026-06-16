"""Native PDF parsing with table candidate extraction."""

from __future__ import annotations

import contextlib
import io
import logging
from hashlib import sha256
from pathlib import Path

import fitz

from ..schemas import (
    ParsedDocument,
    ParsedPage,
    ParseStatus,
    RawTableRecord,
    SourceMode,
)

LOGGER = logging.getLogger(__name__)

COURT_MARKERS = (
    "COURT OF",
    "SCT",
    "ARBITRATION",
    "DIGITAL ECONOMY COURT",
    "TECHNOLOGY AND CONSTRUCTION DIVISION",
    "ENFORCEMENT",
    "JUDGMENT",
    "JUDGMENTS",
    "ORDERS",
)

LEGAL_MARKERS = (
    "article",
    "section",
    "clause",
    "law",
    "court",
    "judgment",
    "order",
    "decision",
    "regulation",
)


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """Parse one PDF into the canonical parsed-document contract.

    Extracts native text, text blocks, and table candidates from each page
    using PyMuPDF. Every page is marked source_mode=NATIVE, parse_status=PARSED.
    """

    path = Path(pdf_path)
    fitz.TOOLS.mupdf_display_warnings(False)
    fitz.TOOLS.mupdf_display_errors(False)

    with fitz.open(path) as document:
        metadata_title = _extract_metadata_title(document)
        document_title = metadata_title
        document_family = "other_legal_document"
        pages: list[ParsedPage] = []

        for page in document:
            page_number = page.number + 1
            native_text = page.get_text("text").strip()
            text_blocks = _extract_text_blocks(page)
            tables = _extract_table_candidates(path.stem, page_number, page, text_blocks)

            if page_number == 1:
                document_title = metadata_title or _extract_title_from_text(native_text)
                document_family = _classify_document_family(native_text)

            pages.append(
                ParsedPage(
                    doc_id=path.stem,
                    page_number=page_number,
                    document_title=document_title,
                    document_family=document_family,
                    source_mode=SourceMode.NATIVE,
                    parse_status=ParseStatus.PARSED,
                    text=native_text,
                    text_blocks=text_blocks,
                    tables=tables,
                )
            )

    return ParsedDocument(
        doc_id=path.stem,
        filename=path.name,
        sha256=_sha256_for_file(path),
        page_count=len(pages),
        document_title=document_title,
        document_family=document_family,
        parse_status=ParseStatus.PARSED,
        pages=pages,
    )


# ── Metadata helpers ──


def _extract_metadata_title(document: fitz.Document) -> str | None:
    metadata = document.metadata or {}
    title = (metadata.get("title") or "").strip()
    return title or None


def _extract_title_from_text(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None


def _classify_document_family(first_page_text: str) -> str:
    heading = " ".join(line.strip() for line in first_page_text.splitlines()[:4]).upper()
    if "LAW NO." in heading or "DIFC LAW" in heading or heading.endswith(" LAW"):
        return "law_or_regulation"
    if any(marker in heading for marker in COURT_MARKERS):
        return "court_or_arbitration_decision"
    return "other_legal_document"


# ── Block and table extraction ──


def _extract_text_blocks(page: fitz.Page) -> list[str]:
    results: list[str] = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        text = str(block[4]).strip()
        if text:
            results.append(text)
    return results


def _extract_table_candidates(
    doc_id: str,
    page_number: int,
    page: fitz.Page,
    text_blocks: list[str],
) -> list[RawTableRecord]:
    candidates: list[RawTableRecord] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tables = list(page.find_tables().tables)
    except Exception:
        tables = []

    for table_index, table in enumerate(tables, start=1):
        rows = _normalize_rows(_extract_rows(table))
        if not rows:
            continue
        caption = text_blocks[-1] if text_blocks else None
        candidates.append(
            RawTableRecord(
                table_id=f"{doc_id}-p{page_number}-t{table_index}",
                page_number=page_number,
                rows=rows,
                row_page_numbers=[page_number] * len(rows),
                has_header=False,
                raw_markdown=_rows_to_markdown(rows),
                raw_html=_rows_to_html(rows),
                header_signature=list(rows[0]),
                caption=caption,
                preceding_context=text_blocks[:3],
                following_context=text_blocks[-3:],
            )
        )
    return candidates


# ── Row normalization and formatting ──


def _extract_rows(table: object) -> list[list[object]]:
    if hasattr(table, "extract"):
        return table.extract()
    return []


def _normalize_rows(rows: list[list[object]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        normalized.append([str(cell).strip() if cell is not None else "" for cell in row])
    return normalized


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(padded_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded_rows[1:]]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def _rows_to_html(rows: list[list[str]]) -> str:
    html_rows: list[str] = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        html_rows.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in row) + "</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _split_text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# ── File hashing ──


def _sha256_for_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()
