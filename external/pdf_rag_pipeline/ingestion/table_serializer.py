"""Rule-based table detection, merging, and serialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..schemas import (
    ParsedDocument,
    RawTableRecord,
    SerializedTableBlock,
    TableBlock,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TableCandidate:
    """Intermediate detected table before merge/serialization."""

    table_id: str
    page_span: list[int]
    rows: list[list[str]]
    row_page_numbers: list[int]
    has_header: bool
    raw_markdown: str
    raw_html: str
    header_signature: list[str]
    caption: str | None = None
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


def serialize_document_tables(document: ParsedDocument) -> dict[int, list[TableBlock]]:
    """Serialize already-extracted table candidates on a parsed document.

    Returns a mapping from page number to the list of TableBlock objects
    whose page_span includes that page.
    """

    candidates: list[TableCandidate] = []
    for page in document.pages:
        for candidate in page.tables:
            rows = _normalize_rows(candidate.rows)
            has_header, header_signature = _resolve_header(candidate, rows)
            candidates.append(
                TableCandidate(
                    table_id=candidate.table_id,
                    page_span=[candidate.page_number],
                    rows=rows,
                    row_page_numbers=_candidate_row_page_numbers(candidate, rows),
                    has_header=has_header,
                    raw_markdown=candidate.raw_markdown or "",
                    raw_html=candidate.raw_html or "",
                    header_signature=header_signature,
                    caption=candidate.caption,
                    context_before=list(candidate.preceding_context),
                    context_after=list(candidate.following_context),
                )
            )

    merged = merge_adjacent_tables(candidates)
    page_map: dict[int, list[TableBlock]] = {}
    for candidate in merged:
        block = serialize_table(candidate)
        for page_number in candidate.page_span:
            page_map.setdefault(page_number, []).append(block)
    return page_map


def merge_adjacent_tables(candidates: list[TableCandidate]) -> list[TableCandidate]:
    """Merge consecutive-page tables when headers and continuation cues match."""

    if not candidates:
        return []

    merged: list[TableCandidate] = []
    current = candidates[0]
    for candidate in candidates[1:]:
        if _should_merge(current, candidate):
            merged_rows = _merge_rows(current, candidate)
            current = TableCandidate(
                table_id=current.table_id,
                page_span=[*current.page_span, *candidate.page_span],
                rows=merged_rows,
                row_page_numbers=_merge_row_page_numbers(current, candidate),
                has_header=current.has_header,
                raw_markdown=_rows_to_markdown(merged_rows),
                raw_html=_rows_to_html(merged_rows),
                header_signature=current.header_signature,
                caption=current.caption or candidate.caption,
                context_before=current.context_before,
                context_after=candidate.context_after,
            )
            continue

        merged.append(current)
        current = candidate

    merged.append(current)
    return merged


def serialize_table(candidate: TableCandidate) -> TableBlock:
    """Convert a detected table candidate into the canonical TableBlock."""

    width = _width(candidate.rows)
    if candidate.has_header and candidate.header_signature:
        headers = [
            header if header else f"column_{column_index + 1}"
            for column_index, header in enumerate(_pad_row(candidate.header_signature, width))
        ]
    else:
        headers = [f"column_{column_index + 1}" for column_index in range(width)]

    data_start_index = 1 if candidate.has_header and len(candidate.rows) > 1 else 0
    data_rows = candidate.rows[data_start_index:]
    data_row_pages = candidate.row_page_numbers[data_start_index:]
    serialized_blocks: list[SerializedTableBlock] = []
    footnotes = candidate.context_after[:2] if candidate.context_after else []

    for row_index, row in enumerate(data_rows, start=1):
        padded_row = _pad_row(row, width)
        if not any(cell for cell in padded_row):
            continue

        row_anchor = next((cell for cell in padded_row if cell), None)
        pairs: list[str] = []
        for column_index, cell in enumerate(padded_row):
            if not cell:
                continue
            pairs.append(f"{headers[column_index]}: {cell}")

        if not pairs:
            continue

        lead = candidate.caption or "Table"
        block_text = f"{lead}. "
        if row_anchor:
            block_text += f"{row_anchor}. "
        block_text += "; ".join(pairs)
        if footnotes:
            block_text += f" Notes: {' '.join(footnotes)}"

        serialized_blocks.append(
            SerializedTableBlock(
                block_id=f"{candidate.table_id}-row-{row_index}",
                text=block_text.strip(),
                source_page_number=data_row_pages[row_index - 1]
                if row_index - 1 < len(data_row_pages)
                else candidate.page_span[0],
                row_anchor=row_anchor,
                column_context=headers[1:] if len(headers) > 1 else headers,
                table_caption=candidate.caption,
                footnotes=footnotes,
            )
        )

    if not serialized_blocks:
        serialized_blocks.append(
            SerializedTableBlock(
                block_id=f"{candidate.table_id}-summary",
                text=((candidate.caption or "Table") + ". " + (candidate.raw_markdown or "").strip()).strip(),
                source_page_number=candidate.page_span[0],
                table_caption=candidate.caption,
                footnotes=footnotes,
            )
        )

    return TableBlock(
        table_id=candidate.table_id,
        page_span=candidate.page_span,
        raw_markdown=candidate.raw_markdown,
        raw_html=candidate.raw_html,
        header_signature=list(candidate.header_signature),
        serialized_blocks=serialized_blocks,
    )


# ── Internal helpers ──


def _resolve_header(candidate: RawTableRecord, rows: list[list[str]]) -> tuple[bool, list[str]]:
    width = _width(rows)
    explicit_signature = _pad_row(candidate.header_signature, width) if candidate.header_signature else []
    inferred_signature = _pad_row(rows[0], width) if rows else []
    if candidate.has_header:
        return True, explicit_signature or inferred_signature
    if _should_infer_header(rows):
        return True, explicit_signature or inferred_signature
    return False, []


def _candidate_row_page_numbers(candidate: RawTableRecord, rows: list[list[str]]) -> list[int]:
    if candidate.row_page_numbers:
        return list(candidate.row_page_numbers)
    return [candidate.page_number] * len(rows)


def _normalize_rows(rows: list[list[object]]) -> list[list[str]]:
    return [[str(cell).strip() if cell is not None else "" for cell in row] for row in rows]


def _width(rows: list[list[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def _pad_row(row: list[str], width: int) -> list[str]:
    return list(row) + [""] * max(width - len(row), 0)


def _should_infer_header(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False

    width = _width(rows)
    first = _pad_row(rows[0], width)
    second = _pad_row(rows[1], width)
    if not first or any(not cell for cell in first):
        return False

    first_alpha = sum(1 for cell in first if any(char.isalpha() for char in cell))
    if first_alpha != width:
        return False

    if any(_looks_numeric(cell) for cell in second):
        return True

    return any(
        first[column_index].strip().lower() != second[column_index].strip().lower()
        for column_index in range(width)
    )


def _looks_numeric(cell: str) -> bool:
    normalized = cell.replace(",", "").replace("%", "").replace(".", "").strip()
    return bool(normalized) and normalized.isdigit()


def _merge_rows(first: TableCandidate, second: TableCandidate) -> list[list[str]]:
    if not first.rows:
        return second.rows
    if not second.rows:
        return first.rows
    if first.has_header and second.has_header and first.header_signature == second.header_signature:
        return first.rows + second.rows[1:]
    return first.rows + second.rows


def _merge_row_page_numbers(first: TableCandidate, second: TableCandidate) -> list[int]:
    if first.has_header and second.has_header and first.header_signature == second.header_signature:
        return first.row_page_numbers + second.row_page_numbers[1:]
    return first.row_page_numbers + second.row_page_numbers


def _should_merge(previous: TableCandidate, current: TableCandidate) -> bool:
    if not previous.page_span or not current.page_span:
        return False
    if current.page_span[0] != previous.page_span[-1] + 1:
        return False
    if _width(previous.rows) != _width(current.rows):
        return False
    if previous.has_header != current.has_header:
        return False

    previous_caption = (previous.caption or "").strip().lower()
    current_caption = (current.caption or "").strip().lower()
    continuation_cues = ("continued", "continuation", "cont.")
    current_before = " ".join(current.context_before).lower()
    previous_after = " ".join(previous.context_after).lower()
    has_continuation = any(cue in current_before or cue in previous_after for cue in continuation_cues)

    if previous.has_header:
        if previous.header_signature != current.header_signature:
            return False
        return has_continuation or (previous_caption and previous_caption == current_caption)

    return has_continuation


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = _width(rows)
    padded_rows = [_pad_row(row, widths) for row in rows]
    header = "| " + " | ".join(padded_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * widths) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded_rows[1:]]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def _rows_to_html(rows: list[list[str]]) -> str:
    table_rows = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        cells = "".join(f"<{tag}>{cell}</{tag}>" for cell in row)
        table_rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(table_rows) + "</table>"
