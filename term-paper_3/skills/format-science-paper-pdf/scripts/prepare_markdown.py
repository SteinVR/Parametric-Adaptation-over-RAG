#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+\{.*\})?\s*$")
TABLE_CAPTION_RE = re.compile(r"^\*\*(Table\s+[^*]+?)\.\*\*\s*(.*)$")
FIGURE_CAPTION_RE = re.compile(r"^\*(Figure\s+[^*]+?)\.\*\s*$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
CAPTION_NUMBER_RE = re.compile(r"^(?:Table|Figure)\s+[A-Z]?\d+\.?\s*", re.IGNORECASE)
PIPE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
APPENDIX_SUB_RE = re.compile(r"^[A-Z](?:\.\d+)+\s+")
APPENDIX_HEAD_RE = re.compile(r"^Appendix\s+[A-Z][.:]?\s+(.*)$")


def find_start(lines: list[str], heading: str) -> int:
    if not heading:
        return 0
    normalized = heading.strip()
    for i, line in enumerate(lines):
        if line.strip() == normalized:
            return i
    raise SystemExit(f"Start heading not found: {heading}")


def strip_manual_toc(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() == "# table of contents":
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("#"):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def extract_front_matter(lines: list[str]) -> list[str]:
    """Capture the inner lines of a leading YAML block (title/author/keywords/...).

    Returned verbatim so the metadata survives the ``--from-heading`` strip and is
    re-emitted into the prepared manuscript's YAML header.
    """
    if not lines or lines[0].strip() != "---":
        return []
    meta: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        meta.append(line)
    return meta


def extract_abstract(lines: list[str]) -> tuple[list[str], list[str]]:
    if not lines or lines[0].strip().lower() != "# abstract":
        return [], lines

    abstract: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].startswith("# "):
            break
        abstract.append(lines[i])
        i += 1
    body = lines[i:]
    return abstract, body


def promote_headings(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) > 1:
            out.append(match.group(1)[1:] + " " + match.group(2).strip())
        else:
            out.append(line)
    return out


def convert_tables_and_figures(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        table_caption = TABLE_CAPTION_RE.match(line.strip())
        if table_caption:
            j = i + 1
            blanks: list[str] = []
            while j < len(lines) and not lines[j].strip():
                blanks.append(lines[j])
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("|"):
                caption = CAPTION_NUMBER_RE.sub("", table_caption.group(1).strip()).strip()
                trailing = table_caption.group(2).strip()
                if trailing:
                    caption = caption.rstrip(".") + ". " + trailing
                out.append("Table: " + caption.rstrip(".") + ".")
                out.extend(blanks)
                i += 1
                continue

        image = IMAGE_RE.match(line.strip())
        if image:
            j = i + 1
            blanks = []
            while j < len(lines) and not lines[j].strip():
                blanks.append(lines[j])
                j += 1
            if j < len(lines):
                figure_caption = FIGURE_CAPTION_RE.match(lines[j].strip())
                if figure_caption:
                    alt = CAPTION_NUMBER_RE.sub("", figure_caption.group(1).strip()).strip()
                    alt = alt.rstrip(".") + "."
                    out.append(f"![{alt}]({image.group(2)})")
                    i = j + 1
                    continue

        out.append(line)
        i += 1
    return out


def number_sections_and_back_matter(lines: list[str]) -> list[str]:
    """Switch from manually numbered headings to native LaTeX numbering.

    Operates on already-promoted headings (``#`` = section, ``##`` = subsection):

    - Strips the baked-in ``1.``/``1.1`` (and appendix ``A.1``) prefixes so LaTeX
      generates the numbers, producing an aligned ``\\numberline`` table of
      contents.
    - Puts ``References`` and each ``Appendix`` on a fresh page (``\\clearpage``).
    - Marks ``References`` unnumbered.
    - Routes the appendices through ``\\appendix`` (auto-lettered A, B, C, ...)
      under a single bold ``Appendices`` entry in the contents.

    Raw ``\\clearpage``/``\\appendix``/``\\addcontentsline`` lines pass through to
    LaTeX via the Pandoc ``raw_tex`` extension.
    """
    out: list[str] = []
    in_code = False
    appendix_started = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        match = HEADING_RE.match(line)
        if not match:
            out.append(line)
            continue

        hashes = match.group(1)
        title = match.group(2).strip()

        if len(hashes) == 1 and title.lower() == "references":
            out.extend(["\\clearpage", "", "# References {-}"])
            continue

        appendix = APPENDIX_HEAD_RE.match(title) if len(hashes) == 1 else None
        if appendix:
            out.extend(["\\clearpage", ""])
            if not appendix_started:
                out.extend([
                    "\\appendix",
                    "",
                    "\\counterwithin{table}{section}",
                    "",
                    "\\counterwithin{figure}{section}",
                    "",
                    "\\addcontentsline{toc}{section}{Appendices}",
                    "",
                ])
                appendix_started = True
            out.append("# " + appendix.group(1).strip())
            continue

        stripped = (
            APPENDIX_SUB_RE.sub("", title)
            if appendix_started
            else SECTION_NUM_RE.sub("", title)
        )
        out.append(hashes + " " + stripped)
    return out


def number_references(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_refs = False
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code:
            match = HEADING_RE.match(line)
            if match:
                title = match.group(2).strip().lower()
                in_refs = title == "references"
                out.append(line)
                continue
            if in_refs and line.startswith("- "):
                out.append("1. " + line[2:])
                continue
        out.append(line)
    return out


def normalize_pipe_tables(lines: list[str]) -> list[str]:
    def split(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    def visible_len(cell: str) -> int:
        cell = re.sub(r"[*_`\\]", "", cell)
        cell = re.sub(r"<[^>]+>", "", cell)
        return len(cell.strip())

    def join(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    out: list[str] = []
    i = 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and lines[i].lstrip().startswith("|")
            and PIPE_SEPARATOR_RE.match(lines[i + 1])
        ):
            block: list[str] = [lines[i], lines[i + 1]]
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                block.append(lines[j])
                j += 1

            rows = [split(row) for row in block if row.lstrip().startswith("|")]
            header = rows[0]
            if header and not header[0]:
                header[0] = "System"

            column_count = len(header)
            widths: list[int] = []
            for col in range(column_count):
                max_len = max(
                    [visible_len(header[col])]
                    + [
                        visible_len(row[col])
                        for row in rows[2:]
                        if len(row) > col
                    ]
                )
                min_width = 14 if col == 0 else 5
                widths.append(max(min_width, min(max_len, 24)))

            out.append(join(header))
            out.append("| " + " | ".join("-" * width for width in widths) + " |")
            out.extend(block[2:])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return out


def write_yaml(meta: list[str], abstract: list[str], body: list[str], output: Path) -> None:
    text: list[str] = ["---"]
    text.extend(meta)
    text.append("abstract: |")
    abstract_text = "\n".join(abstract).strip()
    if abstract_text:
        for line in abstract_text.splitlines():
            text.append("  " + line)
    else:
        text.append("  ")
    text.extend(["---", ""])
    text.extend(body)
    output.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-heading", default="")
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    lines = [line.replace("+/-", "±") for line in lines]
    meta = extract_front_matter(lines)
    lines = lines[find_start(lines, args.from_heading) :]
    abstract, body = extract_abstract(lines)
    body = strip_manual_toc(body)
    body = promote_headings(body)
    body = convert_tables_and_figures(body)
    body = normalize_pipe_tables(body)
    body = number_sections_and_back_matter(body)
    body = number_references(body)
    write_yaml(meta, abstract, body, args.output)


if __name__ == "__main__":
    main()
