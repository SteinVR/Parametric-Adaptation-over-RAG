---
name: format-science-paper-pdf
description: Format Markdown research manuscripts into classic arXiv-like science paper PDFs using pandoc and a LaTeX engine. Use when Codex needs to convert .md academic papers, term papers, technical reports, appendices, tables, figures, and references into a clean one-column LaTeX PDF with Computer Modern/Latin Modern typography, A4 pages, numbered sections, booktabs tables, figure captions, references, and centered page numbers. Also use when required PDF tooling is missing and a Homebrew installation script is needed.
---

# Format Science Paper PDF

## Core Style

Use the classic arXiv-like LaTeX manuscript style:

- A4, one column, generous margins.
- Serif LaTeX typography with Latin Modern / Computer Modern feel.
- Section hierarchy like `1 Introduction`, `1.1 Problem and Motivation`.
- Minimal decoration: no cards, colored boxes, institutional cover, or magazine layout.
- Tables use booktabs-style horizontal rules and no vertical borders.
- Figures are centered floats with captions below.
- References appear as a numbered list unless a real bibliography file is provided.

## Workflow

1. Check for `pandoc` and `tectonic`.
2. If either is missing on macOS, run `scripts/install-deps-macos.sh`.
3. Render with `scripts/render_science_paper_pdf.sh`.
4. Open several rendered pages as images and inspect layout, captions, table width, figure scaling, and page breaks.

## Rendering Command

Use:

```bash
scripts/render_science_paper_pdf.sh INPUT.md OUTPUT.pdf "# Abstract"
```

The optional third argument is the first heading to include. When `"# Abstract"` is used, content before that heading is dropped.

The renderer:

- extracts the first `# Abstract` section into the LaTeX abstract block;
- drops a manual Markdown `# Table of Contents` block and lets LaTeX generate the contents;
- promotes later headings by one level so `## 1. Introduction` becomes a top-level section;
- converts `**Table N. ...**` immediately before pipe tables into Pandoc table captions;
- combines image lines followed by italic `*Figure N. ...*` paragraphs into proper figure captions;
- converts bullet references under `# References` into a numbered reference list.

## Resources

- `scripts/install-deps-macos.sh`: install `pandoc` and `tectonic` with Homebrew.
- `scripts/prepare_markdown.py`: normalize manuscript Markdown before Pandoc.
- `scripts/render_science_paper_pdf.sh`: run preprocessing and Pandoc.
- `assets/arxiv-like.tex`: Pandoc LaTeX template for the target style.
