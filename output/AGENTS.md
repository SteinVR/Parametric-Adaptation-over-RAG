# Export standards

The project has two distinct PDF export profiles. They share `term-paper_3/Term-Paper-3.md` as the manuscript source, but their layout and deliverables are independent. Do not make one profile inherit formatting from the other merely to reduce build code.

## 1. Pre-paper export

- Canonical file: `Parametric-Adaptation-Methods-RAG.pdf` in the repository root.
- Build with `term-paper_3/scripts/build_pdf.sh`.
- Keep the existing pre-paper/arXiv-like presentation. Do not add the university title page, university typography, university pagination, or a `_uni` suffix.
- `output/arxiv/Parametric-Adaptation-Methods-arXiv.zip` is a separate package produced by `build_arxiv.sh`; it is not a third PDF profile.

## 2. University export

The university submission uses the `_uni` suffix and must comply with `docs/HA_formal_en.pdf`. The supplied LaTeX package under `external/CL_Template_Thesis/` is the visual and structural reference; preserve its official source layout. Project-specific Pandoc integration belongs under `term-paper_3/`, currently in `term-paper_3/templates/pandoc-uni.tex`. When the formal requirements and the pre-paper styling differ, the formal requirements control the university export only.

Produce the university submission as separate deliverables; do not merge them unless the user explicitly requests a combined submission copy:

- `Parametric-Adaptation-Methods-RAG_uni.pdf` — university-formatted body without a title page.
- `Parametric-Adaptation-Methods-RAG_frontpage_uni.pdf` — standalone one-page university title page.

The declaration of academic integrity is entirely user-managed and outside the project export scope. Do not create, edit, populate, render, include, package, or validate it.

### University body requirements

- DIN A4, one-sided, approximately 3 cm margins.
- 12 pt conventional Times- or Arial-like body font; footnotes 10 pt.
- 1.5 line spacing, justified paragraphs, correct hyphenation, and widow/orphan plus stranded-heading protection.
- Page numbers at the bottom, centered or right-aligned. The title page and table of contents are not counted; the Introduction begins at Arabic page 1. Lists or indexes before the Introduction and the appendix use Roman page numbers.
- Number sections and subsections consistently with Arabic or Roman numerals.
- Preserve the required structure: table of contents, Introduction, main part covering Method and Results, Conclusion, bibliography, and optional appendix.
- The Introduction must state the topic, scope and limits, research question, methods, relevant research context or gap, significance, and rationale for the paper structure.
- The main part must describe the study design, method implementation and technical setup, analyse the subject and literature, establish the paper's own position, and present findings against the research question.
- The Conclusion must state the main findings, relate them to prior research, discuss strengths and limitations, and give perspectives for further research.
- The bibliography must contain only cited works, be ordered alphabetically by author or editor surname, and contain correct publication details. The optional appendix may contain supplementary statistics, tables, source code, and references to software, datasets, or corpora.
- Do not embed the university title page in the body PDF.

### University title-page requirements

- Match the layout of `external/CL_Template_Thesis/frontpage_only.pdf`. Treat `cldh/frontpage.tex`, `cldh/preamble.sty`, `cldh/originality-confirm.tex`, and the supplied university assets as read-only vendor files; populate the title page through `config.tex`.
- Treat `frontpage_only.pdf` as a visual layout reference only: its old title, dates, and other rendered values are not canonical.
- Populate every required field before final export: university, department and subject; seminar title and semester; submission date; lecturer name and title; paper title; author name; field and study semester; matriculation number; address; and email.
- Never render an empty `()` after the seminar title. The official layout prints `moduleid` in parentheses, so that configuration value must contain the applicable module identifier or semester label.
- The title-page PDF must contain exactly one unnumbered page and no manuscript body.

## Build and QA rules

- Build scripts must preserve the separation between university body and title page. If a script combines them, fix the script rather than accepting that artifact as canonical.
- Never patch generated PDFs directly. Change the manuscript, university template/configuration, or build script and regenerate the affected profile.
- Validate both profiles independently after relevant manuscript changes; success of one does not validate the other.
- For the university export, verify A4 page size, page count, font sizes, margins, line spacing, pagination transitions, section numbering, table of contents, title-page fields, and absence of empty parentheses.
- Render and visually inspect every final PDF for clipping, overflow, broken tables or figures, bad page breaks, blank pages, and incorrect headers or footers before calling the export complete.
