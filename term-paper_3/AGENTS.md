# Current manuscript

`Term-Paper-3.md` is the canonical manuscript source. The two root PDFs are the generated pre-paper and university-submission artifacts; `output/arxiv/` contains the arXiv package.

## Editing and evidence

- Edit the Markdown manuscript and the relevant build or figure source; never patch a PDF or submission archive directly.
- Keep system names and roles aligned with `README.md`: Base-RAG (`S1`), RAFT-RAG (`S2+R`), CLM-RAG (`S3+R`), Merge-RAG (`S7`, post-hoc), and the closed-book controls.
- Verify quantitative claims against `results/`, not against another prose summary. Preserve the distinction between measured findings, post-hoc analysis, limitations, and speculation.
- Figures referenced by the paper live under `assets/figures/` and are produced from result artifacts. Regenerate their source when data changes instead of editing rendered PNGs.
- Treat the bundled `skills/format-science-paper-pdf/` directory as build tooling; change it only when the rendering pipeline itself is in scope.

## Build and verification

- Run `term-paper_3/scripts/build_pdf.sh` from the repository root to regenerate the root paper PDF.
- Run `term-paper_3/scripts/build_uni_pdf.sh` to regenerate the university-formatted root PDF with the `_uni` suffix, or `term-paper_3/scripts/build_pdfs.sh` to build both versions.
- Run `term-paper_3/scripts/build_arxiv.sh` to regenerate the arXiv archive under `output/arxiv/`; it must not create a third paper PDF under `output/`.
- After a manuscript or build change, inspect the rendered PDF for page flow, figures, tables, equations, links, references, and accidental blank pages. A successful command exit does not establish layout correctness.
- If the required Pandoc, Tectonic, LaTeX, or font dependencies are unavailable, report the build as unverified rather than altering the manuscript to work around an environment-only problem.
