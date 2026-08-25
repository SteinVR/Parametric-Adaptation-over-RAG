#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$paper_dir/.." && pwd)"

"$paper_dir/skills/format-science-paper-pdf/scripts/render_science_paper_pdf.sh" \
  "$paper_dir/Term-Paper-3.md" \
  "$repo_dir/Parametric-Adaptation-Methods-RAG.pdf" \
  "# Abstract"
