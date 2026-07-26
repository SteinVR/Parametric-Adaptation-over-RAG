#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$paper_dir/skills/format-science-paper-pdf/scripts/render_science_paper_pdf.sh" \
  "$paper_dir/Term-Paper-3.md" \
  "$paper_dir/Term-Paper-3.pdf" \
  "# Abstract"
