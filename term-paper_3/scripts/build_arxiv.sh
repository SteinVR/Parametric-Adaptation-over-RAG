#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$paper_dir/.." && pwd)"
skill_dir="$paper_dir/skills/format-science-paper-pdf"
input="$paper_dir/Term-Paper-3.md"
pdf_dir="$repo_dir/output/pdf"
archive_dir="$repo_dir/output/arxiv"
preview_pdf="$pdf_dir/Parametric-Adaptation-Methods-arXiv.pdf"
archive="$archive_dir/Parametric-Adaptation-Methods-arXiv.zip"

for tool in python3 pandoc tectonic zip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done

mkdir -p "$pdf_dir" "$archive_dir"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
package_dir="$work_dir/package"
build_dir="$work_dir/build"
prepared="$work_dir/manuscript.md"
mkdir -p "$package_dir/figures" "$build_dir"

python3 "$skill_dir/scripts/prepare_markdown.py" \
  "$input" \
  "$prepared" \
  --from-heading "# Abstract"

# The submission archive must use relative paths and contain only referenced files.
# The term-paper build starts references and every appendix on a new page; the
# arXiv build stays compact and lets those sections flow naturally.
sed -E \
  -e '/^\\clearpage$/d' \
  -e 's/±/$\\pm$/g' \
  -e 's#\(\.\./assets/figures/([^)]*)\)#(figures/\1)#g' \
  "$prepared" | \
awk '
  $0 == "# Use of Generative AI" {
    print "\\enlargethispage{2\\baselineskip}"
    print ""
  }
  { print }
' > "$work_dir/arxiv-manuscript.md"

for figure in \
  fig01_system_schematic.png \
  fig02_delta_bars.png \
  fig03_judge_criteria.png \
  fig04_per_type_heatmap.png \
  fig05_singledoc_multidoc.png \
  figB1_error_overlap_heatmap.png \
  figB2_seed_stability.png \
  figB3_pairwise_win_rates.png; do
  cp "$repo_dir/assets/figures/$figure" "$package_dir/figures/$figure"
done

pandoc "$work_dir/arxiv-manuscript.md" \
  --from=markdown+smart+pipe_tables+fenced_code_blocks+raw_tex+tex_math_dollars+implicit_figures \
  --to=latex \
  --standalone \
  --template="$skill_dir/assets/arxiv-submission.tex" \
  --metadata=link-citations:true \
  -o "$package_dir/main.tex"

(cd "$package_dir" && tectonic --outdir "$build_dir" main.tex)

cp "$build_dir/main.pdf" "$preview_pdf"
(cd "$package_dir" && zip -X -q -r "$archive" main.tex figures)

echo "$preview_pdf"
echo "$archive"
