#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$paper_dir/.." && pwd)"
skill_dir="$paper_dir/skills/format-science-paper-pdf"
template_dir="$repo_dir/external/CL_Template_Thesis"
input="$paper_dir/Term-Paper-3.md"
output="$repo_dir/Parametric-Adaptation-Methods-RAG_uni.pdf"

for tool in python3 pandoc tectonic; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
prepared="$work_dir/manuscript.md"
build_dir="$work_dir/build"

python3 "$skill_dir/scripts/prepare_markdown.py" \
  "$input" \
  "$prepared" \
  --from-heading "# Abstract" \
  --page-break-policy all

resource_path="$(dirname "$input"):$repo_dir:$template_dir"

(
  cd "$work_dir"
  pandoc manuscript.md \
    --from=markdown+smart+pipe_tables+fenced_code_blocks+raw_tex+tex_math_dollars+implicit_figures \
    --to=latex \
    --standalone \
    --template="$template_dir/pandoc-uni.tex" \
    --toc \
    --toc-depth=2 \
    --resource-path="$resource_path" \
    --extract-media=media \
    --metadata=link-citations:true \
    -o university-paper.tex

  mkdir -p "$build_dir"
  tectonic \
    -Z "search-path=$template_dir" \
    --outdir "$build_dir" \
    university-paper.tex
)

cp "$build_dir/university-paper.pdf" "$output"

echo "$output"
