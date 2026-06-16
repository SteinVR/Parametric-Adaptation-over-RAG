#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 INPUT.md OUTPUT.pdf [START_HEADING]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_dir="$(cd "$script_dir/.." && pwd)"

input="$1"
output="$2"
start_heading="${3:-}"

if ! command -v pandoc >/dev/null 2>&1 || ! command -v tectonic >/dev/null 2>&1; then
  echo "Missing pandoc or tectonic. Run: $skill_dir/scripts/install-deps-macos.sh" >&2
  exit 1
fi

input_abs="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
output_abs="$(cd "$(dirname "$output")" && pwd)/$(basename "$output")"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

prepared="$work_dir/manuscript.md"
python3 "$skill_dir/scripts/prepare_markdown.py" "$input_abs" "$prepared" --from-heading "$start_heading"

resource_path="$(dirname "$input_abs"):$(pwd)"

pandoc "$prepared" \
  --from=markdown+smart+pipe_tables+fenced_code_blocks+raw_tex+tex_math_dollars+implicit_figures \
  --pdf-engine=tectonic \
  --template="$skill_dir/assets/arxiv-like.tex" \
  --toc \
  --toc-depth=2 \
  --resource-path="$resource_path" \
  --metadata=link-citations:true \
  -o "$output_abs"

echo "$output_abs"
