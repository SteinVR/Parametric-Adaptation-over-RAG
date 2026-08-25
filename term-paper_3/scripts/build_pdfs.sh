#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/build_pdf.sh"
"$script_dir/build_uni_pdf.sh"
