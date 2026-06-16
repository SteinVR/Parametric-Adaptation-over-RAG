#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is not installed. Install it from https://brew.sh and rerun this script." >&2
  exit 1
fi

missing=()
for tool in pandoc tectonic; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing+=("$tool")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  brew install "${missing[@]}"
fi

pandoc --version | head -n 1
tectonic --version | head -n 1
