# SPEC: EXP-001 — Data Audit + Goldset Merge + Split Freeze

**Status:** Complete | **Wave:** 1 | **Depends on:** Nothing | **Blocks:** EXP-002..008

## Goal
Validate corpus and goldset, merge 2 QA batches, create frozen 150 train / 50 eval split.

## Frozen Artifacts
- `data/corpus/doc{1-8}_*.pdf` — 8 documents
- `data/goldset/goldset.benchmark.json` — merged 200 QA
- `data/splits/split_v1.json` — 150 S2-train / 50 eval, stratified, near-duplicates grouped

## Legacy (not used by downstream experiments)
- `data/manifests/corpus_manifest.csv` — 65-doc manifest from original audit, superseded by 8-doc corpus

## Key Results
- 200 QA, 8 docs, 176 pages, ~141K tokens
- 1 near-duplicate pair (grouped in same split)
- All answer types and difficulty levels represented in both splits
- Each document fits Doc-to-LoRA single pass
