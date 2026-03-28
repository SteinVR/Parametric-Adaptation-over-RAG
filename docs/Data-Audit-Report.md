# EXP-001: Data Audit Report

**Date:** 2026-03-26
**Status:** Complete

## Corpus

- Documents: 65 PDFs
- Total pages: 1,594
- Total tokens (est): ~1,275,200 (at ~800 tokens/page)
- Pages per doc: min=2, median=9, max=537
- Domain: DIFC legal (statutes, case law, cross-case)

## Goldset

- QA pairs: 150 (100 warmup + 50 full)
- Unique docs referenced in gold_retrieval: 65
- All gold doc_ids present in corpus: YES
- Schema validation: all 150 entries have required fields

### Answer type distribution

| Type | Count |
|------|-------|
| free_text | 44 |
| boolean | 42 |
| number | 25 |
| name | 21 |
| names | 11 |
| date | 7 |

### Difficulty distribution

| Difficulty | Count |
|------------|-------|
| easy | 101 |
| medium | 43 |
| hard | 6 |

## Near-Duplicate Analysis

Found **6 near-duplicate pairs** (Jaccard > 0.8 on word sets).
All pairs are structurally similar questions about different cases/entities (e.g., "What is the Date of Issue of SCT 011/2025?" vs "...SCT 459/2024?").

These are **not true duplicates** — they ask the same question pattern about different documents, so answers differ. However, they could inflate apparent generalization if split across dev/test.

**Resolution:** grouped all pairs to land in the same split. No cross-split leakage.

## Split

| Split | Size | Answer types | Difficulty |
|-------|------|-------------|------------|
| dev | 120 | boolean:33, free_text:35, number:23, name:16, names:8, date:5 | easy:82, medium:34, hard:4 |
| test | 30 | boolean:9, free_text:9, name:5, names:3, number:2, date:2 | easy:19, medium:9, hard:2 |

- Stratified by (answer_type, difficulty)
- Near-duplicates grouped within same split
- Saved: `data/splits/split_v1.json`
- **FROZEN** — do not modify after this point

## Capacity Audit

| Metric | Value |
|--------|-------|
| Total corpus tokens (est) | ~1,275,200 |
| Doc-to-LoRA single-pass limit | ~32,000 |
| Ratio | **39.9x** over limit |
| Estimated segments (full corpus) | ~40 |
| Estimated segments per cluster (k=4) | ~10 |
| Individual docs fitting one pass | 55/65 |

**Monolithic single-pass: NOT feasible.**
Both S3 and S4 require chunked processing + adapter merge.

## Key Findings

1. **Data is clean and consistent.** All doc_ids cross-validate, schema is complete.
2. **Corpus is ~40x larger than one Doc-to-LoRA pass.** Chunking + merge is mandatory.
3. **55/65 individual documents fit** a single D2L pass. The remaining 10 large documents need sub-document segmentation.
4. **Near-duplicates handled.** 6 pairs grouped, no leakage.
5. **Sparse strata:** date (7 total, 2 in test) and hard (6 total, 2 in test) — results on these will be noisy.

## Decision

Proceed to EXP-002 (S1 Classical RAG baseline).
