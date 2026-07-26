# Experiment Report: EXP-004 - S3 Doc-to-LoRA Monolithic

**Date:** 2026-03-30
**Status:** Completed with methodology deviation

## 1. Goal

- Generate 8 per-document Doc-to-LoRA adapters, validate each on its own train-document deterministic subset, merge them by simple average, and evaluate the monolithic adapter without retrieval.

## 2. Validation

- Corpus documents: 8
- Goldset references: 200
- S2-train questions: 150
- Eval questions: 50
- Checkpoint status: present

## Spec Deviation Warning

- This run used chunk-level adapter generation with an internal merge back into one per-document adapter.
- That introduces an extra merge level before the frozen 8-adapter monolithic merge, so the result is not a strict EXP-004 implementation.
- Treat all metrics in this report as engineering diagnostics for the memory workaround, not as the official EXP-004 result.

## 3. Per-Document Packaging

| Doc | Pages | Words | Ctx toks | Chunks | Chunked | Gen sec | Peak VRAM MB | Peak RSS MB | Adapter bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 23 | 7271 | 11900 | 12 | yes | 452.0 | 2131.1 | 3601.2 | 9592301 |
| 2 | 26 | 13351 | 20091 | 20 | yes | 674.4 | 2131.1 | 3747.3 | 9592301 |
| 3 | 25 | 11541 | 15903 | 16 | yes | 566.2 | 2131.1 | 3715.0 | 9592301 |
| 4 | 23 | 10201 | 14613 | 15 | yes | 531.4 | 2131.1 | 3705.0 | 9592301 |
| 5 | 21 | 7297 | 11489 | 12 | yes | 446.7 | 2131.1 | 3679.0 | 9592301 |
| 6 | 24 | 6675 | 11746 | 12 | yes | 448.9 | 2131.1 | 3668.9 | 9592301 |
| 7 | 19 | 8832 | 12034 | 12 | yes | 450.6 | 2131.1 | 3679.9 | 9592301 |
| 8 | 15 | 6198 | 8410 | 9 | yes | 362.0 | 2131.1 | 3657.7 | 9592301 |

## 4. Sanity Check

| Doc | Train refs | Deterministic refs | S_det | Malformed |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | 12 | 0.1667 | 2 |
| 2 | 13 | 13 | 0.3077 | 2 |
| 3 | 12 | 12 | 0.2500 | 1 |
| 4 | 9 | 9 | 0.0000 | 2 |
| 5 | 12 | 12 | 0.2500 | 1 |
| 6 | 12 | 12 | 0.1667 | 3 |
| 7 | 11 | 11 | 0.0909 | 0 |
| 8 | 8 | 8 | 0.0000 | 2 |

## 5. Merge Summary

- Merge seconds: 0.0
- Source adapters: 8
- Output dir: `models/d2l/monolithic`

## 6. Eval

| Metric | Value |
| --- | ---: |
| Q_main | 0.2100 |
| S_det | 0.1351 |
| S_asst | 0.3846 |
| Grounding F_beta | N/A |

- Eval questions scored: 50
- Free-text judge calls completed: 13

## 7. System Metrics

- Peak infer VRAM MB: 3071.98193359375
- TTFT median ms: 55.757019501470495
- Latency median ms: 179.3751754994446

## 8. Breakdown by Answer Type

| Answer type | Count | Metric | Score | Malformed |
| --- | ---: | --- | ---: | ---: |
| number | 7 | S_det | 0.0000 | 0.4286 |
| free_text | 13 | S_asst | 0.3846 | 0.0000 |
| name | 8 | S_det | 0.1250 | 0.0000 |
| boolean | 12 | S_det | 0.3333 | 0.0000 |
| date | 5 | S_det | 0.0000 | 1.0000 |
| names | 5 | S_det | 0.0000 | 0.0000 |
| _unanswerable | 3 | S_det | 0.3333 | 0.0000 |

## 9. Merge Viability Assessment

- Technical viability: yes. All 8 adapters generated, loaded, merged, and the monolithic adapter completed the full 50-question eval.
- Quality viability: weak. Mean per-doc sanity S_det was 0.1540, and merged S3 reached Q_main=0.2100, S_det=0.1351, S_asst=0.3846.
- Interpretation: the frozen simple-average merge is operationally feasible but not competitive as a standalone no-retrieval system on this benchmark. Treat this as a negative control and carry the merged adapter forward as the fixed S3 artifact for downstream comparison.

## 10. Ablation: ΔW-Average Merge

- Merge method: ΔW-avg + SVD re-decomposition (rank=8)
- Mean explained variance: 0.9974
- Merge seconds: 124.9

| Metric | Factor-wise | ΔW-avg | Delta |
| --- | ---: | ---: | ---: |
| q_main | 0.2100 | 0.2100 | +0.0000 |
| s_det | 0.1351 | 0.1351 | +0.0000 |
| s_asst | 0.3846 | 0.3846 | +0.0000 |

## 11. Artifacts

- Generation records: `results/EXP-004/document_generation.json`
- Sanity results: `results/EXP-004/doc_sanity.json`
- Merge summary: `results/EXP-004/merge_summary.json`
- Validation summary: `results/EXP-004/validation.json`
- Predictions: `results/EXP-004/predictions.json`
- Eval outputs: `results/EXP-004`
- Adapters: `models/d2l`
