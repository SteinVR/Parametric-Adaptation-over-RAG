# Experiment Report: EXP-010 — Adapter Merge (CLM + RAFT)

**Date:** 2026-03-31
**Status:** Completed

## 1. Goal

Test whether a linear merge of CLM and RAFT adapters combines their strengths
(CLM's S_asst advantage + RAFT's S_det advantage) without retraining.

## 2. Setup

- Merge: α * CLM + (1−α) * RAFT, α=0.5
- Seeds: 42, 123, 777 (same-seed matching)
- Backbone: `google/gemma-2-2b-it` + merged adapter
- Retrieval: S1 pipeline (Qdrant hybrid + reranker + evidence compression)
- No training — linear interpolation of existing adapter weights

## 3. Results

- Q_main: 0.7045 ± 0.0345
- S_det: 0.6790 ± 0.0481
- S_asst: 0.7641 ± 0.0178
- G (F_β=2.5): 0.5667 ± 0.0000

## 4. Per-Seed Summary

| Seed | Q_main | S_det | S_asst | G | Peak infer VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7338 | 0.7252 | 0.7538 | 0.5667 | 3063.9 |
| 123 | 0.6665 | 0.6291 | 0.7538 | 0.5667 | 3072.0 |
| 777 | 0.7132 | 0.6826 | 0.7846 | 0.5667 | 3072.0 |

## 5. Deltas

### vs S1 (RAG, no adapter)

- Q_main: +0.0620
- S_det: +0.0776
- S_asst: +0.0256

### vs S2+R (RAFT + retrieval)

- Q_main: +0.0356
- S_det: +0.0310
- S_asst: +0.0462

### vs S3+R (CLM + retrieval)

- Q_main: +0.0374
- S_det: +0.0799
- S_asst: -0.0615

## 6. Interpretation

TODO: Fill after results.

## 7. Artifacts

- Aggregate summary: `/home/xeliaray/Projects/Term-Paper/results/EXP-010/alpha_0.5/aggregate_summary.json`
- Seed outputs: `/home/xeliaray/Projects/Term-Paper/results/EXP-010/alpha_0.5`
