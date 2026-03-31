# Experiment Report: EXP-003b - S2 QLoRA Closed-Book (Axis 1)

**Date:** 2026-03-29
**Status:** Completed

## 1. Hypothesis

- S2 closed-book tests whether supervised QA-pair training injects document knowledge into a 2B model.
- Expected to underperform S1 (RAG) — model cannot memorize 176 pages from 150 QA pairs.

## 2. Setup

- Seeds: 42, 123, 777
- Dataset: `/home/xeliaray/Projects/Term-Paper/data/processed/closed_book_train.jsonl`
- Backbone: `google/gemma-2-2b-it`
- **No retrieval at inference.**

## 3. Results

- Q_main: 0.2630 ± 0.0046
- S_det: 0.2703 ± 0.0000
- S_asst: 0.2462 ± 0.0154

## 4. Per-seed Summary

| Seed | Q_main | S_det | S_asst | Train sec | Peak train VRAM MB | Peak infer VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.2630 | 0.2703 | 0.2462 | 86.9 | 4724.7 | 3055.7 |
| 123 | 0.2584 | 0.2703 | 0.2308 | 87.7 | 4724.7 | 3072.0 |
| 777 | 0.2677 | 0.2703 | 0.2615 | 89.0 | 4724.7 | 3072.0 |

## 5. Delta vs S1 (Axis 1)

- Q_main delta: -0.3794
- S_det delta: -0.3311
- S_asst delta: -0.4923

## 6. Delta vs S2+R (Axis 2: retrieval contribution)

- Q_main delta: -0.4059
- S_det delta: -0.3777
- S_asst delta: -0.4718

Negative delta = S2 closed-book worse than S2+R → retrieval helps.

## 7. Breakdown By Answer Type

### _unanswerable

- count: 3.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### boolean

- count: 12.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.7500 ± 0.0000

### date

- count: 5.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### free_text

- count: 13.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_asst_mean: 0.2462 ± 0.0154

### name

- count: 8.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### names

- count: 5.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### number

- count: 7.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.1429 ± 0.0000

## 8. Artifacts

- Aggregate summary: `/home/xeliaray/Projects/Term-Paper/results/EXP-003b/aggregate_summary.json`
- Seed outputs: `/home/xeliaray/Projects/Term-Paper/results/EXP-003b`
- Adapters: `/home/xeliaray/Projects/Term-Paper/models/qlora_closed`
