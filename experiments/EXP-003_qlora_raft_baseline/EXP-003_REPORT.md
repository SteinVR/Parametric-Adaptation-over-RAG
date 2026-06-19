# Experiment Report: EXP-003 - S2 QLoRA RAFT-style Baseline

**Date:** 2026-03-29
**Status:** Completed

## 1. Hypothesis

- QLoRA with frozen RAFT-style open-book supervision should establish a supervised parametric baseline over the 8-document corpus.

## 2. Setup

- Seeds: 42, 123, 777
- RAFT dataset: `data/processed/raft_train.jsonl`
- Backbone: `google/gemma-2-2b-it`

## 3. Results

- Q_main: 0.6689 ± 0.0137
- S_det: 0.6479 ± 0.0150
- S_asst: 0.7179 ± 0.0178
- G_f_beta: 0.5667 ± 0.0000

## 4. Per-seed Summary

| Seed | Q_main | S_det | S_asst | G_f_beta | Train sec | Peak train VRAM MB | Peak infer VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6727 | 0.6577 | 0.7077 | 0.5667 | 1239.8 | 6827.9 | 3063.9 |
| 123 | 0.6537 | 0.6306 | 0.7077 | 0.5667 | 1188.3 | 6827.9 | 3072.0 |
| 777 | 0.6804 | 0.6555 | 0.7385 | 0.5667 | 1188.5 | 6827.9 | 3072.0 |

## 5. Delta vs S1

- Q_main delta: 0.0265
- S_det delta: 0.0466
- S_asst delta: -0.0205
- G_f_beta delta: 0.0000

## 6. Breakdown By Answer Type

### _unanswerable

- count: 3.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.3333 ± 0.0000

### boolean

- count: 12.0000 ± 0.0000
- grounding_f_beta_mean: 0.4963 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.8889 ± 0.0481

### date

- count: 5.0000 ± 0.0000
- grounding_f_beta_mean: 0.2693 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.4000 ± 0.0000

### free_text

- count: 13.0000 ± 0.0000
- grounding_f_beta_mean: 0.7092 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_asst_mean: 0.7179 ± 0.0178

### name

- count: 8.0000 ± 0.0000
- grounding_f_beta_mean: 0.5406 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.6250 ± 0.0000

### names

- count: 5.0000 ± 0.0000
- grounding_f_beta_mean: 0.4893 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.2614 ± 0.0091

### number

- count: 7.0000 ± 0.0000
- grounding_f_beta_mean: 0.7202 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.7143 ± 0.0000

## 7. Artifacts

- Aggregate summary: `results/EXP-003/aggregate_summary.json`
- Seed outputs: `results/EXP-003`
- Adapters: `models/qlora`

## 8. Notes

- This report is generated from saved run artifacts.
- The implementation follows the experiment spec; this report summarizes produced artifacts only.
