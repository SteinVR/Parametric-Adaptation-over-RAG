# Experiment Report: EXP-004 - S3 CLM Continued Pretraining

**Date:** 2026-03-30
**Status:** Completed

## 1. Goal

- Train QLoRA adapter on corpus document text with causal LM loss (next-token prediction).
- Evaluate as pure parametric control (no retrieval).
- Compare with S1 (RAG), S2+R (RAFT), and S2 (closed-book).

## 2. Historical Note

- EXP-004 originally used Doc-to-LoRA (D2L) hypernetwork. D2L was non-viable:
  documents exceeded hypernetwork context, chunk workaround yielded Q_main=0.210.
- Architecture pivoted to CLM in v9.0. D2L results archived.

## 3. Setup

- Seeds: 42, 123, 777
- Backbone: `google/gemma-2-2b-it`
- Training data: 8 documents (~115K tokens), causal LM loss
- PEFT: QLoRA rank=32, alpha=32, target q_proj+v_proj
- **No retrieval at inference.**

## 4. Results

- Q_main: 0.1854 ± 0.0027
- S_det: 0.1351 ± 0.0000
- S_asst: 0.3026 ± 0.0089

## 5. Per-Seed Summary

| Seed | Q_main | S_det | S_asst | Train sec | Peak train VRAM MB | Peak infer VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.1823 | 0.1351 | 0.2923 | 580.6 | 6327.7 | 3072.0 |
| 123 | 0.1869 | 0.1351 | 0.3077 | 581.8 | 6327.7 | 3080.1 |
| 777 | 0.1869 | 0.1351 | 0.3077 | 581.9 | 6327.7 | 3080.1 |

## 6. Delta vs S1 (RAG baseline)

- Q_main delta: -0.4571
- S_det delta: -0.4662
- S_asst delta: -0.4359

## 7. Delta vs S2+R (RAFT + retrieval)

- Q_main delta: -0.4836
- S_det delta: -0.5128
- S_asst delta: -0.4154

## 8. Delta vs S2 (closed-book)

- Q_main delta: -0.0777
- S_det delta: -0.1351
- S_asst delta: +0.0564

Positive delta = CLM pretraining outperforms supervised closed-book on this metric.

## 9. Breakdown By Answer Type

### _unanswerable

- count: 3.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.3333 ± 0.0000

### boolean

- count: 12.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.3333 ± 0.0000

### date

- count: 5.0000 ± 0.0000
- malformed_rate: 1.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### free_text

- count: 13.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_asst_mean: 0.3026 ± 0.0089

### name

- count: 8.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.1250 ± 0.0000

### names

- count: 5.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### number

- count: 7.0000 ± 0.0000
- malformed_rate: 0.7143 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

## 10. Artifacts

- Aggregate summary: `/home/xeliaray/Projects/Term-Paper/results/EXP-004_clm/aggregate_summary.json`
- Seed outputs: `/home/xeliaray/Projects/Term-Paper/results/EXP-004_clm`
- Adapters: `/home/xeliaray/Projects/Term-Paper/models/clm`
