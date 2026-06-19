# Experiment Report: EXP-004b - S3+R CLM + Retrieval (Headline)

**Date:** 2026-03-30
**Status:** Completed

## 1. Goal

- Evaluate CLM adapter as retrieval-conditioned generator inside S1 RAG pipeline.
- Symmetric comparison with S2+R: same retrieval, same PEFT arch, different training signal.
- No new training — inference-only using CLM adapters from EXP-004.

## 2. Setup

- Seeds: 42, 123, 777
- Backbone: `google/gemma-2-2b-it` + CLM adapter
- Retrieval: S1 pipeline (Qdrant hybrid + reranker + evidence compression)
- Prompt: RAG template (retrieved context + question)

## 3. Results

- Q_main: 0.6671 ± 0.0229
- S_det: 0.5991 ± 0.0156
- S_asst: 0.8256 ± 0.0622

## 4. Per-Seed Summary

| Seed | Q_main | S_det | S_asst | G | Peak infer VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6514 | 0.5811 | 0.8154 | 0.5667 | 3063.9 |
| 123 | 0.6934 | 0.6081 | 0.8923 | 0.5667 | 3072.0 |
| 777 | 0.6564 | 0.6081 | 0.7692 | 0.5667 | 3072.0 |

## 5. Delta vs S1 (RAG baseline, no adapter)

- Q_main delta: +0.0246
- S_det delta: -0.0023
- S_asst delta: +0.0872

## 6. Delta vs S2+R (RAFT + retrieval)

- Q_main delta: -0.0019
- S_det delta: -0.0488
- S_asst delta: +0.1077

## 7. Delta vs S3 (CLM no retrieval — retrieval contribution)

- Q_main delta: +0.4817
- S_det delta: +0.4640
- S_asst delta: +0.5231

Positive delta = retrieval helps CLM system.

## 8. Breakdown By Answer Type

### _unanswerable

- count: 3.0000 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.0000 ± 0.0000

### boolean

- count: 12.0000 ± 0.0000
- grounding_f_beta_mean: 0.4963 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.8333 ± 0.0000

### date

- count: 5.0000 ± 0.0000
- grounding_f_beta_mean: 0.2693 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.2000 ± 0.0000

### free_text

- count: 13.0000 ± 0.0000
- grounding_f_beta_mean: 0.7092 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_asst_mean: 0.8256 ± 0.0622

### name

- count: 8.0000 ± 0.0000
- grounding_f_beta_mean: 0.5406 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.5833 ± 0.0722

### names

- count: 5.0000 ± 0.0000
- grounding_f_beta_mean: 0.4893 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.3000 ± 0.0000

### number

- count: 7.0000 ± 0.0000
- grounding_f_beta_mean: 0.7202 ± 0.0000
- malformed_rate: 0.0000 ± 0.0000
- s_det_mean: 0.7143 ± 0.0000

## 9. Artifacts

- Aggregate summary: `results/EXP-004b/aggregate_summary.json`
- Seed outputs: `results/EXP-004b`
- CLM adapters: `models/clm`
