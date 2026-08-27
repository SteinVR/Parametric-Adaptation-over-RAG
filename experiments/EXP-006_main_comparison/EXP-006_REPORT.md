# EXP-006: Main Comparison

**Date:** 2026-08-27

## Scope

- Systems: S1, S2+R, S3+R, S7 (post-hoc from EXP-010), S2, S3, S3-legacy (D2L).
- S6 (Naive RAG) is intentionally excluded from EXP-006 outputs.

## Table 1: Unified System Metrics

| System | Class | Q_main | S_det | S_asst | G | TTFT median (ms) | Generation median (ms) | Recorded peak VRAM (MB) | Offline cost (s) |
|--------|-------|--------|-------|--------|---|------------------|---------------------|----------------------|------------------|
| S1 | Headline | 0.6425 | 0.6014 | 0.7385 | 0.5667 | 334.8 | 479.3 | 5200.5 | 0.0 |
| S2+R | Headline | 0.6689 ± 0.0137 | 0.6479 ± 0.0150 | 0.7179 ± 0.0178 | 0.5667 | 318.5 ± 0.2 | 492.0 ± 2.0 | 3069.3 ± 4.7 | 1205.5 ± 29.7 |
| S3+R | Headline | 0.6671 ± 0.0229 | 0.5991 ± 0.0156 | 0.8256 ± 0.0622 | 0.5667 | 315.8 ± 0.8 | 525.3 ± 17.6 | 3069.3 ± 4.7 | 581.4 ± 0.7 |
| S7 | Post-hoc | 0.7045 ± 0.0345 | 0.6790 ± 0.0481 | 0.7641 ± 0.0178 | 0.5667 | 334.5 ± 1.2 | 527.2 ± 19.0 | 3069.3 ± 4.7 | N/A |
| S2 | Control | 0.2630 ± 0.0046 | 0.2703 | 0.2462 ± 0.0154 | N/A | 50.8 ± 0.3 | 257.1 ± 33.7 | 3066.6 ± 9.4 | 87.9 ± 1.0 |
| S3 | Control | 0.1854 ± 0.0027 | 0.1351 | 0.3026 ± 0.0089 | N/A | 57.8 ± 0.4 | 195.2 ± 1.8 | 3077.4 ± 4.7 | 581.4 ± 0.7 |
| S3-legacy | Control | 0.2100 | 0.1351 | 0.3846 | N/A | 55.8 | 179.4 | 3072.0 | 3932.3 |

## Table 2: Per-Type Score (S_det for deterministic, S_asst for free_text)

| Answer type | S1 | S2+R | S3+R | S7 | S2 | S3 | S3-legacy |
|---|---|---|---|---|---|---|---|
| boolean | 0.8333 | 0.8889 | 0.8333 | 0.8889 | 0.7500 | 0.3333 | 0.3333 |
| number | 0.7143 | 0.7143 | 0.7143 | 0.8095 | 0.1429 | 0.0000 | 0.0000 |
| name | 0.5000 | 0.6250 | 0.5833 | 0.7083 | 0.0000 | 0.1250 | 0.1250 |
| names | 0.4500 | 0.2614 | 0.3000 | 0.2244 | 0.0000 | 0.0000 | 0.0000 |
| date | 0.2000 | 0.4000 | 0.2000 | 0.4000 | 0.0000 | 0.0000 | 0.0000 |
| free_text | 0.7385 | 0.7179 | 0.8256 | 0.7641 | 0.2462 | 0.3026 | 0.3846 |

## Table 3: Single-Doc vs Multi-Doc Q_main

| System | Single-doc (n=41) | Multi-doc (n=9) | Δ (multi - single) |
|---|---:|---:|---:|
| S1 | 0.7117 ± 0.0000 | 0.2787 ± 0.0000 | -0.4330 |
| S2+R | 0.7098 ± 0.0066 | 0.3846 ± 0.0159 | -0.3252 |
| S3+R | 0.7380 ± 0.0261 | 0.2787 ± 0.0000 | -0.4593 |
| S7 | 0.7347 ± 0.0223 | 0.4629 ± 0.0889 | -0.2718 |
| S2 | 0.2456 ± 0.0058 | 0.3025 ± 0.0346 | +0.0569 |
| S3 | 0.2090 ± 0.0029 | 0.1200 ± 0.0000 | -0.0890 |
| S3-legacy | 0.2307 ± 0.0000 | 0.1800 ± 0.0000 | -0.0507 |

## Key Deltas

- **S2+R_vs_S1**: Q_main=+0.0265, S_det=+0.0466, S_asst=-0.0205
- **S3+R_vs_S1**: Q_main=+0.0246, S_det=-0.0023, S_asst=+0.0872
- **S7_vs_S1**: Q_main=+0.0620, S_det=+0.0776, S_asst=+0.0256
- **S2+R_vs_S3+R**: Q_main=+0.0019, S_det=+0.0488, S_asst=-0.1077
- **S7_vs_S2+R**: Q_main=+0.0356, S_det=+0.0310, S_asst=+0.0462
- **S7_vs_S3+R**: Q_main=+0.0374, S_det=+0.0799, S_asst=-0.0615
- **S2+R_vs_S2**: Q_main=+0.4059, S_det=+0.3777, S_asst=+0.4718
- **S3+R_vs_S3**: Q_main=+0.4817, S_det=+0.4640, S_asst=+0.5231
- **S3_vs_S3-legacy**: Q_main=-0.0246, S_det=+0.0000, S_asst=-0.0821
- **S3+R_vs_S3-legacy**: Q_main=+0.4571, S_det=+0.4640, S_asst=+0.4410

## Interpretation

- S7 (adapter merge from EXP-010) is strongest by Q_main: 0.7045.
- S2+R vs S3+R remains trade-off shaped: ΔQ_main=+0.0019, ΔS_det=+0.0488, ΔS_asst=-0.1077.
- Retrieval contribution stays dominant: S2→S2+R +0.4059, S3→S3+R +0.4817.
- Legacy D2L anchor shows stronger no-retrieval baseline than S3: S3-S3-legacy -0.0246, while retrieval-adapted S3+R regains a large margin over legacy (+0.4571, Q_main).
- Latency excludes retrieval. S1's saved 5200.5 MB process peak includes retrieval/reranking, while the other peak values are generator-only and reset immediately before generation.
- S7 has no directly comparable offline-cost entry because it inherits the training effort of both source adapters.

## Artifacts

- `results/EXP-006/main_results.csv`
- `results/EXP-006/per_type_breakdown.csv`
- `results/EXP-006/single_vs_multi_doc.csv`
- `results/EXP-006/deltas.json`
- `results/EXP-006/gradient_plot.png`
