# EXP-006: Main Comparison

**Date:** 2026-03-31

## Table 1: All Systems

| System | Class | Q_main | S_det | S_asst | G |
|--------|-------|--------|-------|--------|---|
| S1 | Headline | 0.6425 | 0.6014 | 0.7385 | 0.5667 |
| S2+R | Headline | 0.6689 ± 0.014 | 0.6479 | 0.7179 | N/A |
| S3+R | Headline | 0.6671 ± 0.023 | 0.5991 | 0.8256 | N/A |
| S2 | Control | 0.2630 ± 0.005 | 0.2703 | 0.2462 | N/A |
| S3 | Control | 0.1854 ± 0.003 | 0.1351 | 0.3026 | N/A |
| S6 | Ablation | 0.6335 | 0.6149 | 0.6769 | 0.4891 |

## Key Deltas

- **S2+R_vs_S1:** Q_main=+0.0265, S_det=+0.0466, S_asst=-0.0205
- **S3+R_vs_S1:** Q_main=+0.0246, S_det=-0.0023, S_asst=+0.0872
- **S2+R_vs_S3+R:** Q_main=+0.0019, S_det=+0.0488, S_asst=-0.1077
- **S2+R_vs_S2:** Q_main=+0.4059, S_det=+0.3777, S_asst=+0.4718
- **S3+R_vs_S3:** Q_main=+0.4817, S_det=+0.4640, S_asst=+0.5231
- **S1_vs_S6:** Q_main=+0.0090, S_det=-0.0135, S_asst=+0.0615

## Hypothesis Interpretation

- **H1** (adapter > RAG): **CONFIRMED.** S2+R (+0.027) and S3+R (+0.025) both beat S1.
- **H2** (RAFT > CLM): **AMBIGUOUS.** Δ Q_main=+0.0019. Trade-off: S2+R wins S_det, S3+R wins S_asst.
- **H3** (pure parametric << S1): **CONFIRMED.** S2=0.263, S3=0.185 vs S1=0.643.
- **H4** (S1 dominates deterministic): **PARTIAL.** S2+R beats S1 on S_det (0.648 vs 0.601).
- **H5** (quantifying limits valid): **CONFIRMED.**

## S6 Ablation

- Δ(S1, S6): Q_main=+0.0090. Full pipeline gives modest quality gain.
- Grounding: S1=0.5667 vs S6=0.4891. Pipeline adds +0.078 page precision.
