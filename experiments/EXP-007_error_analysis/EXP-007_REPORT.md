# EXP-007: Error Analysis + Trade-off

**Date:** 2026-03-31

## Scope

- Consolidation includes S1, S2+R, S3+R, S7, S2, S3, S3-legacy (D2L from EXP-004).
- S6 (Naive RAG) intentionally excluded from this refresh.

## Practical Winner Call (S2+R vs S3+R)

| Metric | S2+R | S3+R | Winner |
|--------|------|------|--------|
| Q_main | 0.669 | 0.667 | Tie |
| S_det | 0.648 | 0.599 | S2+R |
| S_asst | 0.718 | 0.826 | S3+R |
| G | 0.567 | 0.567 | Tie |
| Offline cost (s) | 1205.5 | 581.4 | S3+R |

**Verdict:** No single practical winner

## EXP-010 Impact (S7)

- S7 reaches `Q_main=0.7045`, best among all included systems.
- Relative to S2+R: `ΔQ_main=+0.0356`, `ΔS_det=+0.0310`, `ΔS_asst=+0.0462`.

## Control Systems (with Legacy Anchor)

| System | Q_main | S_det | S_asst | Offline cost (s) |
|--------|--------|-------|--------|------------------|
| S2 | 0.263 | 0.270 | 0.246 | 87.9 |
| S3 | 0.185 | 0.135 | 0.303 | 581.4 |
| S3-legacy (D2L) | 0.210 | 0.135 | 0.385 | 3932.3 |

- ΔQ_main (S3 - S3-legacy): -0.0246.
- ΔQ_main (S3+R - S3-legacy): +0.4571.

## Error Analysis

See `results/EXP-007/error_analysis.md`.

## Deep Analysis

See `results/EXP-007/deep_analysis.md`.

## Figures

- `results/figures/main_results_table.png`
- `results/figures/cost_quality_scatter.png`
- `results/figures/per_type_heatmap.png`
- `results/figures/latency_grounding_scatter.png`
- `results/figures/error_overlap_heatmap.png`
- `results/figures/pairwise_win_heatmap.png`
- `results/figures/difficulty_profile.png`
- `results/figures/judge_criteria_profile.png`
- `results/figures/seed_stability.png`
- `results/figures/pareto_frontier.png`

## Mandatory Caveats

- Bounded to this corpus, benchmark split, backbone, and hardware setup.
- CLM-based systems use supervision-free document exposure, not QA supervision.
- S7 is a post-hoc adapter-merge result; it is not a separately retrained system.
