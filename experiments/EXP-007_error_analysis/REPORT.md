# EXP-007: Error Analysis + Trade-off

**Date:** 2026-03-31

## Scope

- Consolidation includes S1, S2+R, S3+R, S7, S2, S3.
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

## Error Analysis

See `/home/xeliaray/Projects/Term-Paper/results/EXP-007/error_analysis.md`.

## Deep Analysis

See `/home/xeliaray/Projects/Term-Paper/results/EXP-007/deep_analysis.md`.

## Figures

- `/home/xeliaray/Projects/Term-Paper/results/figures/main_results_table.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/cost_quality_scatter.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/per_type_heatmap.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/latency_grounding_scatter.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/error_overlap_heatmap.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/pairwise_win_heatmap.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/difficulty_profile.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/judge_criteria_profile.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/seed_stability.png`
- `/home/xeliaray/Projects/Term-Paper/results/figures/pareto_frontier.png`

## Mandatory Caveats

- Bounded to this corpus, benchmark split, backbone, and hardware setup.
- CLM-based systems use supervision-free document exposure, not QA supervision.
- S7 is a post-hoc adapter-merge result; it is not a separately retrained system.
