# EXP-007: Error Analysis + Trade-off

**Date:** 2026-03-31

## Practical Winner Call

| Metric | S2+R | S3+R | Winner |
|--------|------|------|--------|
| Q_main | 0.669 | 0.667 | Tie (Δ=0.002) |
| S_det | 0.648 | 0.599 | S2+R (+0.049) |
| S_asst | 0.718 | 0.826 | S3+R (+0.108) |
| G | 0.567 | 0.567 | Tie |
| Supervision | 150 QA pairs | None (doc text) | S3+R |
| Training time | ~320s | ~580s | S2+R |

**Verdict: No single practical winner.** Trade-off is genuine:
- **S2+R** for deterministic accuracy (structured answers, compliance).
- **S3+R** when no labeled QA data available or free-text quality is priority.
- Both viable on consumer hardware (RTX 4060 8GB).

## Error Analysis

See `results/EXP-007/error_analysis.md`.

## Figures

- `results/figures/cost_quality_scatter.png`
- `results/figures/per_type_heatmap.png`

## Mandatory Caveats

- Bounded to this corpus (8 DIFC legal docs), goldset (200 QA), backbone (Gemma-2-2b-it), hardware (RTX 4060 8GB).
- CLM uses only corpus text, no QA supervision.
- D2L hypernetwork was non-viable (archived, v9.0).
