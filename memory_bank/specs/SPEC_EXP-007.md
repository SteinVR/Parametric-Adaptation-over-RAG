# SPEC: EXP-007 — Error Analysis + Cost/Quality/Grounding Trade-off

**Systems:** S1, S2+R, S3+R, S2, S3 | **Class:** Analysis | **Wave:** 3 | **Depends on:** EXP-006 | **Blocks:** EXP-009 (conditional refresh only)

## Goal

Consolidate mandatory-system results from EXP-002..006 into the default thesis tables and figures. Error analysis on system failures. Cost/quality/grounding trade-off analysis.

If EXP-008 later runs, EXP-009 owns the post-S6 refresh. EXP-007 itself must be completable without waiting for any later wave.

This spec also owns the **practical winner call** between S2+R and S3+R: declare the final best practical hybrid only as a reporting conclusion, never as a new independently validated system.

No fresh inference — reuse existing artifacts only.

## Analysis Steps

1. **Consolidate mandatory results** from EXP-002..006. S2/S2+R: use mean ± std across 3 seeds.
2. **Error categorization:** label failures by type (see protocol below)
3. **Cost/quality scatter:** offline packaging cost (x) vs Q_main (y) per system
4. **Per-type breakdown:** S_det per answer_type per system (heatmap)
5. **Grounding analysis:** G per system (S1, S2+R, S3+R); latency vs G scatter
6. **Practical winner call:** compare S2+R vs S3+R on Q_main, grounding, latency, and offline packaging cost; either name the final best practical hybrid or explicitly state that the trade-off is ambiguous.

## Error Analysis Protocol

Hypotheses for manual inspection (verify each manually — not deterministic labels):
- Questions ALL systems got wrong → hypothesis: ambiguous, too hard, or goldset issue
- Questions ONLY S1 got right → hypothesis: retrieval-critical; parametric systems lack this info
- Questions ONLY S2+R got right → hypothesis: supervised adapter gave an edge; check if answer was in training set
- Questions ONLY S3+R got right → hypothesis: CLM document exposure helped where RAFT didn't
- S1 vs S6 belongs to EXP-009 manual refresh scope if EXP-008 is triggered

Manually inspect top-5 worst failures per headline system (S1, S2+R, S3+R). Document: question text, expected answer, system answer, likely failure cause.

## Key Outputs

| Output | Content |
|--------|---------|
| Table A | Final comparison: headline systems (S1, S2+R, S3+R) × all metrics |
| Table B | Controls (S2, S3) — parametric limits |
| Table C | Per answer_type S_det heatmap (all systems) |
| Table D | Cost/quality/grounding trade-off: packaging cost, Q_main, G, latency |
| Callout | Final best practical hybrid: S2+R or S3+R, or "no single winner" |
| `error_analysis.md` | Categorized failures, top-5 per headline system |

## Output

- `results/EXP-007/consolidated_results.csv`
- `results/EXP-007/error_analysis.md`
- `results/figures/main_results_table.png`
- `results/figures/cost_quality_scatter.png`
- `results/figures/per_type_heatmap.png`
- `experiments/EXP-007/REPORT.md`

## Definition of Done

- [ ] `consolidated_results.csv` has all mandatory systems (S1, S2+R, S3+R, S2, S3) × all metrics
- [ ] Error analysis: top-5 worst failures per headline system documented in `error_analysis.md`
- [ ] Error hypotheses from spec checked (all wrong, only S1 right, only S2+R right, etc.)
- [ ] Cost/quality/grounding scatter generated (`cost_quality_scatter.png`)
- [ ] Per-type heatmap generated (`per_type_heatmap.png`)
- [ ] Tables A–D present in REPORT.md for the mandatory system set
- [ ] REPORT.md states the final best practical hybrid call between S2+R and S3+R, or explicitly states that no single winner exists
- [ ] All results committed to git
- [ ] `experiments/EXP-007/REPORT.md` written with final conclusions and mandatory caveats
