# SPEC: EXP-007 — Error Analysis + Cost/Quality/Grounding Trade-off

**Systems:** All (S1, S2+R, S3+R, S2, S3, S4-doc, S4-cluster; S6 if triggered) | **Class:** Analysis | **Wave:** 4 | **Depends on:** EXP-006 | **Blocks:** Nothing (terminal after S6 trigger is resolved)

## Goal

Consolidate results from all prior experiments into final thesis tables and figures. Error analysis on system failures. Cost/quality/grounding trade-off analysis.

This spec has two modes:
- **Default path:** if EXP-008 is skipped, finalize from EXP-002..006 artifacts.
- **Ablation path:** if EXP-008 runs, refresh the consolidated tables/figures after EXP-008 completes so the final thesis package includes S6.

No fresh inference — reuse existing artifacts only.

## Analysis Steps

1. **Consolidate mandatory results** from EXP-002..006. S2/S2+R: use mean ± std across 3 seeds.
2. **Error categorization:** label failures by type (see protocol below)
3. **Cost/quality scatter:** offline packaging cost (x) vs Q_main (y) per system
4. **Per-type breakdown:** S_det per answer_type per system (heatmap)
5. **Grounding analysis:** G per system (S1, S2+R, S3+R, S6 if triggered); latency vs G scatter
6. **If EXP-008 ran:** re-open `consolidated_results.csv`, append S6 outputs, and refresh every figure/table that depends on the full system set before marking this spec complete.

## Error Analysis Protocol

Hypotheses for manual inspection (verify each manually — not deterministic labels):
- Questions ALL systems got wrong → hypothesis: ambiguous, too hard, or goldset issue
- Questions ONLY S1 got right → hypothesis: retrieval-critical; parametric systems lack this info
- Questions ONLY S2+R got right → hypothesis: supervised adapter gave an edge; check if answer was in training set
- S4-doc failures on multi-doc questions → hypothesis: routing limitation; confirm routed doc was wrong
- S3 vs per-doc adapter on same question → hypothesis: merge degradation
- S1 vs S6 (if triggered) → hypothesis: full pipeline (hybrid+rerank+compression+chunk topology) accounts for Δ vs naive dense RAG

Manually inspect top-5 worst failures per headline system (S1, S2+R, S3+R). Document: question text, expected answer, system answer, likely failure cause.

## Key Outputs

| Output | Content |
|--------|---------|
| Table A | Final comparison: headline systems (S1, S2+R, S3+R) × all metrics |
| Table B | Controls (S2, S3, S4-doc, S4-cluster) — parametric limits |
| Table C | Per answer_type S_det heatmap (all systems) |
| Table D | Cost/quality/grounding trade-off: packaging cost, Q_main, G, latency |
| `error_analysis.md` | Categorized failures, top-5 per headline system |

## Output

- `results/EXP-007/consolidated_results.csv`
- `results/EXP-007/error_analysis.md`
- `results/figures/main_results_table.png`
- `results/figures/cost_quality_scatter.png`
- `results/figures/per_type_heatmap.png`
- `results/figures/merge_route_gradient.png`
- `experiments/EXP-007/REPORT.md`

## Definition of Done

- [ ] `consolidated_results.csv` has all mandatory systems × all metrics; if EXP-008 ran, S6 is appended before final handoff
- [ ] Error analysis: top-5 worst failures per headline system documented in `error_analysis.md`
- [ ] Error hypotheses from spec checked (all wrong, only S1 right, only S2+R right, etc.)
- [ ] Cost/quality/grounding scatter generated (`cost_quality_scatter.png`)
- [ ] Per-type heatmap generated (`per_type_heatmap.png`)
- [ ] Merge↔Route gradient figure generated (`merge_route_gradient.png`)
- [ ] Tables A–D present in REPORT.md and refreshed after EXP-008 if S6 was triggered
- [ ] All results committed to git
- [ ] `experiments/EXP-007/REPORT.md` written with final conclusions and mandatory caveats
