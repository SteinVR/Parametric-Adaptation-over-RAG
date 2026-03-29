# SPEC: EXP-008 — Locked Test + Error Analysis

**Systems:** All (S1, S2, S3, S4-doc, S4-cluster, S5, S6 if triggered) | **Wave:** 5 | **Depends on:** EXP-007 | **Incorporates:** EXP-009 (if it ran) | **Blocks:** Nothing (terminal)

## Goal

Final consolidated evaluation on 50 eval questions (same set used in EXP-002..007). Error analysis. Publication-ready tables and figures.

## Pipeline

1. **Consolidate** existing results from EXP-002..007 (and EXP-009 if it ran) — no fresh inference, reuse artifacts. S2 results use the mean across 3 seeds.
2. Score: Q_main, S_det, S_asst, G (for S1, S2, S5, S6 if triggered)
3. Assemble final comparison tables
4. Error analysis on test failures

## Error Analysis Protocol

Hypotheses for manual inspection (not deterministic labels — verify each manually):
- Questions ALL systems got wrong → **hypothesis:** ambiguous, too hard, or goldset issue
- Questions ONLY S1 got right → **hypothesis:** retrieval-critical, parametric systems lack this info
- Questions ONLY S2 got right → **hypothesis:** supervision gave an edge. Check if answer was in training set.
- S4-doc failures on multi-doc questions → **hypothesis:** routing limitation. Confirm routed doc was wrong.
- S3 vs per-doc adapter on same question → **hypothesis:** merge degradation
- S1 vs S6 (if triggered) → **hypothesis:** full pipeline (hybrid+rerank+compression+chunk topology) accounts for Δ vs naive dense RAG

Manually inspect top-5 worst failures per system. Document with: question text, expected answer, system answer, likely failure cause.

## Output

- `results/EXP-008/test_results.csv`
- `results/EXP-008/cross_system_comparison.csv`
- `results/EXP-008/error_analysis.md`
- `experiments/EXP-008/REPORT.md`
- Publication figures in `results/figures/`:
  - `main_results_table.png`
  - `merge_route_gradient.png`
  - `per_type_heatmap.png`
  - `routing_confusion_matrix.png`

## Definition of Done

- [ ] `test_results.csv` consolidates all system results (no fresh inference)
- [ ] `cross_system_comparison.csv` with all systems × all metrics
- [ ] Error analysis: top-5 worst failures per system documented in `error_analysis.md`
- [ ] Error hypotheses from spec checked (all wrong, only S1 right, only S2 right, etc.)
- [ ] 4 publication figures generated in `results/figures/`
- [ ] All results committed to git
- [ ] `experiments/EXP-008/REPORT.md` written with final conclusions and mandatory caveats
