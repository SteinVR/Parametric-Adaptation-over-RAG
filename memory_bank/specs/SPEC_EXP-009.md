# SPEC: EXP-009 — Refresh Final Thesis Package with S6 (Conditional)

**Systems:** S6 refresh into final package | **Class:** Analysis | **Wave:** 4 (conditional) | **Depends on:** EXP-007, EXP-008 | **Blocks:** Nothing

## Goal

If EXP-008 runs, refresh the final thesis package so shared tables and figures include S6 without reopening Wave 4. This resolves the sequential-wave constraint: EXP-007 completes the mandatory path, then EXP-009 performs the post-S6 refresh.

No fresh inference — reuse artifacts from EXP-007 and EXP-008 only.

## Analysis Steps

1. **Load mandatory package** from EXP-007 (`consolidated_results.csv`, figures, REPORT tables)
2. **Load S6 outputs** from EXP-008
3. **Append S6** to the consolidated results artifact with all applicable metrics
4. **Refresh shared figures/tables** that depend on the full system set:
   - final comparison table
   - cost/quality/grounding trade-off figure/table
   - per-type heatmap
5. **Add S1 vs S6 interpretation** and explicit caveat that S6 is a conditional e2e ablation, not part of the mandatory headline comparison

## Output

- Updated `results/EXP-007/consolidated_results.csv`
- Updated `results/figures/main_results_table.png`
- Updated `results/figures/cost_quality_scatter.png`
- Updated `results/figures/per_type_heatmap.png`
- `experiments/EXP-009/REPORT.md`

## Definition of Done

- [ ] EXP-008 trigger verified as executed; if skipped, document that EXP-009 is skipped too
- [ ] `results/EXP-007/consolidated_results.csv` refreshed to include S6
- [ ] Shared figures/tables that depend on the full system set regenerated with S6
- [ ] S1 vs S6 interpretation added with mandatory conditional-ablation caveat
- [ ] `experiments/EXP-009/REPORT.md` written
- [ ] All refreshed results committed to git
