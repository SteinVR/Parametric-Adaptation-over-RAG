# SPEC: EXP-007 — Error Analysis + Cost/Quality/Grounding Trade-off

**Systems:** S1, S2+R, S3+R, S7, S2, S3, S3-legacy (D2L) | **Class:** Analysis | **Wave:** 3 | **Depends on:** EXP-006 | **Blocks:** None

## Goal

Consolidate the final thesis tables and figures from completed experiments and perform error analysis.

This spec owns:
- practical winner call between S2+R and S3+R,
- explicit post-hoc impact block for S7,
- mandatory legacy control visibility for S3-legacy (D2L).

No fresh inference — reuse existing artifacts only.

## Analysis Steps

1. **Consolidate results** from EXP-002..006 (+ S7 from EXP-010 and D2L legacy row from EXP-004 report).
2. **Error categorization:** label failures by type.
3. **Cost/quality scatter:** offline packaging cost (x) vs Q_main (y) per system.
4. **Per-type breakdown:** score profile by answer_type per system.
5. **Grounding analysis:** G per retrieval-aware system (S1, S2+R, S3+R, S7).
6. **Practical winner call:** compare S2+R vs S3+R on Q_main, G, latency, and offline cost.
7. **Post-hoc block:** quantify S7 deltas vs S1/S2+R/S3+R and state non-retraining caveat.

## Error Analysis Protocol

Hypotheses for manual inspection:
- Questions all systems got wrong → ambiguous/hard/goldset issue
- Questions only S1 got right → retrieval-critical
- Questions only S2+R got right → supervised RAFT edge
- Questions only S3+R got right → CLM exposure edge
- Questions only S7 got right → merge-level complementarity effect

Manually inspect top-5 worst failures per headline family (S1, S2+R, S3+R, S7).

## Key Outputs

| Output | Content |
|--------|---------|
| Table A | Final comparison: headline + post-hoc (S1, S2+R, S3+R, S7) |
| Table B | Controls table including S2, S3, S3-legacy (D2L) |
| Table C | Per answer_type score heatmap (all systems) |
| Table D | Cost/quality/grounding trade-off: packaging cost, Q_main, G, latency |
| Callout | Practical winner call between S2+R and S3+R (or no single winner) |
| Block | S7 post-hoc impact summary with caveats |
| `error_analysis.md` | Categorized failures, top-5 per headline family |

## Output

- `results/EXP-007/consolidated_results.csv`
- `results/EXP-007/error_analysis.md`
- `results/figures/main_results_table.png`
- `results/figures/cost_quality_scatter.png`
- `results/figures/per_type_heatmap.png`
- `experiments/EXP-007_error_analysis/REPORT.md`

## Definition of Done

- [x] `consolidated_results.csv` includes S1, S2+R, S3+R, S7, S2, S3, S3-legacy
- [x] Legacy control row S3-legacy (D2L) present in control tables/narrative
- [x] Error analysis documented in `error_analysis.md`
- [x] Cost/quality/grounding scatter generated
- [x] Per-type heatmap generated
- [x] Tables A–D present in REPORT.md
- [x] Practical winner call between S2+R and S3+R explicitly stated
- [x] S7 post-hoc impact block included with non-retraining caveat
- [x] Results committed to git
- [x] `experiments/EXP-007_error_analysis/REPORT.md` written
