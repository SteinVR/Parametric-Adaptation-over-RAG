# SPEC: EXP-006 — Main Cross-Paradigm Comparison

**Systems:** S1, S2, S3, S4-doc, S4-cluster | **Wave:** 4 | **Depends on:** EXP-002, EXP-003, EXP-004, EXP-005a, EXP-005b | **Blocks:** EXP-007

## Goal

Unified comparison of all systems. Produce the main results table.

**Evaluation scope:** ALL systems evaluated on the same 50 eval questions. S2 never saw these during training (trained on 150 train). No contamination for any system.

## Pipeline

1. **Collect** all system outputs from EXP-002..005b (already evaluated)
2. **Normalize** into common results format: one row per (system, question_id) with predicted_answer, metrics, timing
3. **Score** each system: Q_main, S_det, S_asst. G (F_β=2.5) for S1, S2 (retrieval-based). S3, S4-doc, S4-cluster: G = N/A.
4. **Breakdowns:** by answer_type (6 types: boolean, number, name, names, date, free_text), by difficulty (3 levels), by single/multi-doc, by unanswerable (flag, reported separately)
5. **Systems metrics table:** TTFT, end-to-end latency, peak VRAM, offline packaging cost
6. **Merge↔Route gradient plot:** x-axis = number of adapters (1, 4, 8), y-axis = Q_main for S3, S4-cluster, S4-doc
7. **Best single adapter selection for S5:** rank S2 and S3 only by eval Q_main → select top for S5. S4-doc/S4-cluster are routed multi-adapter systems and cannot be used as a single adapter in S5.

## Key Tables

| Table | Content |
|-------|---------|
| Table 1 | All systems × {Q_main, S_det, S_asst, G (where applicable), latency, VRAM} |
| Table 2 | Per answer_type S_det breakdown (heatmap) |
| Table 3 | Single-doc vs multi-doc Q_main per system |
| Table 4 | Merge↔Route gradient (S3, S4-cluster, S4-doc) |
| Table 5 | Offline cost comparison (training time / adapter gen time / index build time) |

## Best Adapter Selection Rule

S5 candidates: **S2 and S3 only** (single adapters). Select by eval Q_main on 50 questions. If tie (within 1pp): prefer lower inference latency → prefer S2.

**Known limitation:** adapter selection uses the same eval-50 set on which S5 is later evaluated. This introduces mild optimistic bias — S5 inherits an adapter that was chosen *because* it scored higher on this specific set. With only 2 candidates the selection pressure is minimal, but the bias must be acknowledged in the paper. A separate validation split was not created because 50 eval questions is already small and further splitting would reduce statistical power.

## Output

- `results/EXP-006/main_results.csv`
- `results/EXP-006/per_type_breakdown.csv`
- `results/EXP-006/gradient_plot.png`
- `experiments/EXP-006/REPORT.md`
- Frozen best adapter ID for S5

## Definition of Done

- [ ] All system results (S1-S4) collected and normalized into `main_results.csv`
- [ ] Per answer_type breakdown in `per_type_breakdown.csv` (6 types × 5 systems)
- [ ] Merge↔Route gradient plot generated (`gradient_plot.png`)
- [ ] Best adapter for S5 selected and documented (S2 or S3, by Q_main)
- [ ] Tables 1-5 from spec all present in REPORT.md
- [ ] All results committed to git
- [ ] `experiments/EXP-006/REPORT.md` written with interpretation per hypothesis
