# SPEC: EXP-006 — Main Comparison: S1 vs S2+R vs S3+R + Controls

**Systems:** S1, S2+R, S3+R (headline) + S2, S3 (controls) | **Wave:** 3 | **Depends on:** EXP-002, EXP-003, EXP-003b, EXP-004, EXP-004b | **Blocks:** EXP-007

## Goal

Unified comparison of all systems. Produce the main results table answering RQ1 (headline) and RQ2 (controls). No fresh inference — collect and normalize results from prior experiments.

**Evaluation scope:** ALL systems evaluated on the same 50 eval questions. S2/S2+R never saw these during training (trained on 150 train). No contamination for any system.

## Analysis Steps

1. **Collect** all system outputs from EXP-002, EXP-003, EXP-003b, EXP-004, EXP-004b (already evaluated, no fresh inference)
2. **Normalize** into common results format: one row per (system, question_id) with predicted_answer, metrics, timing
3. **Score** each system: Q_main, S_det, S_asst. G (F_β=2.5) for S1, S2+R, S3+R (retrieval-aware). S2, S3: G = N/A.
4. **Breakdowns:** by answer_type (6 types), by difficulty (3 levels), by single/multi-doc, by unanswerable (cross-cutting flag, reported separately)
5. **Systems metrics table:** TTFT, end-to-end latency, peak VRAM, offline packaging cost

## Key Tables

| Table | Content |
|-------|---------|
| Table 1 | All systems × {Q_main, S_det, S_asst, G (where applicable), latency, VRAM} |
| Table 2 | Per answer_type S_det breakdown (heatmap) |
| Table 3 | Single-doc vs multi-doc Q_main per system |
| Table 4 | Offline cost comparison (training time / index build time) |

## Key Deltas

- Δ(S2+R, S1) = value of supervised adapter on top of RAG (RQ1)
- Δ(S3+R, S1) = value of CLM adapter on top of RAG (RQ1)
- Δ(S2+R, S3+R) = supervised vs supervision-free adapter source, same retrieval + PEFT (RQ1)
- Δ(S2+R, S2) = retrieval contribution to supervised system (RQ2)
- Δ(S3+R, S3) = retrieval contribution to CLM system (RQ2)

## Output

- `results/EXP-006/main_results.csv`
- `results/EXP-006/per_type_breakdown.csv`
- `results/EXP-006/gradient_plot.png`
- `experiments/EXP-006/REPORT.md`

## Definition of Done

- [ ] All system results (S1, S2+R, S3+R, S2, S3) collected and normalized into `main_results.csv`
- [ ] Per answer_type breakdown in `per_type_breakdown.csv` (6 types × 5 systems)
- [ ] Key deltas (RQ1/RQ2) computed and documented
- [ ] Tables 1-4 from spec all present in REPORT.md
- [ ] All results committed to git
- [ ] `experiments/EXP-006/REPORT.md` written with interpretation per hypothesis
