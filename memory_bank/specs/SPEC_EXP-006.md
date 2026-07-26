# SPEC: EXP-006 — Main Comparison: Headline + Post-hoc + Controls

**Systems:** S1, S2+R, S3+R (headline) + S7 (post-hoc) + S2, S3, S3-legacy (controls) | **Wave:** 3 | **Depends on:** EXP-002, EXP-003, EXP-003b, EXP-004, EXP-004b, EXP-010 | **Blocks:** EXP-007

## Goal

Unified comparison of all documented systems. Produce the main results table answering RQ1 (headline), RQ2 (controls), and post-hoc merge impact (S7).

No fresh inference — collect and normalize results from prior experiments.

**Evaluation scope:** all compared systems are reported on the same 50 eval questions. S2/S2+R never saw these during training (trained on 150 train).

## Analysis Steps

1. **Collect** outputs from EXP-002, EXP-003, EXP-003b, EXP-004, EXP-004b, EXP-010, and legacy D2L report from `experiments/EXP-004_d2l_monolithic/REPORT.md`.
2. **Normalize** into common results format: one row per (system, question_id) where artifacts permit.
3. **Score** each system: Q_main, S_det, S_asst. G (F_β=2.5) for retrieval-aware systems (S1, S2+R, S3+R, S7). Controls S2, S3, S3-legacy: G = N/A.
4. **Breakdowns:** by answer_type (6 types), by difficulty (3 levels), by single/multi-doc, by unanswerable (cross-cutting flag).
5. **Systems metrics table:** TTFT, end-to-end latency, peak VRAM, offline packaging cost.

## Key Tables

| Table | Content |
|-------|---------|
| Table 1 | All systems × {Q_main, S_det, S_asst, G (where applicable), latency, VRAM} |
| Table 2 | Per answer_type score breakdown |
| Table 3 | Single-doc vs multi-doc Q_main per system |
| Table 4 | Offline cost comparison (training/index/merge) |

## Key Deltas

- Δ(S2+R, S1) = value of supervised adapter on top of RAG
- Δ(S3+R, S1) = value of CLM adapter on top of RAG
- Δ(S7, S2+R), Δ(S7, S3+R), Δ(S7, S1) = post-hoc merge impact
- Δ(S2+R, S3+R) = supervised vs supervision-free adapter source, same retrieval + PEFT
- Δ(S2+R, S2) = retrieval contribution to supervised system
- Δ(S3+R, S3) = retrieval contribution to CLM system
- Legacy anchor: S3-legacy (D2L) must be present as control row in all comparison tables

## Output

- `results/EXP-006/main_results.csv`
- `results/EXP-006/per_type_breakdown.csv`
- `results/EXP-006/single_vs_multi_doc.csv`
- `results/EXP-006/deltas.json`
- `results/EXP-006/gradient_plot.png`
- `experiments/EXP-006_main_comparison/REPORT.md`

## Definition of Done

- [x] All system results (S1, S2+R, S3+R, S7, S2, S3, S3-legacy) consolidated into `main_results.csv`
- [x] Per answer_type breakdown produced in `per_type_breakdown.csv`
- [x] Key deltas (headline, post-hoc, controls) computed and documented
- [x] S3-legacy (D2L) row included as mandatory legacy control in comparison narrative
- [x] Tables 1-4 present in REPORT.md
- [x] Results committed to git
- [x] `experiments/EXP-006_main_comparison/REPORT.md` written with interpretation
