# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** All experiments complete. Ready for thesis writing.
**Blocker:** None
**Next action:** Thesis draft / final packaging

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Backbone: Gemma-2-2b-it | Kept for fairness with completed S1/S2+R |
| 2026-03-26 | No CV; 3 seeds for S2/S3 | Sufficient for variance estimation |
| 2026-03-26 | Judge: gpt-5.4-mini (OpenAI API) | Cheap, reliable, version-pinnable |
| 2026-03-27 | Corpus narrowed to 8 docs | Frozen before experiments |
| 2026-03-27 | 200 QA custom goldset | User-authored, 2 batches × 100, covers all 8 docs |
| 2026-03-27 | Split 150/50 (needed for S2) | S2 trains on QA → needs leakage prevention |
| 2026-03-29 | Outlines only for boolean/names | Constrained decoding distorts logits on date/number/name |
| 2026-03-30 | Architecture v8.0 → v9.0 | D2L→CLM pivot. S4/old RQ2 dropped. |
| 2026-03-31 | CLM hyperparams tuned | LR=5e-5, epochs=5, warmup=0.1, seq_len=512 |
| 2026-03-31 | EXP-008 run by user request | S6 ablation despite trigger not met |
| 2026-03-31 | Practical winner: no single winner | S2+R wins S_det, S3+R wins S_asst. Trade-off is genuine. |

## System Readiness

| System | Class | Status | Q_main |
|--------|-------|--------|--------|
| S1 | Headline | **Done** | 0.6425 |
| S2+R | Headline | **Done** | 0.669±0.014 |
| S3+R | Headline | **Done** | 0.667±0.023 |
| S2 | Control | **Done** | 0.263±0.005 |
| S3 | Control | **Done** | 0.185±0.003 |
| S6 | Ablation | **Done** | 0.6335 |

## Experiment History

| ID | Date | Status | Notes |
|----|------|--------|-------|
| EXP-001 | 2026-03-28 | Done | Data audit. 8 docs, 200 QA, split 150/50. |
| EXP-002 | 2026-03-29 | Done | S1 baseline. Q_main=0.6425. |
| EXP-003 | 2026-03-29 | Done | S2+R RAFT. Q_main=0.669±0.014. |
| EXP-003b | 2026-03-30 | Done | S2 closed-book. Q_main=0.263±0.005. |
| EXP-004 D2L | 2026-03-30 | Archived | Negative finding. Q_main=0.210. |
| EXP-004 CLM | 2026-03-31 | Done | S3 CLM. Q_main=0.185±0.003. |
| EXP-004b | 2026-03-31 | Done | S3+R. Q_main=0.667±0.023. |
| EXP-006 | 2026-03-31 | Done | Main comparison. Tables, deltas, hypotheses. |
| EXP-007 | 2026-03-31 | Done | Error analysis, figures, practical winner call. |
| EXP-008 | 2026-03-31 | Done | S6 naive dense. Q_main=0.634. Δ(S1)=−0.009. |

## Known Issues

| Issue | Status |
|-------|--------|
| CLM seq_len=512 due to logit OOM | Accepted |
| Gemma-2-2b-it weak on dates | Confirmed |
| PYTORCH_CUDA_ALLOC_CONF required | QLoRA on 8GB |
