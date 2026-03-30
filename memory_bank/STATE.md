# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 3 — Comparison + Analysis
**Blocker:** None
**Next action:** EXP-006 (main comparison table) → EXP-007 (error analysis)

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
| 2026-03-31 | CLM hyperparams tuned for continued pretraining | LR=5e-5, epochs=5, warmup=0.1, seq_len=512 |
| 2026-03-31 | EXP-008 run despite trigger not met | User requested S6 ablation for retrieval engineering analysis |

## System Readiness

| System | Class | Status | Notes |
|--------|-------|--------|-------|
| S1 | Headline | **Done (EXP-002)** | Q_main=0.6425 |
| S2+R | Headline | **Done (EXP-003)** | Q_main=0.669±0.014 |
| S3+R | Headline | **Done (EXP-004b)** | Q_main=0.667±0.023 |
| S2 | Control | **Done (EXP-003b)** | Q_main=0.263±0.005 |
| S3 | Control | **Done (EXP-004)** | Q_main=0.185±0.003 |
| S6 | Ablation | **Done (EXP-008)** | Q_main=0.6335, G=0.4891 |

## Experiment History

| ID | Date | Class | Result | Notes |
|----|------|-------|--------|-------|
| EXP-001 | 2026-03-28 | — | Done | 8 docs, ~115K tokens. 200 QA. Split frozen 150/50. |
| EXP-002 | 2026-03-29 | Headline | Done | S1. Q_main=0.6425. |
| EXP-003 | 2026-03-29 | Headline | Done | S2+R RAFT. Q_main=0.669±0.014. |
| EXP-003b | 2026-03-30 | Control | Done | S2 closed-book. Q_main=0.263±0.005. |
| EXP-004 (D2L) | 2026-03-30 | — | **Archived** | D2L negative finding. Q_main=0.210. |
| EXP-004 (CLM) | 2026-03-31 | Control | Done | S3 CLM. Q_main=0.185±0.003. |
| EXP-004b | 2026-03-31 | Headline | Done | S3+R. Q_main=0.667±0.023. S_asst=0.826. |
| EXP-008 | 2026-03-31 | Ablation | Done | S6 naive dense. Q_main=0.634. Δ(S1)=−0.009. Jaccard=0.40. |

## Known Issues

| Issue | Status |
|-------|--------|
| D2L hypernetwork non-viable for corpus | **Resolved** — archived, pivoted to CLM |
| CLM seq_len=512 due to full-seq logit OOM | Accepted — 208 chunks cover full corpus |
| Gemma-2-2b-it weak on dates | Confirmed — model limitation |
| PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True | Required for QLoRA on 8GB VRAM |
