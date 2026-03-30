# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 2 — Parametric Feasibility (CLM pivot)
**Blocker:** None
**Next action:** EXP-004 (S3 CLM training) → then EXP-004b (S3+R headline system)

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
| 2026-03-30 | Architecture v8.0: headline/control split | Main RQ = value of parametric adaptation on top of RAG |
| 2026-03-30 | **D2L → CLM pivot (v9.0)** | D2L hypernetwork non-viable (Q_main=0.210). CLM replaces D2L. S4 and old RQ2 (D2L routing study) dropped; old RQ3 (parametric limits) renumbered to RQ2. CLM PEFT matches S2+R (rank=32, q_proj+v_proj). |

## System Readiness

| System | Class | Status | Notes |
|--------|-------|--------|-------|
| S1 | Headline | **Done (EXP-002)** | Q_main=0.6425 |
| S2+R | Headline | **Done (EXP-003)** | Q_main=0.669±0.014. Delta vs S1: +0.027 |
| S3+R | Headline | **Not started** | Needs EXP-004 (CLM training) then EXP-004b |
| S2 | Control | **Done (EXP-003b)** | Q_main=0.263±0.005. Parametric limit confirmed. |
| S3 | Control | **Not started** | Part of EXP-004 (CLM pivot) |

## Experiment History

| ID | Date | Class | Result | Notes |
|----|------|-------|--------|-------|
| EXP-001 | 2026-03-28 | — | Done | 8 docs, ~115K tokens. 200 QA validated. Split frozen 150/50. |
| EXP-002 | 2026-03-29 | Headline | Done | S1 baseline. Q_main=0.6425. |
| EXP-003 | 2026-03-29 | Headline | Done | S2+R RAFT. Q_main=0.669±0.014. |
| EXP-003b | 2026-03-30 | Control | Done | S2 closed-book. Q_main=0.263±0.005. Δ(S2+R,S2)=+0.406. |
| EXP-004 (D2L) | 2026-03-30 | — | **Archived** | D2L negative finding. Q_main=0.210. Artifacts in `results/EXP-004/` (archived, not overwritten by CLM). |

## Known Issues

| Issue | Status |
|-------|--------|
| D2L hypernetwork non-viable for corpus | **Resolved** — archived as negative finding, pivoted to CLM (v9.0) |
| Gemma-2-2b-it weak on dates (0.2 S_det in S1) | Confirmed — model limitation |
| No cross-batch multi-doc questions | Accepted — noted in SPEC-data |
| PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True required | QLoRA OOMs without it on 8GB VRAM |
