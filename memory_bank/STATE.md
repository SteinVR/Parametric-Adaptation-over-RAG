# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 2 — Parametric Feasibility
**Blocker:** None
**Next action:** EXP-004 (S3 D2L packaging) → then EXP-004b (S3+R headline system)

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Backbone: Gemma-2-2b-it | Only model with released Doc-to-LoRA hypernetwork |
| 2026-03-26 | No CV; 3 seeds for S2 | S1/S3/S4 don't benefit from CV; seeds sufficient for variance |
| 2026-03-26 | Judge: gpt-5.4-mini (OpenAI API) | Cheap, reliable, version-pinnable |
| 2026-03-26 | Doc-to-LoRA not retrained | Hardware constraint; use pre-trained hypernetwork as-is |
| 2026-03-27 | Corpus narrowed to 8 docs | Each fits D2L single pass; no merge-of-40 problem |
| 2026-03-27 | 200 QA custom goldset | User-authored, 2 batches × 100, covers all 8 docs |
| 2026-03-27 | Split 150/50 (needed for S2) | S2 trains on QA → needs leakage prevention |
| 2026-03-29 | Outlines only for boolean/names | Constrained decoding distorts logits on date/number/name |
| 2026-03-30 | **Architecture v8.0: headline/control split** | Advisor review: Axis 1 (isolation) is control, not main story. Main RQ = value of parametric adaptation on top of RAG. S2+R promoted to headline, S3+R added as headline. |
| 2026-03-30 | S3+R = D2L adapter + S1 retrieval | Symmetric comparison with S2+R: same retrieval, different adapter source |

## System Readiness

| System | Class | Status | Notes |
|--------|-------|--------|-------|
| S1 | Headline | **Done (EXP-002)** | Q_main=0.6425 |
| S2+R | Headline | **Done (EXP-003)** | Q_main=0.669±0.014. Delta vs S1: +0.027 |
| S3+R | Headline | **Not started** | Needs EXP-004 (adapters) then EXP-004b (retrieval eval) |
| S2 | Control | **Done (EXP-003b)** | Q_main=0.263±0.005. Parametric limit confirmed. |
| S3 | Control | Not started | Part of EXP-004 |
| S4-doc | Control | Not started | Depends on EXP-004 adapters |
| S4-cluster | Control | Not started | Depends on EXP-005 |

## Experiment History

| ID | Date | Class | Result | Notes |
|----|------|-------|--------|-------|
| EXP-001 | 2026-03-28 | — | Done | 8 docs, ~115K tokens. 200 QA validated. Split frozen 150/50. |
| EXP-002 | 2026-03-29 | Headline | Done | S1 baseline. Q_main=0.6425. |
| EXP-003 | 2026-03-29 | Headline | Done | S2+R RAFT. Q_main=0.669±0.014. |
| EXP-003b | 2026-03-30 | Control | Done | S2 closed-book. Q_main=0.263±0.005. Δ(S2+R,S2)=+0.406. |

## Known Issues

| Issue | Status |
|-------|--------|
| D2L merge strategy: simple average may destroy info | Open — resolve at EXP-004 |
| Gemma-2-2b-it weak on dates (0.2 S_det in S1) | Confirmed — model limitation |
| No cross-batch multi-doc questions | Accepted — noted in SPEC-data |
| PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True required | QLoRA OOMs without it on 8GB VRAM |
