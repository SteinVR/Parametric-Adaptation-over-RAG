# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 2 — Parametric Feasibility
**Blocker:** None
**Next action:** EXP-003 (S2 QLoRA) and EXP-004 (S3 Doc-to-LoRA) — can run in parallel

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Backbone: Gemma-2-2b-it | Only model with released Doc-to-LoRA hypernetwork |
| 2026-03-26 | No CV; 3 seeds for S2 | S1/S3/S4 don't benefit from CV; seeds sufficient for variance |
| 2026-03-26 | Judge: gpt-5.4-mini (OpenAI API) | Cheap, reliable, version-pinnable |
| 2026-03-26 | RAFT-style for S2 | Open-book training more realistic than closed-book |
| 2026-03-26 | Doc-to-LoRA not retrained | Hardware constraint; use pre-trained hypernetwork as-is |
| 2026-03-27 | Corpus narrowed to 8 docs | Each fits D2L single pass; no merge-of-40 problem |
| 2026-03-27 | 200 QA custom goldset | User-authored, 2 batches × 100, covers all 8 docs |
| 2026-03-27 | Split 150/50 (needed for S2) | S2 trains on QA → needs leakage prevention |
| 2026-03-29 | Outlines only for boolean/names | Constrained decoding distorts logits on date/number/name |
| 2026-03-29 | S5 adapter selection limitation acknowledged | eval-50 used for both selection and evaluation; documented as caveat |

## System Readiness

| System | Status | Notes |
|--------|--------|-------|
| S1 Classical RAG | **Done (EXP-002)** | Q_main=0.6425, S_det=0.6014, S_asst=0.7385, G=0.5667 |
| S2 QLoRA | Not started | Need RAFT-style data formatting |
| S3 Doc-to-LoRA mono | Not started | 8 per-doc adapters → merge |
| S4 Cluster-routed D2L | Not started | Depends on EXP-004 adapters |
| S5 Hybrid | Not started | Depends on EXP-006 best adapter selection |

## Experiment History

| ID | Date | Result | Notes |
|----|------|--------|-------|
| EXP-001 | 2026-03-28 | Done | 8 docs, ~115K tokens, all fit D2L. 200 QA validated. Split frozen 150/50. |
| EXP-002 | 2026-03-29 | Done | S1 baseline. Q_main=0.6425. Staged retrieval, reranker bf16, outlines boolean/names. |

## Known Issues

| Issue | Status |
|-------|--------|
| Doc-to-LoRA merge strategy: simple average may destroy info | Open — resolve at EXP-004 |
| Gemma-2-2b-it weak on dates (0.2 S_det) | Confirmed at EXP-002 — model limitation |
| No cross-batch multi-doc questions | Accepted limitation — noted in SPEC-data |
| S5 adapter selection uses eval-50 | Acknowledged — documented as mandatory caveat |
