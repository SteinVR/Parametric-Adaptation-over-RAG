# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 1 — Foundation
**Blocker:** None
**Next action:** Create split on new goldset, then EXP-002 (S1 Classical RAG baseline)

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
| 2026-03-27 | Split 160/40 (needed for S2) | S2 trains on QA → needs leakage prevention |

## System Readiness

| System | Status | Notes |
|--------|--------|-------|
| S1 Classical RAG | Not started | Need retriever setup on Gemma-2-2b-it |
| S2 QLoRA | Not started | Need RAFT-style data formatting |
| S3 Doc-to-LoRA mono | Not started | 8 per-doc adapters → merge (feasible at this scale) |
| S4 Cluster-routed D2L | Not started | 4 clusters × 2 docs; merge of 2 per cluster |
| S5 Hybrid | Not started | Depends on S2-S4 best adapter selection |

## Experiment History

| ID | Date | Result | Notes |
|----|------|--------|-------|
| EXP-001 | 2026-03-28 | Done | 8 docs, 115K tokens, all fit D2L. 200 QA validated. Split frozen 160/40. |

## Known Issues

| Issue | Status |
|-------|--------|
| Doc-to-LoRA merge strategy undefined | Open — resolve at EXP-004 |
| Gemma-2-2b-it quality on legal domain unknown | Open — resolve at EXP-002 |
| No cross-batch multi-doc questions | Accepted limitation — noted in SPEC-data |
| 1 near-duplicate pair in goldset | Handled — will group at split creation |
