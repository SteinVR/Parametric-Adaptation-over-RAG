# Project State

> Compact operational status. Keep under 100 lines.

---

## Current Phase

**Phase:** Wave 1 — Foundation
**Blocker:** None
**Next action:** EXP-002 (S1 Classical RAG baseline)

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Backbone: Gemma-2-2b-it | Only model with released Doc-to-LoRA hypernetwork |
| 2026-03-26 | No CV; 3 seeds for S2 | S1/S3/S4 don't benefit from CV; seeds sufficient for variance |
| 2026-03-26 | Judge: gpt-5.4-mini (OpenAI API) | Cheap, reliable, version-pinnable |
| 2026-03-26 | RAFT-style for S2 | Open-book training more realistic than closed-book |
| 2026-03-26 | Doc-to-LoRA not retrained | Hardware constraint; use pre-trained hypernetwork as-is |
| 2026-03-26 | Document-level clustering for S4 | Simpler, more interpretable than chunk-level |

## System Readiness

| System | Status | Notes |
|--------|--------|-------|
| S1 Classical RAG | Not started | Need retriever setup on Gemma-2-2b-it |
| S2 QLoRA | Not started | Need RAFT-style data formatting |
| S3 Doc-to-LoRA mono | Not started | Need packaging feasibility (merge strategy) |
| S4 Cluster-routed D2L | Not started | Depends on S3 feasibility + clustering |
| S5 Hybrid | Not started | Depends on S2-S4 best adapter selection |

## Experiment History

| ID | Date | Result | Notes |
|----|------|--------|-------|
| EXP-001 | 2026-03-26 | Done | 65 docs, ~1.27M tokens, 40x D2L limit. Split frozen 120/30. 6 dupe pairs grouped. |

## Known Issues

| Issue | Status |
|-------|--------|
| Doc-to-LoRA merge strategy undefined | Open — resolve at EXP-004 |
| Gemma-2-2b-it quality on legal domain unknown | Open — resolve at EXP-002 |
| 65 docs may exceed single D2L pass (~32K limit) | Expected — S3/S4 design accounts for this |
