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
| 2026-03-26 | No CV; 3 seeds for S2/S3 families | Sufficient for variance estimation |
| 2026-03-27 | Corpus narrowed to 8 docs | Frozen before experiments |
| 2026-03-27 | Split 150/50 | Leakage-safe setup for QA-trained systems |
| 2026-03-30 | D2L → CLM pivot (architecture v9.0) | D2L quality non-viable (`Q_main=0.2100`) |
| 2026-03-31 | S7 adapter merge validated | `alpha=0.5` merge improved Q_main without retraining |
| 2026-03-31 | S6 moved to archived set | Removed from active thesis narrative by user request |

## System Readiness (Active + Post-hoc + Controls)

| System | Class | Status | Q_main |
|--------|-------|--------|--------|
| S1 | Headline | **Done** | 0.6425 |
| S2+R | Headline | **Done** | 0.6689 ± 0.0137 |
| S3+R | Headline | **Done** | 0.6671 ± 0.0229 |
| S7 | Post-hoc | **Done** | 0.7045 ± 0.0345 |
| S2 | Control | **Done** | 0.2630 ± 0.0046 |
| S3 | Control | **Done** | 0.1854 ± 0.0027 |
| S3-legacy (D2L) | Control (legacy) | **Done** | 0.2100 |

## Archived Systems

| System | Status | Note |
|--------|--------|------|
| S6 | Archived | EXP-008 completed, but excluded from active thesis comparisons |

## Experiment History

| ID | Date | Status | Notes |
|----|------|--------|-------|
| EXP-001 | 2026-03-28 | Done | Data audit. 8 docs, 200 QA, split 150/50. |
| EXP-002 | 2026-03-29 | Done | S1 baseline. Q_main=0.6425. |
| EXP-003 | 2026-03-29 | Done | S2+R RAFT. Q_main=0.6689±0.0137. |
| EXP-003b | 2026-03-30 | Done | S2 closed-book. Q_main=0.2630±0.0046. |
| EXP-004 D2L | 2026-03-30 | Done (legacy) | Valid negative finding. Q_main=0.2100. |
| EXP-004 CLM | 2026-03-31 | Done | S3 CLM. Q_main=0.1854±0.0027. |
| EXP-004b | 2026-03-31 | Done | S3+R. Q_main=0.6671±0.0229. |
| EXP-006 | 2026-03-31 | Done | Main comparison tables incl. S7 row. |
| EXP-007 | 2026-03-31 | Done | Error analysis + trade-off call. |
| EXP-008 | 2026-03-31 | Archived | S6 historical ablation only. |
| EXP-009 | 2026-03-31 | Archived | Conditional S6 refresh deprecated/out-of-scope. |
| EXP-010 | 2026-03-31 | Done | S7 merge. New best Q_main=0.7045±0.0345. |

## Known Issues

| Issue | Status |
|-------|--------|
| CLM seq_len=512 due to logit OOM | Accepted |
| Gemma-2-2b-it weak on dates | Confirmed |
| `PYTORCH_CUDA_ALLOC_CONF` required | QLoRA on 8GB |
