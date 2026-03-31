# Task Board

> Ordered task list. Details in `memory_bank/tasks/{TASK_ID}.md`.
> Architecture v9.2: Headline systems (S1, S2+R, S3+R), post-hoc S7, controls (S2, S3, S3-legacy D2L).

## Active Queue

### Wave 1: Foundation
- [x] EXP-001 — Data audit, goldset merge, split freeze ([spec](./specs/SPEC_EXP-001.md) | [task](./tasks/EXP-001.md))
- [x] EXP-002 — S1 Classical RAG baseline (Headline) ([spec](./specs/SPEC_EXP-002.md))

### Wave 2: Parametric Feasibility
- [x] EXP-003 — S2+R QLoRA RAFT + retrieval, 3 seeds (Headline) ([spec](./specs/SPEC_EXP-003.md))
- [x] EXP-003b — S2 QLoRA closed-book, 3 seeds (Control) ([spec](./specs/SPEC_EXP-003b.md))
- [x] EXP-004 — S3 CLM continued pretraining, 3 seeds (Control) ([spec](./specs/SPEC_EXP-004.md) | [task](./tasks/EXP-004.md))
- [x] EXP-004b — S3+R CLM + retrieval (Headline) ([spec](./specs/SPEC_EXP-004b.md))

### Wave 3: Comparison + Analysis
- [x] EXP-006 — Main comparison: headline + controls + post-hoc S7 row ([spec](./specs/SPEC_EXP-006.md))
- [x] EXP-007 — Error analysis + cost/quality/grounding trade-off ([spec](./specs/SPEC_EXP-007.md))

### Wave 5: Post-hoc
- [x] EXP-010 — S7 adapter merge (0.5×CLM + 0.5×RAFT), eval-only, no retraining ([spec](./specs/SPEC_EXP-010.md))

## Archived / Out of Scope

- [x] EXP-008 — S6 naive dense RAG (archived from active thesis narrative) ([spec](./specs/SPEC_EXP-008.md))
- [x] EXP-009 — Conditional S6 refresh (archived/deprecated) ([spec](./specs/SPEC_EXP-009.md))

## Dropped (v9.0)

- ~~EXP-005a — S4-doc~~
- ~~EXP-005b — S4-cluster~~

## Reference Docs (in memory_bank/)

- [SPEC-systems](./SPEC-systems.md) — Detailed system definitions (Headline / Post-hoc / Control / Archived)
- [SPEC-evaluation](./SPEC-evaluation.md) — Evaluation protocol, judge config, scoring rules
- [SPEC-data](./SPEC-data.md) — Corpus, goldset, splits, leakage rules
