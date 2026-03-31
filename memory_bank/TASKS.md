# Task Board

> Ordered task list. Details in `memory_bank/tasks/{TASK_ID}.md`.
> Architecture v9.0: Headline systems (S1, S2+R, S3+R) vs Controls (S2, S3).

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
- [x] EXP-006 — Main comparison: all systems ([spec](./specs/SPEC_EXP-006.md))
- [x] EXP-007 — Error analysis + cost/quality/grounding trade-off ([spec](./specs/SPEC_EXP-007.md))

### Wave 4: Ablation
- [x] EXP-008 — S6 naive dense RAG (run by user request) ([spec](./specs/SPEC_EXP-008.md))

### Not triggered
- ~~EXP-009 — Refresh with S6~~ (S6 results integrated directly in EXP-006/007)

### Dropped (v9.0)
- ~~EXP-005a — S4-doc~~ ([spec](./specs/SPEC_EXP-005a.md))
- ~~EXP-005b — S4-cluster~~ ([spec](./specs/SPEC_EXP-005b.md))

## Reference Docs (in memory_bank/)
- [SPEC-systems](./SPEC-systems.md) — Detailed system definitions (Headline / Control)
- [SPEC-evaluation](./SPEC-evaluation.md) — Evaluation protocol, judge config, scoring rules
- [SPEC-data](./SPEC-data.md) — Corpus, goldset, splits, leakage rules
