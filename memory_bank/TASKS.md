# Task Board

> Ordered task list. Details in `memory_bank/tasks/{TASK_ID}.md`.
> Architecture v8.0: Headline systems (S1, S2+R, S3+R) vs Controls (S2, S3, S4).

## Active Queue

### Wave 1: Foundation
- [x] EXP-001 — Data audit, goldset merge, split freeze ([spec](./specs/SPEC_EXP-001.md) | [task](./tasks/EXP-001.md))
- [x] EXP-002 — S1 Classical RAG baseline (Headline) ([spec](./specs/SPEC_EXP-002.md))

### Wave 2: Parametric Feasibility
- [x] EXP-003 — S2+R QLoRA RAFT + retrieval, 3 seeds (Headline) ([spec](./specs/SPEC_EXP-003.md))
- [x] EXP-003b — S2 QLoRA closed-book, 3 seeds (Control) ([spec](./specs/SPEC_EXP-003b.md))
- [ ] EXP-004 — S3 Doc-to-LoRA per-doc generation + monolithic merge (Control) ([spec](./specs/SPEC_EXP-004.md))
- [ ] EXP-004b — S3+R Doc-to-LoRA + retrieval (Headline) ([spec](./specs/SPEC_EXP-004b.md))

### Wave 3: Routing (Controls, RQ2)
- [ ] EXP-005a — S4-doc: per-document routing, hard top-1 ([spec](./specs/SPEC_EXP-005a.md))
- [ ] EXP-005b — S4-cluster: cluster routing, k=4 ([spec](./specs/SPEC_EXP-005b.md))

### Wave 4: Comparison + Analysis
- [ ] EXP-006 — Main comparison: headline S1 vs S2+R vs S3+R + all controls ([spec](./specs/SPEC_EXP-006.md))
- [ ] EXP-007 — Error analysis + cost/quality/grounding trade-off ([spec](./specs/SPEC_EXP-007.md))

### Wave 5: Conditional
- [ ] EXP-008 — S6 E2E naive dense RAG ablation (conditional: S2+R and S3+R < S1) ([spec](./specs/SPEC_EXP-008.md))

## Reference Docs (in memory_bank/)
- [SPEC-systems](./SPEC-systems.md) — Detailed system definitions (Headline / Control)
- [SPEC-evaluation](./SPEC-evaluation.md) — Evaluation protocol, judge config, scoring rules
- [SPEC-data](./SPEC-data.md) — Corpus, goldset, splits, leakage rules

## Backlog
- [ ] [BL-001] Soft weighted routing (addresses multi-doc limitation)
- [ ] [BL-002] Merge degradation curve (Q_main vs number of adapters merged)
- [ ] [BL-003] Adapter similarity matrix (interpretability heatmap)
- [ ] [BL-004] Metadata-based clustering vs k-means comparison
