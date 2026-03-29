# Task Board

> Ordered task list. Details in `memory_bank/tasks/{TASK_ID}.md`.

## Active Queue

### Wave 1: Foundation
- [x] EXP-001 — Data audit, goldset merge, split freeze ([spec](./specs/SPEC_EXP-001.md) | [task](./tasks/EXP-001.md))
- [x] EXP-002 — S1 Classical RAG baseline ([spec](./specs/SPEC_EXP-002.md))

### Wave 2: Parametric Feasibility
- [ ] EXP-003 — S2 QLoRA RAFT-style baseline, 3 seeds ([spec](./specs/SPEC_EXP-003.md))
- [ ] EXP-004 — S3 Doc-to-LoRA per-doc generation + monolithic merge ([spec](./specs/SPEC_EXP-004.md))

### Wave 3: Routing
- [ ] EXP-005a — S4-doc: per-document routing, hard top-1 ([spec](./specs/SPEC_EXP-005a.md))
- [ ] EXP-005b — S4-cluster: cluster routing, k=4 ([spec](./specs/SPEC_EXP-005b.md))

### Wave 4: Comparison
- [ ] EXP-006 — Main cross-paradigm comparison (S1, S2, S3, S4-doc, S4-cluster) ([spec](./specs/SPEC_EXP-006.md))

### Wave 5: Hybrid + Final
- [ ] EXP-007 — S5 Hybrid: RAG + best adapter + HyDE ([spec](./specs/SPEC_EXP-007.md))
- [ ] EXP-008 — Locked test + error analysis + final tables ([spec](./specs/SPEC_EXP-008.md))
- [ ] EXP-009 — S6 E2E naive dense RAG ablation (conditional: S5 < S1) ([spec](./specs/SPEC_EXP-009.md))

## Reference Docs (in memory_bank/)
- [SPEC-systems](./SPEC-systems.md) — Detailed system definitions
- [SPEC-evaluation](./SPEC-evaluation.md) — Evaluation protocol, judge config, scoring rules
- [SPEC-data](./SPEC-data.md) — Corpus, goldset, splits, leakage rules
- [BACKLOG-routing-research](./BACKLOG-routing-research.md) — Full routing/merge research space

## Backlog
- See [BACKLOG-routing-research.md](../BACKLOG-routing-research.md) for full research space
- [ ] [BL-001] Soft weighted routing (addresses multi-doc limitation)
- [ ] [BL-002] Merge degradation curve (Q_main vs number of adapters merged)
- [ ] [BL-003] Adapter similarity matrix (interpretability heatmap)
- [ ] [BL-004] Metadata-based clustering vs k-means comparison
- [ ] [BL-005] RAFT open-book vs closed-book ablation for S2
