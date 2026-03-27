# Task Board

> Ordered task list. Details in `memory_bank/tasks/{TASK_ID}.md`.

## Active Queue

### Wave 1: Foundation
- [x] [EXP-001](./tasks/EXP-001.md) Data audit: corpus manifest, goldset validation, split freeze, capacity audit
- [ ] [EXP-002](./tasks/EXP-002.md) S1 Classical RAG baseline on Gemma-2-2b-it

### Wave 2: Parametric Feasibility
- [ ] [EXP-003](./tasks/EXP-003.md) S2 QLoRA feasibility + baseline (3 seeds, RAFT-style)
- [ ] [EXP-004](./tasks/EXP-004.md) S3 Doc-to-LoRA monolithic feasibility + merge strategy selection

### Wave 3: Routing + Comparison
- [ ] [EXP-005](./tasks/EXP-005.md) S4 Clustering study + cluster-routed Doc-to-LoRA
- [ ] [EXP-006](./tasks/EXP-006.md) Main cross-paradigm comparison S1-S4 on dev

### Wave 4: Hybrid + Final
- [ ] [EXP-007](./tasks/EXP-007.md) S5 Hybrid (S5a raw + S5b HyDE)
- [ ] [EXP-008](./tasks/EXP-008.md) Locked test evaluation + error analysis + final tables

## Specs (Reference)
- [SPEC-systems](./tasks/SPEC-systems.md) — Detailed system definitions
- [SPEC-evaluation](./tasks/SPEC-evaluation.md) — Evaluation protocol, judge config, scoring rules
- [SPEC-data](./tasks/SPEC-data.md) — Corpus, goldset, splits, leakage rules

## Backlog
- [ ] [BL-001] Chunk-level clustering ablation for S4 (if document-level proves inadequate)
- [ ] [BL-002] Learned router for S4 (appendix only)
- [ ] [BL-003] Second backbone confirmatory run (if time permits)
- [ ] [BL-004] RAFT open-book vs closed-book ablation for S2 (appendix)
