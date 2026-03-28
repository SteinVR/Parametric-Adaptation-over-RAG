# SSOT: Knowledge Injection for Document-Grounded QA on Consumer Hardware

**Version:** 4.0 | **Updated:** 2026-03-27 | **Shorthand:** `knowledge-injection-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/tasks/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** How do four knowledge-injection paradigms compare on consumer hardware for document-grounded QA over a fixed corpus — nonparametric retrieval, supervised parametric adaptation, downstream supervision-free hypernetwork-based parametric adaptation, and hybrid — when each paradigm operates under its natural informational and computational constraints?

**RQ2 (inner study, S3 vs S4):** Does cluster-routed Doc-to-LoRA outperform monolithic Doc-to-LoRA, and is the gain better explained by capacity relief / specialization than by simple adapter count increase?

---

## 2. System Inventory (Frozen)

| ID | System | Family | Info input | What is adapted |
|----|--------|--------|------------|-----------------|
| S1 | Classical RAG | Nonparametric | Full corpus via index | Nothing (retrieval only) |
| S2 | QLoRA (RAFT-style) | Supervised parametric | 200 QA pairs + gold chunks | One LoRA adapter on backbone |
| S3 | Doc-to-LoRA (monolithic) | Supervision-free parametric | 8 docs via hypernetwork (per-doc → merge) | One merged LoRA from per-doc passes |
| S4 | Cluster-routed Doc-to-LoRA | Supervision-free parametric + routing | 8 docs via hypernetwork (per-cluster) | Per-cluster LoRA + router |
| S5 | Hybrid: RAG + best adapter | Hybrid | Retrieval + best adapter (S2-S4) | Best adapter + retrieval pipeline |

S5 sub-variants: **S5a** (raw-query retrieval + adapter), **S5b** (adapter-generated HyDE + retrieval + adapter). HyDE is only evaluated inside S5.

See `SPEC-systems.md` for detailed system definitions, packaging strategies, and merge rules.

---

## 3. Working Hypotheses

- **H1.** S5 Hybrid gives best practical trade-off (retrieval evidence + adapted generation).
- **H2.** S2 QLoRA shows best format discipline but is bounded by goldset coverage (~200 QA over 8 docs).
- **H3.** S3/S4 Doc-to-LoRA competitive on facts outside goldset scope; each doc fits one D2L pass cleanly.
- **H4.** S4 cluster-routed beats S3 monolithic via capacity relief and specialization.
- **H5.** S1 RAG dominates on deterministic lookup (date, number, name).
- **H6.** Even if parametric systems don't beat RAG, quantifying their limits is a valid result.

---

## 4. Frozen Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Backbone | **Gemma-2-2b-it** | Only backbone with released Doc-to-LoRA hypernetwork checkpoint; used by all S1-S5 for fairness |
| Hypernetwork | **SakanaAI Doc-to-LoRA** (checkpoint-80000) | Pre-trained, not retrained in this project |
| Corpus | **8 PDF documents** (DIFC legal, ~141K tokens total) | Each fits D2L single pass; frozen before experiments |
| Goldset | **200 human-authored QA pairs** (100 per batch of 4 docs) | `data/goldset/goldset.benchmark.json` |
| Split | **160 dev / 40 locked test**, stratified by answer_type + difficulty | Single fixed split, no CV; split needed for S2 leakage prevention |
| S2 variance | **3 random seeds**, report mean ± std | Replaces CV for supervised system |
| S2 training format | **RAFT-style open-book** (question + gold chunks → answer) | Context-aware, not closed-book |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for QLoRA training | Standard QLoRA recipe |
| Embedding model | Same for retrieval index and document clustering | Unless strong reason to decouple |
| Clustering (S4) | **k=4, document-level, k-means, cosine nearest centroid** | Simple, interpretable, no learned router |

---

## 5. Evaluation Protocol (Compact)

**Primary metric (all systems):**
`Q_main = 0.7 × S_det + 0.3 × S_asst`

- `S_det`: deterministic accuracy (number, boolean, name, names, date, null)
- `S_asst`: LLM-judge score on free_text (5 binary criteria, gpt-5.4-mini)

**Retrieval-aware (S1, S5 only):**
`G = F_β(β=2.5)` on page-level grounding

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training / adapter generation time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Quality claims require discussing cost + grounding trade-off, not quality alone.
- S2 performance reflects supervised adaptation quality, not whole-corpus knowledge.
- S3/S4 performance reflects hypernetwork packaging quality within this benchmark.
- S1 grounding is the reference; S5 must not degrade grounding materially.

See `SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

---

## 6. Terminology Rules

1. In precise prose use **"downstream supervision-free"** not "unsupervised" (hypernetwork is pre-trained upstream).
2. S2 learns from **goldset-style supervision**, not "the whole corpus."
3. No claim of **full corpus internalization** — conclusions bounded to this benchmark, backbone, and hardware.
4. "Unsupervised parametric" acceptable only in tables/diagrams as shorthand.

---

## 7. Data Summary

| Attribute | Value |
|-----------|-------|
| Documents | 8 PDFs (4 statutes/regs + 4 cases), DIFC legal, 176 pages, ~141K tokens |
| QA pairs | 200 (2 batches × 100) |
| Answer types | free_text: 53, boolean: 48, number: 36, name: 30, names: 17, date: 16 |
| Difficulty | easy: 98, medium: 71, hard: 31 |
| Multi-doc questions | 26 (13%) — all within same-batch doc pairs |
| Negative/unanswerable | 17 (8.5%) — 9 deterministic null + 8 free_text negative |

See `SPEC-data.md` for split protocol, schema, leakage rules.

---

## 8. Experiment Phases

| Phase | Goal | Key output |
|-------|------|------------|
| EXP-001 | Data audit + goldset merge + split freeze | `data/goldset/`, `data/splits/` |
| EXP-002 | S1 Classical RAG baseline | Nonparametric baseline metrics |
| EXP-003 | S2 QLoRA feasibility + baseline (3 seeds) | Supervised parametric baseline |
| EXP-004 | S3 Doc-to-LoRA monolithic feasibility + packaging | Hypernetwork baseline |
| EXP-005 | S4 Clustering study + routed Doc-to-LoRA | Routed parametric system |
| EXP-006 | Main comparison S1-S4 on dev | Cross-paradigm results |
| EXP-007 | S5 Hybrid (S5a + S5b) | Hybrid top-line |
| EXP-008 | Locked test + error analysis | Final thesis tables |

---

## 9. Technology Stack

- **Python 3.11+**, `uv` for env management
- **DL:** `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`
- **Retrieval:** `sentence-transformers`, `faiss` or existing vector DB
- **Doc-to-LoRA:** SakanaAI repo as dependency/submodule
- **Evaluation:** custom metrics + OpenAI API client for judge
- **Viz:** `matplotlib`

---

## 10. Change Control

Updates to this file required before new experiments if changing:
research questions, system inventory, backbone, goldset size/split, Doc-to-LoRA packaging strategy, routing protocol, primary metric definition.
