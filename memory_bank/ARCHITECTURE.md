# SSOT: Knowledge Injection for Document-Grounded QA on Consumer Hardware

**Version:** 7.0 | **Updated:** 2026-03-30 | **Shorthand:** `knowledge-injection-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** How do three knowledge-injection paradigms compare **in isolation** on consumer hardware for document-grounded QA — nonparametric retrieval (S1), supervised parametric adaptation (S2), and supervision-free hypernetwork-based parametric adaptation (S3/S4)?

**RQ2 (inner study, S3 vs S4):** Does cluster-routed Doc-to-LoRA outperform monolithic Doc-to-LoRA, and is the gain better explained by capacity relief / specialization than by simple adapter count increase?

**RQ3 (retrieval augmentation):** Are paradigms complementary — does adding retrieval to a parametric system improve over either paradigm in isolation?

---

## 2. Comparison Axes

### Axis 1 — Paradigm in Isolation (RQ1)

Each system operates **only** through its own paradigm. Parametric systems use no retrieval at inference.

| ID | System | Paradigm | Retrieval | What is adapted |
|----|--------|----------|-----------|-----------------|
| S1 | Classical RAG | Nonparametric | Yes | Nothing (retrieval only) |
| S2 | QLoRA closed-book | Supervised parametric | **No** | One LoRA adapter on backbone |
| S3 | Doc-to-LoRA (monolithic) | Supervision-free parametric | **No** | One merged LoRA (8 → 1) |
| S4-doc | Doc-to-LoRA per-doc routed | Supervision-free parametric + routing | **No** | Per-doc adapter, hard top-1 routing |
| S4-cluster | Doc-to-LoRA cluster-routed | Supervision-free parametric + routing | **No** | Per-cluster adapter (k=4) |

### Axis 2 — Retrieval Augmentation (RQ3)

Does adding retrieval to a parametric system improve results?

| ID | System | Configuration | Compares against |
|----|--------|--------------|-----------------|
| S2+R | QLoRA RAFT + retrieval | Supervised adapter trained with context, S1 retrieval at inference | S2 (same adapter paradigm, ± retrieval) |
| S5 | Hybrid: RAG + best adapter | S1 retrieval + best single adapter from Axis 1 | Best parametric from Axis 1 |

**Key deltas:**
- Δ(S2+R, S2) = retrieval contribution to supervised system
- Δ(S5, best parametric) = retrieval contribution to best parametric system
- Δ(S5, S1) = adapter contribution to RAG

### Conditional Ablation

| ID | System | Trigger |
|----|--------|---------|
| S6 | Naive Dense RAG | Only if headline S5 < S1 on eval Q_main |

S6 measures combined retrieval engineering + chunk topology contribution via end-to-end naive baseline.

See `memory_bank/SPEC-systems.md` for detailed system definitions.

---

## 3. Working Hypotheses

- **H1.** S5 Hybrid gives best practical trade-off (retrieval evidence + adapted generation).
- **H2.** S2 closed-book will underperform S1 — a 2B model cannot memorize 176 pages from 150 QA pairs alone.
- **H2b.** S2+R (RAFT + retrieval) will significantly outperform S2 closed-book, demonstrating that retrieval and supervised adaptation are complementary.
- **H3.** S3/S4 Doc-to-LoRA competitive on facts outside goldset scope; each doc fits one D2L pass cleanly.
- **H4.** S4-doc (per-doc routing) beats S3 (full merge) on single-doc questions; S3 may win on multi-doc.
- **H4b.** S4-cluster sits between S3 and S4-doc — tests whether partial merge + routing is the sweet spot.
- **H5.** S1 RAG dominates on deterministic lookup (date, number, name).
- **H6.** Even if parametric systems don't beat RAG, quantifying their limits is a valid result.

---

## 4. Frozen Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Backbone | **Gemma-2-2b-it** | Only backbone with released Doc-to-LoRA hypernetwork checkpoint; used by all systems for fairness |
| Hypernetwork | **SakanaAI Doc-to-LoRA** (checkpoint-80000) | Pre-trained, not retrained in this project |
| Corpus | **8 PDF documents** (DIFC legal, ~115K tokens total) | Each fits D2L single pass; frozen before experiments |
| Goldset | **200 human-authored QA pairs** (100 per batch of 4 docs) | `data/goldset/goldset.benchmark.json` |
| Split | **150 S2-train / 50 eval**, stratified by answer_type + difficulty | All systems evaluated on same 50. S2/S2+R train on 150. No CV. |
| S2 variance | **3 random seeds**, report mean ± std | Replaces CV for supervised systems |
| S2 training format | **Closed-book** (question → answer, no context) | Pure parametric test for Axis 1 |
| S2+R training format | **RAFT-style open-book** (question + gold chunks → answer) | Context-aware for Axis 2 |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for QLoRA training | Standard QLoRA recipe |
| Embedding model | **Qwen3-Embedding-0.6B** for retrieval index and document routing | Shared across S1/S2+R/S4/S5/S6 |
| Reranker | **Qwen3-Reranker-0.6B** cross-encoder for S1 retrieval pipeline | Lexical fallback if model fails |
| S1 retrieval stack | **Full hybrid pipeline** from `external/pdf_rag_pipeline/` | Dense+sparse, RRF, reranker, evidence compressor |
| Routing (S4-doc) | **Hard top-1, cosine similarity to document embeddings** | Simplest per-doc routing |
| Clustering (S4-cluster) | **k=4, document-level, k-means, cosine nearest centroid** | Simple, interpretable, no learned router |

---

## 5. Evaluation Protocol (Compact)

**Primary metric (all systems):**
`Q_main = 0.7 × S_det + 0.3 × S_asst`

- `S_det`: deterministic accuracy (number, boolean, name, names, date). Unanswerable: expected `[]`, system returns `[]` → 1.0.
- `S_asst`: LLM-judge score on free_text (5 binary criteria, gpt-5.4-mini)

**Grounding (retrieval-aware systems only: S1, S2+R, S5, S6):**
`G = F_β(β=2.5)` on page-level grounding. Not computed for S2, S3, S4 (no retrieval → no grounding signal).

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training / adapter generation time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Quality claims require discussing cost + grounding trade-off, not quality alone.
- S2 performance reflects supervised adaptation quality on limited QA data, not whole-corpus knowledge.
- S3/S4 performance reflects hypernetwork packaging quality within this benchmark.
- S2+R performance reflects supervised adaptation + retrieval; delta vs S2 isolates retrieval contribution.
- S1 grounding is the reference; S5 must not degrade grounding materially.

See `memory_bank/SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

---

## 6. Terminology Rules

1. In precise prose use **"downstream supervision-free"** not "unsupervised" (hypernetwork is pre-trained upstream).
2. S2 learns from **goldset-style supervision**, not "the whole corpus."
3. No claim of **full corpus internalization** — conclusions bounded to this benchmark, backbone, and hardware.
4. "Unsupervised parametric" acceptable only in tables/diagrams as shorthand.
5. **S2** always refers to closed-book. **S2+R** always refers to RAFT + retrieval. Never use "S2" alone for the RAFT variant.

---

## 7. Data Summary

| Attribute | Value |
|-----------|-------|
| Documents | 8 PDFs (4 statutes/regs + 4 cases), DIFC legal, 176 pages, ~115K tokens |
| QA pairs | 200 (2 batches × 100) |
| Answer types | free_text: 53, boolean: 48, number: 36, name: 30, names: 17, date: 16 |
| Difficulty | easy: 98, medium: 71, hard: 31 |
| Multi-doc questions | 26 (13%) — all within same-batch doc pairs |
| Unanswerable | 17 (8.5%) — 9 with answer=`null` (expected response `[]`) + 8 free_text negative (expected response: text stating info absent) |

See `memory_bank/SPEC-data.md` for split protocol, schema, leakage rules.

---

## 8. Experiment Phases

| Phase | Axis | Goal | Key output |
|-------|------|------|------------|
| EXP-001 | — | Data audit + goldset merge + split freeze | `data/goldset/`, `data/splits/` |
| EXP-002 | 1 | S1 Classical RAG baseline | Nonparametric baseline metrics |
| EXP-003 | **2** | S2+R QLoRA RAFT + retrieval (3 seeds) | Retrieval-augmented supervised baseline |
| EXP-003b | **1** | S2 QLoRA closed-book (3 seeds) | Supervised parametric baseline |
| EXP-004 | 1 | S3 Doc-to-LoRA monolithic feasibility + packaging | Hypernetwork baseline |
| EXP-005 | 1 | S4 Clustering study + routed Doc-to-LoRA | Routed parametric system |
| EXP-006 | 1 | Main comparison S1-S4 on 50 eval | Cross-paradigm results (Axis 1) |
| EXP-007 | 2 | S5 Hybrid (S5a + S5b) | Hybrid top-line (Axis 2) |
| EXP-008 | — | Error analysis + final thesis tables | Combined Axis 1 + 2 analysis |
| EXP-009 | 2 | S6 E2E naive dense RAG ablation (conditional: S5 < S1) | Full pipeline value vs naive baseline |

---

## 9. Technology Stack

- **Python 3.12**, `uv` for env management, single `.venv` for all experiments
- **DL:** `torch==2.6.0+cu124`, `transformers==4.51.3`, `accelerate==1.6.0`, `peft`, `bitsandbytes` (versions pinned for Doc-to-LoRA compatibility)
- **Retrieval:** `sentence-transformers`, `qdrant-client`, `faiss` (S6 ablation only)
- **Doc-to-LoRA:** SakanaAI repo cloned to `external/doc-to-lora/`, installed as editable `--no-deps` (skips vllm/deepspeed); re-install after every `uv sync`
- **Evaluation:** custom metrics + OpenAI API client (`openai`) for judge
- **Viz:** `matplotlib`

---

## 10. Change Control

Updates to this file required before new experiments if changing:
research questions, system inventory, comparison axes, backbone, goldset size/split, Doc-to-LoRA packaging strategy, routing protocol, primary metric definition.
