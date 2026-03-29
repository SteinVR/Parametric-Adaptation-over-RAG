# SSOT: Knowledge Injection for Document-Grounded QA on Consumer Hardware

**Version:** 6.0 | **Updated:** 2026-03-29 | **Shorthand:** `knowledge-injection-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** How do four knowledge-injection paradigms compare on consumer hardware for document-grounded QA over a fixed corpus — nonparametric retrieval, supervised parametric adaptation, downstream supervision-free hypernetwork-based parametric adaptation, and hybrid — when each paradigm operates under its natural informational and computational constraints?

**RQ2 (inner study, S3 vs S4):** Does cluster-routed Doc-to-LoRA outperform monolithic Doc-to-LoRA, and is the gain better explained by capacity relief / specialization than by simple adapter count increase?

---

## 2. System Inventory (Frozen)

| ID | System | Family | Info input | What is adapted |
|----|--------|--------|------------|-----------------|
| S1 | Classical RAG | Nonparametric | Full corpus via hybrid index | Nothing (retrieval only) |
| S2 | QLoRA (RAFT-style) | Supervised parametric | 150 train QA + gold chunks | One LoRA adapter on backbone |
| S3 | Doc-to-LoRA (monolithic) | Supervision-free parametric | 8 docs via hypernetwork → merge all | One merged LoRA (8 → 1) |
| S4-doc | Doc-to-LoRA per-doc routed | Supervision-free parametric + routing | 8 docs → 8 adapters, hard top-1 routing | Per-doc adapter, zero merge |
| S4-cluster | Doc-to-LoRA cluster-routed | Supervision-free parametric + routing | 8 docs → 4 clusters, merge 2 per cluster, route | Per-cluster adapter (k=4) |
| S5 | Hybrid: RAG + best single adapter | Hybrid | Retrieval + best of S2 or S3 | Single adapter + retrieval pipeline |
| S6 | Naive Dense RAG (e2e ablation) | Nonparametric | Corpus via dense-only index | Nothing (e2e ablation of S1) |

S5 sub-variants: **S5a** (raw-query retrieval + adapter), **S5b** (adapter-generated HyDE + retrieval + adapter). HyDE is only evaluated inside S5.

**S6** is conditional: run only if headline S5 (best of S5a/S5b) < S1 on eval Q_main. End-to-end naive dense RAG ablation (microchunk-only, dense FAISS, no reranker/hybrid/compression). Delta(S1, S6) measures combined retrieval engineering + chunk topology contribution.

See `memory_bank/SPEC-systems.md` for detailed system definitions, packaging strategies, and merge rules.

---

## 3. Working Hypotheses

- **H1.** S5 Hybrid gives best practical trade-off (retrieval evidence + adapted generation).
- **H2.** S2 QLoRA shows best format discipline but is bounded by goldset coverage (~200 QA over 8 docs).
- **H3.** S3/S4 Doc-to-LoRA competitive on facts outside goldset scope; each doc fits one D2L pass cleanly.
- **H4.** S4-doc (per-doc routing) beats S3 (full merge) on single-doc questions; S3 may win on multi-doc.
- **H4b.** S4-cluster sits between S3 and S4-doc — tests whether partial merge + routing is the sweet spot.
- **H5.** S1 RAG dominates on deterministic lookup (date, number, name).
- **H6.** Even if parametric systems don't beat RAG, quantifying their limits is a valid result.

---

## 4. Frozen Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Backbone | **Gemma-2-2b-it** | Only backbone with released Doc-to-LoRA hypernetwork checkpoint; used by all S1-S5 for fairness |
| Hypernetwork | **SakanaAI Doc-to-LoRA** (checkpoint-80000) | Pre-trained, not retrained in this project |
| Corpus | **8 PDF documents** (DIFC legal, ~115K tokens total) | Each fits D2L single pass; frozen before experiments |
| Goldset | **200 human-authored QA pairs** (100 per batch of 4 docs) | `data/goldset/goldset.benchmark.json` |
| Split | **150 S2-train / 50 eval**, stratified by answer_type + difficulty | All systems evaluated on same 50. S2 trains on 150. No CV. |
| S2 variance | **3 random seeds**, report mean ± std | Replaces CV for supervised system |
| S2 training format | **RAFT-style open-book** (question + gold chunks → answer) | Context-aware, not closed-book |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for QLoRA training | Standard QLoRA recipe |
| Embedding model | **Qwen3-Embedding-0.6B** for retrieval index and document routing | Shared across S1/S2/S4/S5/S6 (S2 and S5 via S1 pipeline) |
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

**Retrieval-aware (S1, S2, S5, S6 if triggered):**
`G = F_β(β=2.5)` on page-level grounding. P = deduplicated union of (doc_id, page_number) from **final evidence chunks** (after rerank + compression, not raw candidates).

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training / adapter generation time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Quality claims require discussing cost + grounding trade-off, not quality alone.
- S2 performance reflects supervised adaptation quality, not whole-corpus knowledge.
- S3/S4 performance reflects hypernetwork packaging quality within this benchmark.
- S1 grounding is the reference; S5 must not degrade grounding materially.

See `memory_bank/SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

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
| Documents | 8 PDFs (4 statutes/regs + 4 cases), DIFC legal, 176 pages, ~115K tokens |
| QA pairs | 200 (2 batches × 100) |
| Answer types | free_text: 53, boolean: 48, number: 36, name: 30, names: 17, date: 16 |
| Difficulty | easy: 98, medium: 71, hard: 31 |
| Multi-doc questions | 26 (13%) — all within same-batch doc pairs |
| Unanswerable | 17 (8.5%) — 9 with answer=`null` (expected response `[]`) + 8 free_text negative (expected response: text stating info absent) |

See `memory_bank/SPEC-data.md` for split protocol, schema, leakage rules.

---

## 8. Experiment Phases

| Phase | Goal | Key output |
|-------|------|------------|
| EXP-001 | Data audit + goldset merge + split freeze | `data/goldset/`, `data/splits/` |
| EXP-002 | S1 Classical RAG baseline | Nonparametric baseline metrics |
| EXP-003 | S2 QLoRA feasibility + baseline (3 seeds) | Supervised parametric baseline |
| EXP-004 | S3 Doc-to-LoRA monolithic feasibility + packaging | Hypernetwork baseline |
| EXP-005 | S4 Clustering study + routed Doc-to-LoRA | Routed parametric system |
| EXP-006 | Main comparison S1-S4 on 50 eval | Cross-paradigm results |
| EXP-007 | S5 Hybrid (S5a + S5b) | Hybrid top-line |
| EXP-008 | Locked test + error analysis | Final thesis tables |
| EXP-009 | S6 E2E naive dense RAG ablation (conditional: S5 < S1) | Full pipeline value vs naive baseline |

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
research questions, system inventory, backbone, goldset size/split, Doc-to-LoRA packaging strategy, routing protocol, primary metric definition.
