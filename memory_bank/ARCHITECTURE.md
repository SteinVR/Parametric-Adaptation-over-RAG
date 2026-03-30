# SSOT: Parametric Adaptation for Document-Grounded QA on Consumer Hardware

**Version:** 8.0 | **Updated:** 2026-03-30 | **Shorthand:** `parametric-adaptation-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** On a compact legal corpus under consumer hardware constraints, does parametric adaptation add value on top of a strong RAG baseline, and which adapter source — supervised RAFT-style QLoRA or Doc-to-LoRA hypernetwork packaging — is more effective as a retrieval-conditioned generator?

**RQ2 (inner study, D2L packaging):** Does cluster-routed Doc-to-LoRA outperform monolithic Doc-to-LoRA, and is the gain better explained by capacity relief / specialization than by simple adapter count increase?

**RQ3 (limits of parametric memory):** How far can pure parametric systems go without retrieval on this benchmark, and where does retrieval remain irreplaceable?

---

## 2. System Structure

### Headline Systems (main results table)

These systems form the core comparison for RQ1.

| ID | System | Role | Retrieval | Adapter source |
|----|--------|------|-----------|----------------|
| S1 | Classical RAG | Strong nonparametric baseline | Yes | None |
| S2+R | QLoRA RAFT + retrieval | Supervised adapter inside RAG | Yes | 150 QA RAFT training |
| S3+R | Doc-to-LoRA + retrieval | Hypernetwork adapter inside RAG | Yes | 8 docs via D2L → merge |

**Key comparison:** S2+R vs S3+R — same retrieval backbone, different adapter source (supervised QA vs supervision-free document exposure). Delta isolates the value of each adaptation strategy.

### Controls (secondary analysis, RQ3)

Pure parametric systems without retrieval. Measure limits of internal memory, not expected to beat headline systems.

| ID | System | Role | Retrieval |
|----|--------|------|-----------|
| S2 | QLoRA closed-book | Negative control: supervised parametric limit | No |
| S3 | Doc-to-LoRA (monolithic) | Control: hypernetwork parametric limit | No |
| S4-doc | Doc-to-LoRA per-doc routed | Inner study: routing vs merge (RQ2) | No |
| S4-cluster | Doc-to-LoRA cluster-routed | Inner study: partial merge + routing (RQ2) | No |

**Key deltas:**
- Δ(S2+R, S2) = retrieval contribution to supervised system
- Δ(S3+R, S3) = retrieval contribution to hypernetwork system
- Δ(S2+R, S1) = value of supervised adapter on top of RAG
- Δ(S3+R, S1) = value of hypernetwork adapter on top of RAG

### Conditional Ablation

| ID | System | Trigger |
|----|--------|---------|
| S6 | Naive Dense RAG | Only if both S2+R and S3+R < S1 on eval Q_main |

S6 measures combined retrieval engineering + chunk topology contribution.

See `memory_bank/SPEC-systems.md` for detailed system definitions.

---

## 3. Working Hypotheses

- **H1.** At least one of S2+R or S3+R outperforms S1, demonstrating that parametric adaptation adds value on top of RAG.
- **H2.** S2+R outperforms S3+R — supervised RAFT training with gold contexts gives a stronger adapter than supervision-free D2L packaging.
- **H3.** Pure parametric controls (S2, S3, S4) substantially underperform S1 — retrieval is a necessary memory component.
- **H4.** S4-doc (per-doc routing) beats S3 (full merge) on single-doc questions; S3 may win on multi-doc.
- **H4b.** S4-cluster sits between S3 and S4-doc — partial merge + routing is the D2L sweet spot.
- **H5.** S1 RAG dominates on deterministic lookup (date, number, name) even over augmented systems.
- **H6.** Even if no adapter improves RAG, quantifying the limits of parametric adaptation is a valid result.

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
| S2 training format | **Closed-book** (question → answer, no context) | Pure parametric control |
| S2+R training format | **RAFT-style open-book** (question + gold chunks → answer) | Retrieval-conditioned headline system |
| S3+R inference | S3 monolithic adapter + S1 retrieval pipeline | Symmetric to S2+R: same retrieval, different adapter |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for QLoRA training | Standard QLoRA recipe |
| Embedding model | **Qwen3-Embedding-0.6B** for retrieval index and document routing | Shared across retrieval-aware systems |
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

**Grounding (retrieval-aware systems only: S1, S2+R, S3+R, S6):**
`G = F_β(β=2.5)` on page-level grounding. Not computed for controls S2, S3, S4 (no retrieval).

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training / adapter generation time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Quality claims require discussing cost + grounding trade-off, not quality alone.
- Headline comparison is S1 vs S2+R vs S3+R (same retrieval, different adapter). Delta isolates adapter value.
- Controls (S2, S3, S4) measure limits of parametric memory, not expected to win.
- If S1 beats all augmented systems, this is a valid finding: retrieval engineering dominates over adaptation.

See `memory_bank/SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

---

## 6. Terminology Rules

1. In precise prose use **"downstream supervision-free"** not "unsupervised" (hypernetwork is pre-trained upstream).
2. S2 learns from **goldset-style supervision**, not "the whole corpus."
3. No claim of **full corpus internalization** — conclusions bounded to this benchmark, backbone, and hardware.
4. "Unsupervised parametric" acceptable only in tables/diagrams as shorthand.
5. **S2** = closed-book (control). **S2+R** = RAFT + retrieval (headline). **S3** = D2L mono (control). **S3+R** = D2L + retrieval (headline).

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

| Phase | Role | Goal | Key output |
|-------|------|------|------------|
| EXP-001 | — | Data audit + goldset merge + split freeze | `data/goldset/`, `data/splits/` |
| EXP-002 | Headline | S1 Classical RAG baseline | Nonparametric baseline metrics |
| EXP-003 | Headline | S2+R QLoRA RAFT + retrieval (3 seeds) | Supervised retrieval-augmented baseline |
| EXP-003b | Control | S2 QLoRA closed-book (3 seeds) | Supervised parametric limit |
| EXP-004 | Control + prep | S3 Doc-to-LoRA monolithic packaging + feasibility | D2L adapters + pure parametric control |
| EXP-004b | Headline | S3+R Doc-to-LoRA + retrieval | Hypernetwork retrieval-augmented system |
| EXP-005 | Control | S4 Clustering study + routed Doc-to-LoRA | Routing vs merge inner study (RQ2) |
| EXP-006 | Analysis | Main comparison: headline S1 vs S2+R vs S3+R + all controls | Cross-system results table |
| EXP-007 | Analysis | Error analysis + cost/quality/grounding trade-off | Final thesis tables |
| EXP-008 | Ablation | S6 E2E naive dense RAG (conditional: S2+R and S3+R < S1) | Retrieval engineering value |

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
research questions, system inventory, headline/control classification, backbone, goldset size/split, Doc-to-LoRA packaging strategy, routing protocol, primary metric definition.
