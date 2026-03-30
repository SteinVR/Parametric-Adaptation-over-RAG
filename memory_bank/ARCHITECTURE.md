# SSOT: Parametric Adaptation for Document-Grounded QA on Consumer Hardware

**Version:** 9.0 | **Updated:** 2026-03-30 | **Shorthand:** `parametric-adaptation-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** On a compact legal corpus under consumer hardware constraints, does parametric adaptation add value on top of a strong RAG baseline, and which adapter source — supervised RAFT-style QLoRA or CLM continued pretraining — is more effective as a retrieval-conditioned generator?

**RQ2 (limits of parametric memory):** How far can pure parametric systems go without retrieval on this benchmark, and where does retrieval remain irreplaceable?

---

## 2. System Structure

### Headline Systems (main results table)

These systems form the core comparison for RQ1.

| ID | System | Role | Retrieval | Adapter source |
|----|--------|------|-----------|----------------|
| S1 | Classical RAG | Strong nonparametric baseline | Yes | None |
| S2+R | QLoRA RAFT + retrieval | Supervised adapter inside RAG | Yes | 150 QA RAFT training |
| S3+R | CLM + retrieval | Continued pretraining adapter inside RAG | Yes | Corpus-wide CLM (8 docs, causal LM) |

**Key comparison:** S2+R vs S3+R — same retrieval backbone, same PEFT architecture (QLoRA rank=32, q_proj+v_proj), different training signal (supervised QA vs unsupervised document text). Delta isolates the value of each adaptation strategy.

### Controls (secondary analysis, RQ2)

Pure parametric systems without retrieval. Measure limits of internal memory, not expected to beat headline systems.

| ID | System | Role | Retrieval |
|----|--------|------|-----------|
| S2 | QLoRA closed-book | Negative control: supervised parametric limit | No |
| S3 | CLM (monolithic) | Control: continued pretraining parametric limit | No |

**Key deltas:**
- Δ(S2+R, S2) = retrieval contribution to supervised system
- Δ(S3+R, S3) = retrieval contribution to CLM system
- Δ(S2+R, S1) = value of supervised adapter on top of RAG
- Δ(S3+R, S1) = value of CLM adapter on top of RAG
- Δ(S2+R, S3+R) = supervised vs supervision-free adapter, same retrieval + same PEFT

### Conditional Ablation

| ID | System | Trigger |
|----|--------|---------|
| S6 | Naive Dense RAG | Only if both S2+R and S3+R < S1 on eval Q_main |

S6 measures combined retrieval engineering + chunk topology contribution.

See `memory_bank/SPEC-systems.md` for detailed system definitions.

---

## 3. Working Hypotheses

- **H1.** At least one of S2+R or S3+R outperforms S1, demonstrating that parametric adaptation adds value on top of RAG.
- **H2.** S2+R outperforms S3+R — supervised RAFT training with gold contexts gives a stronger adapter than supervision-free CLM continued pretraining.
- **H3.** Pure parametric controls (S2, S3) substantially underperform S1 — retrieval is a necessary memory component.
- **H4.** S1 RAG dominates on deterministic lookup (date, number, name) even over augmented systems.
- **H5.** Even if no adapter improves RAG, quantifying the limits of parametric adaptation is a valid result.

---

## 4. Frozen Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Backbone | **Gemma-2-2b-it** | Kept for fairness with completed S1/S2+R experiments; used by all systems |
| Corpus | **8 PDF documents** (DIFC legal, ~115K tokens total) | Frozen before experiments |
| Goldset | **200 human-authored QA pairs** (100 per batch of 4 docs) | `data/goldset/goldset.benchmark.json` |
| Split | **150 S2-train / 50 eval**, stratified by answer_type + difficulty | All systems evaluated on same 50. S2/S2+R train on 150. No CV. |
| S2/S3 variance | **3 random seeds** (42, 123, 777), report mean ± std | Replaces CV for trained systems |
| S2 training format | **Closed-book** (question → answer, no context) | Pure parametric control |
| S2+R training format | **RAFT-style open-book** (question + gold chunks → answer) | Retrieval-conditioned headline system |
| S3 training objective | **Causal LM** on corpus text (next-token prediction) | Supervision-free document exposure |
| S3 PEFT config | **QLoRA rank=32, alpha=32, q_proj+v_proj, 4-bit NF4** | Matches S2+R exactly — isolates training signal difference |
| S3 training data | **8 documents combined** (~115K tokens), no QA pairs | Corpus-wide continued pretraining, no per-doc split |
| S3+R inference | S3 CLM adapter + S1 retrieval pipeline | Symmetric to S2+R: same retrieval, same PEFT arch, different adapter source |
| S5 practical slot | **Reporting-only best practical hybrid** chosen between S2+R and S3+R | No separate training/eval; summary conclusion after headline comparison |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for all QLoRA training and inference | Standard QLoRA recipe |
| Embedding model | **Qwen3-Embedding-0.6B** for retrieval index | Shared across retrieval-aware systems |
| Reranker | **Qwen3-Reranker-0.6B** cross-encoder for S1 retrieval pipeline | Lexical fallback if model fails |
| S1 retrieval stack | **Full hybrid pipeline** from `external/pdf_rag_pipeline/` | Dense+sparse, RRF, reranker, evidence compressor |

---

## 5. Evaluation Protocol (Compact)

**Primary metric (all systems):**
`Q_main = 0.7 × S_det + 0.3 × S_asst`

- `S_det`: deterministic accuracy (number, boolean, name, names, date). Unanswerable: expected `[]`, system returns `[]` → 1.0.
- `S_asst`: LLM-judge score on free_text (5 binary criteria, gpt-5.4-mini)

**Grounding (retrieval-aware systems only: S1, S2+R, S3+R, S6):**
`G = F_β(β=2.5)` on page-level grounding. Not computed for controls S2, S3 (no retrieval).

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Quality claims require discussing cost + grounding trade-off, not quality alone.
- Headline comparison is S1 vs S2+R vs S3+R (same retrieval, different adapter). Delta isolates adapter value.
- Final best practical hybrid is a reporting conclusion over S2+R vs S3+R, not a separately validated system row.
- Controls (S2, S3) measure limits of parametric memory, not expected to win.
- If S1 beats all augmented systems, this is a valid finding: retrieval engineering dominates over adaptation.

See `memory_bank/SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

---

## 6. Terminology Rules

1. S3/S3+R use **"supervision-free continued pretraining"** — the CLM adapter sees only document text, no QA labels.
2. S2 learns from **goldset-style supervision**, not "the whole corpus."
3. No claim of **full corpus internalization** — conclusions bounded to this benchmark, backbone, and hardware.
4. "Unsupervised parametric" acceptable only in tables/diagrams as shorthand for CLM.
5. **S2** = closed-book (control). **S2+R** = RAFT + retrieval (headline). **S3** = CLM (control). **S3+R** = CLM + retrieval (headline).

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
| EXP-004 | Control | S3 CLM continued pretraining (3 seeds) | CLM adapter + pure parametric control |
| EXP-004b | Headline | S3+R CLM + retrieval | CLM retrieval-augmented system |
| EXP-006 | Analysis | Main comparison: headline S1 vs S2+R vs S3+R + controls S2, S3 | Cross-system results table |
| EXP-007 | Analysis | Error analysis + cost/quality/grounding trade-off | Final thesis tables + practical winner call |
| EXP-008 | Ablation | S6 E2E naive dense RAG (conditional: S2+R and S3+R < S1) | Retrieval engineering value |
| EXP-009 | Analysis | Refresh final thesis package with S6 after EXP-008 (conditional) | Final thesis tables including S6 |

---

## 9. Technology Stack

- **Python 3.12**, `uv` for env management, single `.venv` for all experiments
- **DL:** `torch==2.6.0+cu124`, `transformers==4.51.3`, `accelerate==1.6.0`, `peft`, `bitsandbytes`
- **Retrieval:** `sentence-transformers`, `qdrant-client`, `faiss` (S6 ablation only)
- **CLM training:** same QLoRA stack as S2+R — `peft` LoRA + `bitsandbytes` 4-bit + `transformers` Trainer
- **Evaluation:** custom metrics + OpenAI API client (`openai`) for judge
- **Viz:** `matplotlib`

---

## 10. Change Control

Updates to this file required before new experiments if changing:
research questions, system inventory, headline/control classification, backbone, goldset size/split, CLM training strategy, primary metric definition.

### Change Log

| Version | Date | Change |
|---------|------|--------|
| 8.0 | 2026-03-30 | Headline/control split. S2+R and S3+R promoted to headline. |
| 9.0 | 2026-03-30 | **D2L → CLM pivot.** Doc-to-LoRA hypernetwork non-viable for corpus (EXP-004 negative finding, Q_main=0.210). S3 redefined as CLM continued pretraining. S4-doc, S4-cluster, RQ2 (routing study) dropped. CLM PEFT matches S2+R exactly (rank=32, q_proj+v_proj) to isolate training signal difference. |
