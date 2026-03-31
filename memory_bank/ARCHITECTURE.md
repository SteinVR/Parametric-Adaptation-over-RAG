# SSOT: Parametric Adaptation for Document-Grounded QA on Consumer Hardware

**Version:** 9.2 | **Updated:** 2026-03-31 | **Shorthand:** `parametric-adaptation-consumer-hw`

> Authoritative source for scope, systems, metrics, and frozen decisions.
> Detail specs live in `memory_bank/SPEC-*.md`.
> Deviations require updating this file first.

---

## 1. Research Questions

**RQ1 (main):** On a compact legal corpus under consumer hardware constraints, does parametric adaptation add value on top of a strong RAG baseline, and which adapter source — supervised RAFT-style QLoRA or CLM continued pretraining — is more effective as a retrieval-conditioned generator?

**RQ2 (limits of parametric memory):** How far can pure parametric systems go without retrieval on this benchmark, and where does retrieval remain irreplaceable?

---

## 2. System Structure

### Headline Systems (main thesis comparison)

These systems form the core comparison for RQ1.

| ID | System | Role | Retrieval | Adapter source |
|----|--------|------|-----------|----------------|
| S1 | Classical RAG | Strong nonparametric baseline | Yes | None |
| S2+R | QLoRA RAFT + retrieval | Supervised adapter inside RAG | Yes | 150 QA RAFT training |
| S3+R | CLM + retrieval | Continued pretraining adapter inside RAG | Yes | Corpus-wide CLM (8 docs, causal LM) |

**Key comparison:** S2+R vs S3+R — same retrieval backbone, same PEFT architecture (QLoRA rank=32, q_proj+v_proj), different training signal (supervised QA vs unsupervised document text). Delta isolates the value of each adaptation strategy.

### Post-hoc System (reported separately)

| ID | System | Role | Retrieval |
|----|--------|------|-----------|
| S7 | CLM+QLoRA adapter merge | Post-hoc merged adapter (no retraining) | Yes |

S7 comes from EXP-010 (`alpha=0.5`, seed-matched linear interpolation), evaluated with the same S1 retrieval stack. It is reported as post-hoc champion, not as a new training pipeline.

### Controls (secondary analysis, RQ2)

Pure parametric systems without retrieval. Measure limits of internal memory, not expected to beat headline systems.

| ID | System | Role | Retrieval |
|----|--------|------|-----------|
| S2 | QLoRA closed-book | Negative control: supervised parametric limit | No |
| S3 | CLM (monolithic) | Control: continued pretraining parametric limit | No |
| S3-legacy (D2L) | Doc-to-LoRA monolithic | Legacy diagnostic control (negative finding) | No |

`S3-legacy (D2L)` is retained as a valid negative finding from `experiments/EXP-004_d2l_monolithic/REPORT.md` (`Q_main=0.2100`) and stays in documentation as a historical control.

### Archived / Not Active in Thesis Narrative

| ID | System | Status | Note |
|----|--------|--------|------|
| S6 | Naive Dense RAG | Archived | Run in EXP-008 by user request; excluded from active thesis system set |

See `memory_bank/SPEC-systems.md` for detailed system definitions.

---

## 3. Working Hypotheses

- **H1.** At least one of S2+R or S3+R outperforms S1, demonstrating that parametric adaptation adds value on top of RAG.
- **H2.** S2+R outperforms S3+R — supervised RAFT training with gold contexts gives a stronger adapter than supervision-free CLM continued pretraining.
- **H3.** Pure parametric controls (S2, S3, S3-legacy D2L) substantially underperform S1 — retrieval is a necessary memory component.
- **H4.** S1 RAG dominates on deterministic lookup (date, number, name) even over augmented systems.
- **H5.** Even if no adapter improves RAG, quantifying the limits of parametric adaptation is a valid result.
- **H6 (post-hoc).** Adapter interpolation (S7) may combine complementary strengths of S2+R and S3+R without additional training.

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
| S7 merge rule | **alpha=0.5** linear interpolation of CLM and RAFT adapters (same seed) | EXP-010 post-hoc eval-only; no retraining |
| S5 practical slot | **Reporting-only best practical hybrid** chosen between S2+R and S3+R | No separate training/eval; summary conclusion after headline comparison |
| S6 policy | **Archived from active thesis system set** | Avoids noise in core narrative while preserving historical trace |
| Judge model | **gpt-5.4-mini** (OpenAI API, medium reasoning), version-pinned | External, not self-judging |
| Hardware | **RTX 4060 8GB VRAM, 32GB RAM** | Hard constraint; QLoRA 4-bit default |
| Quantization | **4-bit NF4** for all QLoRA training and inference | Standard QLoRA recipe |
| Embedding model | **Qwen3-Embedding-0.6B** for retrieval index | Shared across retrieval-aware systems |
| Reranker | **Qwen3-Reranker-0.6B** cross-encoder for S1 retrieval pipeline | Lexical fallback if model fails |
| S1 retrieval stack | **Full hybrid pipeline** from `external/pdf_rag_pipeline/` | Dense+sparse, RRF, reranker, evidence compressor |

---

## 5. Evaluation Protocol (Compact)

**Primary metric (all active systems):**
`Q_main = 0.7 × S_det + 0.3 × S_asst`

- `S_det`: deterministic accuracy (number, boolean, name, names, date). Unanswerable: expected `[]`, system returns `[]` → 1.0.
- `S_asst`: LLM-judge score on free_text (5 binary criteria, gpt-5.4-mini)

**Grounding (retrieval-aware systems only):**
`G = F_β(β=2.5)` on page-level grounding for S1, S2+R, S3+R, S7.

Grounding is not computed for controls S2, S3, S3-legacy (no retrieval).

**Systems metrics (all):**
TTFT, end-to-end latency, peak VRAM, offline packaging cost (index build / training time)

**Breakdowns:** every metric reported aggregate + by answer_type.

**Interpretation guidelines** (not targets):
- Headline comparison is S1 vs S2+R vs S3+R.
- S7 is reported as post-hoc adapter-merge result, not a retrained fourth headline system.
- Final best practical hybrid call remains a reporting conclusion over S2+R vs S3+R.
- Controls (S2, S3, S3-legacy D2L) measure limits of parametric memory.
- If S1 beats all augmented systems, this is still a valid finding.

See `memory_bank/SPEC-evaluation.md` for scoring rules, judge rubric, and reporting format.

---

## 6. Terminology Rules

1. S3/S3+R use **"supervision-free continued pretraining"** — CLM adapter sees only document text, no QA labels.
2. S2 learns from **goldset-style supervision**, not "the whole corpus".
3. No claim of **full corpus internalization** — conclusions bounded to this benchmark, backbone, and hardware.
4. "Unsupervised parametric" is acceptable only as CLM shorthand in tables/diagrams.
5. **S3 = CLM control.** **S3-legacy (D2L) = historical negative control from EXP-004 D2L.**
6. **S7 = post-hoc CLM+RAFT merge**, eval-only, no retraining.

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
| EXP-001 | Foundation | Data audit + goldset merge + split freeze | `data/goldset/`, `data/splits/` |
| EXP-002 | Headline | S1 Classical RAG baseline | Nonparametric baseline metrics |
| EXP-003 | Headline | S2+R QLoRA RAFT + retrieval (3 seeds) | Supervised retrieval-augmented baseline |
| EXP-003b | Control | S2 QLoRA closed-book (3 seeds) | Supervised parametric limit |
| EXP-004 D2L | Legacy control | Doc-to-LoRA monolithic feasibility | Negative finding (`Q_main=0.2100`) |
| EXP-004 CLM | Control | S3 CLM continued pretraining (3 seeds) | CLM pure-parametric control |
| EXP-004b | Headline | S3+R CLM + retrieval | CLM retrieval-augmented system |
| EXP-006 | Analysis | Main comparison with headline + controls (+ post-hoc S7 row) | Cross-system results table |
| EXP-007 | Analysis | Error analysis + cost/quality/grounding trade-off | Final thesis tables + practical winner call |
| EXP-008 | Archived | S6 e2e naive dense RAG | Historical ablation result (not active thesis set) |
| EXP-009 | Archived | Conditional refresh with S6 | Archived/out-of-scope for thesis narrative |
| EXP-010 | Post-hoc | CLM+RAFT adapter merge (S7), eval-only | Post-hoc champion row (`Q_main=0.7045±0.0345`) |

---

## 9. Technology Stack

- **Python 3.12**, `uv` for env management, single `.venv` for all experiments
- **DL:** `torch==2.6.0+cu124`, `transformers==4.51.3`, `accelerate==1.6.0`, `peft`, `bitsandbytes`
- **Retrieval:** `sentence-transformers`, `qdrant-client`
- **Training:** QLoRA 4-bit pipeline for S2/S3 families
- **Evaluation:** custom metrics + OpenAI API client (`openai`) for judge
- **Viz:** `matplotlib`

---

## 10. Change Control

Updates to this file are required before new experiments if changing:
research questions, system inventory, headline/control classification, backbone, goldset size/split, training strategy, primary metric definition.

### Change Log

| Version | Date | Change |
|---------|------|--------|
| 8.0 | 2026-03-30 | Headline/control split. S2+R and S3+R promoted to headline. |
| 9.0 | 2026-03-30 | D2L → CLM pivot. S4/old RQ2 dropped. |
| 9.1 | 2026-03-31 | Added post-hoc S7 (EXP-010) to project narrative and reporting stack. |
| 9.2 | 2026-03-31 | S6 archived from active thesis set; restored S3-legacy (D2L) as documented negative control; aligned EXP-006/007/010 system taxonomy. |
