# SPEC: System Definitions

> Detail spec for all systems. Parent: `memory_bank/ARCHITECTURE.md` (v9.2)
> Systems are classified as **Headline** (main results), **Post-hoc** (reported separately), or **Control** (secondary/limits analysis).

---

## HEADLINE SYSTEMS

### S1 — Classical RAG (Headline)

- **Role:** Strong nonparametric baseline
- **Pipeline:** full hybrid retrieval stack from `external/pdf_rag_pipeline/`:
  ingestion (PyMuPDF + table serializer) → corpus assembly → hierarchical chunking (page, section, clause, microchunk, table) → Qdrant hybrid index (dense + sparse) → RRF fusion → cross-encoder reranking → page-diverse evidence compression → generation
- **Backbone:** Gemma-2-2b-it (same as all systems)
- **Embedding:** Qwen3-Embedding-0.6B (prompt_name=document for indexing, prompt_name=query for retrieval)
- **Sparse:** BM25 Okapi (k1=1.5, b=0.75)
- **Index:** Qdrant hybrid (dense cosine + sparse BM25)
- **Fusion:** RRF (k=60, dense_weight=1.0, sparse_weight=1.0)
- **Reranker:** Qwen3-Reranker-0.6B cross-encoder (lexical fallback if model fails)
- **Evidence selection:** page-diverse compressor
- **Prompt:** RAG answer prompt with retrieved evidence as context
- **No model adaptation.** Index + retrieval pipeline are the only offline artifacts.
- **Metrics scope:** Q_main + grounding G + systems metrics

### S2+R — QLoRA RAFT + Retrieval (Headline)

- **Role:** Supervised adapter inside RAG. Tests whether RAFT-style fine-tuning improves RAG quality.
- **Training format:** RAFT-style open-book
  - Input: question + gold page-family chunks (from gold_retrieval pages) + 2 distractor page-family chunks (from non-gold docs)
  - Output: answer in expected format
- **Adapter config:** QLoRA 4-bit NF4, rank 32, alpha 32, dropout 0.05, target q_proj + v_proj, lr 2e-4, paged AdamW 8-bit, 3 epochs. No sweep.
- **Split:** 150 S2-train / 50 eval (all systems evaluated on same 50)
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Inference:** S1 retrieval pipeline → retrieved evidence + RAFT-adapted generator → answer
- **Interpretation:** S2+R measures whether supervised context-aware fine-tuning improves answer extraction from retrieved passages. Delta vs S1 = adapter value. Delta vs S3+R = supervised vs supervision-free adapter.
- **Results:** frozen from EXP-003. See `memory_bank/specs/SPEC_EXP-003.md`.
- **Metrics scope:** Q_main + grounding G + systems metrics

### S3+R — CLM + Retrieval (Headline)

- **Role:** Continued pretraining adapter inside RAG. Tests whether CLM document exposure improves RAG quality without QA supervision.
- **Adapter:** S3 CLM adapter (QLoRA trained on corpus text with causal LM loss)
- **Adapter config:** QLoRA 4-bit NF4, rank 32, alpha 32, dropout 0.05, target q_proj + v_proj. Same PEFT architecture as S2+R — isolates training signal difference.
- **Inference:** S1 retrieval pipeline → retrieved evidence + CLM-adapted generator → answer. Same retrieval backbone as S1 and S2+R.
- **Prompt:** Same RAG prompt template as S1 and S2+R — retrieved context provided, model generates answer.
- **Note:** CLM adapter was trained on raw document text with next-token prediction, not with this RAG prompt.
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Interpretation:** S3+R measures whether supervision-free document exposure improves RAG generation. Delta vs S1 = CLM adapter value.
- **Metrics scope:** Q_main + grounding G + systems metrics

### S5 — Final Best Practical Hybrid (Reporting Alias)

- **Role:** Reporting-only practical winner label after headline comparison.
- **Definition:** S5 is not a separately trained or separately evaluated system.
- **Selection rule:** choose between S2+R and S3+R using Q_main + grounding + latency + offline cost.
- **Guardrail:** if trade-off is ambiguous, report "no single practical winner".
- **Usage:** may be referenced in thesis conclusions, but must not be added as an independent experimental row.

---

## POST-HOC SYSTEMS

### S7 — CLM+QLoRA Adapter Merge (Post-hoc)

- **Role:** Post-hoc merged adapter to test whether CLM and RAFT strengths can be combined without retraining.
- **Source experiment:** `EXP-010` (`experiments/EXP-010_adapter_merge/REPORT.md`)
- **Merge rule:** `merged = alpha * CLM + (1 - alpha) * RAFT`, with **alpha=0.5**.
- **Seed matching:** merge is performed per matching seed pair (42, 123, 777).
- **Inference:** same S1 retrieval pipeline as headline systems.
- **Training:** none (adapter interpolation only).
- **Results:** `Q_main=0.7045 ± 0.0345`, `S_det=0.6790 ± 0.0481`, `S_asst=0.7641 ± 0.0178`, `G=0.5667`.
- **Interpretation caveat:** S7 is post-hoc and eval-only; it is not a separately retrained headline pipeline.
- **Metrics scope:** Q_main + grounding G + systems metrics

---

## CONTROL SYSTEMS

### S2 — QLoRA Closed-Book (Control: Supervised Parametric Limit)

- **Role:** Negative control. Measures how far supervised QA-pair training can go without retrieval.
- **Training format:** closed-book
  - Input: question only (no retrieved context, no gold chunks)
  - Output: answer in expected format
  - Dataset: 150 S2-train QA pairs from goldset
- **Adapter config:** same as S2+R
- **Split/variance:** same as S2+R
- **Inference:** question → adapted model → answer. **No retrieval.**
- **Interpretation:** S2 tests the limit of supervised parametric memory in a 2B model. Δ(S2+R, S2) = retrieval contribution.
- **Results:** frozen from EXP-003b.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

### S3 — CLM Monolithic (Control: Continued Pretraining Parametric Limit)

- **Role:** Negative control. Measures how far continued pretraining on document text can go without retrieval.
- **Training objective:** causal LM on all 8 corpus documents combined (~115K tokens)
- **Adapter config:** QLoRA 4-bit NF4, rank 32, alpha 32, dropout 0.05, target q_proj + v_proj (same as S2+R and S3+R)
- **Training hyperparams:** lr 5e-5, batch 1, grad_accum 4, 5 epochs, warmup 0.1, weight_decay 0.01, cosine schedule, paged AdamW 8-bit
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Inference:** no retrieved context — adapter parameters only. Question → adapted model → answer.
- **Interpretation:** S3 tests the CLM parametric limit. Δ(S3+R, S3) = retrieval contribution to CLM system.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

### S3-legacy (D2L) — Doc-to-LoRA Monolithic (Control: Legacy Diagnostic)

- **Role:** Historical negative control retained for thesis traceability.
- **Source report:** `experiments/EXP-004_d2l_monolithic/REPORT.md`
- **Method note:** used chunk-level D2L generation + internal merge workaround due context limits.
- **Observed result:** `Q_main=0.2100`, `S_det=0.1351`, `S_asst=0.3846`.
- **Interpretation:** quality-nonviable for this benchmark; valid negative finding that motivated D2L→CLM pivot.
- **Status:** legacy/diagnostic only; no further training planned.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

---

## ARCHIVED / DEPRECATED

### S6 — End-to-End Naive Dense RAG (Archived)

- **Status:** archived from active thesis system set
- **Reason:** user requested removal from active narrative to avoid documentation noise
- **History:** executed in EXP-008; kept only as historical record
- **Usage:** do not include in active comparison tables unless explicitly requested
