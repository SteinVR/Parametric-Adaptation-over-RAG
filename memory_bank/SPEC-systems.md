# SPEC: System Definitions

> Detail spec for all systems. Parent: `memory_bank/ARCHITECTURE.md` (v9.0)
> Systems are classified as **Headline** (main results) or **Control** (secondary/limits analysis).

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
- **Note:** The CLM adapter was NOT trained with this prompt — it was trained on raw document text with next-token prediction. Whether the adapter nonetheless improves generation from retrieved context is the central question of this experiment.
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Interpretation:** S3+R measures whether supervision-free document exposure improves RAG generation. Delta vs S1 = CLM adapter value. Symmetric comparison with S2+R (same retrieval, same PEFT, different training data/objective).
- **Metrics scope:** Q_main + grounding G + systems metrics

### S5 — Final Best Practical Hybrid (Reporting-Only Headline Conclusion)

- **Role:** Optional deployment-oriented conclusion after the headline comparison is complete.
- **Definition:** S5 is **not** a separately trained or separately evaluated system. It is the reporting alias for the stronger practical hybrid among **S2+R** and **S3+R** after both headline systems have been evaluated on the frozen benchmark.
- **Selection rule:** choose between S2+R and S3+R using the already reported trade-off package:
  - primary signal: Q_main
  - required qualifiers: grounding G, end-to-end latency, offline packaging cost
  - if the trade-off is ambiguous, report **no single practical winner**
- **Guardrail:** no retuning, no extra inference pass, no choosing on eval-50 and then presenting S5 as an independently validated result. S5 is a **summary label**, not a fourth experimental row.
- **Interpretation:** this is the answer to the practical question "which retrieval+adapter combination would we carry forward on consumer hardware?" It must be discussed only after the direct S2+R vs S3+R comparison is reported.

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
- **Interpretation:** S2 tests the limit of supervised parametric memory in a 2B model. Expected to substantially underperform S1. Δ(S2+R, S2) = retrieval contribution.
- **Results:** frozen from EXP-003b.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

### S3 — CLM Monolithic (Control: Continued Pretraining Parametric Limit)

- **Role:** Negative control. Measures how far continued pretraining on document text can go without retrieval.
- **Training objective:** Causal language modeling (next-token prediction) on all 8 corpus documents combined (~115K tokens)
- **Adapter config:** QLoRA 4-bit NF4, rank 32, alpha 32, dropout 0.05, target q_proj + v_proj (same as S2+R and S3+R)
- **Training hyperparams:** lr 2e-4, batch 1, grad_accum 4, 3 epochs, warmup 0.03, weight_decay 0.01, cosine schedule, paged AdamW 8-bit
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Inference:** no retrieved context — adapter parameters only. Question → adapted model → answer.
- **Interpretation:** S3 tests the limit of CLM parametric memory. Δ(S3+R, S3) = retrieval contribution to CLM system.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

---

## CONDITIONAL ABLATION

### S6 — End-to-End Naive Dense RAG (Conditional)

- **Trigger:** run only if both S2+R and S3+R < S1 on eval Q_main.
- **Pipeline:** maximally simplified end-to-end RAG:
  - Same ingestion and corpus assembly as S1
  - Chunking: **microchunk family only** (300 tokens / 50 overlap)
  - Embedding: Qwen3-Embedding-0.6B dense only (no BM25 sparse)
  - Index: FAISS IndexFlatIP (inner product on normalized vectors = cosine)
  - Retrieval: naive top-5 by cosine similarity, fixed k=5
  - No reranker, no evidence compression, no page-diverse selection
- **Generation:** same Gemma-2-2b-it 4-bit, same prompt template, same answer parser
- **Interpretation:** Delta(S1, S6) measures the **combined** contribution of S1's retrieval engineering (hybrid search, RRF, reranking, evidence compression) AND chunk topology. E2e comparison, not single-variable ablation.
- **Metrics scope:** Q_main + grounding G + systems metrics
