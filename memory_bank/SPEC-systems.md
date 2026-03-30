# SPEC: System Definitions

> Detail spec for all systems. Parent: `memory_bank/ARCHITECTURE.md` (v8.0)
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

### S3+R — Doc-to-LoRA + Retrieval (Headline)

- **Role:** Hypernetwork adapter inside RAG. Tests whether D2L packaging improves RAG quality without downstream QA supervision.
- **Adapter:** S3 monolithic adapter (8 per-doc D2L adapters merged into one)
- **Inference:** S1 retrieval pipeline → retrieved evidence + D2L-adapted generator → answer. Same retrieval backbone as S1 and S2+R.
- **Prompt:** Same RAG prompt template as S1 and S2+R — retrieved context provided, model generates answer.
- **Interpretation:** S3+R measures whether document-level hypernetwork packaging improves RAG generation, even without QA supervision. Delta vs S1 = D2L adapter value. Symmetric comparison with S2+R (same retrieval, different adapter source).
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
- **Interpretation:** S2 tests the limit of supervised parametric memory in a 2B model. Expected to substantially underperform S1. Δ(S2+R, S2) = retrieval contribution.
- **Results:** frozen from EXP-003b.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

### S3 — Doc-to-LoRA Monolithic (Control: Hypernetwork Parametric Limit)

- **Hypernetwork:** SakanaAI Doc-to-LoRA, Gemma-2-2b-it checkpoint
- **Target modules:** MLP layers (as defined by hypernetwork)
- **Packaging pipeline:**
  1. Process each of 8 documents individually through hypernetwork (each fits single pass)
  2. Get 8 per-document LoRA adapters
  3. Merge 8 adapters into one global adapter
- **Merge strategy:** simple average of delta weights (frozen).
- **Inference:** no retrieved context — adapter parameters only. Question → adapted model → answer.
- **Key concern:** merge of 8 adapters may degrade quality. S3 is explicitly the *monolithic baseline*.
- **Interpretation:** S3 tests the limit of hypernetwork parametric memory. Δ(S3+R, S3) = retrieval contribution.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

### S4-doc — Per-Document Routed Doc-to-LoRA (Control: RQ2 Inner Study)

- **Adapters:** 8 per-document adapters from EXP-004 (no merge)
- **Router:** embed question → cosine similarity to 8 document embeddings → hard top-1
- **Inference:** no retrieved context — routed adapter parameters only. Question → router → selected doc adapter → answer.
- **Interpretation:** maximum specialization, zero merge. Expected to excel on single-doc questions, fail on multi-doc.
- **Metrics scope:** Q_main + systems metrics + routing accuracy analysis (no grounding — no retrieval)

### S4-cluster — Cluster-Routed Doc-to-LoRA (Control: RQ2 Inner Study)

- **Clustering:**
  - Granularity: **document-level** (8 docs → 4 clusters of ~2 docs each)
  - Natural clustering: statutes / regulations / first-instance cases / appeal cases
  - Alternative: embedding-based k-means (k=4) for data-driven clusters
  - Diagnostics: cluster balance, semantic coherence, silhouette (diagnostic only)
- **Per-cluster adapter generation:**
  - For each cluster: process cluster documents through D2L → merge per-cluster adapters
  - Merge of 2 adapters per cluster (much more tractable than S3's 8-way merge)
- **Router:**
  - Embed user question with same embedding model
  - Cosine similarity to cluster centroids
  - Activate top-1 cluster adapter
  - Log similarity scores for all centroids
- **Fallback:** if routing confidence is very low (all similarities near-equal), log warning but still pick top-1
- **No learned router** in core scope
- **Inference:** no retrieved context — routed adapter parameters only.
- **Interpretation:** S4 tests whether document-level sharding + routing preserves more information than global merge (S3)
- **Metrics scope:** Q_main + systems metrics + routing distribution analysis (no grounding — no retrieval)

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
