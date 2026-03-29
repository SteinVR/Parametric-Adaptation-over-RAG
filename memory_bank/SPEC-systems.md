# SPEC: System Definitions

> Detail spec for S1-S6. Parent: `memory_bank/ARCHITECTURE.md`

---

## S1 — Classical RAG

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

## S2 — QLoRA Fine-Tuned (RAFT-style)

- **Training format:** RAFT-style open-book
  - Input: question + gold page-family chunks (from gold_retrieval pages) + 2 distractor page-family chunks (from non-gold docs)
  - Output: answer in expected format
- **Adapter config:** QLoRA 4-bit NF4, rank 32, alpha 32, dropout 0.05, target q_proj + v_proj, lr 2e-4, paged AdamW 8-bit, 3 epochs. No sweep.
- **Split:** 150 S2-train / 50 eval (all systems evaluated on same 50)
- **Variance:** 3 random seeds (42, 123, 777), report mean ± std on 50 eval
- **Inference:** S2 uses the same retrieval pipeline as S1 — retrieved evidence + adapted generator → answer. RAFT-style: trained with context, infers with context.
- **Interpretation:** S2 is "RAG with supervised adapter." Learns context-use and answer formatting. Does NOT internalize full corpus.
- **Metrics scope:** Q_main + grounding G + systems metrics

## S3 — Doc-to-LoRA (Monolithic)

- **Hypernetwork:** SakanaAI Doc-to-LoRA, Gemma-2-2b-it checkpoint
- **Target modules:** MLP layers (as defined by hypernetwork)
- **Packaging pipeline:**
  1. Process each of 8 documents individually through hypernetwork (each fits single pass)
  2. Get 8 per-document LoRA adapters
  3. Merge 8 adapters into one global adapter
- **Merge strategy:** simple average of delta weights (frozen).
- **Inference:** no retrieved context — adapter parameters only. Question → adapted model → answer.
- **Key concern:** merge of 8 adapters may degrade quality. S3 is explicitly the *monolithic baseline*.
- **Interpretation:** S3 tests whether merged multi-document adapter retains useful knowledge.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval)

## S4-doc — Per-Document Routed Doc-to-LoRA

- **Adapters:** 8 per-document adapters from EXP-004 (no merge)
- **Router:** embed question → cosine similarity to 8 document embeddings → hard top-1
- **Inference:** no retrieved context — routed adapter parameters only. Question → router → selected doc adapter → answer.
- **Interpretation:** maximum specialization, zero merge. Expected to excel on single-doc questions, fail on multi-doc.
- **Metrics scope:** Q_main + systems metrics + routing accuracy analysis (no grounding — no retrieval)

## S4-cluster — Cluster-Routed Doc-to-LoRA

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
- **Inference:** no retrieved context — routed adapter parameters only. Question → router → selected adapter → answer.
- **Interpretation:** S4 tests whether document-level sharding + routing preserves more information than global merge (S3)
- **Metrics scope:** Q_main + systems metrics + routing distribution analysis (no grounding — no retrieval)

## S5 — Hybrid: RAG + Best Adapter

- **Adapter selection:** best **single** adapter from S2 or S3 on eval Q_main. S4-doc/S4-cluster are multi-adapter routed systems — not eligible. Ties broken by: latency → simplicity.
- **Two sub-variants:**
  - **S5a:** raw question → S1 retrieval pipeline → best adapter generates answer with retrieved context
  - **S5b:** best adapter generates HyDE text → HyDE used as retrieval query → best adapter generates answer with retrieved context
- **HyDE is only evaluated here.** No standalone vanilla-HyDE on S1.
- **Headline S5:** best of S5a/S5b on eval.
- **Metrics scope:** Q_main + grounding G + systems metrics

## S6 — End-to-End Naive Dense RAG Ablation (Conditional)

- **Trigger:** run only if headline S5 (best of S5a/S5b, per EXP-007) < S1 on eval Q_main.
- **Pipeline:** maximally simplified end-to-end RAG:
  - Same ingestion and corpus assembly as S1
  - Chunking: **microchunk family only** (300 tokens / 50 overlap)
  - Embedding: Qwen3-Embedding-0.6B dense only (no BM25 sparse)
  - Index: FAISS IndexFlatIP (inner product on normalized vectors = cosine)
  - Retrieval: naive top-5 by cosine similarity, fixed k=5
  - No reranker, no evidence compression, no page-diverse selection
- **Generation:** same Gemma-2-2b-it 4-bit, same prompt template, same answer parser
- **Interpretation:** end-to-end naive RAG ablation. Delta(S1, S6) measures the **combined** contribution of S1's retrieval engineering (hybrid search, RRF, reranking, evidence compression) AND chunk topology (5 families vs microchunk-only). This is deliberately an e2e comparison, not a single-variable ablation.
- **Metrics scope:** Q_main + grounding G + systems metrics
