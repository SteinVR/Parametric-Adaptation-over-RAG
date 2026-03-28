# SPEC: System Definitions

> Detail spec for S1-S5. Parent: `memory_bank/ARCHITECTURE.md`

---

## S1 — Classical RAG

- **Pipeline:** existing ingestion → chunking → embedding → vector index → retrieval → generation
- **Backbone:** Gemma-2-2b-it (same as all systems)
- **Retriever:** dense embeddings, top-k retrieval, k TBD at EXP-002
- **Prompt:** RAG answer prompt with retrieved chunks as context
- **No model adaptation.** Index is the only offline artifact.
- **Metrics scope:** Q_main + grounding G + systems metrics

## S2 — QLoRA Fine-Tuned (RAFT-style)

- **Training format:** RAFT-style open-book
  - Input: question + gold evidence chunk(s) + optional distractor chunk(s)
  - Output: answer in expected format
- **Adapter config:**
  - Method: QLoRA (4-bit NF4 quantization)
  - Rank: 32
  - Target modules: `q_proj`, `v_proj` (default; `o_proj` only if consistent across all supervised runs)
  - Alpha: sweep [16, 32, 64]
  - Dropout: sweep [0.0, 0.05, 0.1]
  - Optimizer: AdamW / paged AdamW
  - LR sweep: [5e-5, 1e-4, 2e-4, 4e-4]
  - Early stopping on dev Q_main
- **Variance:** 3 random seeds, report mean ± std
- **Interpretation:** S2 learns context-use and answer formatting from human supervision. It does NOT internalize the full corpus — only ~150 supervised examples.
- **Metrics scope:** Q_main + systems metrics (no grounding — no retrieval at inference)

## S3 — Doc-to-LoRA (Monolithic)

- **Hypernetwork:** SakanaAI Doc-to-LoRA, Gemma-2-2b-it checkpoint
- **Target modules:** MLP layers (as defined by hypernetwork)
- **Packaging pipeline:**
  1. Process each of 8 documents individually through hypernetwork (each fits single pass)
  2. Get 8 per-document LoRA adapters
  3. Merge 8 adapters into one global adapter
- **Merge strategy:** TBD at EXP-004 feasibility. Candidates:
  - Simple average of delta weights
  - Weighted average by document token count
  - TIES-Merging if compatible
- **Key concern:** merge of 8 adapters may degrade quality. S3 is explicitly the *monolithic baseline*.
- **Interpretation:** S3 tests whether merged multi-document adapter retains useful knowledge.
- **Metrics scope:** Q_main + systems metrics

## S4 — Cluster-Routed Doc-to-LoRA

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
- **Interpretation:** S4 tests whether document-level sharding + routing preserves more information than global merge (S3)
- **Metrics scope:** Q_main + systems metrics + routing distribution analysis

## S5 — Hybrid: RAG + Best Adapter

- **Adapter selection:** best of S2-S4 on dev Q_main. Ties broken by: latency → variance → simplicity.
- **Two sub-variants:**
  - **S5a:** raw question → retrieval → best adapter generates answer with retrieved context
  - **S5b:** best adapter generates HyDE text → HyDE used as retrieval query → best adapter generates answer with retrieved context
- **HyDE is only evaluated here.** No standalone vanilla-HyDE on S1.
- **Headline S5:** best of S5a/S5b on dev.
- **Metrics scope:** Q_main + grounding G + systems metrics
