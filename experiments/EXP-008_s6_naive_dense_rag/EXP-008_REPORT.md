# Experiment Report: EXP-008 - S6 Naive Dense RAG Ablation

**Date:** 2026-03-30
**Status:** Completed

## 1. Goal

- Quantify the combined contribution of S1's retrieval engineering
  (hybrid search, RRF, reranker, evidence compression) AND chunk topology.
- Delta(S1, S6) = value of full pipeline over naive dense RAG.

## 2. Setup

- Backbone: `google/gemma-2-2b-it` (no adapter)
- Embedding: `Qwen/Qwen3-Embedding-0.6B` (dense only, no BM25)
- Index: FAISS IndexFlatIP (cosine via normalized IP)
- Retrieval: top-5, no reranker, no compression
- Chunking: microchunk only (300 tokens / 50 overlap)

## 3. Results

- Q_main: 0.6335
- S_det: 0.6149
- S_asst: 0.6769
- G (F_β=2.5): 0.4891

## 4. Delta vs S1 (full hybrid RAG)

- Q_main: -0.0090
- S_det: +0.0135
- S_asst: -0.0615
- G: -0.0776

Negative delta = S6 worse than S1 → full pipeline adds value.

## 5. Retrieval Overlap with S1

- Mean Jaccard similarity: 0.3999

## 6. Artifacts

- FAISS index: `results/EXP-008/faiss_index`
- Results: `results/EXP-008/main`
