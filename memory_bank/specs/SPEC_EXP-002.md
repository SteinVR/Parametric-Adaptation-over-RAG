# SPEC: EXP-002 — S1 Classical RAG Baseline

**System:** S1 | **Wave:** 1 | **Depends on:** EXP-001 | **Blocks:** EXP-003, EXP-006, EXP-007

## Goal

Build working RAG pipeline on Gemma-2-2b-it over 8-doc corpus using full hybrid retrieval stack from external project (`external/pdf_rag_pipeline/`). Establish nonparametric baseline. Freeze shared infrastructure (prompt template, answer parser) used by all downstream experiments.

## Pipeline

1. **Text extraction:** PyMuPDF via `ingestion/pdf_parser.py` — all 8 docs, page-level text with page metadata, table candidate extraction
2. **Table serialization:** `ingestion/table_serializer.py` — cross-page merge, row-level serialization into self-contained text blocks
3. **Corpus assembly:** `ingestion/corpus_builder.py` — `CanonicalPageRecord` per page with TEXT + TABLE `ContentBlock`s
4. **Chunking:** hierarchical structure-aware via `indexing/chunking.py` — 5 families: page, section, clause, microchunk (300 tokens / 50 overlap), table. Each `IndexChunk` carries `(doc_id, page_span, parent_page_numbers)` for grounding computation.
5. **Embedding:** `Qwen3-Embedding-0.6B` dense (`indexing/embeddings.py`, prompt_name=document for indexing, prompt_name=query for retrieval) + BM25 sparse (`BM25SparseEncoder`)
6. **Index:** Qdrant hybrid (dense cosine + sparse BM25) via `indexing/qdrant_store.py`. Each point stores full chunk payload with metadata.
7. **Retrieval:** `retrieval/hybrid_search.py` — dense + sparse with RRF fusion (k=60, equal weights) → `retrieval/reranker.py` — Qwen3-Reranker-0.6B cross-encoder (lexical fallback) → `retrieval/evidence_compressor.py` — page-diverse evidence selection → `retrieval/page_lifter.py` — page-level grounding references
8. **Generation:** Gemma-2-2b-it (4-bit NF4 via bitsandbytes), prompt template below
9. **Answer parsing:** rule-based parser per answer_type (see below)
10. **Scoring:** Q_main, G, systems metrics on 50 eval questions

## External Pipeline Integration

- **Source:** `external/pdf_rag_pipeline/` (imported from separate project)
- **At experiment start:** pin source commit hash in REPORT.md
- **Config parameters to document:**
  - From `PipelineConfig`: `token_chunk_size`, `token_chunk_overlap`, `enabled_chunk_families`, `candidate_budget`, `candidate_multiplier`, `dense_weight`, `sparse_weight`, `rrf_k`
  - From `RetrievalService` init: `rerank_budget`, `evidence_budget`, `min_rerank_score`
  - From `HybridSearchEngine` init: `rerank_budget` (if inline reranking used)

## Prompt Template (frozen for all systems)

```
<start_of_turn>user
Answer the question using ONLY the provided context. If the information is not in the context, respond with [] for factual questions or state that the information is not available for free-text questions.

Context:
{retrieved_chunks}

Question: {question}
Expected answer format: {answer_type_instruction}
<end_of_turn>
<start_of_turn>model
```

`answer_type_instruction` per type:
- `boolean` → "Answer true or false."
- `number` → "Answer with a number only."
- `name` → "Answer with the exact name."
- `names` → "Answer with a JSON array of names, e.g. [\"Name1\", \"Name2\"]."
- `date` → "Answer with a date in YYYY-MM-DD format."
- `free_text` → "Answer in 1-3 sentences (max 280 characters)."

## Answer Parser

Per answer_type:
- `boolean`: search for "true"/"false" in output, case-insensitive
- `number`: extract first numeric value (int or float)
- `name`: strip whitespace, take first line
- `names`: try JSON parse; fallback: split by commas/newlines
- `date`: regex `\d{4}-\d{2}-\d{2}`, take first match
- `free_text`: take full output, truncate at 280 chars
- If parsing fails → `_malformed_` (distinct from intentional `[]` abstention; scored as 0)

## Frozen Decisions

| Decision | Value |
|----------|-------|
| Embedding model | Qwen3-Embedding-0.6B (prompt_name: document/query) |
| Sparse encoder | BM25 Okapi (k1=1.5, b=0.75) |
| Index type | Qdrant hybrid (dense cosine + sparse BM25) |
| Search fusion | RRF (k=60, dense_weight=1.0, sparse_weight=1.0) |
| Reranker | Qwen3-Reranker-0.6B cross-encoder (lexical fallback) |
| Evidence selection | Page-diverse compressor |
| Chunking families | page, section, clause, microchunk, table |
| Chunk size | 300 tokens / 50 overlap (microchunk) |
| Generation model | Gemma-2-2b-it 4-bit NF4 |
| Prompt template | As above |
| Answer parser | Rule-based per type |

## VRAM Budget (sequential model loading)

| Stage | Model | ~VRAM |
|-------|-------|-------|
| Embedding (index + query) | Qwen3-Embedding-0.6B | ~1.2 GB |
| Reranking | Qwen3-Reranker-0.6B | ~1.2 GB |
| Generation | Gemma-2-2b-it 4-bit | ~1.5 GB |
| **Peak** (if sequential) | | **~3-4 GB** |

Models loaded/unloaded sequentially to stay within 8 GB RTX 4060 budget.

## Metrics

- Q_main = 0.7 × S_det + 0.3 × S_asst
- Grounding G (F_β=2.5) on page-level (doc_id, page_number) from final evidence set (post-compression `page_references`)
- S_det breakdown by answer_type
- S_asst via gpt-5.4-mini judge
- TTFT (median, p95), end-to-end latency, peak VRAM
- Malformed output rate
- Retrieval diagnostics: candidate count, rerank count, evidence count per query

## Output

- `src/retrieval/` — integration wrappers around external pipeline
- `src/generation/` — Gemma loader, prompt builder, answer parser
- `src/evaluation/` — Q_main scorer, grounding scorer, judge client
- Eval baseline metrics CSV in `results/EXP-002/`
- `experiments/EXP-002_s1_rag_baseline/REPORT.md`
