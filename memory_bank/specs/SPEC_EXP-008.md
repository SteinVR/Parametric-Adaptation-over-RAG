# SPEC: EXP-008 — S6 End-to-End Naive Dense RAG Ablation (Conditional)

**System:** S6 | **Wave:** 5 (conditional) | **Depends on:** EXP-006 (trigger evaluation) | **Blocks:** Nothing (terminal ablation)

## Trigger Condition

Run **only if** both S2+R AND S3+R < S1 on eval Q_main (from EXP-006). If either S2+R or S3+R ≥ S1, skip this experiment entirely and document the skip reason.

## Goal

End-to-end ablation: quantify the combined contribution of S1's retrieval engineering (hybrid search, RRF fusion, cross-encoder reranking, page-diverse evidence compression) AND its chunk topology (5 families vs microchunk-only). Delta(S1, S6) measures the full gap between a maximally simplified dense RAG and the production-grade hybrid RAG. This is deliberately an e2e comparison, not a single-variable ablation.

## Pipeline

1. **Ingestion + corpus:** reuse from EXP-002 (same `CanonicalPageRecord` set)
2. **Chunking:** `indexing/chunking.py` with `enabled_chunk_families={"microchunk"}` only. Same 300-token / 50-overlap parameters. No page, section, clause, or table chunks.
3. **Embedding:** Qwen3-Embedding-0.6B dense only (prompt_name=document for indexing, prompt_name=query at retrieval). No BM25 sparse.
4. **Index:** FAISS IndexFlatIP over L2-normalized dense vectors (inner product = cosine on normalized vectors). Each indexed chunk carries metadata: `(doc_id, page_number)` via side-loaded dict keyed by FAISS row index.
5. **Retrieval:** top-5 chunks by cosine similarity, fixed k=5. No reranker, no RRF, no evidence compression.
6. **Generation:** Gemma-2-2b-it (4-bit NF4), same prompt template and answer parser as EXP-002.
7. **Scoring:** Q_main, G, systems metrics on same 50 eval questions.

## Frozen Decisions

| Decision | Value |
|----------|-------|
| Embedding model | Qwen3-Embedding-0.6B (dense only) |
| Index type | FAISS IndexFlatIP |
| Top-k | 5 |
| Chunking | microchunk only (300 tokens / 50 overlap) |
| Reranker | None |
| Evidence compression | None |
| Generation model | Gemma-2-2b-it 4-bit NF4 |
| Prompt template | Same as EXP-002 |
| Answer parser | Same as EXP-002 |

All parameters above must be defined in a local `S6Config` dataclass (not inline magic numbers).

## Analysis

- **Primary:** Delta(S1, S6) on Q_main — quantifies combined value of retrieval engineering + chunk topology
- **Grounding:** Delta(S1, S6) on G — does the full pipeline improve page-level grounding?
- **Per answer_type:** which question types benefit most from the full pipeline?
- **Retrieval overlap:** Jaccard similarity between S1 and S6 retrieved page sets per question
- **Latency:** S6 should be faster (no reranker, simpler search)
- **Note:** Delta confounds two factors (retrieval stack + chunk families). If Delta is large and further isolation is needed, this becomes a backlog item (single-variable ablations).

## Metrics

- Q_main, S_det, S_asst, G (F_β=2.5)
- Delta vs S1 on all metrics
- TTFT, end-to-end latency, peak VRAM
- Breakdown by answer_type

## Output

- FAISS index: `results/EXP-008/faiss_index/`
- Eval results: `results/EXP-008/`
- `experiments/EXP-008/REPORT.md`

## Definition of Done

- [ ] Trigger condition verified: both S2+R AND S3+R < S1 on Q_main (otherwise skip + document)
- [ ] FAISS index built with microchunk-only chunking — `results/EXP-008/faiss_index/`
- [ ] Full 50-question eval — `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API
- [ ] Q_main, S_det, S_asst, G reported
- [ ] Delta vs S1 computed on all metrics
- [ ] Retrieval overlap: Jaccard between S1 and S6 page sets per question
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-008/REPORT.md` written
