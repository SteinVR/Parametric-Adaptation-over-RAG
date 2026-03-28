# SPEC: EXP-005a — S4-doc Per-Document Routing

**System:** S4-doc | **Wave:** 3 | **Depends on:** EXP-004 (8 per-doc adapters) | **Blocks:** EXP-006

## Goal

Evaluate per-document routing with zero merge. At inference, route each question to the single most relevant document adapter.

## Pipeline

### Step 1: Document embeddings
- For each of 8 docs: embed all chunks via Qwen3-Embedding-0.6B → mean-pool → 1 document embedding per doc
- Store 8 document embeddings as routing index

### Step 2: Router
- Embed question via Qwen3-Embedding-0.6B
- Cosine similarity to 8 document embeddings
- Hard top-1: select the doc with highest similarity
- Log all 8 similarity scores per question

### Step 3: Generation
- Load selected doc's adapter (from EXP-004)
- No retrieved context — adapter parameters only
- Same no-retrieval prompt template as EXP-004
- Same answer parser

### Step 4: Evaluation on 50 eval questions

## Analysis

- **Routing accuracy:** % of questions where router selected a document that appears in gold_retrieval.
  - **Single-doc questions:** correct if routed doc = gold doc
  - **Multi-doc questions:** correct if routed doc ∈ gold docs (any hit counts)
  - **Unanswerable (gold_retrieval=[]):** excluded from routing accuracy (no correct route exists)
- **Single-doc Q_main vs multi-doc Q_main:** quantify routing penalty on multi-doc questions
- **Routing confusion matrix:** 8×8 heatmap, **single-doc subset only** (multi-doc has no single-label truth)
- **Confidence distribution:** histogram of max cosine similarity scores

## Metrics

- Q_main, S_det, S_asst
- Routing accuracy (overall, single-doc, multi-doc)
- TTFT, end-to-end latency (routing overhead: embedding + cosine)
- Peak VRAM at inference
- Breakdown by answer_type

## Output

- Routing logs: `results/EXP-005a/routing_log.csv` (question_id, gold_docs, routed_doc, similarities, correct)
- `experiments/EXP-005a/REPORT.md`
