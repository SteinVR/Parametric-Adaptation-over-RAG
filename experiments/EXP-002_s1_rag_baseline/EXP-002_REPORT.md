# Experiment Report: EXP-002 - S1 Classical RAG Baseline

**Date:** 2026-03-31  
**Status:** Completed

## 1. Goal

- Build the S1 nonparametric baseline: hybrid retrieval over 8-doc corpus + Gemma-2-2b-it generation.
- Freeze shared retrieval/prompt/parser infrastructure for downstream experiments.

## 2. External Pipeline Integration

- Source: `src/rag_pipeline/`
- Pinned commit hash: `8c9a8f70846dcbd697f5e1e0fa180130dc3549b9`

## 3. Runtime Configuration

### PipelineConfig (from `src/rag_pipeline/config.py`)

- `token_chunk_size`: 300
- `token_chunk_overlap`: 50
- `enabled_chunk_families`: `["page", "section", "clause", "microchunk", "table"]`
- `candidate_budget`: 10
- `candidate_multiplier`: 3
- `dense_weight`: 1.0
- `sparse_weight`: 1.0
- `rrf_k`: 60

### Retrieval stage settings (used in `src/retrieval/staged.py`)

- `rerank_budget`: 5
- `evidence_budget`: 3
- `min_rerank_score`: 0.0
- `HybridSearchEngine.rerank_budget` (inline rerank mode): 0 (not used in staged path)

### Generation / eval

- Backbone: `google/gemma-2-2b-it` (`nf4`, 4-bit)
- Judge model: `gpt-5.4-mini` (`reasoning=medium`)
- `Q_main = 0.7 * S_det + 0.3 * S_asst`
- Grounding metric: `F_beta` with `beta=2.5`

## 4. DoD Verification

- [x] RAG index persisted: `results/EXP-002/index/`
- [x] Eval completed: `predictions.json` has 50 entries
- [x] Free-text judged via API: 13/13 have non-null `s_asst` (no all-zero fallback)
- [x] Required artifacts exist: `predictions.json`, `eval_report.json`, `eval_summary.csv`, `systems_metrics.json`
- [x] Aggregate metrics reported: `Q_main`, `S_det`, `S_asst`, `G`
- [x] Breakdown includes all 6 answer types in `eval_summary.csv`
- [x] Malformed rate < 5% (`2.00%`)
- [x] Results tracked in git
- [x] This `REPORT.md` is present

## 5. Aggregate Results (Eval, n=50)

- `Q_main`: 0.6425
- `S_det`: 0.6014
- `S_asst`: 0.7385
- `G (F_beta=2.5)`: 0.5667
- `Malformed rate`: 2.00%

## 6. Systems Metrics

- `TTFT median`: 334.8 ms
- `TTFT p95`: 749.8 ms
- `Latency median`: 479.3 ms
- `Latency p95`: 2089.0 ms
- `Peak VRAM`: 5200.5 MB

## 7. Breakdown by Answer Type

| Answer type | Count | Main score | Grounding F_beta mean | Malformed rate |
| --- | ---: | ---: | ---: | ---: |
| boolean | 12 | S_det = 0.8333 | 0.4963 | 0.0000 |
| number | 7 | S_det = 0.7143 | 0.7202 | 0.1429 |
| name | 8 | S_det = 0.5000 | 0.5406 | 0.0000 |
| names | 5 | S_det = 0.4500 | 0.4893 | 0.0000 |
| date | 5 | S_det = 0.2000 | 0.2693 | 0.0000 |
| free_text | 13 | S_asst = 0.7385 | 0.7092 | 0.0000 |

## 8. Interpretation

- S1 establishes a stable baseline (`Q_main=0.6425`) with strong assistant quality on free-text (`S_asst=0.7385`).
- Deterministic accuracy is uneven: best on `boolean`/`number`, weakest on `date` and multi-name extraction.
- Grounding is moderate (`G=0.5667`), indicating retrieval is often useful but still misses/over-selects pages on harder queries.
- Malformed outputs are low (2%), below the 5% threshold.

## 9. Artifacts

- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/predictions.json`
- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/eval_report.json`
- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/eval_results.json`
- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/eval_summary.csv`
- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/systems_metrics.json`
- `/home/xeliaray/Projects/Term-Paper/results/EXP-002/index/`
