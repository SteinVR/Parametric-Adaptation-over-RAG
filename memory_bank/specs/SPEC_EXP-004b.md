# SPEC: EXP-004b — S3+R CLM + Retrieval (Headline)

**System:** S3+R | **Class:** Headline | **Wave:** 2 | **Depends on:** EXP-002 (retrieval index), EXP-004 (CLM adapters) | **Blocks:** EXP-006

## Goal

Evaluate the CLM continued pretraining adapter as a retrieval-conditioned generator inside the S1 RAG pipeline. This is the symmetric counterpart to S2+R: same retrieval backbone, same PEFT architecture (QLoRA rank=32, q_proj+v_proj), different training signal (supervised QA vs unsupervised document text).

No training — inference-only experiment using existing artifacts from EXP-004.

## Prerequisites

| Artifact | Source | Path |
|----------|--------|------|
| Qdrant index | EXP-002 | `results/EXP-002/index/` |
| S3 CLM adapters (3 seeds) | EXP-004 | `models/clm/seed_{42,123,777}/` |
| Retrieval pipeline | EXP-002 | `external/pdf_rag_pipeline/` + `src/retrieval/staged.py` |
| Eval split | EXP-001 | `data/splits/split_v1.json` (50 eval questions) |

## Pipeline

1. Run S1 retrieval pipeline on 50 eval questions (reuse staged retrieval from EXP-002 index — embed, search, rerank, compress). Cache retrieval results.
2. For each seed (42, 123, 777):
   a. Load Gemma-2-2b-it (4-bit NF4, `bnb_4bit_use_double_quant=True`) + CLM adapter via PEFT.
   b. For each question: RAG prompt (retrieved context + question) → CLM-adapted model → generate → parse.
   c. Score on 50 eval questions with judge.
3. Aggregate results: mean ± std over 3 seeds.

## Prompt Template

Same RAG prompt as S1 and S2+R (`src/generation/prompt.py::PROMPT_TEMPLATE`):
```
Answer the question using ONLY the provided context. ...

Context:
{context}

Question: {question}
Expected answer format: {answer_type_instruction}
```

The CLM adapter was NOT trained with this prompt — it was trained on raw document text with next-token prediction loss. Whether the adapter nonetheless improves generation from retrieved context is the central question of this experiment.

## Metrics (on 50 eval questions, per seed → aggregate mean ± std)

- Q_main, S_det, S_asst, G (F_β=2.5) — full retrieval-aware metrics
- Peak VRAM during inference
- TTFT, end-to-end latency
- Breakdown by answer_type
- Delta vs S1 (CLM adapter contribution to RAG)
- Delta vs S2+R (CLM vs QLoRA RAFT as adapter source)
- Delta vs S3 control (retrieval contribution to CLM system)

Delta vs S3 is a post-hoc reporting step computed after both EXP-004 and EXP-004b complete.

## Output

- Predictions per seed: `results/EXP-004b/predictions_seed_{42,123,777}.json` (50 entries each)
- Eval report: `results/EXP-004b/eval_report.json`
- Systems metrics: `results/EXP-004b/systems_metrics.json`
- `experiments/EXP-004b_clm_retrieval/REPORT.md`

## Definition of Done

- [ ] S1 retrieval run on 50 eval questions — cached retrieval contexts
- [ ] S3 CLM adapter loaded and applied to backbone (all 3 seeds)
- [ ] Full 50-question inference per seed with CLM adapter + retrieved context
- [ ] Judge scored all free_text questions via OpenAI API (per seed)
- [ ] Q_main, S_det, S_asst, G reported as mean ± std over 3 seeds
- [ ] Delta vs S1 computed and reported
- [ ] Delta vs S2+R computed and reported
- [ ] Delta vs S3 computed and reported
- [ ] Peak VRAM, TTFT, latency logged
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-004b_clm_retrieval/REPORT.md` written
