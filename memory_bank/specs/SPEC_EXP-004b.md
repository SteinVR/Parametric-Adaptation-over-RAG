# SPEC: EXP-004b — S3+R Doc-to-LoRA + Retrieval (Headline)

**System:** S3+R | **Class:** Headline | **Wave:** 2 | **Depends on:** EXP-002 (retrieval index), EXP-004 (monolithic adapter) | **Blocks:** EXP-006

## Goal

Evaluate the Doc-to-LoRA monolithic adapter as a retrieval-conditioned generator inside the S1 RAG pipeline. This is the symmetric counterpart to S2+R: same retrieval backbone, different adapter source (supervision-free D2L vs supervised RAFT).

No training — inference-only experiment using existing artifacts.

## Prerequisites

| Artifact | Source | Path |
|----------|--------|------|
| Qdrant index | EXP-002 | `results/EXP-002/index/` |
| S3 monolithic adapter | EXP-004 | `models/d2l/monolithic/` |
| Retrieval pipeline | EXP-002 | `external/pdf_rag_pipeline/` + `src/retrieval/staged.py` |
| Eval split | EXP-001 | `data/splits/split_v1.json` (50 eval questions) |

## Pipeline

1. Run S1 retrieval pipeline on 50 eval questions (reuse staged retrieval from EXP-002 index — embed, search, rerank, compress). Cache retrieval results.
2. Load Gemma-2-2b-it (4-bit NF4, `bnb_4bit_use_double_quant=True`) + S3 monolithic D2L adapter via PEFT.
3. For each question: RAG prompt (retrieved context + question) → D2L-adapted model → generate → parse.
4. Score on 50 eval questions with judge.

## Prompt Template

Same RAG prompt as S1 and S2+R (`src/generation/prompt.py::PROMPT_TEMPLATE`):
```
Answer the question using ONLY the provided context. ...

Context:
{context}

Question: {question}
Expected answer format: {answer_type_instruction}
```

The D2L adapter was NOT trained with this prompt — it was trained via hypernetwork on raw document text. Whether the adapter nonetheless improves generation from retrieved context is the central question of this experiment.

## Adapter Note

D2L adapters target **MLP layers** (as defined by the hypernetwork), while S2+R targets **q_proj + v_proj** (attention). This is an inherent difference between the two adaptation methods, not a confound — the comparison answers "which practical method gives better results," not "which layer subset is better."

## Metrics (on 50 eval questions)

- Q_main, S_det, S_asst, G (F_β=2.5) — full retrieval-aware metrics
- Peak VRAM during inference
- TTFT, end-to-end latency
- Breakdown by answer_type
- Delta vs S1 (adapter contribution to RAG)
- Delta vs S2+R (D2L vs QLoRA as adapter source)
- Delta vs S3 control (retrieval contribution to D2L system)

Delta vs S3 is a post-hoc reporting step computed after both EXP-004 and EXP-004b complete.

## Output

- Predictions: `results/EXP-004b/predictions.json` (50 entries)
- Eval report: `results/EXP-004b/eval_report.json`
- Systems metrics: `results/EXP-004b/systems_metrics.json`
- `experiments/EXP-004b_d2l_retrieval/REPORT.md`

## Definition of Done

- [ ] S1 retrieval run on 50 eval questions — cached retrieval contexts
- [ ] S3 monolithic adapter loaded and applied to backbone
- [ ] Full 50-question inference with D2L adapter + retrieved context — `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API
- [ ] Q_main, S_det, S_asst, G reported
- [ ] Delta vs S1 computed and reported
- [ ] Delta vs S2+R computed and reported
- [ ] Peak VRAM, TTFT, latency logged
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-004b_d2l_retrieval/REPORT.md` written
