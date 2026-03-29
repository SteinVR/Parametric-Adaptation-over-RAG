# SPEC: EXP-007 — S5 Hybrid (RAG + Best Adapter)

**System:** S5 | **Wave:** 5 | **Depends on:** EXP-006 (best adapter selected) | **Blocks:** EXP-008, EXP-009

## Goal

Combine S1 retriever with the best-performing **single** adapter (S2 or S3, selected in EXP-006). S4-doc/S4-cluster are multi-adapter routed systems — not eligible for S5. Test raw query (S5a) and HyDE (S5b).

## Pipeline

### S5a — Raw query + adapter
1. Question → S1 retrieval pipeline (hybrid search + RRF + reranker + evidence compressor)
2. Retrieved evidence + question → best adapter on Gemma-2-2b-it → answer
3. Prompt: same RAG template as EXP-002 (with context)

### S5b — HyDE + adapter
1. Question → best adapter on Gemma-2-2b-it → hypothetical answer (HyDE generation)
2. HyDE text → S1 retrieval pipeline (hybrid search + RRF + reranker + evidence compressor)
3. Retrieved evidence + question → best adapter → final answer
4. HyDE prompt:

```
<start_of_turn>user
Write a brief answer to this legal question as if you had the source document in front of you.
Question: {question}
<end_of_turn>
<start_of_turn>model
```

Then use the generated text as retrieval query (not the original question).

### Evaluation
Both S5a and S5b on 50 eval questions. Headline S5 = variant with higher Q_main.

## Metrics

- Q_main, S_det, S_asst, G (F_β=2.5) for both variants
- Grounding delta: S5a vs S1, S5b vs S1
- TTFT, end-to-end latency (S5b has extra generation step)
- HyDE retrieval quality: Recall@k improvement vs raw query (k = evidence_budget from S1 pipeline config)
- Breakdown by answer_type

## Output

- S5a and S5b results: `results/EXP-007/`
- `experiments/EXP-007/REPORT.md`
