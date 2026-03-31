# SPEC: EXP-010 — S7 CLM+QLoRA Adapter Merge (Post-hoc)

**System:** S7 | **Class:** Post-hoc analysis | **Wave:** 5 | **Depends on:** EXP-003, EXP-004, EXP-004b | **Blocks:** None

## Goal

Test whether linear interpolation of existing CLM and RAFT adapters can combine complementary behavior (higher `S_asst` from CLM and higher `S_det` from RAFT) without retraining.

## Pipeline

1. Load paired adapters from CLM and RAFT runs by matching seed (`42, 123, 777`).
2. Validate adapter compatibility (same LoRA keys/shapes and base model family).
3. Merge weights via linear interpolation:
   - `merged = alpha * CLM + (1 - alpha) * RAFT`
4. Save merged adapter artifacts per seed under alpha-specific output directory.
5. Run retrieval-aware evaluation on the frozen 50-question eval set using the S1 retrieval stack.
6. Aggregate mean ± std over seeds for Q_main, S_det, S_asst, G and systems metrics.

## Frozen Decisions

| Decision | Value |
|----------|-------|
| Merge alpha | 0.5 |
| Seeds | 42, 123, 777 |
| Base model | `google/gemma-2-2b-it` |
| Retrieval stack | Same as S1 (`external/pdf_rag_pipeline`) |
| Training | None (eval-only merge) |
| Scoring | Same protocol as `SPEC-evaluation.md` |

## Metrics

- Q_main, S_det, S_asst (mean ± std over seeds)
- Grounding `G = F_β(β=2.5)`
- TTFT, end-to-end latency, peak inference VRAM
- Deltas vs S1, S2+R, S3+R

## Results (Observed)

- `Q_main = 0.7045 ± 0.0345`
- `S_det = 0.6790 ± 0.0481`
- `S_asst = 0.7641 ± 0.0178`
- `G = 0.5667`

## Output

- `results/EXP-010/alpha_0.5/aggregate_summary.json`
- `results/EXP-010/alpha_0.5/seed_*/`
- `experiments/EXP-010_adapter_merge/REPORT.md`

## Definition of Done

- [x] Merge rule implemented and alpha validated
- [x] Adapter compatibility checks added before merge
- [x] Seed-matched merges completed for 42/123/777
- [x] Retrieval-aware eval completed on frozen 50-question set
- [x] Q_main, S_det, S_asst, G reported as mean ± std
- [x] Deltas vs S1/S2+R/S3+R documented
- [x] Artifacts saved under alpha-scoped paths
- [x] `experiments/EXP-010_adapter_merge/REPORT.md` written
