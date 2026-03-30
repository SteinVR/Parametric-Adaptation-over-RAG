# SPEC: EXP-004 — S3 Doc-to-LoRA Monolithic

**System:** S3 | **Class:** Control + prep | **Wave:** 2 | **Depends on:** EXP-002 (prompt, parser) | **Blocks:** EXP-004b, EXP-005a, EXP-005b, EXP-006

## Goal

Generate 8 per-document LoRA adapters via Doc-to-LoRA hypernetwork. Sanity-check each individually. Merge all 8 into one monolithic adapter. Evaluate.

## Pipeline (inference)

1. Load Gemma-2-2b-it + monolithic merged adapter (4-bit NF4)
2. For each question: no-retrieval prompt → generate → parse
3. Score on 50 eval questions

## Packaging

### Adapter generation
- For each of 8 documents: extract full text → feed to Doc-to-LoRA hypernetwork → save LoRA adapter
- Expected output: 8 adapter files, each containing delta weights
- Log: generation time per doc, adapter file size, peak VRAM

### Per-doc sanity check
- For each doc adapter: load it, run inference on questions from that document only (from S2-train set, not eval)
- Score per-doc **S_det only** (no judge calls — this is a diagnostic, not a final result). Free_text excluded.
- Validates that the hypernetwork works before merge

### Merge
**Primary strategy: simple average.**
For each LoRA matrix: `merged_W = (1/8) × Σ(adapter_i_W)`

**Merge strategy is frozen to simple average.** No fallback selection on eval set (that would be tuning on evaluation data). If simple average produces poor Q_main, report as negative result — this IS the finding.

## Prompt Template (S3/S4 — no retrieval)

```
<start_of_turn>user
Answer the question based on your knowledge. If you are not confident in the answer, respond with [].

Question: {question}
Expected answer format: {answer_type_instruction}
<end_of_turn>
<start_of_turn>model
```

Same `answer_type_instruction` as EXP-002.

**Unanswerable handling:** S3/S4 have no retrieved context, so for unanswerable questions the model must rely on its uncertainty signal. Scoring: if model outputs `[]` → correct (1.0); if model hallucinates an answer → incorrect (0.0). This is an expected disadvantage of parametric-only systems.

## Metrics

- Per-doc adapter S_det (sanity check, per-document subset, deterministic types only)
- Monolithic Q_main, S_det, S_asst on 50 eval
- Adapter generation time per doc (seconds)
- Merge compute time
- Peak VRAM at inference
- Breakdown by answer_type

## Output

- 8 per-doc adapters: `models/d2l/doc{1-8}/`
- Monolithic merged adapter: `models/d2l/monolithic/`
- Per-doc sanity check results table
- **Frozen merge strategy** for EXP-005b
- `experiments/EXP-004/REPORT.md`

## Definition of Done

- [ ] 8 per-doc adapters generated — `models/d2l/doc{1-8}/` each contain adapter weights
- [ ] Per-doc sanity check: S_det reported for each doc adapter on its own questions
- [ ] Monolithic merged adapter saved — `models/d2l/monolithic/`
- [ ] Full 50-question eval on monolithic adapter — `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API
- [ ] Q_main, S_det, S_asst reported for monolithic adapter
- [ ] Adapter generation time per doc and merge time logged
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-004/REPORT.md` written (including sanity check table and merge viability assessment)

## Risk

Merge of 8 adapters may destroy information. If so, S3 is declared non-viable — document this as finding, proceed with S4-doc and S4-cluster.
