# SPEC: EXP-004 — S3 Doc-to-LoRA Monolithic

**System:** S3 | **Wave:** 2 | **Depends on:** EXP-002 (prompt, parser) | **Blocks:** EXP-005a, EXP-005b, EXP-006

## Goal

Generate 8 per-document LoRA adapters via Doc-to-LoRA hypernetwork. Sanity-check each individually. Merge all 8 into one monolithic adapter. Evaluate.

## Pipeline

### Step 1: Setup
- Clone/install SakanaAI doc-to-lora repo
- Download Gemma-2-2b-it checkpoint and hypernetwork weights (checkpoint-80000)
- Verify: load model + hypernetwork, run on one test document, confirm adapter is generated

### Step 2: Per-document adapter generation
- For each of 8 documents: extract full text → feed to hypernetwork → save LoRA adapter
- Expected output: 8 adapter files, each containing delta weights for MLP layers
- Log: generation time per doc, adapter file size, peak VRAM

### Step 3: Per-doc sanity check
- For each doc adapter: load it, run inference on questions from that document only (from S2-train set, not eval)
- Score per-doc Q_main — does the adapter help on its own document's questions?
- This validates that the hypernetwork actually works before attempting merge

### Step 4: Merge
**Primary strategy: simple average.**
For each LoRA matrix, average the corresponding delta weights across all 8 adapters:
`merged_W = (1/8) × Σ(adapter_i_W)`

**Merge strategy is frozen to simple average.** No fallback selection on eval set (that would be tuning on evaluation data). If simple average produces poor Q_main, report as negative result — this IS the finding.

### Step 5: Monolithic evaluation
- Load merged adapter, run on 50 eval questions
- No retrieved context — adapter parameters only
- Same prompt template (but without retrieved chunks section), same answer parser

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

- Per-doc adapter Q_main (sanity check, per-document subset)
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

## Risk

Merge of 8 adapters may destroy information. If so, S3 is declared non-viable — document this as finding, proceed with S4-doc and S4-cluster.
