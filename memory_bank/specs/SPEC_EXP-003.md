# SPEC: EXP-003 — S2 QLoRA RAFT-style Baseline

**System:** S2 | **Wave:** 2 | **Depends on:** EXP-002 (retriever, prompt, parser) | **Blocks:** EXP-006, EXP-007

## Goal

Fine-tune Gemma-2-2b-it with QLoRA in RAFT-style. Establish supervised parametric baseline. At inference S2 uses same retriever as S1.

## Split

From 200 QA total:
- **150 S2-train:** used for QLoRA training
- **50 eval:** held out, never seen during S2 training. ALL systems (S1–S5) evaluated on these same 50.

Split stratified by answer_type + difficulty, frozen in `data/splits/split_v1.json`.

## Data Preparation

From 150 train questions, build RAFT-style training examples. All examples are oracle (gold context provided):

```
Input:  prompt_template(question, [gold_chunk, distractor_1, distractor_2])
Output: answer
```

- **Distractors:** exactly 2 random pages from documents OTHER than the gold document
- **Distractors sampled once** and frozen in training set (not re-sampled per epoch)
- **No distractor-only examples** (2B model cannot answer domain-specific legal facts without gold context)
- **Answer formatting:** same format as parser expects (true/false, number, date ISO, `[]` for unanswerable, etc.)

## QLoRA Configuration (Fixed)

| Parameter | Value |
|-----------|-------|
| Base model | Gemma-2-2b-it |
| Quantization | 4-bit NF4 (double quantization enabled) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Optimizer | paged AdamW 8-bit |
| LR scheduler | cosine |
| Max seq length | 4096 |
| Batch size | 4 (gradient accumulation to effective 8 if needed) |
| Epochs | 3 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |

No hyperparameter sweep. Fixed values.

## Training Protocol

1. Prepare RAFT dataset from 150 train questions (150 oracle examples, 2 distractors each)
2. Train with fixed hyperparams
3. Repeat with 3 seeds: 42, 123, 777
4. Report mean ± std on 50 eval questions

## Inference

S2 at inference = S1 retriever (top-5 chunks) + QLoRA-adapted Gemma-2-2b-it. Same RAG prompt template, same answer parser.

## Metrics (on 50 eval questions)

- Q_main, S_det, S_asst, G (F_β=2.5)
- Variance: mean ± std across 3 seeds
- Training time per seed, peak VRAM during training
- Breakdown by answer_type
- Delta vs S1 baseline

## Output

- RAFT dataset: `data/processed/raft_train.jsonl`
- 3 adapter checkpoints: `models/qlora/seed_{42,123,777}/`
- `experiments/EXP-003/REPORT.md`
