# SPEC: EXP-003b — S2 QLoRA Closed-Book (Axis 1)

**System:** S2 | **Axis:** 1 (Paradigm in Isolation) | **Wave:** 2 | **Depends on:** EXP-002 (prompt/parser reuse), EXP-003 (shared QLoRA config, delta computation) | **Blocks:** EXP-006, EXP-007

## Goal

Fine-tune Gemma-2-2b-it with QLoRA on question→answer pairs **without any retrieved context**. At inference, the model answers from weights alone — no retrieval pipeline. Establishes supervised parametric baseline for Axis 1 paradigm comparison.

Key delta: Δ(S2+R, S2) = retrieval contribution to the supervised system (computed against EXP-003 results).

## Split

Same frozen split as all experiments:
- **150 S2-train:** used for QLoRA training
- **50 eval:** held out, never seen during training. ALL systems evaluated on these same 50.

Split in `data/splits/split_v1.json`.

## Data Preparation

From 150 S2-train questions, build **closed-book** training examples:

```
Input:  closed_book_prompt(question, answer_type)
Output: answer
```

### Prompt template (closed-book)

```
Answer the following question based on your knowledge.

Question: {question}
Expected answer format: {answer_type_instruction}
```

No context field. No mention of "provided context." Answer type instructions from `src/generation/prompt.py::ANSWER_TYPE_INSTRUCTIONS`:

| answer_type | Instruction |
|-------------|-------------|
| boolean | "Answer true or false." |
| number | "Answer with a number only." |
| name | "Answer with the exact name." |
| names | `Answer with a JSON array of names, e.g. ["Name1", "Name2"].` |
| date | "Answer with a date in YYYY-MM-DD format." |
| free_text | "Answer in 1-3 sentences (max 280 characters)." |

### Answer formatting

Same format as all systems: true/false for boolean, YYYY-MM-DD for date, number for number, `[]` for unanswerable, etc. Identical to parser expectations.

### Dataset file

`data/processed/closed_book_train.jsonl` — 150 entries, each with:
```json
{"question_id": "...", "answer_type": "...", "prompt": "...", "answer": "..."}
```

## QLoRA Configuration (Fixed — identical to EXP-003)

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
| Batch size | 1 (gradient accumulation 4 → effective 4) |
| Epochs | 3 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |

Learning rate, optimizer, scheduler, rank, alpha, dropout, epochs, and warmup are identical to S2+R. Batch size is effective 4 here (micro=1, accum=4) vs effective 4 in EXP-003 (same actual config despite EXP-003 spec text suggesting effective 8; 8GB VRAM forced micro=1 for both). The only controlled variable is the training data format (closed-book vs RAFT open-book).

## Training Protocol

1. Generate `data/processed/closed_book_train.jsonl` from 150 S2-train questions
2. Train with fixed hyperparams
3. Repeat with 3 seeds: 42, 123, 777
4. Report mean ± std on 50 eval questions

## Inference

S2 closed-book at inference: question → same closed-book prompt template → QLoRA-adapted Gemma-2-2b-it → answer. **No retrieval, no Qdrant, no reranker.** Same answer parser as all systems.

## Metrics (on 50 eval questions)

- Q_main, S_det, S_asst (**no grounding G** — no retrieval)
- Variance: mean ± std across 3 seeds
- Training time per seed, peak VRAM during training and inference
- Breakdown by answer_type
- Delta vs S1 baseline (Axis 1 comparison)
- Delta vs S2+R (Axis 2 ablation: retrieval contribution). This is a post-hoc reporting step computed after both EXP-003 and EXP-003b complete; it does not block EXP-003b eval.

## Output

- Closed-book dataset: `data/processed/closed_book_train.jsonl`
- 3 adapter checkpoints: `models/qlora_closed/seed_{42,123,777}/`
- Per-seed eval results: `results/EXP-003b/seed_{42,123,777}/predictions.json` (50 entries each)
- Per-seed eval reports: `results/EXP-003b/seed_{42,123,777}/eval_report.json`
- Aggregate: `results/EXP-003b/aggregate_summary.json`
- `experiments/EXP-003b_qlora_closed/REPORT.md`

## Definition of Done

- [ ] Closed-book dataset generated — `data/processed/closed_book_train.jsonl` has 150 entries, no context in prompts
- [ ] 3 adapters trained — `models/qlora_closed/seed_{42,123,777}/` each contain adapter weights
- [ ] Full 50-question eval for ALL 3 seeds — each `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API for each seed
- [ ] Mean ± std reported for Q_main, S_det, S_asst across 3 seeds
- [ ] Delta vs S1 baseline computed and reported
- [ ] Delta vs S2+R (EXP-003) computed and reported
- [ ] Training time (seconds) and peak VRAM (MB) logged per seed
- [ ] Breakdown by answer_type for each seed
- [ ] All results committed to git
- [ ] `experiments/EXP-003b_qlora_closed/REPORT.md` written
