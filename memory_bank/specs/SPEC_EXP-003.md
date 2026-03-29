# SPEC: EXP-003 — S2+R QLoRA RAFT + Retrieval (Axis 2)

> **Architecture note:** This experiment produces **S2+R** results (Axis 2: retrieval augmentation).
> For Axis 1 paradigm comparison (S2 closed-book), see EXP-003b.

**System:** S2+R | **Axis:** 2 (Retrieval Augmentation) | **Wave:** 2 | **Depends on:** EXP-002 (retriever, prompt, parser) | **Blocks:** EXP-006, EXP-007

## Goal

Fine-tune Gemma-2-2b-it with QLoRA in RAFT-style. At inference S2+R uses same retriever as S1. Results serve as retrieval-augmented ablation: Δ(S2+R, S2) isolates retrieval contribution to the supervised system.

## Split

From 200 QA total:
- **150 S2-train:** used for QLoRA training
- **50 eval:** held out, never seen during S2 training. ALL systems (S1–S5) evaluated on these same 50.

Split stratified by answer_type + difficulty, frozen in `data/splits/split_v1.json`.

## Data Preparation

From 150 train questions, build RAFT-style training examples. Use oracle gold context when `gold_retrieval` is available:

```
Input:  prompt_template(question, [gold_chunks..., distractor_1, distractor_2])
Output: answer
```

### Gold chunk selection policy

Goldset provides `gold_retrieval` at page level: `[{doc_id, page_numbers}]`. The EXP-002 index contains chunks across 5 families (page, section, clause, microchunk, table). Training context uses **page-family chunks only:**

1. For each gold page in `gold_retrieval` → find the corresponding `page`-family `IndexChunk` from the EXP-002 index (one chunk per page, contains full page text)
2. Gold chunks = ordered list of these page chunks, sorted by (doc_id, page_number)
3. If a gold page has no matching page-family chunk (should not happen) → fall back to raw page text from `CanonicalPageRecord`

Rationale: page-family chunks are the most complete representation of gold evidence. Using finer-grained chunks (clause, microchunk) would require arbitrary selection among multiple candidates per gold page.

### Distractor policy

- **Distractors:** exactly 2 **page-family chunks** from documents OTHER than any gold document
- Sampled uniformly at random from all page-family chunks of non-gold documents
- **Distractors sampled once** and frozen in training set (not re-sampled per epoch)

### Context assembly

Training prompt `retrieved_chunks` field = gold chunks first, then distractors, concatenated with `\n\n---\n\n` separator. Order: `[gold_1, gold_2, ..., distractor_1, distractor_2]`. No shuffling — model should learn to extract from any position.

### Other rules

- If a train question has empty `gold_retrieval`, keep it in the 150-example training set and build a **distractor-only** context with the same 2 distractors. These are rare negative examples required to preserve the frozen split.
- **Answer formatting:** same format as parser expects (true/false, number, date ISO, `[]` for unanswerable, etc.)
- **Multi-doc questions:** all gold pages from all gold documents included as gold chunks

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
| Batch size | 4 ( to effective 8 if needed) |
| Epochs | 3 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |

No hyperparameter sweep. Fixed values.

## Training Protocol

1. Prepare RAFT dataset from 150 train questions (oracle when gold exists, distractor-only when `gold_retrieval` is empty, always 2 distractors)
2. Train with fixed hyperparams
3. Repeat with 3 seeds: 42, 123, 777
4. Report mean ± std on 50 eval questions

## Inference

S2 at inference = S1 retrieval pipeline (hybrid search + RRF + reranker + evidence compressor) + QLoRA-adapted Gemma-2-2b-it. Reuses the Qdrant index built in EXP-002 (no rebuild). Same RAG prompt template, same answer parser.

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

## Definition of Done

- [ ] RAFT dataset generated — `data/processed/raft_train.jsonl` has 150 entries, each with exactly 2 distractors and gold pages when available
- [ ] 3 adapters trained — `models/qlora/seed_{42,123,777}/` each contain adapter weights
- [ ] Full 50-question eval for ALL 3 seeds — each `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API for each seed
- [ ] Mean ± std reported for Q_main, S_det, S_asst, G across 3 seeds
- [ ] Delta vs S1 baseline computed and reported
- [ ] Training time (seconds) and peak VRAM (MB) logged per seed
- [ ] Breakdown by answer_type for each seed
- [ ] All results committed to git
- [ ] `experiments/EXP-003/REPORT.md` written
