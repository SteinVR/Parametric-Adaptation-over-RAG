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

- **No distractor-only examples** (2B model cannot answer domain-specific legal facts without gold context)
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
