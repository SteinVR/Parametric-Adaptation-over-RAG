# SPEC: EXP-004 — S3 CLM Continued Pretraining

**System:** S3 | **Class:** Control | **Wave:** 2 | **Depends on:** EXP-002 (prompt, parser) | **Blocks:** EXP-004b, EXP-006

## Goal

Train a QLoRA adapter on corpus document text with causal language modeling loss (next-token prediction). Evaluate as pure parametric control without retrieval. 3 seeds for variance estimation.

## Historical Note

EXP-004 originally targeted Doc-to-LoRA (D2L) hypernetwork packaging. D2L was non-viable: documents exceeded the hypernetwork's context window, requiring chunking (12–20 chunks/doc) and double merge. Result: Q_main=0.210, worse than S2 closed-book (0.263). Per-doc sanity S_det=0.154. Archived as negative finding in `experiments/EXP-004_d2l_monolithic/REPORT.md`. Architecture pivoted to CLM in v9.0.

## Pipeline

1. Extract text from 8 PDFs via `src/d2l/corpus.py::load_frozen_corpus_documents()` (module path is legacy from D2L; function is corpus-agnostic)
2. Concatenate all document texts into a single training corpus (~115K tokens)
3. Tokenize as plain text — no chat template, no prompt/answer masking. Labels = input_ids (standard CLM).
4. QLoRA training with causal LM loss
5. Eval on 50 questions, closed-book prompt (no retrieval), 3 seeds

## Training Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| PEFT method | QLoRA | Matches S2+R for fair comparison |
| Rank | 32 | Same as S2+R |
| Alpha | 32 | Same as S2+R |
| Dropout | 0.05 | Same as S2+R |
| Target modules | q_proj, v_proj | Same as S2+R — isolates training signal |
| Quantization | 4-bit NF4, double quant | Standard QLoRA recipe |
| Learning rate | 5e-5 | Standard for continued pretraining; 2e-4 too aggressive on 106K tokens |
| Batch size | 1 | Hardware constraint (512-tok sequences on 8GB) |
| Gradient accumulation | 4 | Effective batch = 4 |
| Epochs | 5 | Diffuse CLM signal needs more passes; safe at low LR |
| Warmup ratio | 0.1 | Longer warmup stabilizes low-LR continued pretraining |
| Weight decay | 0.01 | Same as S2+R |
| Scheduler | Cosine | Same as S2+R |
| Optimizer | Paged AdamW 8-bit | Same as S2+R |
| Max sequence length | 512 | CLM computes loss on all tokens → (seq × 256K vocab) fp32 logit tensor. 512 fits 8 GB; 2048 does not (OOM at cross_entropy). |
| Seeds | 42, 123, 777 | Same as S2+R |

## Prompt Template (S3 — no retrieval)

```
<start_of_turn>user
Answer the question based on your knowledge. If you are not confident in the answer, respond with [].

Question: {question}
Expected answer format: {answer_type_instruction}
<end_of_turn>
<start_of_turn>model
```

Same `answer_type_instruction` as EXP-002.

**Unanswerable handling:** S3 has no retrieved context, so for unanswerable questions the model must rely on its uncertainty signal. Scoring: if model outputs `[]` → correct (1.0); if model hallucinates an answer → incorrect (0.0).

## Metrics

- Q_main, S_det, S_asst on 50 eval — mean ± std over 3 seeds
- Training time per seed (seconds)
- Training loss curve (final loss)
- Peak VRAM during training
- Peak VRAM during inference
- TTFT, end-to-end latency
- Breakdown by answer_type

## Output

- 3 adapters: `models/clm/seed_{42,123,777}/`
- Per-seed results: `results/EXP-004_clm/seed_{42,123,777}/` each containing:
  - `predictions_seed_*.json` (50 entries)
  - `eval_report.json`
  - `systems_metrics.json`
  - `training_metrics.json`
- Aggregate: `results/EXP-004_clm/aggregate_summary.json`
- `experiments/EXP-004_clm_pretraining/REPORT.md`

**Note:** D2L archived results remain in `results/EXP-004/` (not overwritten).

## Definition of Done

- [ ] CLM training module implemented (`src/training/clm.py`)
- [ ] 3 adapters trained (seeds 42, 123, 777) — `models/clm/seed_*/` each contain adapter weights
- [ ] Full 50-question eval per seed — each `seed_*/predictions_seed_*.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API (per seed)
- [ ] Q_main, S_det, S_asst reported as mean ± std over 3 seeds
- [ ] Training time per seed and peak training VRAM logged
- [ ] Inference VRAM, TTFT, latency logged
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-004_clm_pretraining/REPORT.md` written

## Risk

- **115K tokens may be too small for meaningful CLM adaptation.** 3 epochs gives ~345K token exposures. If Q_main ≈ base model (no improvement), this is a valid finding: continued pretraining on a small corpus does not inject useful parametric knowledge.
- **Overfitting.** Low data volume + 3 epochs may cause memorization without generalization. Monitor training loss for collapse. If loss → 0 too fast, reduce to 1 epoch.
