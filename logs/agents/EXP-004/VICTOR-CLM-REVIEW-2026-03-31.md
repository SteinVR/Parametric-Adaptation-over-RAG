# Agent Log — Victor Code Reviewer
**Task:** EXP-004 CLM implementation review
**Date:** 2026-03-31
**Files reviewed:**
- `src/training/clm.py`
- `experiments/EXP-004_clm_pretraining/config.py`
- `experiments/EXP-004_clm_pretraining/main_exp.py`
**Spec:** `memory_bank/specs/SPEC_EXP-004.md`
**Reference:** `experiments/EXP-003b_qlora_closed/main_exp.py`, `src/training/qlora.py`

---

## Findings

### P1 — HIGH

**1. `_resolve_seeds` ignores `--all-seeds` flag**
File: `experiments/EXP-004_clm_pretraining/main_exp.py`, lines 255–260

Current code:
```python
def _resolve_seeds(seed: int | None, all_seeds: bool, smoke: bool) -> list[int]:
    if seed is not None:
        return [seed]
    if smoke:
        return [cfg.DEFAULT_SEED]
    return list(exp_cfg.TRAIN_SEEDS)
```

The `all_seeds` parameter is accepted but never used. The function always returns all seeds when no single seed is given and smoke=False. This means `--all-seeds` is silently a no-op. The reference EXP-003b has the same bug, but it is still a defect — the `--all-seeds` CLI argument is advertised to users but has no effect.

**2. `EXP003_AGGREGATE_PATH` points to wrong experiment**
File: `experiments/EXP-004_clm_pretraining/config.py`, line 17

```python
EXP003_AGGREGATE_PATH = base_cfg.RESULTS_DIR / "EXP-003" / "aggregate_summary.json"
```

This resolves to `results/EXP-003/aggregate_summary.json`, which is the S2+R (RAFT + retrieval) aggregate. EXP-004 correctly labels this as `delta_vs_s2r` in main_exp.py, so the intent is right. However the spec says the delta target is "S2+R (EXP-003)" — this path IS correct. No bug here, but note:

`EXP003B_AGGREGATE_PATH` on line 18 resolves to `results/EXP-003b/aggregate_summary.json` — this path IS correct and the file exists.

**3. `DataCollatorForLanguageModeling` with variable-length chunks: no padding specified**
File: `src/training/clm.py`, lines 139–142

```python
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

`DataCollatorForLanguageModeling` with `mlm=False` does pad batches. However it defaults to the tokenizer's `padding_side` and uses `pad_to_multiple_of=None`. More importantly: since all chunks are exactly `max_seq_length` (except possibly the last one, which is filtered out by `MIN_CHUNK_TOKENS` check and the `break` statement), the dataset produces uniform-length chunks. There is no truncation in the collator — the collator will receive lists of equal-length token sequences. This is safe because the `break` statement discards tail chunks shorter than MIN_CHUNK_TOKENS. This is not a bug, but worth noting.

### P2 — MEDIUM

**4. Predictions saved with per-seed filename inside flat result_dir, but eval_report saved without seed in name**
File: `experiments/EXP-004_clm_pretraining/main_exp.py`, lines 162–173

Predictions are saved to `result_dir / f"predictions_seed_{seed}.json"` where `result_dir = RESULTS_DIR / f"seed_{seed}"`. This creates double-seed encoding in the path: `results/EXP-004_clm/seed_42/predictions_seed_42.json`. The spec says the output should be `results/EXP-004_clm/predictions_seed_{42,123,777}.json` (flat in the results dir). The current output puts predictions inside per-seed subdirectories, which does not match the spec's output specification.

Similarly `eval_runner.save_report(report, result_dir)` saves to the per-seed `result_dir`, which is consistent internally but deviates from the spec's flat output layout.

**5. Spec adapter output path: `models/clm/seed_{42,123,777}/` vs actual `models/clm/seed_42/`**
File: `experiments/EXP-004_clm_pretraining/main_exp.py`, line 81 and `config.py` line 13

`MODELS_DIR = base_cfg.MODELS_DIR / "clm"` and adapter_dir is `MODELS_DIR / f"seed_{seed}"`. This gives `models/clm/seed_42/`, which matches spec exactly.

**6. `train_result.peak_vram_mb` may be `None` when logging**
File: `experiments/EXP-004_clm_pretraining/main_exp.py`, line 108

```python
train_result.peak_vram_mb or 0
```
This is fine — the `or 0` guard handles `None`. No issue.

### P3 — LOW / OBSERVATIONS

**7. `vram_before` computed but never used in return value**
File: `src/training/clm.py`, lines 153–154

```python
vram_before = torch.cuda.memory_allocated() / 1024 / 1024
logger.info("VRAM before training: %.1f MB", vram_before)
```
Purely informational logging. Variable is computed but only used in the log line. Not a bug, but a dead variable.

**8. `total_steps` computation in `clm.py` does not account for fractional last batch**
File: `src/training/clm.py`, lines 156–163

```python
total_steps = len(dataset) * config.epochs // (
    config.per_device_batch_size * config.gradient_accumulation_steps
)
```
This is integer division that may undercount by 1 optimizer step. This only affects the pre-train log message, not actual training (Trainer computes its own steps). No functional impact.

**9. `all_seeds` argument in `_resolve_seeds` is never `False` in practice (EXP-003b parity bug)**
The reference EXP-003b has the same issue, so this is inherited. Low priority.

**10. Smoke mode: `--smoke` runs only 1 seed but still runs full aggregation**
When `args.smoke` is True and aggregation runs, `_write_report` is not called (correct guard at line 230), but `save_seed_aggregate` is called with smoke output dir. This is correct and intentional.

---

## Verdict
CHANGES REQUIRED (P2 item 4: output path deviates from spec)
P1 item 1 (--all-seeds no-op) is a pre-existing bug from EXP-003b — lower priority to fix now.
