# [EXP-006-007-REFRESH] [d2l-legacy-refresh]

## Metadata + Exact Request
- Logged at: 31-03-2026 18:46
- Agent identity: Codex
- Task: EXP-006-007-REFRESH
- Branch / worktree: main / /home/xeliaray/Projects/Term-Paper
- Scope: Refresh EXP-006 and EXP-007 to include legacy D2L control row S3-legacy from EXP-004 artifacts.
- Exact user request or delegated objective:
  > Обнови EXP-006 и EXP-007 с учетом результатов D2L (experiments/EXP-004_d2l_monolithic, results/EXP-004) S3-legacy (D2L).

## Task Setup
- Context used:
- `memory_bank/specs/SPEC_EXP-006.md`, `memory_bank/specs/SPEC_EXP-007.md`
- `experiments/EXP-006_main_comparison/{config.py,main_exp.py,REPORT.md}`
- `experiments/EXP-007_error_analysis/{config.py,main_exp.py,REPORT.md}`
- `results/EXP-004/*` + `experiments/EXP-004_d2l_monolithic/REPORT.md`
- Constraints:
- Keep S6 excluded.
- Do not update Memory Bank files.
- Reuse existing artifacts only.
- Planned approach:
- Add S3-legacy data source wiring in EXP-006/007 configs.
- Update analysis scripts to ingest single-run legacy metrics and eval outputs.
- Rebuild artifacts and verify S3-legacy appears in final outputs and reports.

## Implementation Log
1. Added S3-legacy inputs into EXP-006 config and EXP-007 config (EXP-004 eval/systems/doc-generation/merge paths).
2. Updated EXP-006 pipeline to load S3-legacy metrics, per-type, single-vs-multi, and delta comparisons; regenerated report narrative with legacy interpretation.
3. Updated EXP-007 pipeline to include S3-legacy in consolidated metrics and control narrative, including offline D2L packaging cost from generation+merge summaries.
4. Re-ran both experiment scripts and refreshed all CSV/PNG/REPORT outputs.

## Verification
- `.venv/bin/python -m py_compile experiments/EXP-006_main_comparison/main_exp.py experiments/EXP-007_error_analysis/main_exp.py experiments/EXP-006_main_comparison/config.py experiments/EXP-007_error_analysis/config.py`
- `.venv/bin/python experiments/EXP-006_main_comparison/main_exp.py`
- `.venv/bin/python experiments/EXP-007_error_analysis/main_exp.py`
- Manual checks:
- `results/EXP-006/main_results.csv` includes `S3-legacy`
- `results/EXP-007/consolidated_results.csv` includes `S3-legacy`
- both `REPORT.md` files include `S3-legacy (D2L)` control narrative
- Result:
- Refresh completed successfully with S3-legacy integrated into EXP-006 and EXP-007 outputs.

## Issues / Resolutions
- Issue:
- Inference VRAM key differs across systems metrics (`peak_vram_mb` vs `peak_infer_vram_mb`).
- Resolution:
- Added robust dual-key extraction in both scripts.
- Issue:
- Legacy D2L offline packaging cost was not directly available as one scalar in existing outputs.
- Resolution:
- Computed cost as `sum(document_generation.generation_seconds) + merge_summary.merge_seconds` from `results/EXP-004`.

## Artifacts
- Files touched:
- `experiments/EXP-006_main_comparison/config.py`
- `experiments/EXP-006_main_comparison/main_exp.py`
- `experiments/EXP-006_main_comparison/REPORT.md`
- `experiments/EXP-007_error_analysis/config.py`
- `experiments/EXP-007_error_analysis/main_exp.py`
- `experiments/EXP-007_error_analysis/REPORT.md`
- `results/EXP-006/*`
- `results/EXP-007/*`
- `results/figures/*` (regenerated plots)
- Reports, plots, logs, or other evidence:
- `results/EXP-006/main_results.csv`
- `results/EXP-007/consolidated_results.csv`
- `experiments/EXP-006_main_comparison/REPORT.md`
- `experiments/EXP-007_error_analysis/REPORT.md`
- Verification artifacts:
- Successful py_compile and both refresh runs in this session.

## Result / Conclusions (Exact Answer to User)
> EXP-006 и EXP-007 обновлены с учетом legacy D2L (S3-legacy) из EXP-004: строка S3-legacy добавлена в консолидацию, сравнительные таблицы, дельты, single-vs-multi и итоговые отчеты; артефакты и графики пересобраны.
