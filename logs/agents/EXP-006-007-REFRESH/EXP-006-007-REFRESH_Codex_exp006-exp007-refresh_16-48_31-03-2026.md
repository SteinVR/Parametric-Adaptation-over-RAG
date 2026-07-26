# [EXP-006-007-REFRESH] [exp006-exp007-refresh]

## Metadata + Exact Request
- Logged at: 31-03-2026 16:48
- Agent identity: Codex
- Task: EXP-006-007-REFRESH
- Branch / worktree: main / /home/xeliaray/Projects/Term-Paper
- Scope: Rebuild EXP-006 and EXP-007 outputs with EXP-010 (S7), exclude S6, and deliver deep result analysis with expanded visual analytics.
- Exact user request or delegated objective:
  > Выполни повторно EXP-006 и EXP-007 с учетом EXP-010 и выкинь от туда результаты S6 (Naive RAG). Убедись, что все корректно. Затем сделай глубокую и масштабную обработку/интерпретацию результатов с дополнительными агрегациями и графиками.

## Task Setup
- Context used:
- SPEC_EXP-006, SPEC_EXP-007, SPEC-evaluation, SPEC-systems, SPEC-data
- Existing EXP-006/EXP-007 scripts and current results artifacts
- EXP-010 report and aggregate outputs
- Constraints:
- No Memory Bank updates unless explicitly requested
- Keep S6 out of EXP-006/EXP-007 outputs
- Reuse existing artifacts only (analysis-only, no fresh inference)
- Planned approach:
- Fix EXP-006 aggregation correctness and regenerate artifacts
- Implement reproducible EXP-007 analysis pipeline with mandatory + deep outputs
- Generate extended figures and interpretation reports

## Implementation Log
1. Reworked `experiments/EXP-006_main_comparison/main_exp.py` + `config.py`: fixed per-type metric extraction, added unified systems metrics, added single-vs-multi analysis, generated gradient plot, refreshed report.
2. Implemented full `experiments/EXP-007_error_analysis/main_exp.py` and updated `config.py`: consolidated metrics table, per-question aggregation, pairwise wins, error-overlap, seed-stability, difficulty/judge profiles, error analysis markdown, deep analysis markdown, and expanded figure set.
3. Reran pipelines and refreshed reports/artifacts for EXP-006 and EXP-007; confirmed S6 is excluded and S7/EXP-010 is integrated.

## Verification
- `.venv/bin/python experiments/EXP-007_error_analysis/main_exp.py`
- `python experiments/EXP-006_main_comparison/main_exp.py`
- `python -m py_compile experiments/EXP-006_main_comparison/main_exp.py experiments/EXP-007_error_analysis/main_exp.py`
- `rg -n "S6|Naive" results/EXP-006 results/EXP-007 experiments/EXP-006_main_comparison/REPORT.md experiments/EXP-007_error_analysis/REPORT.md`
- Result:
- EXP-006/EXP-007 artifacts regenerated successfully
- No S6 entries in refreshed outputs (only explicit exclusion note in reports)
- EXP-010/S7 included in consolidated analysis and interpretation

## Issues / Resolutions
- Issue: default `python` environment lacked `pandas` required by new EXP-007 analysis pipeline.
- Resolution: ran EXP-007 via project runtime `.venv/bin/python` (where pandas/numpy/matplotlib are available).
- Issue: matplotlib cache directory warning under unwritable default config path.
- Resolution: set `MPLCONFIGDIR` in EXP-007 script to `/tmp/matplotlib`.

## Artifacts
- Files touched:
- experiments/EXP-006_main_comparison/main_exp.py
- experiments/EXP-006_main_comparison/config.py
- experiments/EXP-006_main_comparison/REPORT.md
- experiments/EXP-007_error_analysis/main_exp.py
- experiments/EXP-007_error_analysis/config.py
- experiments/EXP-007_error_analysis/REPORT.md
- results/EXP-006/* (main_results, per_type_breakdown, deltas, single_vs_multi_doc, gradient_plot)
- results/EXP-007/* (consolidated_results, error/deep analysis, aggregation CSVs)
- results/figures/* (expanded analytics figure set)
- Reports, plots, logs, or other evidence:
- `results/EXP-007/deep_analysis.md`
- `results/EXP-007/error_analysis.md`
- `results/figures/{main_results_table,cost_quality_scatter,per_type_heatmap,latency_grounding_scatter,error_overlap_heatmap,pairwise_win_heatmap,difficulty_profile,judge_criteria_profile,seed_stability,pareto_frontier}.png`
- Verification artifacts:
- terminal run logs + compiled scripts + grep validation for S6 exclusion

## Result / Conclusions (Exact Answer to User)
> EXP-006 и EXP-007 полностью пересобраны с учетом EXP-010 (S7), S6 исключен из итоговых таблиц/анализа. Дополнительно выполнен расширенный глубокий анализ с новыми агрегациями и 10 графиками (trade-off, overlap, pairwise wins, difficulty, judge criteria, stability, pareto и др.), а также обновлены отчеты и все ключевые CSV-артефакты.
