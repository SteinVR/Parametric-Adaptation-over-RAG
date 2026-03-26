## Skill
- Use skill `apm-eda` for the workflow and report format.

## Expected structure
- `src/` -- main EDA scripts folder
- `results/figures/` -- saved plots (PNG)
- `results/tables/` -- saved tables (CSV)
- `results/deep/figures/` -- deep low-level diagnostic plots
- `results/deep/tables/` -- deep low-level diagnostic tables
- `reports/` -- EDA reports folder

## Conventions
- Name plots descriptively: `{analysis}_{feature}.png`.
- Keep findings reproducible from scripts in `eda/src/`.
- Keep `EDA-Report.md` concise and high-level.
- Keep `EDA-Insights.md` exhaustive and low-level; each major insight must point to a concrete artifact path.
- Keep task-specific implementation notes in `memory_bank/tasks/{TASK_ID}.md`.

## Guardrails
- Do not run experiments from this directory (use `experiments/`).
- Do not modify `src/` code directly from EDA scripts; extract reusable functions to `src/` separately.
