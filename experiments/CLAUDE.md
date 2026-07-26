## Skill
- Use skill `apm-ds-exp` for hypothesis-driven experiments.

## Expected structure
Each experiment lives in its own directory:
```
experiments/EXP-XXX_{description}/
  main_exp.py      # Experiment pipeline (cell-like blocks)
  config.py        # Experiment-specific hyperparameters
  REPORT.md        # Experiment report (from apm-ds-exp template)
```

## Naming convention
`EXP-{number}_{short-description}/` -- e.g., `EXP-001_baseline/`, `EXP-002_xgboost_tuning/`.

## Creating a new experiment
1. Create directory: `experiments/EXP-XXX_{description}/`
2. Create or copy template files: `config.py`, `main_exp.py`, `REPORT.md`
3. Link the run to a task in `memory_bank/TASKS.md` and mirror details in `memory_bank/tasks/{TASK_ID}.md`

## Running experiments
Each `main_exp.py` uses cell-like separators (`# %%`) for block-by-block execution.
Import reusable functions from `src/` modules (`src/data.py`, `src/features.py`, `src/models.py`, `src/evaluation.py`).


## Conventions
- Number experiments sequentially: `EXP-001`, `EXP-002`, etc.
- Always set and log random seeds for reproducibility.
- Keep high-level task status in `memory_bank/TASKS.md`.
- Keep experiment-level planning and notes in `memory_bank/tasks/{TASK_ID}.md`.

## Guardrails
- Do not run full training without user approval; smoke-test first.
- Keep experiment code self-contained; shared logic belongs in `src/`.
