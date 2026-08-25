# Experiment work

The existing experiment set is complete. Use `experiments/README.md` for the current experiment-to-system map and dependency graph. Before changing an existing experiment, read its code, local report, and outputs under `results/`. Memory Bank specs are historical context, not active execution instructions.

## Layout and naming

- Preserve existing folder IDs and names, including suffixes such as `EXP-003b`, `EXP-004b`, and parallel `EXP-004_*` controls. Do not infer the next ID or rename historical folders for consistency.
- An experiment may use `run.py` or `main_exp.py`, an optional local `config.py`, and an existing report named `EXP-*_REPORT*.md`. Preserve the local pattern rather than forcing a template.
- Experiment entrypoints are regular scripts; `# %%` cell markers are optional.
- Store reusable logic in the appropriate package under `src/`: `data`, `training`, `retrieval`, `rag_pipeline`, `generation`, `evaluation`, or `d2l`.
- Write empirical outputs under the matching `results/EXP-*` directory. Model adapters and weights belong under the gitignored `models/` tree.

Creating a new experiment, changing an experiment ID, or revising the frozen system design requires explicit user direction. Do not create or update historical Memory Bank records as a side effect.

## Reproducibility and execution

- Run entrypoints from the repository root, preferably with `uv run python experiments/<folder>/<entrypoint>.py`.
- Use global defaults from `config.py` and experiment-local overrides from the local `config.py`. Preserve the configured seeds (`42`, `123`, `777`) and record any deliberate deviation.
- Reuse valid indexes, adapters, predictions, and judge responses. Never overwrite a completed baseline merely to smoke-test a code path.
- Start with import, compilation, configuration, or reduced-data validation. Multi-seed training, a full paid OpenAI judge pass, destructive result regeneration, and large checkpoint downloads require explicit execution scope from the user.
- QLoRA workflows require the documented CUDA environment; D2L additionally requires Linux-compatible CUDA dependencies. On unsupported hardware, perform static validation and leave the hardware run clearly unverified.

## Reports and result integrity

- Reports must distinguish measured results from interpretation and proposed follow-up work.
- Keep `S1`, `S2+R`, `S3+R`, `S7`, `S2`, `S3`, and `S3-legacy` terminology aligned with `README.md`.
- Do not manually adjust prediction, metric, aggregate-summary, or chart values. Change the producing code or source artifact, regenerate the narrowest affected output, and document any invalidated downstream tables or figures.
