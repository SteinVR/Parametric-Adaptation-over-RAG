## Structure
- `data/raw/` -- original, unmodified data files.
- `data/processed/` -- cleaned and transformed data ready for modeling.
- `data/external/` -- third-party or supplementary datasets.

## Conventions
- Never modify files in `raw/`; treat them as immutable.
- Processed data should be reproducible from raw via scripts in `src/`.

## Guardrails
- Do not commit large data files to version control; use `.gitignore`.