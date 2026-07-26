## Skill
- Follow skill `apm-logs` for application logging conventions.

## Expected structure
- `logs/project/runtime/` -- application runtime logs produced by code (training, evaluation, metrics, errors).
- `logs/project/reports/` -- report documents (test, review, model, general).
- `logs/agents/{TASK_ID}/` -- agent session logs.

## Conventions
- Log format for application logs: `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`.
- Store DS application runtime output under `logs/project/runtime/`.
- Store report documents under `logs/project/reports/`.

## Guardrails
- Do not store model artifacts here (use `models/`).
