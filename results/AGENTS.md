# Result integrity

`results/` contains empirical evidence produced by experiment code. Treat predictions, judge outputs, metrics, aggregates, tables, and figures as generated artifacts rather than editable documentation.

- Do not hand-edit numeric results to match a report or manuscript. Correct the upstream code or source artifact and regenerate the narrowest affected output.
- Preserve experiment IDs, system IDs, seed identity, sample counts, and question IDs. Do not combine runs with different configurations without recording the distinction.
- Before overwriting a completed result, confirm that regeneration is explicitly in scope and that the required index, adapter, model, API judge, and hardware environment are available.
- Reuse cached outputs only when their inputs and configuration remain valid. If provenance cannot be established, label the result uncertain instead of silently treating it as current.
- When an upstream result changes, identify and regenerate dependent aggregate tables, publication figures under `assets/figures/`, README values, and manuscript claims as applicable.
- Model weights do not belong here; they are local artifacts under `models/`.
