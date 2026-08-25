# Dataset scope

- `corpus/` contains the active eight DIFC legal PDFs used by the experiments.
- `goldset/goldset.benchmark.json` is the frozen 200-question benchmark; `goldset.questions.json` is its question inventory.
- `splits/split_v1.json` is the frozen 150-train / 50-evaluation split.
- `processed/` contains derived RAFT and closed-book training tables.
- `corpus4-100/` and `corpus4_2-100/` preserve the two four-document, 100-question construction batches that were merged into the active benchmark. The `100` denotes questions, not documents.
- `manifests/` records corpus metadata. `old_corpus/` is historical provenance and is not an active experiment input.

Use the paths in root `config.py` and the loaders under `src/data/` rather than selecting files by a similar-looking name.

## Frozen-data guardrails

- Do not modify the active PDFs, benchmark, split, or derived training tables unless the user explicitly requests a dataset revision.
- Preserve question IDs, answer types, reference answers, evidence page references, document IDs, and the established semantics for unanswerable items.
- Never allow the 50 evaluation questions or their answers to enter supervised training data. Validate schema and leakage against the frozen benchmark, split, audit artifacts, and loaders under `src/data/`.
- Treat `corpus4-*` and `old_corpus/` as provenance. Do not silently substitute them for `corpus/` or `goldset/` in current experiments.
- A legitimate frozen-data change invalidates downstream artifacts. Re-run the relevant audit, update manifests and derived tables, and identify which experiment results and paper claims must be regenerated.
- Prefer deterministic scripts for transformations. Do not hand-edit large JSON/JSONL/CSV outputs when the producing script can express the change.
