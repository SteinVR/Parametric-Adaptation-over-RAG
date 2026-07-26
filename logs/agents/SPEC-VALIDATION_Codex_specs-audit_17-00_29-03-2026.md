# [SPEC-VALIDATION] [specs-audit]

## Metadata + Exact Request
- Logged at: 29-03-2026 17:00
- Agent identity: Codex
- Task: SPEC-VALIDATION
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Audit project specs for internal consistency and alignment with repo state
- Exact user request or delegated objective:
  > Проверь спеки проекта на корректность и валидность

## Task Setup
- Context used:
  - `AGENTS.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/TASKS.md`
  - `memory_bank/STATE.md`
  - `memory_bank/SPEC-*.md`
  - `memory_bank/specs/SPEC_EXP-*.md`
  - `config.py`, `pyproject.toml`
  - `src/evaluation/*`, `src/data/io.py`
  - `external/pdf_rag_pipeline/*`
  - `data/goldset/goldset.benchmark.json`
  - `data/splits/split_v1.json`
  - `data/manifests/corpus_manifest_v2.csv`
  - `results/EXP-002/*`
- Constraints:
  - Do not update Memory Bank files without explicit user request
  - Treat this as a review/audit, not an implementation task
- Planned approach:
  - Cross-check SSOT docs against each other
  - Validate factual claims against data artifacts and code
  - Report ranked findings with file references

## Implementation Log
1. Read review/report skill instructions and mapped spec sources in `memory_bank/` and `docs/project_preparation/`.
2. Cross-checked architecture, systems, data, evaluation, task board, and state documents for contradictions.
3. Validated key claims against repository artifacts: goldset schema/counts, split sizes, corpus manifest totals, evaluation outputs, and retrieval/eval code paths.

## Verification
- Checked `data/goldset/goldset.benchmark.json` structure and counts
- Checked `data/splits/split_v1.json` keys and lengths
- Checked `data/manifests/corpus_manifest_v2.csv` page/token totals
- Checked `results/EXP-002/` for completed eval artifacts
- Checked `src/evaluation/runner.py` against reporting requirements
- Result:
  - Confirmed several internal contradictions and one high-severity leakage contradiction
  - Confirmed some specs are already stale relative to repo state

## Issues / Resolutions
- Issue: S5 adapter-selection rule uses eval set despite explicit leakage prohibition.
- Resolution: Reported as highest-severity spec defect; requires spec correction before continuing experiments.
- Issue: SSOT documents disagree on split ratio and corpus token count.
- Resolution: Reported with ground truth from `data/manifests/corpus_manifest_v2.csv` and `data/splits/split_v1.json`.
- Issue: `STATE.md` and `TASKS.md` are stale relative to `results/EXP-002/`.
- Resolution: Reported as SSOT drift, but not edited due to instruction constraints.

## Artifacts
- Files touched:
  - `logs/agents/SPEC-VALIDATION_Codex_specs-audit_17-00_29-03-2026.md`
- Reports, plots, logs, or other evidence:
  - `results/EXP-002/eval_report.json`
  - `results/EXP-002/eval_summary.csv`
  - `data/manifests/corpus_manifest_v2.csv`
- Verification artifacts:
  - `src/evaluation/runner.py`
  - `src/data/io.py`
  - `external/pdf_rag_pipeline/retrieval/service.py`

## Result / Conclusions (Exact Answer to User)
> Найден один критичный дефект спецификаций и несколько средних расхождений: leakage в выборе адаптера для S5, конфликтующие значения split/token count, устаревший `STATE.md`/`TASKS.md`, и неверное описание `is_unanswerable` как поля goldset. Спеки в текущем виде нельзя считать полностью валидными как SSOT.
