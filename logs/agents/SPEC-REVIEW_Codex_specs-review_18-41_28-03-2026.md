# [SPEC-REVIEW] specs-review

## Metadata + Exact Request
- Logged at: 28-03-2026 18:41
- Agent identity: Codex
- Task: SPEC-REVIEW
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Review experiment specifications in `memory_bank/specs` for methodological correctness and ambiguity
- Exact user request or delegated objective:
  > Выполни review спецификаций к задачам. memory_bank/specs
  >
  > Они должны быть методологически корректны, полностью однозначны

## Task Setup
- Context used:
  - `memory_bank/specs/SPEC_EXP-001.md` ... `memory_bank/specs/SPEC_EXP-008.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/SPEC-data.md`
  - `memory_bank/SPEC-evaluation.md`
  - `.codex/skills/apm-critical-review/SKILL.md`
- Constraints:
  - Do not update Memory Bank files unless explicitly requested
  - Review should focus on methodology, ambiguity, and evaluation validity
- Planned approach:
  - Read all experiment specs
  - Cross-check against architecture/data/evaluation SSOT
  - Produce ranked findings with concrete replacements

## Implementation Log
1. Enumerated all spec files under `memory_bank/specs/`.
2. Reviewed experiment specs and cross-checked them against architecture, data, and evaluation specs.
3. Extracted line-referenced findings focused on invalid comparisons, undefined protocols, and contradictory scoring contracts.

## Verification
- Checked every spec file in `memory_bank/specs/`
- Cross-checked evaluation and data assumptions against `memory_bank/SPEC-evaluation.md` and `memory_bank/SPEC-data.md`
- Result: multiple blocking ambiguities and methodological inconsistencies identified

## Issues / Resolutions
- Cross-system comparison is not methodologically valid as written because S2 is selected on a different evaluation set size and distribution than the other candidate adapters.
- Evaluation spec mixes answer types inconsistently (`free_text negative`, deterministic `null`, six-type vs seven-type reporting), making scoring and tables underdefined.
- Several experiment specs contain decision logic driven by undefined baselines (`random baseline`, fallback thresholds, tie rules), which prevents reproducible execution.

## Artifacts
- Files touched:
  - `/home/xeliaray/Projects/Term-Paper/logs/agents/SPEC-REVIEW_Codex_specs-review_18-41_28-03-2026.md`
- Reports, plots, logs, or other evidence:
  - Review based on line-referenced inspection of specs and SSOT docs
- Verification artifacts:
  - Shell reads with numbered lines for all reviewed specs

## Result / Conclusions (Exact Answer to User)
> Спецификации сейчас не проходят критерий полной однозначности. Главные блокеры: некорректное сравнение S2 с остальными системами на разных выборках, противоречивый контракт по типам ответов и null/free_text-negative, а также несколько неоперационализированных правил выбора и fallback-логики. Ниже пользователю возвращается ранжированное ревью с точечными ссылками и тем, что именно нужно заменить.
