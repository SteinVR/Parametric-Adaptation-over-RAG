# [CRISIS-REVIEW-CHECK-2026-03-30] [wave-refresh-fix]

## Metadata + Exact Request
- Logged at: 30-03-2026 13:23
- Agent identity: Codex
- Task: CRISIS-REVIEW-CHECK-2026-03-30
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Remove the Wave 4 ↔ Wave 5 cycle introduced by post-S6 refresh requirements.
- Exact user request or delegated objective:
  > "1-е требует переобучения? Если да - забей. По второму замечанию - скорректируй"

## Task Setup
- Context used:
  - `AGENTS.md`
  - `memory_bank/TASKS.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/specs/SPEC_EXP-007.md`
  - `memory_bank/specs/SPEC_EXP-008.md`
  - `memory_bank/specs/SPEC_EXP-009.md`
- Constraints:
  - Keep Wave Protocol sequential.
  - Do not require retraining for the seed-schema concern.
  - Update Memory Bank only within the requested orchestration fix.
- Planned approach:
  - Separate mandatory Wave 4 analysis from conditional post-S6 refresh.
  - Reuse `EXP-009` as the dedicated refresh stage.
  - Reconcile task board, architecture registry, and affected specs.

## Implementation Log
1. Re-read `TASKS.md`, `SPEC_EXP-007.md`, `SPEC_EXP-008.md`, and `AGENTS.md` to validate the reported Wave Protocol contradiction.
2. Converted `EXP-007` back into a self-contained Wave 4 terminal analysis for the mandatory system set only.
3. Reintroduced `EXP-009` as a conditional Wave 5 refresh task that updates the final thesis package after `EXP-008` completes.

## Verification
- `git diff --check -- memory_bank/TASKS.md memory_bank/ARCHITECTURE.md memory_bank/specs/SPEC_EXP-007.md memory_bank/specs/SPEC_EXP-008.md memory_bank/specs/SPEC_EXP-009.md`
- Result: passed with no whitespace or patch-format issues.

## Issues / Resolutions
- Issue: `EXP-007` had been made dependent on a later-wave ablation refresh, creating a task-order cycle under the documented sequential wave protocol.
- Resolution: moved the refresh ownership into a dedicated conditional `EXP-009` stage while keeping `EXP-007` complete on its own.
- Issue: reviewer also flagged seed normalization in `EXP-006`.
- Resolution: not changed in this patch because the user asked to skip anything that would imply retraining; current answer explains that this is a downstream schema issue, not a retraining requirement.

## Artifacts
- Files touched:
  - `memory_bank/TASKS.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/specs/SPEC_EXP-007.md`
  - `memory_bank/specs/SPEC_EXP-008.md`
  - `memory_bank/specs/SPEC_EXP-009.md`
- Reports, plots, logs, or other evidence:
  - This activity log
- Verification artifacts:
  - clean `git diff --check` on touched files

## Result / Conclusions (Exact Answer to User)
> Первое замечание не требует переобучения: это проблема downstream schema/aggregation, а не train protocol. Второе замечание исправлено структурно: `EXP-007` снова завершает Wave 4 сам по себе, а conditional post-S6 refresh вынесен в отдельный `EXP-009`, так что цикла между Wave 4 и Wave 5 больше нет.
