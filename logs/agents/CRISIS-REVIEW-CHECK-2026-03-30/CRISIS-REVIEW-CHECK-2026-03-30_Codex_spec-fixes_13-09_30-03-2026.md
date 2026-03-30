# [CRISIS-REVIEW-CHECK-2026-03-30] spec-fixes

## Metadata + Exact Request
- Logged at: 30-03-2026 13:09
- Agent identity: Codex
- Task: CRISIS-REVIEW-CHECK-2026-03-30
- Branch / worktree: current workspace on existing branch
- Scope: align updated architecture/spec docs with the Crisis Review follow-up findings
- Exact user request or delegated objective:
  > "Внеси поправки"

## Task Setup
- Context used:
  - `docs/Crisis_Review.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/SPEC-systems.md`
  - `memory_bank/SPEC-data.md`
  - `memory_bank/specs/SPEC_EXP-003.md`
  - `memory_bank/specs/SPEC_EXP-007.md`
  - `memory_bank/specs/SPEC_EXP-008.md`
- Constraints:
  - Change only the specific mismatches previously identified.
  - Do not update unrelated Memory Bank files.
- Planned approach:
  1. Fix split/training ownership wording.
  2. Fix the `S2+R` inference naming slip.
  3. Clarify `EXP-007` finalization flow when conditional `EXP-008` runs.

## Implementation Log
1. Updated `memory_bank/SPEC-data.md` so Train-150 / `S2-train` is explicitly shared by `S2` and `S2+R`, and leakage rules now mention both supervised systems.
2. Updated `memory_bank/specs/SPEC_EXP-003.md` so the inference section correctly names `S2+R`, not `S2`.
3. Updated `memory_bank/specs/SPEC_EXP-007.md` to define default vs ablation paths and require refreshing final outputs if `EXP-008` runs.

## Verification
- Ran `git diff --check -- memory_bank/SPEC-data.md memory_bank/specs/SPEC_EXP-003.md memory_bank/specs/SPEC_EXP-007.md`
- Result: no whitespace or patch formatting issues

## Issues / Resolutions
- Issue: `EXP-007` and `EXP-008` were conceptually aligned but operational ownership of final tables was still ambiguous.
- Resolution: clarified that `EXP-007` remains the synthesis spec and must refresh outputs after `EXP-008` if `S6` is triggered.

## Artifacts
- Files touched:
  - `memory_bank/SPEC-data.md`
  - `memory_bank/specs/SPEC_EXP-003.md`
  - `memory_bank/specs/SPEC_EXP-007.md`
- Reports, plots, logs, or other evidence:
  - `logs/agents/CRISIS-REVIEW-CHECK-2026-03-30/CRISIS-REVIEW-CHECK-2026-03-30_Codex_spec-fixes_13-09_30-03-2026.md`
- Verification artifacts:
  - clean `git diff --check` for the touched files

## Result / Conclusions (Exact Answer to User)
> Внесены точечные правки в `SPEC-data`, `SPEC_EXP-003` и `SPEC_EXP-007`: убрана двусмысленность по Train-150 / `S2-train`, исправлен `S2`→`S2+R` в inference-описании и явно прописан refresh финальных таблиц после `EXP-008`, если условная абляция запускается.
