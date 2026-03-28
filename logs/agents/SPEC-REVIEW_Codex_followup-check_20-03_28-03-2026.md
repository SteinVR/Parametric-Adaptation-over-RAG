# [SPEC-REVIEW] followup-check

## Metadata + Exact Request
- Logged at: 28-03-2026 20:03
- Agent identity: Codex
- Task: SPEC-REVIEW
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Re-check updated experiment specs and SSOT docs after prior review
- Exact user request or delegated objective:
  > Проверь, все ли было поправлено и не осталось ли чего

## Task Setup
- Context used:
  - `memory_bank/specs/SPEC_EXP-002.md` ... `memory_bank/specs/SPEC_EXP-008.md`
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/SPEC-evaluation.md`
- Constraints:
  - Review only; do not modify Memory Bank files
  - Focus on remaining methodological gaps and ambiguity
- Planned approach:
  - Compare current specs to prior findings
  - Confirm closed items
  - Report remaining blockers with exact references

## Implementation Log
1. Re-read the updated experiment specs and supporting SSOT docs.
2. Verified that several prior blockers were fixed: common eval set, S5 candidate pool restricted to single adapters, frozen S3 merge rule, and clearer S4-doc/S4-cluster routing analysis.
3. Identified remaining inconsistencies around locked test usage and unanswerable-response contract.

## Verification
- Checked current text of updated specs and architecture/evaluation specs with numbered lines
- Result: major improvements, but not fully clean yet

## Issues / Resolutions
- Remaining issue: no true locked final test exists if EXP-006 and EXP-007 already use the same 50-question eval set that EXP-008 calls "Locked Test"
- Remaining issue: prompts still instruct `null`, while architecture/evaluation now expect `[]` for deterministic unanswerable cases
- Remaining issue: reporting still names the unanswerable bucket `null`, conflicting with the newer `unanswerable` terminology

## Artifacts
- Files touched:
  - `/home/xeliaray/Projects/Term-Paper/logs/agents/SPEC-REVIEW_Codex_followup-check_20-03_28-03-2026.md`
- Reports, plots, logs, or other evidence:
  - Line-referenced follow-up review of current specs and SSOT
- Verification artifacts:
  - Shell reads with line numbers for relevant files

## Result / Conclusions (Exact Answer to User)
> Большая часть замечаний исправлена, но не всё. Остались как минимум три существенных несостыковки: нет настоящего locked test, потому что EXP-008 использует тот же 50-question eval, что и предыдущие этапы; контракт по unanswerable всё ещё расходится между prompt и scoring (`null` против `[]`); и в reporting-format bucket всё ещё назван `null`, хотя выше уже введён `unanswerable`.
