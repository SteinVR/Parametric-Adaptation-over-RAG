# [SPEC-REVIEW] fixcheck-1-3

## Metadata + Exact Request
- Logged at: 29-03-2026 01:49
- Agent identity: Codex
- Task: SPEC-REVIEW
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Verify whether fixes 1-3 from the previous spec review were applied
- Exact user request or delegated objective:
  > Проверь исправления 1-3

## Task Setup
- Context used:
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/SPEC-evaluation.md`
  - `memory_bank/SPEC-systems.md`
  - `memory_bank/specs/SPEC_EXP-002.md`
  - `memory_bank/specs/SPEC_EXP-003.md`
  - `memory_bank/specs/SPEC_EXP-009.md`
  - `external/pdf_rag_pipeline/retrieval/service.py`
- Constraints:
  - Review only; do not modify Memory Bank files
  - Focus only on items 1-3 from the previous review
- Planned approach:
  - Re-check the updated grounding contract
  - Re-check S2 train-data construction policy
  - Re-check S6 ablation framing and scope

## Implementation Log
1. Re-read the updated grounding rules in architecture and evaluation specs.
2. Re-read the updated S2 train-data construction rules and system definition.
3. Re-read the updated S6 ablation framing and compared it against its prior methodological issue.

## Verification
- Checked numbered-line excerpts for all three previously flagged issues
- Result: fixes 1-3 are materially applied

## Issues / Resolutions
- Grounding now explicitly uses final evidence output after the retrieval pipeline, aligning with `RetrievalService.retrieve().page_references`.
- S2 training context now has a concrete page-family chunk selection policy and distractor construction rule.
- S6 is now explicitly framed as an end-to-end naive dense RAG ablation, not a single-variable retrieval-engineering isolation.

## Artifacts
- Files touched:
  - `/home/xeliaray/Projects/Term-Paper/logs/agents/SPEC-REVIEW_Codex_fixcheck-1-3_01-49_29-03-2026.md`
- Reports, plots, logs, or other evidence:
  - Follow-up line-referenced review of SSOT/specs
- Verification artifacts:
  - Numbered shell reads of the updated docs

## Result / Conclusions (Exact Answer to User)
> Исправления 1-3 по существу внесены. Grounding contract синхронизирован с реальным `RetrievalService`, S2 train-data construction стал операционализирован, а S6 больше не притворяется чистой абляцией retrieval engineering и честно описан как end-to-end naive dense RAG ablation. Остался только мелкий семантический хвост: в `ARCHITECTURE.md` у S2 ещё стоит общее "gold chunks", тогда как в детальных спеках уже зафиксировано "gold page-family chunks", но это уже не методологический дефект.
