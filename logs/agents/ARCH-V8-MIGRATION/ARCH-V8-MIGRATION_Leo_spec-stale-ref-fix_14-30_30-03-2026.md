# ARCH-V8-MIGRATION spec-stale-ref-fix

## Metadata + Exact Request
- Logged at: 30-03-2026 14:30
- Agent identity: Leo (apm-dev)
- Task: ARCH-V8-MIGRATION
- Branch / worktree: main (no worktree)
- Scope: Fix all stale references in downstream spec files after Architecture v8.0 migration
- Exact user request or delegated objective:
  > Fix all stale references in downstream spec files after Architecture v8.0 migration. ARCHITECTURE.md and SPEC-systems.md are already correct (v8.0). Seven downstream files need updates including rewrites of EXP-007 and EXP-008, deletion of EXP-009 purpose, and targeted line-level fixes to SPEC-evaluation.md, SPEC-data.md, ARCHITECTURE.md, and SPEC_EXP-004b.md.

## Task Setup
- Context used: ARCHITECTURE.md (v8.0), SPEC-systems.md (v8.0), all seven target files read before editing
- Constraints: Minimal targeted edits except EXP-007 and EXP-008 which require full rewrites; do not touch BACKLOG or historical notes; no Memory Bank file updates unless explicitly part of scope
- Planned approach: Read all targets first, then edit sequentially, then verify with final grep

## Implementation Log

1. Read all target files: ARCHITECTURE.md, SPEC-evaluation.md, SPEC-data.md, SPEC_EXP-006.md, SPEC_EXP-007.md, SPEC_EXP-008.md, SPEC_EXP-009.md, SPEC_EXP-004b.md
2. Rewrote SPEC_EXP-006.md — replaced "Main Cross-Paradigm Comparison (Axis 1)" framing with v8.0 headline S1/S2+R/S3+R + controls structure; added EXP-004b to dependencies; removed S5 best adapter selection section and DoD item; added Key Deltas section mapping to RQ1/RQ2/RQ3
3. Rewrote SPEC_EXP-007.md — completely replaced old "S5 Hybrid (RAG + Best Adapter)" spec with Error Analysis + Cost/Quality/Grounding Trade-off spec; no fresh inference; depends on EXP-006; outputs final thesis tables and error_analysis.md
4. Rewrote SPEC_EXP-008.md — replaced old "Locked Test + Error Analysis" spec with S6 conditional ablation (previously EXP-009); trigger: both S2+R AND S3+R < S1; preserved pipeline, frozen decisions, and analysis from EXP-009 spec; updated all output paths from EXP-009/ to EXP-008/
5. Replaced SPEC_EXP-009.md content with single-line deprecation note pointing to EXP-008
6. Fixed SPEC-evaluation.md:
   - Line 11: "S1-S5 (and S6)" → explicit system list
   - Line 83: "S1/S2+R/S5" → "S1/S2+R/S3+R"
   - Interpretation table: replaced "S5 best on Q_main" row with "S2+R or S3+R best on Q_main"; updated S2/S3 rows to S2+R/S3+R framing
   - Mandatory caveats: removed S5 adapter selection leakage caveat; de-duplicated S2 format caveat
7. Fixed SPEC-data.md:
   - Section header: "S2 Training Data Format (RAFT-style)" → "S2+R Training Data Format (RAFT-style)"
   - Leakage rule 1: removed S5 exception clause
8. Fixed ARCHITECTURE.md:
   - Line 33: "unsupervised document exposure" → "supervision-free document exposure"
   - Line 67 (H2): "unsupervised D2L packaging" → "supervision-free D2L packaging"
9. Fixed SPEC_EXP-004b.md line 7: "unsupervised D2L" → "supervision-free D2L"
10. Discovered and fixed an additional stale S5 reference in SPEC_EXP-003.md line 16 ("ALL systems (S1–S5)")

## Verification

- Final grep for "S5" across all memory_bank/**/*.md: 0 matches
- Final grep for "best adapter selection" across all memory_bank/**/*.md: 0 matches
- Final grep for "unsupervised" across memory_bank/**/*.md: only 2 legitimate occurrences remain — the terminology rule definition in ARCHITECTURE.md §6 (the rule itself quotes the term to prohibit it) and SPEC-systems.md (user-confirmed as already correct v8.0)
- Re-read all modified files to confirm changes are coherent and no duplicate/corrupted content

## Issues / Resolutions

- Duplicate caveat in SPEC-evaluation.md created by edit sequence (old line 134 was kept while replacement was inserted as line 135). Detected during self-review; fixed by collapsing to single line.
- SPEC_EXP-003.md contained an unlisted stale S5 reference ("S1–S5") — found by final grep, fixed in scope.
- SPEC-evaluation.md line 100 contains "S2-S4" as a table scope shorthand for training/packaging cost rows. This is accurate (S2, S3, S4-doc, S4-cluster all have training or adapter-gen costs) and is not a system inventory claim — left as-is.

## Artifacts

- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-006.md` — rewritten
- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-007.md` — rewritten
- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-008.md` — rewritten
- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-009.md` — replaced with deprecation note
- `/home/xeliaray/Projects/Term-Paper/memory_bank/SPEC-evaluation.md` — 4 targeted edits
- `/home/xeliaray/Projects/Term-Paper/memory_bank/SPEC-data.md` — 2 targeted edits
- `/home/xeliaray/Projects/Term-Paper/memory_bank/ARCHITECTURE.md` — 2 targeted edits (lines 33, 67)
- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-004b.md` — 1 targeted edit (line 7)
- `/home/xeliaray/Projects/Term-Paper/memory_bank/specs/SPEC_EXP-003.md` — 1 targeted edit (line 16, bonus fix)

## Result / Conclusions (Exact Answer to User)

All stale references from the pre-v8.0 system inventory have been removed from downstream spec files. Zero S5 references remain in memory_bank/. EXP-006 now describes the headline S1/S2+R/S3+R comparison with controls. EXP-007 is now Error Analysis + trade-off consolidation. EXP-008 is now the S6 conditional ablation (trigger: both S2+R AND S3+R < S1). EXP-009 is deprecated. All "unsupervised" references in prose have been changed to "supervision-free" per §6 terminology rules.
