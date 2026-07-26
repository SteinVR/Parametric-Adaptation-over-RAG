# Activity Log

- Task ID: term-paper-citation-fixes
- Agent: Codex
- Date: 2026-04-01 00:18 Europe/Moscow

## Scope
Close review findings for citation recall and bibliography precision in `term-paper_2/Term-Paper_Draft.md`.

## Actions
- Added implementation-oriented QLoRA citations in Section 2.2 using sources already present in `docs/Literature.md`.
- Expanded Section 5.6 to cite task-aware and online adapter-composition context from the provided bibliography.
- Normalized ambiguous reference entries by adding identifiers or source descriptors already present in `docs/Literature.md`.
- Preserved the rule that the manuscript cites only works drawn from `docs/Literature.md`.

## Files Changed
- `term-paper_2/Term-Paper_Draft.md`

## Verification
- Re-checked citation numbering after inserting new references.
- Verified new references [18]-[21] are both defined and cited where needed.
- Re-read updated Sections 2.2, 5.6, and the `References` block.

## Outcome
The two P3 recall gaps are closed, and the `References` block is less ambiguous for review purposes.
