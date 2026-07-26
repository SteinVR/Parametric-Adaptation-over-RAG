# [TERM-PAPER-FIGURES] reuse-swap

## Metadata + Exact Request
- Logged at: 01-04-2026 01:22
- Agent identity: Codex
- Task: TERM-PAPER-FIGURES
- Branch / worktree: main workspace `/home/xeliaray/Projects/Term-Paper`
- Scope: Replace the current `term-paper_2` figure package with reusable legacy assets and sync manuscript links.
- Exact user request or delegated objective:
  > "Тогда очисти term-paper_2/figures от всего что там есть сейчас, перенеси и замени из term-paper/figures что можно и сделай нейминг ссылки в работе. То, что нельзя заменить, пока убери - нет времени создавать"

## Task Setup
- Context used:
  - `term-paper/figures/*`
  - `term-paper_2/figures/*`
  - `term-paper_2/Term-Paper_Draft.md`
- Constraints:
  - Remove all current assets from `term-paper_2/figures`.
  - Reuse only legacy figures that already meet the current paper's needs.
  - Remove references to unavailable figures instead of recreating them.
- Planned approach:
  - Wipe the directory, copy reusable legacy PNGs under current canonical names, then patch manuscript figure links and references.

## Implementation Log
1. Cleared `term-paper_2/figures`, including stale generators and intermediate assets.
2. Copied reusable legacy PNG assets from `term-paper/figures` into `term-paper_2/figures` under current names for `Figure 2`, `Figure 3`, `Figure 4`, and `Appendix Figure B3`.
3. Patched `term-paper_2/Term-Paper_Draft.md` to remove `Figure 1`/`Appendix Figure B2` mentions, insert actual image links, and align Section 5.5 prose with the reused single-vs-multi figure.

## Verification
- Listed final files in `term-paper_2/figures`.
- Verified all manuscript image links resolve to existing file names.
- Verified no remaining mentions of unavailable `Figure 1` or `Appendix Figure B2` remain in the manuscript.

## Issues / Resolutions
- Issue: Initial cleanup command hit a leftover `__pycache__` directory.
- Resolution: Performed explicit recursive cleanup and reran the copy step.
- Issue: Legacy `Figure 1` and appendix error-overlap figure were not suitable for reuse.
- Resolution: Removed them from the current draft instead of keeping placeholders.

## Artifacts
- Files present in `term-paper_2/figures`:
  - `figure_2_delta_to_s1.png`
  - `figure_3_per_type_heatmap.png`
  - `figure_4_single_vs_multi_doc.png`
  - `appendix_figure_b3_judge_criteria_profile.png`
- Files changed:
  - `term-paper_2/Term-Paper_Draft.md`
  - `logs/agents/TERM-PAPER-FIGURES_Codex_reuse-swap_01-22_01-04-2026.md`

## Result / Conclusions (Exact Answer to User)
> `term-paper_2/figures` was cleaned and repopulated only with reusable legacy figure assets, and the manuscript now links only to the figures that actually exist in that directory.
