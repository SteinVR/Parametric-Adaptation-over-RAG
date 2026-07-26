# [TERM-PAPER-FIGURE-REUSE] reuse-audit

## Metadata + Exact Request
- Logged at: 01-04-2026 01:16
- Agent identity: Codex
- Task: TERM-PAPER-FIGURE-REUSE
- Branch / worktree: main workspace `/home/xeliaray/Projects/Term-Paper`
- Scope: Audit whether figures from `term-paper/figures` can be reused for `term-paper_2/figures`.
- Exact user request or delegated objective:
  > "Плохие графики, много что наезжает друг на друга. Есть хорошие в term-paper/figures. Изучи, можно ли их переиспользовать."

## Task Setup
- Context used:
  - `term-paper/figures/*`
  - `term-paper/figures/generate_figures.py`
  - `term-paper_2/figures/*`
  - `term-paper_2/Term-Paper_Blueprint.md`
- Constraints:
  - Audit against current `term-paper_2` figure requirements.
  - Prefer direct reuse when the old asset already matches the new blueprint and current data.
- Planned approach:
  - Compare the old figure set visually and structurally against the current figure package and the `term-paper_2` blueprint.

## Implementation Log
1. Enumerated the available figure assets in `term-paper/figures` and `term-paper_2/figures`.
2. Inspected `term-paper/figures/generate_figures.py` to determine what data and layout assumptions the older assets use.
3. Reviewed all five legacy PNG figures and compared them against the current `term-paper_2` blueprint.

## Verification
- Confirmed the legacy figures are generated from the same `results/EXP-007` source files.
- Confirmed `fig02_delta_bars.png`, `fig04_per_type_heatmap.png`, and `fig05_singledoc_multidoc.png` are aligned with the current main-text needs.
- Confirmed `fig03_judge_criteria.png` aligns with current Appendix Figure B3.
- Confirmed `fig01_system_schematic.png` is only partially reusable because it does not include a separate `S3-legacy` control and collapses the no-retrieval controls.

## Issues / Resolutions
- Issue: The legacy figure generator depends on `pandas`, so script reuse is weaker than asset reuse in the current runtime.
- Resolution: Recommend asset/style reuse rather than blindly adopting the older generator as-is.
- Issue: There is no legacy appendix error-overlap figure in `term-paper/figures`.
- Resolution: `Appendix Figure B2` still needs a dedicated `term-paper_2` implementation.

## Artifacts
- Files audited:
  - `term-paper/figures/fig01_system_schematic.png`
  - `term-paper/figures/fig02_delta_bars.png`
  - `term-paper/figures/fig03_judge_criteria.png`
  - `term-paper/figures/fig04_per_type_heatmap.png`
  - `term-paper/figures/fig05_singledoc_multidoc.png`
  - `term-paper/figures/generate_figures.py`
- Output log:
  - `logs/agents/TERM-PAPER-FIGURE-REUSE_Codex_reuse-audit_01-16_01-04-2026.md`

## Result / Conclusions (Exact Answer to User)
> Yes: four legacy figures are directly reusable as better assets for `term-paper_2` (`fig02`, `fig03`, `fig04`, `fig05`). `fig01` is reusable only as a style/template, not as a final asset, because the current paper needs a more explicit control inventory. There is no reusable legacy equivalent for the appendix error-overlap figure.
