# EXP-004 Doc-to-LoRA Monolithic

## Metadata + Exact Request
- Logged at: 30-03-2026 13:50
- Agent identity: Codex
- Task: EXP-004
- Branch / worktree: main tree, no worktree
- Scope: Implement Doc-to-LoRA monolithic packaging, sanity checks, merge, and no-retrieval eval scaffolding for EXP-004
- Exact user request or delegated objective:
  > Implement SPEC_EXP-004 end-to-end in modular typed code without creating a worktree and without modifying memory_bank/ files. Reuse existing evaluation/generation patterns where appropriate. Current runtime state: ./.venv has dependencies incl. ctx_to_lora, but the expected D2L checkpoint path from config.py (trained_d2l/gemma_demo/checkpoint-80000) is missing; do not download anything, instead implement clear validation/early failure around that path. Needed scope includes: D2L checkpoint loading helpers, document text extraction over the 8-doc corpus, per-doc adapter generation/export to models/d2l/doc{1-8}/, sanity-check flow on s2_train doc-specific deterministic subsets only (no judge calls), frozen 8-way arithmetic-mean merge to models/d2l/monolithic/, closed-book S3 inference/eval wiring for 50 eval questions, experiment entrypoint under experiments/, and report generation scaffolding/artifacts consistent with repo style. Run whatever local validations are possible without the missing checkpoint, and summarize changed files and remaining blockers.

## Task Setup
- Context used: `memory_bank/specs/SPEC_EXP-004.md`, `memory_bank/SPEC-evaluation.md`, `memory_bank/SPEC-systems.md`, `memory_bank/SPEC-data.md`, `memory_bank/ARCHITECTURE.md`, and reusable experiment / eval modules in `src/`
- Constraints: no worktree, no `memory_bank/` edits, no checkpoint download, modular typed code, local validation only where possible
- Planned approach:
  1. Map existing eval / generation / data hooks.
  2. Add D2L-specific helpers for checkpoint loading, corpus extraction, adapter export, merge, and no-retrieval inference.
  3. Add EXP-004 entrypoint/config/report scaffold.
  4. Validate with compile/import checks and a validate-only run.

## Implementation Log
1. Added a new `src/d2l/` package with typed helpers for corpus extraction, checkpoint resolution/loading, PEFT adapter export, LoRA averaging, no-retrieval inference, and deterministic sanity scoring.
2. Added a new EXP-004 experiment directory with config, main entrypoint, and report scaffold. The entrypoint supports `--validate-only` and writes a validation artifact.
3. Ran `compileall` on the new modules and executed `experiments/EXP-004_d2l_monolithic/main_exp.py --validate-only` to verify the run stops cleanly on the missing checkpoint while still validating corpus/split counts.

## Verification
- `./.venv/bin/python -m compileall src/d2l experiments/EXP-004_d2l_monolithic`
- `./.venv/bin/python experiments/EXP-004_d2l_monolithic/main_exp.py --validate-only`
- Result: both passed; validate-only wrote `results/EXP-004/validation.json` and reported the checkpoint blocker clearly

## Issues / Resolutions
- Blocker: local Doc-to-LoRA checkpoint is missing at `trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`
- Resolution: implemented explicit checkpoint validation / early failure and a validate-only path so the rest of the pipeline can still be checked without downloading anything

## Artifacts
- Files touched: `src/d2l/*`, `experiments/EXP-004_d2l_monolithic/*`
- Reports / logs: `experiments/EXP-004_d2l_monolithic/REPORT.md`
- Verification artifacts: `results/EXP-004/validation.json`

## Result / Conclusions (Exact Answer to User)
> EXP-004 implementation is in place, local validation passes, and the run is blocked only by the missing local Doc-to-LoRA checkpoint.

