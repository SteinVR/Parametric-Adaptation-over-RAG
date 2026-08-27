# Project

This repository is the technical companion to *Parametric Adaptation Methods for Document-Grounded Legal QA*. It tests whether parametric adaptation adds value after retrieval is already available on a fixed eight-document DIFC legal benchmark.

The main comparison is Base-RAG (`S1`) versus RAFT-RAG (`S2+R`) and CLM-RAG (`S3+R`). Merge-RAG (`S7`) is a post-hoc result; RAFT-Closed (`S2`), CLM-Closed (`S3`), and D2L-Closed (`S3-legacy`) are controls. The planned experiments are complete, so treat new experiment design, retraining, and benchmark changes as explicit scope changes rather than routine maintenance.

## Sources of truth

- `README.md` is the current public system map, result summary, and reproduction guide.
- `config.py` and the code under `src/` define the implemented defaults and runtime behavior.
- `results/` contains the empirical outputs used by the paper. Do not replace measured values with values copied from prose.
- `term-paper_3/Term-Paper-3.md` is the current manuscript source.
- `memory_bank/` is historical planning and design material. It is not a live task tracker or an active instruction layer.
- Existing experiment reports document experiment-local decisions and deviations.

When these sources disagree, do not silently choose one and propagate the mismatch. Identify whether the conflict concerns intended design, implemented behavior, measured output, or manuscript wording, then surface it to the user. Consult Memory Bank files only when historical design context is relevant; do not update them unless the user explicitly asks.

## Repository map

- `src/rag_pipeline/` — reusable ingestion, indexing, hybrid retrieval, reranking, and evidence-compression engine.
- `src/retrieval/` — project wrappers and staged retrieval orchestration used by experiments.
- `src/data/`, `src/training/`, `src/generation/`, `src/evaluation/`, `src/d2l/` — dataset builders, adaptation, inference, scoring, and the legacy D2L control.
- `data/` — frozen corpus, benchmark, split, derived training tables, and dataset-construction provenance.
- `experiments/` — experiment entrypoints, local configuration, and reports.
- `results/` — generated predictions, metrics, analysis tables, and result figures.
- `assets/figures/` — publication figures generated from result artifacts.
- `term-paper_3/` — current manuscript and paper build tooling.
- `output/` and the root PDF — generated publication packages and rendered paper output.
- `models/` and `trained_d2l/` — local, gitignored model artifacts that may be absent from a checkout.
- `memory_bank/` — historical internal research design and planning material.
- `docs/` and `external/` — supporting notes, reviews, literature, and third-party reference material.

Legacy drafts under `.old/` are scratch history and are not active sources.

## Environment and validation

- Use Python 3.12 and `uv`; run project commands from the repository root. Install or refresh the environment with `uv sync` when dependency work requires it.
- There is no single authoritative full test suite. Validate the smallest relevant surface: compile or import changed Python modules, run a targeted script or smoke path when safe, and verify the exact generated artifact affected by the change.
- Do not assume a macOS checkout can run CUDA, bitsandbytes, or Linux-only D2L workflows. Separate static validation from hardware-backed execution and state what was not run.
- Never print `.env` contents, API keys, tokens, or model credentials.

## Artifact and research integrity

- Treat benchmark data, predictions, metrics, charts, PDFs, archives, and model weights as artifacts with provenance. Edit their source code or source data and regenerate them; do not hand-edit generated values or binaries.
- Preserve system IDs, experiment IDs, question IDs, seed identity, and the distinction between headline, post-hoc, control, and archived systems.
- Reuse cached retrieval, generations, and judge outputs when they remain valid. Do not start multi-seed training, paid judging, or large model downloads unless execution is explicitly requested.
- Keep reusable implementation in `src/`; experiment folders should contain orchestration and experiment-specific configuration rather than copied library code.

## Branch policy: `dev` and `main`

- `dev` is the authoritative working branch. `main` is the public distribution branch and intentionally omits private working material.
- Use the existing separate worktrees for `dev` and `main`; locate them with `git worktree list`. Publish in the `main` worktree without switching the `dev` checkout to `main`. Inspect both worktrees before transferring changes and preserve unrelated uncommitted edits.
- Publish `dev` changes to `main` selectively: transfer only the public files or patches selected for that publication. Never copy the whole `dev` tree into `main` and then delete private material.
- Bring fixes from `main` back into `dev` selectively as well. Cherry-pick in either direction only after checking that the entire commit diff is appropriate for the destination branch.
- Do not merge the branches wholesale in either direction unless the user explicitly requests it. Do not use automatic mirroring that propagates deletions, such as `rsync --delete`.
- Preserve `dev`-only working material unless the user explicitly requests deletion. Treat deletion or absence on `main` as publication filtering unless evidence shows a genuine project deletion; do not propagate publication filtering into `dev`.
- Real project deletions, renames, and relocations are allowed when relevant to the selected change. Review them explicitly, update references, and do not retain stale duplicate trees.
- Before transferring changes, record the destination branch revision. Before committing or pushing, inspect the complete change set against that pre-update revision, including deletions and renames. For publication, verify that only intended public material is included; for updates to `dev`, verify that the private working layer remains usable. A clean patch application or cherry-pick is not sufficient validation.

## Procedure after context auto-compact

- Read our thread with `read_thread` and check everything we discussed: make sure nothing was missed, there are no contradictions, and you retain details from our conversation, your work, and your mistakes.
- This is only a gentle checkpoint: Consider whether the work is progressing reasonably and roughly according to plan. If so, or if any deviation seems modest, simply continue. Only if it appears substantially off course or the time spent feels disproportionate, consider whether changing approach or asking the user would help.
