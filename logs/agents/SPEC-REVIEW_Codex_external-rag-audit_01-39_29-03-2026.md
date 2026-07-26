# [SPEC-REVIEW] external-rag-audit

## Metadata + Exact Request
- Logged at: 29-03-2026 01:39
- Agent identity: Codex
- Task: SPEC-REVIEW
- Branch / worktree: `main` / `/home/xeliaray/Projects/Term-Paper`
- Scope: Review updated specs after importing external RAG ingestion/indexing/retrieval pipeline
- Exact user request or delegated objective:
  > Спецификация изменилась.
  >
  > Я импортировал часть своего RAG пайплайна из другого проекта (Ingestion, Indexing, Retrieval) - не тривиальны. Лежит в @external/pdf_rag_pipeline, и в целом, его можно использовать как as-is.
  >
  > Проверь обновленную спецификацию, что все учтено, нет старых/устаревших, некорректных вещей и хвостов.

## Task Setup
- Context used:
  - `memory_bank/ARCHITECTURE.md`
  - `memory_bank/SPEC-systems.md`
  - `memory_bank/SPEC-data.md`
  - `memory_bank/SPEC-evaluation.md`
  - `memory_bank/specs/SPEC_EXP-001.md` ... `SPEC_EXP-009.md`
  - `external/pdf_rag_pipeline/README.md`
  - `external/pdf_rag_pipeline/config.py`
  - `external/pdf_rag_pipeline/retrieval/service.py`
  - `external/pdf_rag_pipeline/schemas.py`
- Constraints:
  - Review only; do not modify Memory Bank files
  - Focus on methodological correctness, ambiguity, outdated assumptions, and tails after external pipeline import
- Planned approach:
  - Reconstruct new frozen system design around `external/pdf_rag_pipeline`
  - Cross-check specs against actual external code contracts
  - Report only material inconsistencies and stale assumptions

## Implementation Log
1. Reviewed updated SSOT and experiment specs, including new `S6`/`EXP-009` ablation.
2. Compared spec claims about hybrid retrieval, grounding, and budgets against the imported external pipeline code.
3. Isolated remaining blockers introduced or exposed by the external pipeline integration.

## Verification
- Checked current spec text and supporting SSOT with numbered-line reads
- Verified external pipeline behavior from code, not just README
- Result: update is materially better aligned, but several methodological and stale-contract issues remain

## Issues / Resolutions
- Grounding contract still assumes `P` is built from top-k retrieved chunks, while the imported retrieval service emits page references after reranking and evidence compression.
- S2 RAFT training data remains under-specified now that gold annotations are page-level but retrieval context is chunk-level and multi-family.
- S6 ablation changes chunking family in addition to retrieval strategy, so it does not isolate retrieval engineering cleanly.

## Artifacts
- Files touched:
  - `/home/xeliaray/Projects/Term-Paper/logs/agents/SPEC-REVIEW_Codex_external-rag-audit_01-39_29-03-2026.md`
- Reports, plots, logs, or other evidence:
  - Line-referenced audit of updated specs and external pipeline code
- Verification artifacts:
  - Shell reads with line numbers for SSOT docs, experiment specs, and external pipeline modules

## Result / Conclusions (Exact Answer to User)
> Обновление в целом правильное: S1/S2/S5 теперь действительно привязаны к импортированному hybrid RAG stack, старый dense-only baseline вынесен в отдельный S6, malformed-contract у parser исправлен, и большая часть старых хвостов убрана. Но спецификация всё ещё не полностью чистая. Главные остатки: grounding в SSOT считается по top-k retrieved chunks, тогда как реальный external pipeline формирует page references после rerank+compression; S2 training-set generation остаётся недоопределённой для gold page → training chunk(s) при иерархическом chunking; и S6-абляция смешивает абляцию retrieval engineering с изменением chunking family, поэтому не изолирует только retrieval stack.
