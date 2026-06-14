## About project
This research project investigates **knowledge injection techniques for document-grounded question answering (QA) on consumer hardware**. The thesis compares multiple approaches for efficient knowledge integration, including retrieval-augmented generation (RAG), adapter-based fine-tuning (QLoRA), and pretraining strategies, with focus on practical deployment constraints and performance trade-offs. The project includes comprehensive data processing pipelines, baseline experiments, and systematic error analysis to evaluate the effectiveness of different knowledge injection methods.

## Project map
- `memory_bank/` — stable project-level architecture, DS state, and task board.
- `data/` — raw, processed, and external data layers.
- `eda/` — EDA scripts, results, high-level `EDA-Report.md`, and deep `EDA-Insights.md`.
- `experiments/` — hypothesis implementation and reports.
- `models/` — model artifacts and model reports.
- `logs/` — split into `logs/project/` for project logs and `logs/agents/` for agent-session logs.
- `results/` — experiment outputs, benchmarks, and performance reports.
- `src/` — core application source code, data pipelines, training, retrieval, and generation modules.
- `src/retrieval/` — primary RAG pipeline with staged retrieval, indexing, and retrieval factory.
- `external/pdf_rag_pipeline/` — reference RAG implementation with ingestion, indexing, and retrieval stages.
- `term-paper/` — thesis drafts, advisor reviews, writing blueprints, and supporting figures.
- `term-paper_2/` — updated thesis version with structured sections, blueprints, and additional materials.
- `term-paper_3/` — most recent and thoroughly developed thesis version with comprehensive refinements and finalized content.
- `docs/` - additional documents.

## Memory Bank (SSOT)
- Directory name is `memory_bank/`.
- **TASKS.md:** grouped, ordered high-level tasks only (lives directly in `memory_bank/`, not inside `tasks/`).
- **tasks/{TASK_ID}.md:** each active task has its own file with implementation plan, notes, and execution details.
- **STATE.md:** compact operational status for experiments and blockers.
- Do not update Memory Bank files unless the user explicitly asks.
- Keep main headers from templates intact; add sub-sections only when needed.
- **ARCHITECTURE.md:** high-level project architecture, system design decisions, and component relationships.
- **SPEC-*.md:** domain-specific specifications covering data processing, evaluation methodology, and system requirements.
- **specs/:** detailed specification documents and implementation guidelines for specific features and modules.
