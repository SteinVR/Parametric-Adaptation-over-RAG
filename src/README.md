# `src/` — core modules

Reusable code behind the experiments. Each experiment in `experiments/EXP-XXX/` imports
from here; shared logic lives in `src/`, experiment scripts only orchestrate it.

```
src/
├── rag_pipeline/                  # self-contained RAG engine
│   ├── config.py                  # PipelineConfig dataclass (defaults for all stages)
│   ├── schemas.py                 # core data models (chunks, results)
│   ├── ingestion/
│   │   ├── pdf_parser.py          # native PDF parsing + table candidate extraction
│   │   ├── table_serializer.py    # rule-based table detection, merge, serialization
│   │   └── corpus_builder.py      # canonical corpus assembly from parsed docs
│   ├── indexing/
│   │   ├── chunking.py            # structure-aware hierarchical chunking
│   │   ├── embeddings.py          # dense (Qwen3) + sparse (BM25) encoders
│   │   └── qdrant_store.py        # Qdrant hybrid-index persistence
│   └── retrieval/
│       ├── hybrid_search.py       # dense+sparse search with RRF fusion
│       ├── reranker.py            # Qwen3 cross-encoder reranker (lexical fallback)
│       ├── evidence_compressor.py # page-diverse evidence selection
│       ├── page_lifter.py         # chunk → (doc_id, page) lifting for grounding
│       └── service.py             # orchestrates search → rerank → compress → lift
├── retrieval/                     # project wrappers around rag_pipeline
│   ├── factory.py                 # build RetrievalService from PipelineConfig + index
│   ├── indexer.py                 # ingest the 8-doc corpus → Qdrant index
│   └── staged.py                  # 5-stage retrieve with sequential model loading (8 GB)
├── generation/                    # answer generation
│   ├── loader.py                  # Gemma-2-2b-it backbone in 4-bit NF4
│   ├── adapters.py                # adapter-aware model loaders (RAFT / CLM / merged)
│   ├── prompt.py                  # frozen prompt template + per-type instructions
│   ├── pipeline.py                # prompt → constrained generate → parse
│   └── parser.py                  # extract typed values from raw model output
├── training/                      # parametric adaptation
│   ├── qlora.py                   # QLoRA RAFT training (EXP-003)
│   └── clm.py                     # CLM continued pretraining (EXP-004)
├── evaluation/                    # scoring and metrics
│   ├── runner.py                  # EvalRunner: reusable evaluation orchestrator
│   ├── s2_runner.py               # retrieval + adapter generation runner
│   ├── s2_closed_runner.py        # adapter-only (no-retrieval) runner
│   ├── deterministic.py           # per-type deterministic scorers (S_det)
│   ├── judge.py                   # GPT-5.4-mini free-text judge (S_asst)
│   ├── grounding.py               # grounding scorer: F_β on (doc_id, page) pairs
│   ├── seed_stats.py              # multi-seed mean ± std aggregation
│   └── schemas.py                 # pydantic schemas for predictions and results
├── data/                          # dataset construction
│   ├── io.py                      # goldset / JSON load and save helpers
│   ├── splits.py                  # frozen train/eval split loading
│   ├── raft.py                    # RAFT-style training set (gold + distractors)
│   └── closed_book.py             # closed-book training set (question → answer)
└── d2l/                           # Doc-to-LoRA control (EXP-004_d2l_monolithic)
    ├── checkpoint.py              # hypernetwork checkpoint load + HF auto-download
    ├── packaging.py               # generate and export per-document D2L adapters
    ├── adapter_io.py              # export helpers for generated adapters
    ├── corpus.py                  # corpus extraction for D2L packaging
    ├── runner.py                  # no-retrieval inference runner
    ├── prompt.py                  # frozen no-retrieval prompt content
    └── sanity.py                  # per-document deterministic sanity checks
```

## Components

- **`rag_pipeline/`** — the retrieval engine, self-contained and configured by a single
  `PipelineConfig` dataclass. Implements the five stages described in the paper (§3.3):
  - **`ingestion/`** — parse PDFs (PyMuPDF), detect and serialize tables, and assemble a
    canonical corpus with stable `doc_id`s.
  - **`indexing/`** — hierarchical chunking (page / section / clause / microchunk / table),
    dense embeddings (Qwen3-Embedding-0.6B) and sparse BM25, persisted to a Qdrant hybrid
    index.
  - **`retrieval/`** — hybrid search with RRF fusion, Qwen3-Reranker-0.6B cross-encoder
    reranking, page-diverse evidence compression, and page lifting for grounding;
    `service.py` runs the full chain.

- **`retrieval/`** — thin project wrappers that bind `rag_pipeline` to this project's data
  and 8 GB budget. `indexer.py` builds the corpus index; `staged.py` runs retrieval with
  sequential model loading so the embedder, reranker, and generator never co-reside in VRAM.

- **`generation/`** — loads Gemma-2-2b-it in 4-bit NF4, applies the RAFT / CLM / merged
  adapter when present, formats the frozen prompt, generates (greedy; constrained decoding
  via Outlines for boolean and name types), and parses typed answers.

- **`training/`** — the two adaptation signals on an identical QLoRA configuration:
  `qlora.py` for RAFT-style supervised fine-tuning, `clm.py` for supervision-free continued
  pretraining on the raw corpus.

- **`evaluation/`** — `runner.py` orchestrates inference and scoring; `deterministic.py`
  computes `S_det` per answer type; `judge.py` computes `S_asst` via the GPT-5.4-mini judge;
  `grounding.py` computes `G = F_β (β=2.5)`; `seed_stats.py` aggregates across the three
  seeds. Inference variants split into retrieval-aware (`s2_runner.py`) and no-retrieval
  (`s2_closed_runner.py`).

- **`data/`** — goldset I/O, the frozen 150/50 split, and builders for the two training
  formats (`raft.py`: question + gold chunks + 2 distractors → answer; `closed_book.py`:
  question → answer).

- **`d2l/`** — utilities for the Doc-to-LoRA control. `checkpoint.py` loads the SakanaAI
  hypernetwork checkpoint (auto-downloaded from the Hugging Face Hub when absent) with a
  low-memory shim for the 8 GB budget; the rest generate, export, and run per-document
  adapters without retrieval. Requires the `d2l` optional dependency and a CUDA runtime
  (see the repository README).
