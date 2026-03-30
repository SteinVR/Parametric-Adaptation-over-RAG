"""Experiment-specific configuration for EXP-008: S6 Naive Dense RAG."""

from __future__ import annotations

from pathlib import Path

import config as base_cfg

EXPERIMENT_ID = "EXP-008"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

# S6 FAISS index (self-contained, separate from S1 Qdrant)
FAISS_INDEX_DIR = RESULTS_DIR / "faiss_index"

# Delta reference
EXP002_REPORT_PATH = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"

# S6 frozen decisions
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 5
ENABLED_CHUNK_FAMILIES = {"microchunk"}
TOKEN_CHUNK_SIZE = 300
TOKEN_CHUNK_OVERLAP = 50
MAX_NEW_TOKENS = 256
SMOKE_EVAL_QUESTIONS = 2
