"""Experiment-specific configuration for EXP-010 Adapter Merge (CLM + RAFT)."""

from __future__ import annotations

from pathlib import Path

import config as base_cfg

EXPERIMENT_ID = "EXP-010"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

# Adapter sources
CLM_MODELS_DIR = base_cfg.MODELS_DIR / "clm"
RAFT_MODELS_DIR = base_cfg.MODELS_DIR / "qlora"

# Merge weight: α * CLM + (1-α) * RAFT
MERGE_ALPHA = 0.5

# Retrieval index from EXP-002
INDEX_OUTPUT_DIR = base_cfg.RESULTS_DIR / "EXP-002" / "index"

# Delta reference paths
EXP002_REPORT_PATH = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"
EXP003_AGGREGATE_PATH = base_cfg.RESULTS_DIR / "EXP-003" / "aggregate_summary.json"
EXP004B_AGGREGATE_PATH = base_cfg.RESULTS_DIR / "EXP-004b" / "aggregate_summary.json"

TRAIN_SEEDS = tuple(base_cfg.RANDOM_SEEDS)
MAX_NEW_TOKENS = 256
SMOKE_EVAL_QUESTIONS = 2
