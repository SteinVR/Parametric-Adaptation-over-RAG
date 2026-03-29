"""Experiment-specific configuration for EXP-003."""

from __future__ import annotations

from pathlib import Path

import config as base_cfg

EXPERIMENT_ID = "EXP-003"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"
RAFT_DATASET_PATH = base_cfg.DATA_PROCESSED / "raft_train.jsonl"
EXP002_INDEX_DIR = base_cfg.RESULTS_DIR / "EXP-002" / "index"
EXP002_REPORT_PATH = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"
MODELS_DIR = base_cfg.MODELS_DIR / "qlora"

DISTRACTOR_SEED = base_cfg.DEFAULT_SEED
TRAIN_SEEDS = tuple(base_cfg.RANDOM_SEEDS)
MAX_NEW_TOKENS = 256
SMOKE_TRAIN_EXAMPLES = 4
SMOKE_EVAL_QUESTIONS = 2

TRAIN_MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
