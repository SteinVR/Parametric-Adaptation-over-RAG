"""Experiment-specific configuration for EXP-004."""

from __future__ import annotations

from pathlib import Path

import config as base_cfg

EXPERIMENT_ID = "EXP-004"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

MODELS_DIR = base_cfg.MODELS_DIR / "d2l"
DOC_MODELS_DIR = MODELS_DIR
MONOLITHIC_MODEL_DIR = MODELS_DIR / "monolithic"
DOC_MODEL_DIRS = tuple(MODELS_DIR / f"doc{i}" for i in range(1, base_cfg.N_DOCUMENTS + 1))

DOC2LORA_CHECKPOINT_ROOT = base_cfg.PROJECT_ROOT / Path(base_cfg.DOC2LORA_CHECKPOINT)
DOC2LORA_CHECKPOINT_FILE = DOC2LORA_CHECKPOINT_ROOT / "pytorch_model.bin"

EXP002_INDEX_DIR = base_cfg.RESULTS_DIR / "EXP-002" / "index"
EXP002_REPORT_PATH = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"

TRAIN_SPLIT_NAME = "s2_train"
MAX_NEW_TOKENS = 256
SMOKE_DOCS = 2
SMOKE_EVAL_QUESTIONS = 2

