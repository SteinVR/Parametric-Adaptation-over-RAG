"""Experiment-specific configuration for EXP-004 CLM."""

from __future__ import annotations

from pathlib import Path

import config as base_cfg

EXPERIMENT_ID = "EXP-004"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-004_clm"
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"
MODELS_DIR = base_cfg.MODELS_DIR / "clm"

# Delta reference paths
EXP002_REPORT_PATH = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"
EXP003_AGGREGATE_PATH = base_cfg.RESULTS_DIR / "EXP-003" / "aggregate_summary.json"
EXP003B_AGGREGATE_PATH = base_cfg.RESULTS_DIR / "EXP-003b" / "aggregate_summary.json"

TRAIN_SEEDS = tuple(base_cfg.RANDOM_SEEDS)
MAX_NEW_TOKENS = 256
SMOKE_EVAL_QUESTIONS = 2

# CLM training overrides (batch=1 on 8GB VRAM)
# 512 instead of 2048: CLM computes loss on ALL tokens, producing a
# (seq × 256K vocab) logit tensor. cross_entropy internally upcasts to fp32:
# 512 × 256K × 4B ≈ 500 MB. Fits 8 GB with 4-bit model (~3.3 GB).
# RAFT avoids this via logits_to_keep (answer-only suffix loss).
CLM_MAX_SEQ_LENGTH = 512
CLM_LR = 5e-5              # continued pretraining standard; 2e-4 too aggressive for 106K tokens
CLM_EPOCHS = 5             # diffuse signal needs more passes; safe at low LR
CLM_WARMUP_RATIO = 0.1     # longer warmup stabilizes low-LR training
TRAIN_MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
