"""Experiment-specific configuration for EXP-006 Main Comparison Table."""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as base_cfg

EXPERIMENT_ID = "EXP-006"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

# ---------------------------------------------------------------------------
# Source data paths
# ---------------------------------------------------------------------------

# S1 — single run, EXP-002
S1_EVAL_REPORT = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"
S1_SYSTEMS_METRICS = base_cfg.RESULTS_DIR / "EXP-002" / "systems_metrics.json"

# S2+R — 3 seeds, EXP-003
S2R_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-003" / "aggregate_summary.json"

# S3+R — 3 seeds, EXP-004b
S3R_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-004b" / "aggregate_summary.json"

# S2 — 3 seeds, EXP-003b
S2_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-003b" / "aggregate_summary.json"

# S3 — 3 seeds, EXP-004_clm
S3_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-004_clm" / "aggregate_summary.json"

# S6 — single run, EXP-008/main
S6_EVAL_REPORT = base_cfg.RESULTS_DIR / "EXP-008" / "main" / "eval_report.json"
S6_SYSTEMS_METRICS = base_cfg.RESULTS_DIR / "EXP-008" / "main" / "systems_metrics.json"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

MAIN_RESULTS_CSV = RESULTS_DIR / "main_results.csv"
PER_TYPE_BREAKDOWN_CSV = RESULTS_DIR / "per_type_breakdown.csv"
DELTAS_JSON = RESULTS_DIR / "deltas.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_TYPES = ["number", "boolean", "name", "names", "date", "free_text"]

SYSTEMS_ORDER = ["S1", "S2+R", "S3+R", "S2", "S3", "S6"]
