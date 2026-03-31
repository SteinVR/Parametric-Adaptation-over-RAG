"""Experiment-specific configuration for EXP-006 Main Comparison Table."""

from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as base_cfg

EXPERIMENT_ID = "EXP-006"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
FIGURES_DIR = base_cfg.RESULTS_DIR / "figures"
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

# ---------------------------------------------------------------------------
# Source data paths
# ---------------------------------------------------------------------------

# S1 — single run, EXP-002
S1_EVAL_REPORT = base_cfg.RESULTS_DIR / "EXP-002" / "eval_report.json"
S1_EVAL_RESULTS = base_cfg.RESULTS_DIR / "EXP-002" / "eval_results.json"
S1_SYSTEMS_METRICS = base_cfg.RESULTS_DIR / "EXP-002" / "systems_metrics.json"

# S2+R — 3 seeds, EXP-003
S2R_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-003" / "aggregate_summary.json"
S2R_RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-003"

# S3+R — 3 seeds, EXP-004b
S3R_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-004b" / "aggregate_summary.json"
S3R_RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-004b"

# S2 — 3 seeds, EXP-003b
S2_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-003b" / "aggregate_summary.json"
S2_RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-003b"

# S3 — 3 seeds, EXP-004_clm
S3_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-004_clm" / "aggregate_summary.json"
S3_RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-004_clm"

# S3-legacy (D2L) — single run, EXP-004
S3_LEGACY_EVAL_REPORT = base_cfg.RESULTS_DIR / "EXP-004" / "eval_report.json"
S3_LEGACY_EVAL_RESULTS = base_cfg.RESULTS_DIR / "EXP-004" / "eval_results.json"
S3_LEGACY_SYSTEMS_METRICS = base_cfg.RESULTS_DIR / "EXP-004" / "systems_metrics.json"
S3_LEGACY_DOC_GENERATION = base_cfg.RESULTS_DIR / "EXP-004" / "document_generation.json"
S3_LEGACY_MERGE_SUMMARY = base_cfg.RESULTS_DIR / "EXP-004" / "merge_summary.json"

# S7 — 3 seeds, EXP-010 adapter merge
S7_AGGREGATE = base_cfg.RESULTS_DIR / "EXP-010" / "alpha_0.5" / "aggregate_summary.json"
S7_RESULTS_DIR = base_cfg.RESULTS_DIR / "EXP-010" / "alpha_0.5"

SEEDS = (42, 123, 777)
Q_MAIN_WEIGHTS = base_cfg.Q_MAIN_WEIGHTS
GOLDSET_PATH = base_cfg.GOLDSET_PATH
SPLIT_PATH = base_cfg.DATA_SPLITS / "split_v1.json"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

MAIN_RESULTS_CSV = RESULTS_DIR / "main_results.csv"
PER_TYPE_BREAKDOWN_CSV = RESULTS_DIR / "per_type_breakdown.csv"
DELTAS_JSON = RESULTS_DIR / "deltas.json"
SINGLE_MULTI_CSV = RESULTS_DIR / "single_vs_multi_doc.csv"
GRADIENT_PLOT_PATH = RESULTS_DIR / "gradient_plot.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_TYPES = ["boolean", "number", "name", "names", "date", "free_text"]
DETERMINISTIC_TYPES = ["boolean", "number", "name", "names", "date"]

HEADLINE_SYSTEMS = ["S1", "S2+R", "S3+R", "S7"]
CONTROL_SYSTEMS = ["S2", "S3", "S3-legacy"]
SYSTEMS_ORDER = HEADLINE_SYSTEMS + CONTROL_SYSTEMS
