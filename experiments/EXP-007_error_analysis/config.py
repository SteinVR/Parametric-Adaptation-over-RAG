"""Experiment-specific configuration for EXP-007 analysis refresh."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as base_cfg

EXPERIMENT_ID = "EXP-007"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = base_cfg.RESULTS_DIR / EXPERIMENT_ID
FIGURES_DIR = base_cfg.RESULTS_DIR / "figures"
REPORT_PATH = EXPERIMENT_DIR / "REPORT.md"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
GOLDSET_PATH = base_cfg.GOLDSET_PATH
SPLIT_PATH = base_cfg.DATA_SPLITS / "split_v1.json"

# ---------------------------------------------------------------------------
# Source experiment artifacts (S6 intentionally excluded)
# ---------------------------------------------------------------------------
S1_DIR = base_cfg.RESULTS_DIR / "EXP-002"
S2R_DIR = base_cfg.RESULTS_DIR / "EXP-003"
S3R_DIR = base_cfg.RESULTS_DIR / "EXP-004b"
S7_DIR = base_cfg.RESULTS_DIR / "EXP-010" / "alpha_0.5"
S2_DIR = base_cfg.RESULTS_DIR / "EXP-003b"
S3_DIR = base_cfg.RESULTS_DIR / "EXP-004_clm"

S2R_AGGREGATE = S2R_DIR / "aggregate_summary.json"
S3R_AGGREGATE = S3R_DIR / "aggregate_summary.json"
S7_AGGREGATE = S7_DIR / "aggregate_summary.json"
S2_AGGREGATE = S2_DIR / "aggregate_summary.json"
S3_AGGREGATE = S3_DIR / "aggregate_summary.json"

# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------
CONSOLIDATED_RESULTS_CSV = RESULTS_DIR / "consolidated_results.csv"
PER_TYPE_BREAKDOWN_CSV = RESULTS_DIR / "per_type_breakdown.csv"
SEED_STABILITY_CSV = RESULTS_DIR / "seed_stability.csv"
PAIRWISE_WIN_RATE_CSV = RESULTS_DIR / "pairwise_win_rate.csv"
ERROR_OVERLAP_CSV = RESULTS_DIR / "error_overlap_jaccard.csv"
DIFFICULTY_PROFILE_CSV = RESULTS_DIR / "difficulty_profile.csv"
JUDGE_CRITERIA_CSV = RESULTS_DIR / "judge_criteria_profile.csv"
HEADLINE_FAILURES_CSV = RESULTS_DIR / "headline_failure_slices.csv"
DEEP_ANALYSIS_MD = RESULTS_DIR / "deep_analysis.md"
ERROR_ANALYSIS_MD = RESULTS_DIR / "error_analysis.md"

MAIN_RESULTS_TABLE_PNG = FIGURES_DIR / "main_results_table.png"
COST_QUALITY_SCATTER_PNG = FIGURES_DIR / "cost_quality_scatter.png"
PER_TYPE_HEATMAP_PNG = FIGURES_DIR / "per_type_heatmap.png"
LATENCY_GROUNDING_SCATTER_PNG = FIGURES_DIR / "latency_grounding_scatter.png"
ERROR_OVERLAP_HEATMAP_PNG = FIGURES_DIR / "error_overlap_heatmap.png"
PAIRWISE_WIN_HEATMAP_PNG = FIGURES_DIR / "pairwise_win_heatmap.png"
DIFFICULTY_PROFILE_PNG = FIGURES_DIR / "difficulty_profile.png"
JUDGE_CRITERIA_PNG = FIGURES_DIR / "judge_criteria_profile.png"
SEED_STABILITY_PNG = FIGURES_DIR / "seed_stability.png"
PARETO_FRONTIER_PNG = FIGURES_DIR / "pareto_frontier.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = (42, 123, 777)
REPRESENTATIVE_SEED = 42
ANSWER_TYPES = ["boolean", "number", "name", "names", "date", "free_text"]
DETERMINISTIC_TYPES = ["boolean", "number", "name", "names", "date"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
Q_MAIN_WEIGHTS = base_cfg.Q_MAIN_WEIGHTS

HEADLINE_SYSTEMS = ["S1", "S2+R", "S3+R", "S7"]
CONTROL_SYSTEMS = ["S2", "S3"]
ALL_SYSTEMS = HEADLINE_SYSTEMS + CONTROL_SYSTEMS
