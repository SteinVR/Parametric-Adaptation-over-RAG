"""
Global project configuration.

This file contains all global settings, paths, and hyperparameters.
Experiment-specific configs should be in experiments/EXP-XXX/config.py
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# Data Configuration
# =============================================================================

# Target column name
TARGET_COL = "target"

# Train/Val/Test split ratios

# =============================================================================
# Model Configuration (Baseline)
# =============================================================================

BASELINE_MODEL = "random_forest"
BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

# =============================================================================
# Evaluation
# =============================================================================

# Primary metric for model selection
PRIMARY_METRIC = "f1"

# Secondary metrics to track
SECONDARY_METRICS = ["accuracy", "precision", "recall", "auc"]

# Cross-validation folds
CV_FOLDS = 5

# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
