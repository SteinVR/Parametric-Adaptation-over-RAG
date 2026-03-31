"""
EXP-006: Main Comparison Table
Analysis-only: collects results from all completed experiments and produces
the main comparison table, per-answer-type breakdown, key deltas, and REPORT.md.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_json, save_json  # noqa: E402
import config as cfg  # noqa: E402

logging.basicConfig(
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATE_FORMAT,
    level=logging.INFO,
)
log = logging.getLogger(__name__)
