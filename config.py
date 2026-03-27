"""
Global project configuration.

Experiment-specific configs should be in experiments/EXP-XXX/config.py
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "150"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MANIFESTS = PROJECT_ROOT / "data" / "manifests"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEEDS = [42, 123, 777]  # 3 seeds for S2 variance estimation
DEFAULT_SEED = RANDOM_SEEDS[0]

# =============================================================================
# Backbone
# =============================================================================

BACKBONE_MODEL = "google/gemma-2-2b-it"
BACKBONE_QUANTIZATION = "nf4"  # 4-bit for QLoRA

# =============================================================================
# Doc-to-LoRA
# =============================================================================

DOC2LORA_CHECKPOINT = "trained_d2l/gemma_demo/checkpoint-80000"
DOC2LORA_MAX_CONTEXT_TOKENS = 32_000  # approximate safe limit

# =============================================================================
# Corpus & Goldset
# =============================================================================

CORPUS_DIR = DATA_RAW / "documents" / "pdfs"
GOLDSET_PATH = DATA_RAW / "dev-gold-150-v1.benchmark.json"
QUESTIONS_PATH = DATA_RAW / "questions" / "questions.json"

GOLDSET_SIZE = 150
DEV_SIZE = 120
TEST_SIZE = 30

# =============================================================================
# S2 QLoRA Defaults
# =============================================================================

QLORA_RANK = 32
QLORA_TARGET_MODULES = ["q_proj", "v_proj"]
QLORA_ALPHA_CANDIDATES = [16, 32, 64]
QLORA_DROPOUT_CANDIDATES = [0.0, 0.05, 0.1]
QLORA_LR_CANDIDATES = [5e-5, 1e-4, 2e-4, 4e-4]

# =============================================================================
# S4 Clustering
# =============================================================================

N_CLUSTERS = 4
CLUSTERING_METHOD = "kmeans"
ROUTING_METRIC = "cosine"

# =============================================================================
# Evaluation
# =============================================================================

PRIMARY_METRIC = "Q_main"
Q_MAIN_WEIGHTS = {"S_det": 0.7, "S_asst": 0.3}
GROUNDING_BETA = 2.5

# Judge
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_REASONING = "medium"

# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
