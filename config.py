"""
Global project configuration.

Experiment-specific configs should be in experiments/EXP-XXX/config.py
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_CORPUS = PROJECT_ROOT / "data" / "corpus"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_GOLDSET = PROJECT_ROOT / "data" / "goldset"
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

# Pretrained hypernetwork checkpoint on the Hugging Face Hub. When the local path
# above is absent, the loader fetches this file from the Hub (see src/d2l/checkpoint.py).
DOC2LORA_HF_REPO = "SakanaAI/doc-to-lora"
DOC2LORA_HF_CHECKPOINT_SUBFOLDER = "gemma_demo/checkpoint-80000"

# =============================================================================
# Corpus & Goldset
# =============================================================================

CORPUS_DIR = DATA_CORPUS
GOLDSET_PATH = DATA_GOLDSET / "goldset.benchmark.json"
QUESTIONS_PATH = DATA_GOLDSET / "goldset.questions.json"

N_DOCUMENTS = 8
GOLDSET_SIZE = 200
S2_TRAIN_SIZE = 150
EVAL_SIZE = 50

# =============================================================================
# S2 QLoRA Defaults
# =============================================================================

QLORA_RANK = 32
QLORA_ALPHA = 32
QLORA_DROPOUT = 0.05
QLORA_TARGET_MODULES = ["q_proj", "v_proj"]
QLORA_LR = 2e-4
QLORA_EPOCHS = 3
QLORA_MAX_SEQ_LEN = 4096
QLORA_BATCH_SIZE = 4
QLORA_WARMUP_RATIO = 0.03
QLORA_WEIGHT_DECAY = 0.01
RAFT_N_DISTRACTORS = 2  # all examples are oracle (gold + distractors), no distractor-only

# =============================================================================
# Retrieval & Embedding
# =============================================================================

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RETRIEVAL_TOP_K = 5

# =============================================================================
# S4 Clustering & Routing
# =============================================================================

N_CLUSTERS = 4
CLUSTERING_METHOD = "kmeans"
ROUTING_METRIC = "cosine"
ROUTING_STRATEGY = "hard_top1"

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
