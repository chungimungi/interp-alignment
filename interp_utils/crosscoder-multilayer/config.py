import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CROSSCODER_RESULTS_DIR = PACKAGE_DIR / "results"
# HuggingFace `datasets` save_to_disk cache for chat-normalized preference prompts (reused across runs).
NORMALIZED_PROMPTS_CACHE_DIR = PACKAGE_DIR / "cache" / "normalized_prompts"
# Base model activation cache - keyed by (base_model, layer, position, dataset); reused across aligned runs.
BASE_ACTIVATIONS_CACHE_DIR = PACKAGE_DIR / "cache" / "base_activations"

SEED = int(os.environ.get("CROSSCODER_SEED", 42))

# Preference dataset
PREFERENCE_DATASET_NAME = "argilla/ultrafeedback-multi-binarized-preferences-cleaned"
MAX_PROMPT_TOKENS = 512
DISABLE_THINKING = True
VAL_FRACTION = 0.1

# GPU optimization
NUM_WORKERS = 4
PIN_MEMORY = True
CUDA_OPTIMIZATIONS = True
FLUSH_GPU_EVERY_N_EPOCHS = 10
FLUSH_GPU_EVERY_N_BATCHES = 50

LEARNING_RATE = float(os.environ.get("CROSSCODER_LR", 3e-4))
WARMUP_FRACTION = 0.05
BATCH_SIZE = 32
# Full forward through both LMs is VRAM-heavy; raise if you have headroom.
EXTRACT_BATCH_SIZE = 32
PROGRESS_LOG_EVERY_N_BATCHES = int(os.environ.get("CROSSCODER_PROGRESS_EVERY_N_BATCHES", 100))
NUM_EPOCHS = int(os.environ.get("CROSSCODER_NUM_EPOCHS", 4))
# Crosscoder is small; AMP mainly cuts activation memory during topk/linear.
USE_TRAIN_AMP = True
# Batched inference over all samples in analyze.py
ANALYZE_FEATURE_BATCH_SIZE = 64
CHECKPOINT_EVERY = 1

LAMBDA_SPARSITY = float(os.environ.get("CROSSCODER_LAMBDA_SPARSITY", 1e-3))
LAMBDA_CROSS = float(os.environ.get("CROSSCODER_LAMBDA_CROSS", 0.4))
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
LAMBDA_SHARED_MULTIPLIER = float(os.environ.get("CROSSCODER_LAMBDA_SHARED_MULT", 0.05))
FORCED_SHARED_FRACTION = float(os.environ.get("CROSSCODER_FORCED_SHARED_FRAC", 0.06))

# Single-stream LLM hidden activations
TOPK_LLM = int(os.environ.get("CROSSCODER_TOPK", 400))
EXPANSION_FACTOR_LLM = int(os.environ.get("CROSSCODER_EXPANSION", 8))
MULTILAYER_TOPK_MODE = os.environ.get("CROSSCODER_MULTILAYER_TOPK_MODE", "model_balanced_layer_agg")
MULTILAYER_TOPK_MODES = ("model_balanced_layer_agg", "global_sum")

# Quality Control
FVE_THRESHOLD = 0.5
DEAD_NEURON_THRESHOLD = 1   # ideally, as low as possible, 1 being highest. But, currently, the dead neuron ratio is already close 0.98

# GMM / fixed thresholds for rho–theta classification
RHO_BASE_ONLY = 0.15
RHO_ALIGNED_ONLY = 0.85
RHO_SHARED_LOW = 0.35
RHO_SHARED_HIGH = 0.65
THETA_ALIGNED = 0.80
THETA_REDIRECTED = 0.50

SUPERPOSITION_R2_THRESHOLD = 0.8
SUPERPOSITION_MAX_CONSTITUENTS = 50
SUPERPOSITION_TOP_SAMPLES = 100

POSITION_LAST_PROMPT = "last_prompt"
POSITION_MEAN_PROMPT = "mean_prompt"
POSITION_CHOICES = (POSITION_LAST_PROMPT, POSITION_MEAN_PROMPT)
