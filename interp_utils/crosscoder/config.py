from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CROSSCODER_RESULTS_DIR = PROJECT_ROOT / "crosscoder" / "results"
# HuggingFace `datasets` save_to_disk cache for chat-normalized preference prompts (reused across runs).
NORMALIZED_PROMPTS_CACHE_DIR = PROJECT_ROOT / "crosscoder" / "cache" / "normalized_prompts"

SEED = 42

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

LEARNING_RATE = 3e-4
WARMUP_FRACTION = 0.05
BATCH_SIZE = 32
# Full forward through both LMs is VRAM-heavy; raise if you have headroom.
EXTRACT_BATCH_SIZE = 32
NUM_EPOCHS = 4
# Crosscoder is small; AMP mainly cuts activation memory during topk/linear.
USE_TRAIN_AMP = True
# Batched inference over all samples in analyze.py
ANALYZE_FEATURE_BATCH_SIZE = 64
CHECKPOINT_EVERY = 1

LAMBDA_SPARSITY = 1e-3
LAMBDA_CROSS = 0.4
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
LAMBDA_SHARED_MULTIPLIER = 0.05
FORCED_SHARED_FRACTION = 0.06

# Single-stream LLM hidden activations
TOPK_LLM = 400
EXPANSION_FACTOR_LLM = 8

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
