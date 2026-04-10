# ── Compute ───────────────────────────────────────────────────────────────────
GPU = "L40S"
TIMEOUT = 60 * 60 * 24     # 24 hours

# ── Model & Data ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "/root/outputs/SmolLM3-3B-SimPO"
RUN_NAME = "SmolLM3-3B-SimPO"
DISABLE_THINKING = True

# ── SimPO (arXiv:2405.14734) ──────────────────────────────────────────────────
# SimPO uses CPOTrainer with loss_type="simpo" — reference-free.
# Reward = average log probability (length-normalized).
# Objective = Bradley-Terry with target reward margin γ.
#
# Paper recommendations:
#   beta ∈ [2.0, 10.0]  — reward scaling (higher than DPO)
#   gamma_beta_ratio ∈ [0.0, 1.0]  — recommend 0.5
#   gamma = beta * gamma_beta_ratio  — target reward margin
#   cpo_alpha = 0.0  — pure SimPO (no behavior cloning)
BETA = 2.0
GAMMA_BETA_RATIO = 0.5
SIMPO_GAMMA = BETA * GAMMA_BETA_RATIO   # = 1.0
CPO_ALPHA = 0.0

# ── Training ──────────────────────────────────────────────────────────────────
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-7
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
MAX_LENGTH = 512

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
