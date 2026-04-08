# ── Compute ───────────────────────────────────────────────────────────────────
GPU = "L40S"
TIMEOUT = 60 * 60 * 24     # 24 hours

# ── Model & Data ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "/root/outputs/SmolLM3-3B-GRPO"
RUN_NAME = "SmolLM3-3B-GRPO"
DISABLE_THINKING = True

# ── Sequence lengths ──────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS = 1024       # prompt truncation at dataset prep (p99 ≈ 790 words)
MAX_COMPLETION_LENGTH = 768    # max tokens the model generates per completion (p95 ≈ 537 words)

# ── GRPO ──────────────────────────────────────────────────────────────────────
NUM_GENERATIONS = 4            # completions sampled per prompt (G); higher = better signal, more compute

# ── Training ──────────────────────────────────────────────────────────────────
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-6
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
