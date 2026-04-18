# ── Model & Data ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
OUTPUT_DIR = "./outputs/SmolLM3-3B-ultrachat-sft"
RUN_NAME = "SmolLM3-3B-ultrachat-sft"
DISABLE_THINKING = True

# ── Training (SFT) ────────────────────────────────────────────────────────────
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
MAX_LENGTH = 2048

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
