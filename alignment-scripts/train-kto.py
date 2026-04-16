import os
from pathlib import Path
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned-kto"
OUTPUT_DIR = "./outputs/SmolLM3-3B-KTO"
RUN_NAME = "SmolLM3-3B-KTO"
REPO_ID = "JonJacob/SmolLM3-3B-KTO"
MERGED_REPO_ID = "JonJacob/SmolLM3-3B-KTO-merged"

def _to_text(value, tokenizer):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [str(item.get("content", "") if isinstance(item, dict) else item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)

def _normalize_example(example, tokenizer):
    prompt = _to_text(example.get("prompt", ""), tokenizer)
    completion = _to_text(example.get("completion", ""), tokenizer)
    label = bool(example.get("label", False))
    return {"prompt": prompt, "completion": completion, "label": label}

def _latest_checkpoint_path(output_dir: str):
    out_path = Path(output_dir)
    if not out_path.exists():
        return None
    checkpoints = [p for p in out_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])

# ====================== LOGIN ======================
hf_token = os.getenv("HF_TOKEN")
wandb_key = os.getenv("WANDB_API_KEY")

if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not found in environment. You will not be able to push to HF.")

if wandb_key:
    wandb.login(key=wandb_key)
else:
    print("WANDB_API_KEY not found — training will run without Wandb logging.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(_normalize_example, fn_kwargs={"tokenizer": tokenizer})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

kto_config = KTOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_torch",
    max_length=512,
    report_to="wandb" if wandb_key else "none",
    run_name=RUN_NAME,
    remove_unused_columns=False,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = KTOTrainer(
    model=model,
    args=kto_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Resume if checkpoint exists
resume_ckpt = _latest_checkpoint_path(OUTPUT_DIR)
trainer.train(resume_from_checkpoint=resume_ckpt)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished. Model saved locally.")

if hf_token:
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
    trainer.model.push_to_hub(REPO_ID, token=hf_token)
    tokenizer.push_to_hub(REPO_ID, token=hf_token)

    # Merge and push dense model
    merged_model = trainer.model.merge_and_unload()
    dense_output_dir = f"{OUTPUT_DIR}-merged"
    Path(dense_output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(dense_output_dir)
    tokenizer.save_pretrained(dense_output_dir)

    api.create_repo(repo_id=MERGED_REPO_ID, private=False, exist_ok=True)
    merged_model.push_to_hub(MERGED_REPO_ID, token=hf_token)
    tokenizer.push_to_hub(MERGED_REPO_ID, token=hf_token)

    print(f"Pushed to HF:\n   Adapter → {REPO_ID}\n   Merged  → {MERGED_REPO_ID}")
else:
    print("Skipped pushing to HF (no HF_TOKEN).")

print(" KTO training pipeline complete!")
