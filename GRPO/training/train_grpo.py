import os
from pathlib import Path

import modal

from .app import app, image, hf_cache_vol, model_out_vol
from .config import (
    GPU, TIMEOUT,
    MODEL_NAME, DATASET_NAME, OUTPUT_DIR, RUN_NAME,
    NUM_GENERATIONS, MAX_COMPLETION_LENGTH,
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, LOGGING_STEPS,
    SAVE_STEPS, SAVE_TOTAL_LIMIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    image=image,
    secrets=[modal.Secret.from_dotenv(path=".env")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/outputs": model_out_vol,
    },
)
def train_and_push(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
    dry_run: bool = False,
    light_run: bool = False,
) -> str:
    import torch
    import wandb
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig, TaskType
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.dataset import prepare_example
    from training.rewards import make_preference_reward
    from training.utils import latest_checkpoint_path, ProgressLogger

    # ── 1. Credentials ────────────────────────────────────────────────────────
    hf_token = os.environ["HF_TOKEN"]
    wandb_key = os.environ["WANDB_API_KEY"]
    print("[check] env vars found: HF_TOKEN, WANDB_API_KEY")

    login(token=hf_token)
    print("[check] HuggingFace login OK")

    wandb.login(key=wandb_key)
    print("[check] W&B login OK")

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[check] tokenizer loaded: {MODEL_NAME}")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"[check] dataset loaded: {DATASET_NAME} — {len(dataset)} rows")

    dataset = dataset.map(prepare_example, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.select_columns(["prompt", "chosen", "chosen_rating"])
    print(f"[check] dataset prepared, columns: {dataset.column_names}")

    # ── 4. Reward function smoke-test ─────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    preference_reward = make_preference_reward(scorer)

    sample = dataset.select(range(2))
    dummy_completions = [[{"role": "assistant", "content": "test output"}]] * 2
    test_rewards = preference_reward(
        dummy_completions,
        chosen=sample["chosen"],
        chosen_rating=sample["chosen_rating"],
    )
    print(f"[check] reward fn smoke-test passed, sample rewards: {test_rewards}")

    if dry_run:
        return "[dry-run] All checks passed. Model not loaded, no training, no push."

    # ── 5. Model ──────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    print(f"[check] model loaded: {MODEL_NAME}")

    if light_run:
        dataset = dataset.select(range(20))
        print(f"[light-run] dataset sliced to {len(dataset)} rows")

    grpo_config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1 if light_run else LOGGING_STEPS,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
    )
    if light_run:
        grpo_config_kwargs.update({"max_steps": 5, "save_strategy": "no"})
    else:
        grpo_config_kwargs.update({
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "save_strategy": "steps",
            "save_steps": SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
        })
    grpo_config = GRPOConfig(**grpo_config_kwargs)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    total_steps = grpo_config_kwargs.get("max_steps") or (
        len(dataset) // (grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps)
        * grpo_config.num_train_epochs
    )
    progress_logger = ProgressLogger(total_steps=total_steps)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=[preference_reward],
        callbacks=[progress_logger.callback],
    )

    resume_ckpt = None if light_run else latest_checkpoint_path(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    if light_run:
        return "[light-run] Training completed (5 steps, 20 rows). No model saved or pushed."

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    HfApi(token=hf_token).create_repo(repo_id=repo_id, private=private, exist_ok=True)
    trainer.model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    pushed_urls = [f"https://huggingface.co/{repo_id} (adapter)"]

    if push_merged:
        dense_repo_id = merged_repo_id or f"{repo_id}-merged"
        print(f"Merging LoRA adapter into dense model for repo: {dense_repo_id}")

        merged_model = trainer.model.merge_and_unload()
        dense_output_dir = f"{OUTPUT_DIR}-merged"
        Path(dense_output_dir).mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(dense_output_dir)
        tokenizer.save_pretrained(dense_output_dir)

        HfApi(token=hf_token).create_repo(
            repo_id=dense_repo_id, private=private, exist_ok=True
        )
        merged_model.push_to_hub(dense_repo_id, token=hf_token)
        tokenizer.push_to_hub(dense_repo_id, token=hf_token)
        pushed_urls.append(f"https://huggingface.co/{dense_repo_id} (merged dense)")

    model_out_vol.commit()
    return "Finished training and pushed: " + ", ".join(pushed_urls)


@app.local_entrypoint()
def main(
    repo_id: str = "",
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str = "",
    dry_run: bool = False,
    light_run: bool = False,
    local_output_dir: str = "./outputs",
) -> None:
    import subprocess

    if not dry_run and not light_run and not repo_id:
        raise ValueError("--repo-id is required for a full training run")

    resolved_merged_repo_id = merged_repo_id.strip() or None
    message = train_and_push.remote(
        repo_id=repo_id,
        private=private,
        push_merged=push_merged,
        merged_repo_id=resolved_merged_repo_id,
        dry_run=dry_run,
        light_run=light_run,
    )
    print(message)

    if not dry_run and not light_run:
        print(f"Artifacts saved in Modal volume under: {OUTPUT_DIR}")
        print(f"Pulling model from volume to {local_output_dir} ...")
        volume_name = "grpo-model-outputs"
        for remote_path, label in [
            ("SmolLM3-3B-GRPO", "adapter"),
            ("SmolLM3-3B-GRPO-merged", "merged"),
        ]:
            local_path = f"{local_output_dir}/{remote_path}"
            result = subprocess.run(
                ["modal", "volume", "get", volume_name, remote_path, local_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"[pull] {label} → {local_path}")
            else:
                print(f"[pull] {label} failed: {result.stderr.strip()}")
