"""
Local supervised fine-tuning on HuggingFaceH4/ultrachat_200k (train_sft) with TRL SFTTrainer.

Run from repo root:
  python SFT/train.py train --light-run
  python SFT/train.py train --repo-id your/name --dry-run

Optional env: HF_TOKEN (Hub auth / gated assets), WANDB_API_KEY (logging; omitted => report_to none).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

from training.config import (  # noqa: E402
    DATASET_NAME,
    DATASET_SPLIT,
    DISABLE_THINKING,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    RUN_NAME,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
)
from training.utils import (  # noqa: E402
    ProgressLogger,
    configure_pytorch_cuda_for_training,
    latest_checkpoint_path,
    release_cuda_memory,
)


def _valid_messages(example: dict) -> bool:
    msgs = example.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return False
    return any(m.get("role") == "assistant" for m in msgs)


def _messages_to_text(example: dict, tokenizer) -> dict:
    msgs = example["messages"]
    try:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=not DISABLE_THINKING,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
    return {"text": text}


def train_and_push(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
    dry_run: bool = False,
    light_run: bool = False,
) -> str:
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    report_to = "wandb" if wandb_key else "none"

    if hf_token:
        login(token=hf_token)
        print("[check] HuggingFace login OK (HF_TOKEN set)")
    else:
        print("[check] HF_TOKEN not set — using anonymous Hub access where possible")

    if report_to == "wandb":
        import wandb
        wandb.login(key=wandb_key)
        print("[check] W&B login OK")
    else:
        print("[check] WANDB_API_KEY not set — logging disabled (report_to=none)")

    configure_pytorch_cuda_for_training()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[check] tokenizer loaded: {MODEL_NAME}")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, token=hf_token)
    print(f"[check] dataset loaded: {DATASET_NAME} split={DATASET_SPLIT} — {len(dataset)} rows")

    dataset = dataset.filter(_valid_messages)
    print(f"[check] after filter: {len(dataset)} rows")

    if dry_run:
        row = dataset[0]
        text_row = _messages_to_text(row, tokenizer)
        print(f"[dry-run] sample text length={len(text_row['text'])} chars")
        return "[dry-run] All checks passed. Model not loaded, no training, no push."

    dataset = dataset.map(
        lambda ex: _messages_to_text(ex, tokenizer),
        remove_columns=["prompt", "prompt_id", "messages"],
    )
    print(f"[check] dataset prepared, columns: {dataset.column_names}")

    release_cuda_memory()

    use_cuda = torch.cuda.is_available()
    # Native weight dtype only; Trainer fp16/bf16 flags are disabled (no AMP / GradScaler).
    if use_cuda and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    print(f"[check] model loaded: {MODEL_NAME}")

    if light_run:
        dataset = dataset.select(range(min(20, len(dataset))))
        print(f"[light-run] dataset sliced to {len(dataset)} rows")

    sft_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1 if light_run else LOGGING_STEPS,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=RUN_NAME,
        remove_unused_columns=False,
        dataset_text_field="text",
    )
    if light_run:
        sft_kwargs.update({"max_steps": 5, "save_strategy": "no"})
    else:
        sft_kwargs.update(
            {
                "num_train_epochs": NUM_TRAIN_EPOCHS,
                "save_strategy": "steps",
                "save_steps": SAVE_STEPS,
                "save_total_limit": SAVE_TOTAL_LIMIT,
            }
        )
    sft_config = SFTConfig(**sft_kwargs)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    total_steps = sft_kwargs.get("max_steps") or (
        len(dataset)
        // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps)
        * int(sft_config.num_train_epochs)
    )
    progress_logger = ProgressLogger(total_steps=max(1, total_steps))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[progress_logger.callback],
    )

    resume_ckpt = None if light_run else latest_checkpoint_path(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    release_cuda_memory()

    if light_run:
        return "[light-run] Training completed (5 steps, ≤20 rows). No model saved or pushed."

    if not repo_id:
        raise ValueError("repo_id is required after training to save/push (or use --light-run).")

    if not hf_token:
        raise ValueError("HF_TOKEN is required to push to the Hub after a full training run.")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    HfApi(token=hf_token).create_repo(repo_id=repo_id, private=private, exist_ok=True)
    trainer.model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    pushed_urls = [f"https://huggingface.co/{repo_id} (adapter)"]

    if push_merged:
        dense_repo_id = merged_repo_id or f"{repo_id}-merged"
        print(f"Merging LoRA adapter into dense model for repo: {dense_repo_id}")

        release_cuda_memory()
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
        release_cuda_memory()

    return "Finished training and pushed: " + ", ".join(pushed_urls)


def push_from_local(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
) -> str:
    from huggingface_hub import HfApi, login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_pytorch_cuda_for_training()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is required for push_from_local")

    login(token=hf_token)

    pushed_urls: list[str] = []

    print(f"Loading model from {OUTPUT_DIR} ...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, token=hf_token)

    HfApi(token=hf_token).create_repo(repo_id=repo_id, private=private, exist_ok=True)
    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)
    pushed_urls.append(f"https://huggingface.co/{repo_id}")
    print(f"Pushed adapter/model to {repo_id}")
    del model
    release_cuda_memory()

    if push_merged:
        merged_dir = f"{OUTPUT_DIR}-merged"
        dense_repo_id = merged_repo_id or f"{repo_id}-merged"

        print(f"Loading merged model from {merged_dir} ...")
        merged_model = AutoModelForCausalLM.from_pretrained(merged_dir, token=hf_token)
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_dir, token=hf_token)

        HfApi(token=hf_token).create_repo(repo_id=dense_repo_id, private=private, exist_ok=True)
        merged_model.push_to_hub(dense_repo_id, token=hf_token)
        merged_tokenizer.push_to_hub(dense_repo_id, token=hf_token)
        pushed_urls.append(f"https://huggingface.co/{dense_repo_id}")
        print(f"Pushed merged model to {dense_repo_id}")
        del merged_model
        release_cuda_memory()

    return "Pushed: " + ", ".join(pushed_urls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run SFT on ultrachat_200k train_sft")
    train_parser.add_argument("--repo-id", default="")
    train_parser.add_argument("--private", action="store_true")
    train_parser.add_argument("--no-push-merged", dest="push_merged", action="store_false")
    train_parser.add_argument("--merged-repo-id", default="")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--light-run", action="store_true")

    push_parser = subparsers.add_parser("push", help="Re-push from local output dir")
    push_parser.add_argument("--repo-id", required=True)
    push_parser.add_argument("--private", action="store_true")
    push_parser.add_argument("--no-push-merged", dest="push_merged", action="store_false")
    push_parser.add_argument("--merged-repo-id", default="")

    args = parser.parse_args()

    if args.command == "train":
        if not args.dry_run and not args.light_run and not args.repo_id:
            parser.error("--repo-id is required for a full training run")
        message = train_and_push(
            repo_id=args.repo_id,
            private=args.private,
            push_merged=args.push_merged,
            merged_repo_id=args.merged_repo_id.strip() or None,
            dry_run=args.dry_run,
            light_run=args.light_run,
        )
        print(message)

    elif args.command == "push":
        message = push_from_local(
            repo_id=args.repo_id,
            private=args.private,
            push_merged=args.push_merged,
            merged_repo_id=args.merged_repo_id.strip() or None,
        )
        print(message)
