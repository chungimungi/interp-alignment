import os
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "/root/outputs/Qwen3-4B-DPO"
RUN_NAME = "Qwen3-4B-DPO"
DISABLE_THINKING = True

app = modal.App("dpo-qwen3-training")

image = modal.Image.from_registry("python:3.11").pip_install(
    "torch",
    "transformers>=4.45.0",
    "datasets",
    "trl>=0.11.0",
    "peft",
    "accelerate",
    "bitsandbytes",
    "wandb",
    "huggingface_hub",
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_out_vol = modal.Volume.from_name("dpo-model-outputs", create_if_missing=True)


def _to_text(value, tokenizer) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        is_chat_messages = all(isinstance(item, dict) and "role" in item for item in value)
        if is_chat_messages:
            try:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=not DISABLE_THINKING,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                )

        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        content = value.get("content", "")
        return str(content)
    return str(value)


def _as_messages(value):
    if isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    ):
        return value
    return None


def _normalize_example(example, tokenizer):
    prompt_messages = _as_messages(example.get("prompt"))
    chosen_messages = _as_messages(example.get("chosen"))
    rejected_messages = _as_messages(example.get("rejected"))

    # Build prompt/completions from the same chat-template path to avoid
    # prompt-prefix tokenization mismatches seen with Qwen + TRL.
    if prompt_messages and chosen_messages and rejected_messages:
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not DISABLE_THINKING,
            )
            chosen_full = tokenizer.apply_chat_template(
                prompt_messages + chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=not DISABLE_THINKING,
            )
            rejected_full = tokenizer.apply_chat_template(
                prompt_messages + rejected_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=not DISABLE_THINKING,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_full = tokenizer.apply_chat_template(
                prompt_messages + chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_full = tokenizer.apply_chat_template(
                prompt_messages + rejected_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        if chosen_full.startswith(prompt_text) and rejected_full.startswith(prompt_text):
            return {
                "prompt": prompt_text,
                "chosen": chosen_full[len(prompt_text) :],
                "rejected": rejected_full[len(prompt_text) :],
            }

    prompt = _to_text(example.get("prompt", ""), tokenizer)
    chosen = _to_text(example["chosen"], tokenizer)
    rejected = _to_text(example["rejected"], tokenizer)

    if prompt and chosen.startswith(prompt) and rejected.startswith(prompt):
        return {
            "prompt": prompt,
            "chosen": chosen[len(prompt) :],
            "rejected": rejected[len(prompt) :],
        }

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _latest_checkpoint_path(output_dir: str) -> str | None:
    out_path = Path(output_dir)
    if not out_path.exists():
        return None

    checkpoints = [p for p in out_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    image=image,
    secrets=[modal.Secret.from_dotenv(path=".env")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/outputs": model_out_vol,
    },
)
def train_and_push(repo_id: str, private: bool = False) -> str:
    import torch
    import wandb
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hf_token = os.environ["HF_TOKEN"]
    wandb_key = os.environ["WANDB_API_KEY"]

    login(token=hf_token)
    wandb.login(key=wandb_key)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(_normalize_example, fn_kwargs={"tokenizer": tokenizer})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        device_map="auto",
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        max_length=512,
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_ckpt = _latest_checkpoint_path(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    HfApi(token=hf_token).create_repo(repo_id=repo_id, private=private, exist_ok=True)
    trainer.model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    model_out_vol.commit()
    return f"Finished training and pushed to https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def main(repo_id: str, private: bool = False) -> None:
    message = train_and_push.remote(repo_id=repo_id, private=private)
    print(message)

    out_dir = Path(OUTPUT_DIR)
    print(f"Artifacts saved in Modal volume under: {out_dir}")