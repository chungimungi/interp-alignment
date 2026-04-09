import os
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import dotenv

dotenv.load_dotenv()

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-multi-binarized-preferences-cleaned"
OUTPUT_DIR = "results/SmolLM3-3B-GRPO"
RUN_NAME = "SmolLM3-3B-GRPO"
DISABLE_THINKING = True
MAX_PROMPT_TOKENS = 512
CHOSEN_WEIGHT = 3.0
REJECTED_WEIGHT = 1.0


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


def _truncate_prompt(prompt_text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return prompt_text
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt_text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _normalize_prompt(example, tokenizer):
    prompt_messages = _as_messages(example.get("prompt"))
    if prompt_messages:
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not DISABLE_THINKING,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt_text = _to_text(example.get("prompt", ""), tokenizer)

    chosen = _to_text(example.get("chosen", ""), tokenizer)
    rejected = _to_text(example.get("rejected", ""), tokenizer)

    prompt_text = _truncate_prompt(prompt_text, tokenizer, MAX_PROMPT_TOKENS)
    return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}


def simple_preference_reward(prompts, completions, chosen, rejected, **kwargs):
    """
    Continuous similarity reward for GRPO.
    reward = CHOSEN_WEIGHT * sim(completion, chosen) - REJECTED_WEIGHT * sim(completion, rejected)
    Always produces a non-zero gradient signal.
    """
    rewards = []
    for completion, c_target, r_target in zip(completions, chosen, rejected):
        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
            comp_text = completion[-1].get("content", "").strip()
        else:
            comp_text = str(completion).strip()

        c_text = str(c_target).strip()
        r_text = str(r_target).strip()

        chosen_sim = SequenceMatcher(None, comp_text, c_text).ratio()
        rejected_sim = SequenceMatcher(None, comp_text, r_text).ratio()

        reward = CHOSEN_WEIGHT * chosen_sim - REJECTED_WEIGHT * rejected_sim
        rewards.append(reward)

    return rewards


def grpo_reward_funcs():
    """Returns the custom reward function for GRPOTrainer."""
    return [simple_preference_reward]


def _latest_checkpoint_path(output_dir: str) -> str | None:
    out_path = Path(output_dir)
    if not out_path.exists():
        return None

    checkpoints = [p for p in out_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


def train_and_push(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
) -> str:
    import torch
    import wandb
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

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
    tokenizer.padding_side = "left"

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(_normalize_prompt, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "chosen", "rejected"]]
    )

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        hub_model_id=repo_id,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_8bit",
        max_completion_length=256,
        num_generations=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0,
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

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=grpo_reward_funcs(),
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
    trainer.push_to_hub(token=hf_token)
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

    return "Finished training and pushed: " + ", ".join(pushed_urls)


def main(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str = "",
) -> None:
    resolved_merged_repo_id = merged_repo_id.strip() or None
    message = train_and_push(
        repo_id=repo_id,
        private=private,
        push_merged=push_merged,
        merged_repo_id=resolved_merged_repo_id,
    )
    print(message)

    out_dir = Path(OUTPUT_DIR)
    print(f"Artifacts saved locally under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRPO model and push to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. username/model-name")
    parser.add_argument("--private", action="store_true", help="Create/push to a private HF repo")
    parser.add_argument(
        "--no-push-merged",
        action="store_true",
        help="Skip pushing merged dense checkpoint",
    )
    parser.add_argument(
        "--merged-repo-id",
        default="",
        help="Optional HF repo id for merged dense model",
    )
    args = parser.parse_args()

    main(
        repo_id=args.repo_id,
        private=args.private,
        push_merged=not args.no_push_merged,
        merged_repo_id=args.merged_repo_id,
    )
