import os
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import dotenv
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

dotenv.load_dotenv()

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "argilla/ultrafeedback-multi-binarized-preferences-cleaned"

OUTPUT_DIR = "results/Qwen3-4B-Instruct-2507-GRPO-fast"
RUN_NAME = "Qwen3-4B-Instruct-2507-GRPO-fast"

DISABLE_THINKING = True

# Prompt / generation lengths
MAX_PROMPT_TOKENS = 512
MAX_COMPLETION_LENGTH = 256
NUM_GENERATIONS = 4

# Reward preprocessing budget
MAX_REWARD_WORDS = 160

# Reward weights
CHOSEN_WEIGHT = 1.0
REJECTED_WEIGHT = 0.25
LENGTH_PENALTY_START = 1.35
LENGTH_PENALTY_SCALE = 0.05

# Training
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-6
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 2
SEED = 42

# vLLM: enabled by default
USE_VLLM = True
VLLM_MODE = "colocate"  
VLLM_GPU_MEMORY_UTILIZATION = 0.35
VLLM_TP_SIZE = 1
# Cap below the model's advertised 262k context so KV cache fits next to training.
VLLM_MAX_MODEL_LENGTH = max(2048, MAX_PROMPT_TOKENS + MAX_COMPLETION_LENGTH + 512)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


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


def _normalize_reward_text(text: str, max_words: int = MAX_REWARD_WORDS) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = _WORD_RE.findall(text)
    if max_words > 0:
        tokens = tokens[:max_words]
    return " ".join(tokens)


def _normalize_example(example, tokenizer):
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

    chosen_text = _to_text(example.get("chosen", ""), tokenizer)
    rejected_text = _to_text(example.get("rejected", ""), tokenizer)

    prompt_text = _truncate_prompt(prompt_text, tokenizer, MAX_PROMPT_TOKENS)

    chosen_ref = _normalize_reward_text(chosen_text)
    rejected_ref = _normalize_reward_text(rejected_text)

    return {
        "prompt": prompt_text,
        "chosen_ref": chosen_ref,
        "rejected_ref": rejected_ref,
    }


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()

    if isinstance(completion, list):
        if len(completion) == 0:
            return ""
        if all(isinstance(x, dict) and "content" in x for x in completion):
            return " ".join(str(x.get("content", "")).strip() for x in completion).strip()
        return " ".join(str(x).strip() for x in completion).strip()

    return str(completion).strip()


def _token_f1(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    overlap = sum((a_counts & b_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def fast_preference_reward(completions, chosen_ref, rejected_ref, **kwargs):
    rewards = []

    for completion, c_ref, r_ref in zip(completions, chosen_ref, rejected_ref):
        comp_text = _normalize_reward_text(
            _completion_to_text(completion),
            max_words=MAX_REWARD_WORDS,
        )

        if not comp_text:
            rewards.append(-0.25)
            continue

        chosen_score = _token_f1(comp_text, c_ref)
        rejected_score = _token_f1(comp_text, r_ref)

        comp_len = max(len(comp_text.split()), 1)
        chosen_len = max(len(c_ref.split()), 1)

        length_penalty = 0.0
        ratio = comp_len / chosen_len
        if ratio > LENGTH_PENALTY_START:
            length_penalty = LENGTH_PENALTY_SCALE * (ratio - LENGTH_PENALTY_START)

        reward = (
            CHOSEN_WEIGHT * chosen_score
            - REJECTED_WEIGHT * rejected_score
            - length_penalty
        )

        reward = max(-1.0, min(1.0, reward))
        rewards.append(float(reward))

    return rewards


def _latest_checkpoint_path(output_dir: str) -> Optional[str]:
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
    merged_repo_id: Optional[str] = None,
) -> str:
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
    dataset = dataset.map(
        _normalize_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Normalizing dataset",
    )
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "chosen_ref", "rejected_ref"]]
    )

    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        hub_model_id=repo_id,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_8bit",
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards=False,
        temperature=0.8,
        top_p=0.9,
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
        seed=SEED,
        logging_first_step=True,
        save_safetensors=True,
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        vllm_tensor_parallel_size=VLLM_TP_SIZE,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
    )

    grpo_config = GRPOConfig(**config_kwargs)

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
        reward_funcs=[fast_preference_reward],
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

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

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

        api.create_repo(repo_id=dense_repo_id, private=private, exist_ok=True)
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
    print(f"Artifacts saved locally under: {Path(OUTPUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast GRPO training + push to Hugging Face Hub")
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
