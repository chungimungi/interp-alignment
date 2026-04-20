import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _configure_cuda_runtime_paths() -> None:
    """Ensure NVIDIA shared libs from the active venv are visible to dlopen.

    bitsandbytes 8-bit optimizers need `libnvJitLink.so.*` at **optimizer step** time.
    Pip wheels usually place it under `site-packages/nvidia/cu13/lib/` (CUDA 13).
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_ver / "site-packages"
    nvidia_dir = site_packages / "nvidia"

    lib_dirs: list[str] = []
    for preferred in (
        nvidia_dir / "cu13" / "lib",
        nvidia_dir / "nvjitlink" / "lib",
    ):
        if preferred.is_dir():
            lib_dirs.append(str(preferred))

    if nvidia_dir.is_dir():
        for child in sorted(nvidia_dir.iterdir(), key=lambda p: p.name):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                p = str(lib_dir)
                if p not in lib_dirs:
                    lib_dirs.append(p)

    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [p for p in current.split(":") if p]
    merged: list[str] = []
    for p in lib_dirs + existing:
        if p not in merged:
            merged.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


_configure_cuda_runtime_paths()

import dotenv
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer

dotenv.load_dotenv()

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "argilla/ultrafeedback-multi-binarized-preferences-cleaned"

OUTPUT_DIR = "results/Qwen3-4B-Instruct-2507-PPO-fast"
RUN_NAME = "Qwen3-4B-Instruct-2507-PPO-fast"

DISABLE_THINKING = True

MAX_PROMPT_TOKENS = 256

RESPONSE_LENGTH = 256
LOCAL_ROLLOUT_FORWARD_BATCH_SIZE = 4
NUM_PPO_EPOCHS = 1
NUM_MINI_BATCHES = 1
NUM_SAMPLE_GENERATIONS = 0
MISSING_EOS_PENALTY = 1.0
STOP_TOKEN = "eos"
TEMPERATURE = 0.7
KL_COEF = 0.05
KL_ESTIMATOR = "k1"

# Training (batch layout kept similar to the old GRPO script; LR matches PPOConfig default in TRL docs)
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-6
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 2
SEED = 42

REWARD_MODEL_PATH = os.environ.get("PPO_REWARD_MODEL_PATH", MODEL_NAME)
PPO_OPTIM = os.environ.get("PPO_OPTIM", "adamw_torch")
PPO_ATTN_IMPLEMENTATION = os.environ.get(
    "PPO_ATTN_IMPLEMENTATION",
    "flash_attention_2" if torch.cuda.is_available() else "sdpa",
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def _normalize_example(example, tokenizer):
    """Same prompt extraction as before; PPO only needs the prompt column."""
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

    prompt_text = _truncate_prompt(prompt_text, tokenizer, MAX_PROMPT_TOKENS)
    return {"prompt": prompt_text}


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
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"PPO attention implementation: {PPO_ATTN_IMPLEMENTATION}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        _normalize_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Normalizing dataset",
    )

    eval_holdout = min(256, max(1, len(dataset) // 100))
    train_dataset = dataset.select(range(len(dataset) - eval_holdout))
    eval_dataset = dataset.select(range(len(dataset) - eval_holdout, len(dataset)))

    def tokenize_split(ds):
        def tokenize_batch(batch):
            enc = tokenizer(batch["prompt"], padding=False)
            return {"input_ids": enc["input_ids"]}

        return ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing prompts",
        )

    train_dataset = tokenize_split(train_dataset)
    eval_dataset = tokenize_split(eval_dataset)

    model_kwargs = {"token": hf_token, "trust_remote_code": True, "torch_dtype": dtype}
    if use_cuda:
        model_kwargs["attn_implementation"] = PPO_ATTN_IMPLEMENTATION

    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        **model_kwargs,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        **model_kwargs,
    )
    reward_model.requires_grad_(False)
    reward_model.eval()

    ref_policy = None

    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        hub_model_id=repo_id,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=PPO_OPTIM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=RUN_NAME,
        seed=SEED,
        logging_first_step=True,
        push_to_hub=True,
        num_ppo_epochs=NUM_PPO_EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        num_sample_generations=NUM_SAMPLE_GENERATIONS,
        missing_eos_penalty=MISSING_EOS_PENALTY,
        stop_token=STOP_TOKEN,
        kl_coef=KL_COEF,
        kl_estimator=KL_ESTIMATOR,
        reward_model_path=REWARD_MODEL_PATH,
        sft_model_path=MODEL_NAME,
    )

    ppo_config = PPOConfig(**config_kwargs)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    resume_ckpt = _latest_checkpoint_path(OUTPUT_DIR)
    if resume_ckpt:
        print(
            "Note: experimental PPOTrainer may not resume from checkpoints; "
            f"found {resume_ckpt} but starting a full train run."
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer.train()
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

        policy_module = trainer.accelerator.unwrap_model(trainer.model).policy
        merged_model = policy_module.merge_and_unload()
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
    parser = argparse.ArgumentParser(
        description="PPO training (TRL experimental) + push to Hugging Face Hub",
        epilog=(
            "Default attention on GPU is flash_attention_2 (override: PPO_ATTN_IMPLEMENTATION=sdpa). "
            "Example: nohup /home/riftuser/interp/bin/python /home/riftuser/ppo.py "
            "--repo-id user/model > nohup.out 2>&1 &"
        ),
    )
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
