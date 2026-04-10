import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Model & Data ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "./outputs/SmolLM3-3B-SimPO"
RUN_NAME = "SmolLM3-3B-SimPO"
DISABLE_THINKING = True

# ── SimPO (arXiv:2405.14734) ──────────────────────────────────────────────────
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


# ── Dataset ───────────────────────────────────────────────────────────────────

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


# ── Utils ─────────────────────────────────────────────────────────────────────

def _latest_checkpoint_path(output_dir: str) -> str | None:
    out_path = Path(output_dir)
    if not out_path.exists():
        return None
    checkpoints = [p for p in out_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


class _ProgressLogger:
    """Callback-compatible timer that logs step progress, elapsed, and ETA."""

    def __init__(self, total_steps: int):
        from transformers import TrainerCallback

        self.total_steps = total_steps
        self._start: float | None = None

        class _Callback(TrainerCallback):
            def on_train_begin(cb_self, args, state, control, **kwargs):
                self._start = time.time()
                print(f"[timer] Training started — {self.total_steps} steps total")

            def on_step_end(cb_self, args, state, control, **kwargs):
                if self._start is None:
                    return
                step = state.global_step
                elapsed = time.time() - self._start
                pct = step / self.total_steps * 100
                avg_per_step = elapsed / step if step > 0 else 0
                eta = avg_per_step * (self.total_steps - step)

                gpu_info = ""
                if step % 50 == 0 or self.total_steps <= 10:
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        used_gb = mem.used / 1024 ** 3
                        total_gb = mem.total / 1024 ** 3
                        gpu_info = (
                            f" — GPU mem {used_gb:.1f}/{total_gb:.1f}GB"
                            f" — GPU util {util.gpu}%"
                        )
                    except Exception:
                        pass

                print(
                    f"[timer] step {step}/{self.total_steps} "
                    f"({pct:.1f}%) — "
                    f"elapsed {elapsed:.0f}s — "
                    f"ETA {eta:.0f}s"
                    f"{gpu_info}"
                )

            def on_train_end(cb_self, args, state, control, **kwargs):
                if self._start is None:
                    return
                total = time.time() - self._start
                print(f"[timer] Training done in {total:.0f}s ({total/60:.1f} min)")

        self.callback = _Callback()


# ── Train ─────────────────────────────────────────────────────────────────────

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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import CPOConfig, CPOTrainer

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

    dataset = dataset.map(_normalize_example, fn_kwargs={"tokenizer": tokenizer})
    print(f"[check] dataset prepared, columns: {dataset.column_names}")

    if dry_run:
        return "[dry-run] All checks passed. Model not loaded, no training, no push."

    # ── 4. Model ──────────────────────────────────────────────────────────────
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

    cpo_config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        # SimPO-specific
        loss_type="simpo",
        cpo_alpha=CPO_ALPHA,
        simpo_gamma=SIMPO_GAMMA,
        beta=BETA,
        # Training
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1 if light_run else LOGGING_STEPS,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
    )
    if light_run:
        cpo_config_kwargs.update({"max_steps": 5, "save_strategy": "no"})
    else:
        cpo_config_kwargs.update({
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "save_strategy": "steps",
            "save_steps": SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
        })
    cpo_config = CPOConfig(**cpo_config_kwargs)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    total_steps = cpo_config_kwargs.get("max_steps") or (
        len(dataset) // (cpo_config.per_device_train_batch_size * cpo_config.gradient_accumulation_steps)
        * cpo_config.num_train_epochs
    )
    progress_logger = _ProgressLogger(total_steps=total_steps)

    trainer = CPOTrainer(
        model=model,
        args=cpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[progress_logger.callback],
    )

    resume_ckpt = None if light_run else _latest_checkpoint_path(OUTPUT_DIR)
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

    return "Finished training and pushed: " + ", ".join(pushed_urls)


# ── Re-push from local output dir (if HF push failed during training) ─────────

def push_from_local(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
) -> str:
    from huggingface_hub import HfApi, login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    login(token=os.environ["HF_TOKEN"])

    pushed_urls = []

    print(f"Loading model from {OUTPUT_DIR} ...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    HfApi().create_repo(repo_id=repo_id, private=private, exist_ok=True)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    pushed_urls.append(f"https://huggingface.co/{repo_id}")
    print(f"Pushed adapter/model to {repo_id}")

    if push_merged:
        merged_dir = f"{OUTPUT_DIR}-merged"
        dense_repo_id = merged_repo_id or f"{repo_id}-merged"

        print(f"Loading merged model from {merged_dir} ...")
        merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_dir)

        HfApi().create_repo(repo_id=dense_repo_id, private=private, exist_ok=True)
        merged_model.push_to_hub(dense_repo_id)
        merged_tokenizer.push_to_hub(dense_repo_id)
        pushed_urls.append(f"https://huggingface.co/{dense_repo_id}")
        print(f"Pushed merged model to {dense_repo_id}")

    return "Pushed: " + ", ".join(pushed_urls)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run SimPO training")
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
