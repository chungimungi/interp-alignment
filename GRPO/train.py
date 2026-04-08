import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Model & Data ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "./outputs/SmolLM3-3B-GRPO"
RUN_NAME = "SmolLM3-3B-GRPO"
DISABLE_THINKING = True

# ── Sequence lengths ──────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS = 1024
MAX_COMPLETION_LENGTH = 768

# ── GRPO ──────────────────────────────────────────────────────────────────────
NUM_GENERATIONS = 16

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


# ── Dataset ───────────────────────────────────────────────────────────────────

def _prepare_example(example, tokenizer):
    prompt = example.get("prompt", "")
    if isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m for m in prompt):
        try:
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not DISABLE_THINKING,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        formatted = str(prompt)

    # Truncate prompt to MAX_PROMPT_TOKENS (max_prompt_length removed from GRPOConfig in trl>=0.25)
    tokens = tokenizer(formatted, truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors=None)
    formatted = tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)

    # Rename chosen-rating → chosen_rating (hyphen breaks Python kwargs)
    return {
        "prompt": formatted,
        "chosen": example.get("chosen", []),
        "chosen_rating": float(example.get("chosen-rating", 3.0)),
    }


# ── Reward ────────────────────────────────────────────────────────────────────

def _make_preference_reward(scorer):
    """
    Factory that returns a GRPO-compatible reward function.

    Scores each generated completion by ROUGE-L similarity to the chosen
    (high-quality) response, scaled by its GPT-4 quality rating.

    - ROUGE-L captures longest common subsequence overlap — a proxy for
      whether the generated response covers the same content as the chosen one.
    - chosen_rating (1–5) scales the reward: prompts where even the best
      response scored 3.0 contribute less signal than prompts with a 5.0
      gold response. Normalized to [0.5, 1.0] so it never zeroes out reward.
    """

    def preference_reward(completions, chosen, chosen_rating, **kwargs):
        rewards = []
        for completion, ref, rating in zip(completions, chosen, chosen_rating):
            gen_text = completion[0]["content"] if isinstance(completion, list) else completion

            # chosen is a list of messages — extract assistant turn(s)
            if isinstance(ref, list):
                ref_text = " ".join(
                    m["content"] for m in ref if isinstance(m, dict) and m.get("role") == "assistant"
                )
            else:
                ref_text = str(ref)

            rouge_l = scorer.score(gen_text, ref_text)["rougeL"].fmeasure

            # Map rating 1–5 → quality_scale 0.5–1.0
            quality_scale = 0.5 + (float(rating) - 1.0) / 8.0

            rewards.append(rouge_l * quality_scale)
        return rewards

    return preference_reward


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
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

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

    dataset = dataset.map(_prepare_example, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.select_columns(["prompt", "chosen", "chosen_rating"])
    print(f"[check] dataset prepared, columns: {dataset.column_names}")

    # ── 4. Reward function smoke-test ─────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    preference_reward = _make_preference_reward(scorer)

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
    progress_logger = _ProgressLogger(total_steps=total_steps)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=[preference_reward],
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

    train_parser = subparsers.add_parser("train", help="Run GRPO training")
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
