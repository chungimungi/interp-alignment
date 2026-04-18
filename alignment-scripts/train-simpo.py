import argparse
import inspect
import os
import sys
import time
from pathlib import Path


def _configure_cuda_runtime_paths() -> None:
    """Ensure NVIDIA shared libs from the active venv are visible to dlopen.

    Must run before `torch` / `peft` / `bitsandbytes` import, otherwise loading
    the bitsandbytes CUDA extension fails with missing `libnvJitLink.so.13`.
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / py_ver / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.exists():
        return

    lib_dirs = []
    for child in sorted(nvidia_dir.iterdir(), key=lambda p: p.name):
        lib_dir = child / "lib"
        if lib_dir.is_dir():
            lib_dirs.append(str(lib_dir))
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [p for p in current.split(":") if p]
    merged = []
    for p in lib_dirs + existing:
        if p not in merged:
            merged.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


_configure_cuda_runtime_paths()

from dotenv import load_dotenv
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.cpo import CPOConfig, CPOTrainer

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
OUTPUT_DIR = "./outputs/Llama-3.2-3B-Instruct-SimPO"
RUN_NAME = "Llama-3.2-3B-Instruct-SimPO"
DISABLE_THINKING = True

BETA = 2.0
GAMMA_BETA_RATIO = 0.5
SIMPO_GAMMA = BETA * GAMMA_BETA_RATIO   # = 1.0
# When this is 0.0, TRL reports train/nll_loss = 0 always (no BC NLL term). Raise slightly
# if you want a non-zero NLL curve in W&B (changes the objective toward CPO+SimPO hybrid).
CPO_ALPHA = 0.0

PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-7
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
# CPO tokenization can leave prompt+chosen longer than max_length when chosen/rejected
# lengths differ a lot; use a comfortable budget to avoid NaN log-probs / grads.
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 128

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

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


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=not DISABLE_THINKING,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _as_messages(value):
    if isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    ):
        return value
    return None


def _shared_message_prefix(left_messages, right_messages):
    prefix = []
    for left_item, right_item in zip(left_messages, right_messages):
        if left_item != right_item:
            break
        prefix.append(left_item)
    return prefix


def _strip_prompt_prefix(full_text: str, prompt_text: str) -> str:
    for prefix in (prompt_text, prompt_text.rstrip()):
        if prefix and full_text.startswith(prefix):
            return full_text[len(prefix) :].strip()
    return full_text.strip()


def _normalize_example(example, tokenizer):
    chosen_messages = _as_messages(example.get("chosen"))
    rejected_messages = _as_messages(example.get("rejected"))

    if chosen_messages and rejected_messages:
        prompt_messages = _as_messages(example.get("prompt"))
        if prompt_messages is None:
            prompt_messages = _shared_message_prefix(chosen_messages, rejected_messages)

        if (
            prompt_messages
            and len(prompt_messages) < len(chosen_messages)
            and len(prompt_messages) < len(rejected_messages)
        ):
            prompt_text = _apply_chat_template(
                tokenizer, prompt_messages, add_generation_prompt=True
            )
            chosen_full = _apply_chat_template(
                tokenizer, chosen_messages, add_generation_prompt=False
            )
            rejected_full = _apply_chat_template(
                tokenizer, rejected_messages, add_generation_prompt=False
            )
            if chosen_full.startswith(prompt_text) and rejected_full.startswith(prompt_text):
                chosen = _strip_prompt_prefix(chosen_full, prompt_text)
                rejected = _strip_prompt_prefix(rejected_full, prompt_text)
            else:
                chosen = _apply_chat_template(
                    tokenizer, chosen_messages[len(prompt_messages) :], add_generation_prompt=False
                ).strip()
                rejected = _apply_chat_template(
                    tokenizer, rejected_messages[len(prompt_messages) :], add_generation_prompt=False
                ).strip()
            if prompt_text.strip() and chosen and rejected:
                return {
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                }

    prompt = _to_text(example.get("prompt", ""), tokenizer).strip()
    chosen = _to_text(example["chosen"], tokenizer).strip()
    rejected = _to_text(example["rejected"], tokenizer).strip()

    if prompt:
        chosen = _strip_prompt_prefix(chosen, prompt)
        rejected = _strip_prompt_prefix(rejected, prompt)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _is_valid_preference_example(example) -> bool:
    prompt = example["prompt"].strip()
    chosen = example["chosen"].strip()
    rejected = example["rejected"].strip()
    return bool(prompt and chosen and rejected and chosen != rejected)


def _fits_cpo_context(example, tokenizer, max_length: int) -> bool:
    """Keep rows where both branches fit in max_length tokens (same concat TRL uses).

    Without this, asymmetric chosen/rejected lengths can exceed max_length after
    `CPOTrainer.tokenize_row`, which often yields NaN average log-probs and dead training.
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    len_c = len(tokenizer(prompt + chosen, add_special_tokens=False)["input_ids"])
    len_r = len(tokenizer(prompt + rejected, add_special_tokens=False)["input_ids"])
    return len_c <= max_length and len_r <= max_length


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

def train_and_push(repo_id: str) -> str:
    hf_token = os.environ["HF_TOKEN"]
    wandb_key = os.environ["WANDB_API_KEY"]
    print("[check] env vars found: HF_TOKEN, WANDB_API_KEY")

    login(token=hf_token)
    print("[check] HuggingFace login OK")

    wandb.login(key=wandb_key)
    print("[check] W&B login OK")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[check] tokenizer loaded: {MODEL_NAME}")

    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"[check] dataset loaded: {DATASET_NAME} — {len(dataset)} rows")

    dataset = dataset.map(_normalize_example, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.filter(_is_valid_preference_example)
    n_before_ctx = len(dataset)
    dataset = dataset.filter(
        _fits_cpo_context, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH}
    )
    print(
        f"[check] dataset prepared, columns: {dataset.column_names} — "
        f"{len(dataset)} rows (dropped {n_before_ctx - len(dataset)} over max_length={MAX_LENGTH})"
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
        ),
    )
    print(f"[check] model loaded: {MODEL_NAME}")

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
        logging_steps=LOGGING_STEPS,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        gradient_checkpointing=True,
        # reentrant checkpointing avoids NaN grads with some models when use_reentrant=False
        gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )
    supported_cpo_kwargs = set(inspect.signature(CPOConfig).parameters.keys())
    if "max_prompt_length" not in supported_cpo_kwargs:
        cpo_config_kwargs.pop("max_prompt_length", None)
        print("[check] CPOConfig has no max_prompt_length in this TRL version; using defaults.")
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

    resume_ckpt = _latest_checkpoint_path(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    HfApi(token=hf_token).create_repo(repo_id=repo_id, private=False, exist_ok=True)
    trainer.model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    pushed_urls = [f"https://huggingface.co/{repo_id} (adapter)"]

    dense_repo_id = f"{repo_id}-merged"
    print(f"Merging LoRA adapter into dense model for repo: {dense_repo_id}")

    merged_model = trainer.model.merge_and_unload()
    dense_output_dir = f"{OUTPUT_DIR}-merged"
    Path(dense_output_dir).mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(dense_output_dir)
    tokenizer.save_pretrained(dense_output_dir)

    HfApi(token=hf_token).create_repo(
        repo_id=dense_repo_id, private=False, exist_ok=True
    )
    merged_model.push_to_hub(dense_repo_id, token=hf_token)
    tokenizer.push_to_hub(dense_repo_id, token=hf_token)
    pushed_urls.append(f"https://huggingface.co/{dense_repo_id} (merged dense)")

    return "Finished training and pushed: " + ", ".join(pushed_urls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimPO training + push to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF repo id, e.g. username/model-name",
    )
    args = parser.parse_args()
    message = train_and_push(repo_id=args.repo_id)
    print(message)
