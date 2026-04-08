# GRPO Training

Trains a causal LM using **Group Relative Policy Optimization (GRPO)** on the
[`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
dataset (60,917 rows). Runs on [Modal](https://modal.com) with an L40S GPU.

Reward signal: ROUGE-L similarity between the generated completion and the
dataset's `chosen` (GPT-4 preferred) response, scaled by its quality rating
(1–5 → 0.5–1.0).

---

## Setup

### 1. Install dependencies

```bash
uv venv && source .venv/bin/activate
uv pip install modal python-dotenv
```

### 2. Authenticate Modal

```bash
modal setup
```

### 3. Create `.env`

Copy `.env.example` and fill in your keys:

```bash
cp .env.example .env
```

```env
HF_TOKEN=hf_your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
```

- **HF_TOKEN**: from https://huggingface.co/settings/tokens (write access needed to push models)
- **WANDB_API_KEY**: from https://wandb.ai/authorize

---

## Configuration

All tunable parameters are in [`training/config.py`](training/config.py):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `SmolLM3-3B` | Base model to fine-tune |
| `GPU` | `L40S` | Modal GPU type |
| `NUM_GENERATIONS` | `4` | Completions sampled per prompt (G) |
| `MAX_PROMPT_TOKENS` | `1024` | Prompt truncation at dataset prep |
| `MAX_COMPLETION_LENGTH` | `768` | Max tokens generated per completion |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | Batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Effective batch = batch × accum = 16 |
| `LEARNING_RATE` | `1e-6` | |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |

---

## Running

All commands run from the `GRPO/` directory.

### Dry-run — check credentials, dataset, reward fn (no model load)

```bash
modal run -m training.train_grpo --dry-run
```

### Light-run — 5 steps on 20 rows, full pipeline (no push)

```bash
modal run -m training.train_grpo --light-run
```

### Full run

```bash
modal run -m training.train_grpo --repo-id <hf-username>/<repo-name>
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--repo-id` | required | HuggingFace repo to push adapter to |
| `--private` | `false` | Make HF repo private |
| `--push-merged` | `true` | Also push merged dense model |
| `--merged-repo-id` | `<repo-id>-merged` | Repo name for merged model |
| `--local-output-dir` | `./outputs` | Local dir to pull model into after training |

After training completes, the model is automatically pulled from the Modal volume
to `./outputs/` on your local machine, regardless of whether the HF push succeeded.

### Re-push from Modal volume (if HF push failed)

```bash
modal run -m training.push --repo-id <hf-username>/<repo-name>
```

---

## Time & Memory Estimates

Tested on `argilla/ultrafeedback-binarized-preferences-cleaned` (60,917 rows),
batch size 4, grad accum 4 (effective batch 16), G=4, max_completion_length=768.

### SmolLM3-3B

| GPU | VRAM used | Step time | Steps | Est. total |
|---|---|---|---|---|
| L40S (48GB) | ~12 GB (27%) | ~30s | ~3,807 | **~32 hrs** |
| A100-40GB | ~14 GB (35%) | ~71s | ~3,807 | **~75 hrs** |
| L4 (24GB) | ~14 GB (est.) | ~100s (est.) | ~3,807 | **~105 hrs** |

> Training exceeds the 24hr Modal timeout. It auto-resumes from the latest
> checkpoint — just re-run the same command and it picks up where it left off.

### Qwen3-4B (estimated)

| GPU | VRAM used | Step time | Est. total |
|---|---|---|---|
| L40S (48GB) | ~16 GB (est.) | ~40s (est.) | **~42 hrs** |

### Llama-3.2-3B (estimated)

| GPU | VRAM used | Step time | Est. total |
|---|---|---|---|
| L40S (48GB) | ~12 GB (est.) | ~30s (est.) | **~32 hrs** |

> Qwen3 and Llama estimates are extrapolated from SmolLM3-3B benchmarks.
> Run a `--light-run` first to get accurate per-step timing for your target model.

---

## Outputs

| Location | Contents |
|---|---|
| Modal volume `grpo-model-outputs` | Checkpoints + final model (persistent) |
| `./outputs/SmolLM3-3B-GRPO/` | LoRA adapter (pulled locally after run) |
| `./outputs/SmolLM3-3B-GRPO-merged/` | Merged dense model (pulled locally) |
| HuggingFace | Adapter + merged model pushed to your repo |
| W&B | Training metrics at `wandb.ai/<your-entity>/huggingface` |

---

## Checkpoint resumption

Training checkpoints are saved every 100 steps to the Modal volume. If a run
times out or fails mid-way, re-run the same full-run command — it will
automatically detect and resume from the latest checkpoint.
