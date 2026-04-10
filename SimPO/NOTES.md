# SimPO Training

Trains a causal LM using **Simple Preference Optimization (SimPO)** on the
[`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
dataset (60,917 rows). Runs on [Modal](https://modal.com) with an L40S GPU.

SimPO (arXiv:2405.14734) is a reference-free preference optimization method.
It uses the average log probability of a sequence as the implicit reward
(length-normalized) and introduces a target reward margin γ to the Bradley-Terry
objective. No reference model is loaded — saving ~10% memory vs DPO.

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
| `BETA` | `2.0` | Reward scaling (paper: 2.0–10.0) |
| `GAMMA_BETA_RATIO` | `0.5` | γ/β ratio (paper: 0.0–1.0) |
| `SIMPO_GAMMA` | `1.0` | Target reward margin (= β × ratio) |
| `CPO_ALPHA` | `0.0` | Pure SimPO (no BC regularization) |
| `MAX_LENGTH` | `512` | Max sequence length |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | Batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Effective batch = batch × accum = 32 |
| `LEARNING_RATE` | `5e-7` | Paper recommendation for general tasks |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |

---

## Running

All commands run from the `SimPO/` directory.

### Dry-run — check credentials, dataset (no model load)

```bash
modal run -m training.train_simpo --dry-run
```

### Light-run — 5 steps on 20 rows, full pipeline (no push)

```bash
modal run -m training.train_simpo --light-run
```

### Full run

```bash
modal run -m training.train_simpo --repo-id <hf-username>/<repo-name>
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

## Key Differences from DPO

| Aspect | DPO | SimPO |
|---|---|---|
| Reference model | Required (doubles memory) | Not needed |
| Reward signal | Implicit (log ratio) | Length-normalized log prob |
| Margin | None | Target margin γ |
| TRL class | `DPOTrainer` / `DPOConfig` | `CPOTrainer` / `CPOConfig` |
| `loss_type` | N/A | `"simpo"` |
| Memory | ~2× model size | ~1× model size |

---

## Outputs

| Location | Contents |
|---|---|
| Modal volume `simpo-model-outputs` | Checkpoints + final model (persistent) |
| `./outputs/SmolLM3-3B-SimPO/` | LoRA adapter (pulled locally after run) |
| `./outputs/SmolLM3-3B-SimPO-merged/` | Merged dense model (pulled locally) |
| HuggingFace | Adapter + merged model pushed to your repo |
| W&B | Training metrics at `wandb.ai/<your-entity>/huggingface` |

---

## Checkpoint resumption

Training checkpoints are saved every 100 steps to the Modal volume. If a run
times out or fails mid-way, re-run the same full-run command — it will
automatically detect and resume from the latest checkpoint.
