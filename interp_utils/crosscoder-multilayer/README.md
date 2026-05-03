# Crosscoder: base vs aligned LLMs

Train a **sparse cross-coder** (SPARC-style) on **paired hidden states** from the same prompts run through a **base** causal LM and an **alignment-trained** causal LM (DPO, GRPO, SFT, etc.). Then classify features (base-only, aligned-only, shared subclasses), run lightweight analyses, and plot metrics.

---

## What it does

1. **Extract** activations at a chosen decoder layer for every prompt, for both models, with fixed pooling (`last_prompt` or mean over prompt tokens).
2. **Train** `SPARCCrossCoder`: two encoders/decoders with shared global Top‑K, cross-reconstruction, and a small forced-shared subspace on the decoders.
3. **Analyze** decoder geometry (ρ, θ), merge with per-feature “sensitivity” stats, superposition on **aligned-only** features, and aggregate metrics.
4. **Visualize** loss curves, ρ/θ plots, class distributions, CF-style boxplots, superposition scatter.

---

## CLI

From the **repo root** (parent of `crosscoder/`):

```bash
python -m interp_utils.crosscoder-multilayer.main \
  --base-model HuggingFaceTB/SmolLM3-3B \
  --aligned-model /path/to/merged-or-adapter \
  --aligned-run-id my_grpo_run \
  --layer 15 \
  --stage all
```

Ensure that your reconstruction loss goes down and Fraction of Variance Explained (FVE) goes up.

### Multi-layer extraction

Contiguous windows can be specified with a center layer:

```bash
.venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage extract \
  --crosscoder-kind multilayer_sparc \
  --base-model HuggingFaceTB/SmolLM3-3B \
  --aligned-model MInAlA/SmolLM3-3B-PPO-merged \
  --aligned-run-id smollm3-ppo \
  --center-layer 19 \
  --layer-window 1 \
  --trust-remote-code \
  --extract-batch-size 4
```

Arbitrary/non-contiguous layer sets are supported with `--layers`:

```bash
.venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage extract \
  --crosscoder-kind multilayer_sparc \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --aligned-model MInAlA/Llama-3.2-3B-Instruct-KTO-merged \
  --aligned-run-id llama32-3b-kto \
  --layers 10,11,12,13,14,23,24,25,26 \
  --extract-batch-size 2
```

### Manifest on multiple GPUs

Manifest rows use the same fields as the CLI. For multi-layer rows, use either
`"center_layer"` + `"layer_window"` or explicit `"layers"`:

```json
[
  {
    "stage": "extract",
    "crosscoder_kind": "multilayer_sparc",
    "base_model": "HuggingFaceTB/SmolLM3-3B",
    "aligned_model": "MInAlA/SmolLM3-3B-PPO-merged",
    "aligned_run_id": "smollm3-ppo",
    "layers": [17, 18, 19],
    "dataset_name": "argilla/ultrafeedback-binarized-preferences-cleaned",
    "trust_remote_code": true,
    "extract_batch_size": 4
  }
]
```

For true parallel extraction, split base and aligned sides, then assemble a normal
training artifact:

```bash
# Machine/GPU A: base union layers once per base model
.venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage extract \
  --crosscoder-kind multilayer_sparc \
  --extract-side base \
  --base-model HuggingFaceTB/SmolLM3-3B \
  --aligned-run-id smollm3-base-union \
  --layers 16,17,18,19,20 \
  --output-dir output/base_activations/smollm3-union \
  --extract-batch-size 4

# Machine/GPU B: aligned model layers for one run
.venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage extract \
  --crosscoder-kind multilayer_sparc \
  --extract-side aligned \
  --base-model HuggingFaceTB/SmolLM3-3B \
  --aligned-model MInAlA/SmolLM3-3B-PPO-merged \
  --aligned-run-id smollm3-ppo \
  --layers 17,18,19 \
  --trust-remote-code \
  --output-dir output/aligned_activations/smollm3-ppo \
  --extract-batch-size 4

# After copying artifacts onto one machine, slice base union to aligned layers
.venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage assemble \
  --crosscoder-kind multilayer_sparc \
  --base-activations-dir output/base_activations/smollm3-union \
  --aligned-activations-dir output/aligned_activations/smollm3-ppo \
  --output-dir output/crosscoder-multilayer/smollm3-ppo/L17-19
```

`assemble` requires exact matching `sample_ids`, `splits`, `position`,
`dataset_name`, and `max_prompt_tokens`. The aligned layer set must be a subset
of the base layer set; base activations are sliced/reordered to the aligned
layers before writing `activations/activations.pt`.

Equivalent manifest rows:

```json
[
  {
    "stage": "extract",
    "crosscoder_kind": "multilayer_sparc",
    "extract_side": "base",
    "base_model": "HuggingFaceTB/SmolLM3-3B",
    "aligned_run_id": "smollm3-base-union",
    "layers": [16, 17, 18, 19, 20],
    "output_dir": "output/base_activations/smollm3-union",
    "extract_batch_size": 4
  },
  {
    "stage": "extract",
    "crosscoder_kind": "multilayer_sparc",
    "extract_side": "aligned",
    "base_model": "HuggingFaceTB/SmolLM3-3B",
    "aligned_model": "MInAlA/SmolLM3-3B-PPO-merged",
    "aligned_run_id": "smollm3-ppo",
    "layers": [17, 18, 19],
    "output_dir": "output/aligned_activations/smollm3-ppo",
    "trust_remote_code": true,
    "extract_batch_size": 4
  },
  {
    "name": "assemble-smollm3-ppo",
    "stage": "assemble",
    "crosscoder_kind": "multilayer_sparc",
    "base_activations_dir": "output/base_activations/smollm3-union",
    "aligned_activations_dir": "output/aligned_activations/smollm3-ppo",
    "output_dir": "output/crosscoder-multilayer/smollm3-ppo/L17-19"
  }
]
```

Run detached and write parent + per-job logs:

```bash
mkdir -p logs/crosscoder
nohup .venv/bin/python -m interp_utils.crosscoder-multilayer.main \
  --stage manifest \
  --manifest manifests/multilayer_extract.json \
  --gpu-ids 0,1,2,3 \
  --output-root output/crosscoder-multilayer \
  --log-dir logs/crosscoder \
  > logs/crosscoder/manifest.out 2>&1 &
```

Monitor:

```bash
tail -F logs/crosscoder/manifest.out
tail -F logs/crosscoder/*.log
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv
```

Each manifest row is launched as a child CLI process with one `CUDA_VISIBLE_DEVICES`
value from the GPU pool. Per-row logs include the exact command, assigned GPU,
output directory, and periodic plain-text extraction progress.

### Important flags

| Flag | Meaning |
|------|---------|
| `--base-model` | Hugging Face id (or path) for the **base** causal LM. |
| `--aligned-model` | **Merged** full checkpoint dir, HF hub id, or **PEFT** folder (must contain `adapter_config.json`). |
| `--aligned-run-id` | Short slug for the default results directory. **Do not use `__` in this string** (it breaks directory parsing). |
| `--layer` | Integer index into `model.model.layers[layer]` (Llama/SmolLM3-style). |
| `--position` | `last_prompt` (default) or `mean_prompt`. |
| `--dataset-name` | HF dataset; default `argilla/ultrafeedback-multi-binarized-preferences-cleaned`. |
| `--max-prompt-tokens` | Truncation after chat-template (default 512). |
| `--trust-remote-code` | Passed to `from_pretrained` when needed. |
| `--output-dir` | If set, **all** artifacts go here; otherwise `interp_utils/crosscoder-multilayer/results/<slug>/`. |
| `--stage` | `extract` \| `train` \| `analyze` \| `visualize` \| `all` \| `manifest` \| `hypothesis_tests`. |
| `--prompts-cache-dir` | Where to store **reusable** normalized prompts (`datasets` Arrow on disk). Default: `crosscoder-multilayer/cache/normalized_prompts/`. |
| `--no-prompts-cache` | Always load the raw HF dataset and re-run chat normalization (no disk cache). |


---

## Output results

Default run outputs are stored in `--output-dir` as:

`interp_utils/crosscoder-multilayer/results/{base_model}__{aligned_run_id}__L{layer}__{position_slug}`

where `position_slug` is `lastprompt` or `meanprompt`.

Each folder should usually have:

- `activations/activations.pt` - tensors + `sample_ids`, `splits`, provenance.
- `run_meta.json` - run configuration snapshot.
- `checkpoints/final.pt` - trained crosscoder.
- `metrics/training_metrics.json`, `aggregate_metrics.json`, …
- `features/*.csv`, `superposition_analysis.json`
- `plots/*.png`

---

## Auxiliary tools

```bash
# Summarize many result folders into CSVs
python -m interp_utils.crosscoder-multilayer.compute_metrics --results-root interp_utils/crosscoder-multilayer/results

# Compare class counts between two runs
python -m interp_utils.crosscoder-multilayer.plot_class_comparison --dir-a run_a --dir-b run_b --output cmp.png
```

Edit `crosscoder_sweep.py` constants, then run `python -m interp_utils.crosscoder-multilayer.crosscoder_sweep` for expansion/Top‑K sweeps.
