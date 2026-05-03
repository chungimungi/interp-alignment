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
python -m interp_utils.crosscoder-singlelayer.main \
  --base-model HuggingFaceTB/SmolLM3-3B \
  --aligned-model /path/to/merged-or-adapter \
  --aligned-run-id my_grpo_run \
  --layer 15 \
  --stage all
```

Ensure that your reconstruction loss goes down and Fraction of Variance Explained (FVE) goes up.

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
| `--output-dir` | If set, **all** artifacts go here; otherwise `interp_utils/crosscoder-singlelayer/results/<slug>/`. |
| `--stage` | `extract` \| `train` \| `analyze` \| `visualize` \| `all` \| `manifest` \| `hypothesis_tests`. |
| `--prompts-cache-dir` | Where to store **reusable** normalized prompts (`datasets` Arrow on disk). Default: `crosscoder-singlelayer/cache/normalized_prompts/`. |
| `--no-prompts-cache` | Always load the raw HF dataset and re-run chat normalization (no disk cache). |


---

## Output results

Default run outputs are stored in `--output-dir` as:

`interp_utils/crosscoder-singlelayer/results/{base_model}__{aligned_run_id}__L{layer}__{position_slug}`

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
python -m interp_utils.crosscoder-singlelayer.compute_metrics --results-root interp_utils/crosscoder-singlelayer/results

# Compare class counts between two runs
python -m interp_utils.crosscoder-singlelayer.plot_class_comparison --dir-a run_a --dir-b run_b --output cmp.png
```

Edit `crosscoder_sweep.py` constants, then run `python -m interp_utils.crosscoder-singlelayer.crosscoder_sweep` for expansion/Top‑K sweeps.