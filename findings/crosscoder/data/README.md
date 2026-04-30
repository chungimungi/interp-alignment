# Crosscoder data — aggregate summaries

This folder contains the aggregate summary CSVs that drive every plot in
`findings/crosscoder/`. The plots are reproducible from these files alone.

## Files

### `per_feature_summary.csv` (89 rows)
Aggregate stats per (slug × feature_class) for the seed-1 run only. One row per
(cell, primary_class) combination. Used by `findings/plot_per_feature.py` to
generate the 4 single-seed figures (`fig_aligned_only_counts`,
`fig_shift_p95`, `fig_decoder_norm_ratio`, `fig_partition_scatter`).

### `per_feature_summary_all_seeds.csv` (185 rows)
Same schema as above, but with a `run` column (`seed1` / `seed2` / `rescue`) and
all three runs combined. Breakdown:
- `seed1`: 89 rows — full 15-cell sweep at `CROSSCODER_SEED=42`
- `seed2`: 83 rows — re-run at `CROSSCODER_SEED=99` (qwen-simpo seed-2 missing,
  the run was in flight when the source box was recycled)
- `rescue`: 13 rows — qwen-kto and qwen-orpo re-runs at relaxed
  hyperparameters (`LAMBDA_SHARED_MULT=0.01`, `FORCED_SHARED_FRAC=0.02`,
  `NUM_EPOCHS=8`)

Used by `findings/plot_seeds_comparison.py` and
`findings/plot_more_diagnostics.py` to generate the 7 cross-seed and
heatmap figures.

## Schema

| Column | Type | Description |
|---|---|---|
| `run` | string | One of `seed1`, `seed2`, `rescue` (only in `_all_seeds.csv`) |
| `slug` | string | Pair identifier, e.g. `llama-kto`, `qwen-grpo` |
| `layer` | int | Decoder layer index where the crosscoder was trained (per base: smollm=19, llama=11, qwen=24) |
| `class` | string | One of `base_only`, `aligned_only`, `shared_aligned`, `shared_attenuated`, `shared_redirected`, `shared_intermediate`, `other` |
| `n` | int | Number of features in this class |
| `frac` | float | `n / total_features` for the cell |
| `shift_mean_abs` | float | Mean of `|cf_shift|` over features in this class |
| `shift_median_abs` | float | Median of `|cf_shift|` |
| `shift_p95_abs` | float | 95th percentile of `|cf_shift|` |
| `shift_max_abs` | float | Max of `|cf_shift|` |
| `shift_mean_signed` | float | Mean of signed `cf_shift` |
| `norm_ratio_median` | float | Median of `||W_aligned_dec|| / ||W_base_dec||` over features in this class |
| `feature_sharing_ratio` | float | Cell-level metric (same value across all classes within a cell): fraction of features classified as shared rather than {base_only, aligned_only} |

## What's NOT here (and why)

The per-pair raw feature CSVs that produced these summaries are **not in this
folder** because the source box (`Arth-Temporary`, `~/work/interp-alignment/`)
was recycled after the sweep completed. Those files were:

```
output/crosscoder/<slug>/L<layer>/
├── activations/activations.pt              ~700 MB / cell
├── checkpoints/{epoch_1..4,final}.pt       ~50 MB / cell
├── features/
│   ├── merged_classification.csv           per-feature: rho, theta, decoder norms,
│   │                                       cf_base, cf_aligned, cf_shift, primary_class
│   ├── feature_classification.csv          per-feature: rho, theta, primary_class
│   ├── counterfactual_scores.csv           per-feature: cf_base, cf_aligned, cf_shift
│   ├── shared_features_geometry.csv        per-feature: rho, theta, norm_ratio_raw, angle_deg
│   ├── superposition_analysis.json         aligned-only feature decomposition
│   └── visual_evidence_features.json       top-shifted features
├── metrics/
│   ├── aggregate_metrics.json              cell-level summary (class_counts, fve, l0, etc.)
│   ├── training_metrics.json               loss curves per epoch
│   ├── shared_geometry_metrics.json        decoder geometry summary
│   └── cf_shift_by_class.json
├── plots/                                  9 per-cell PNGs
└── run_meta.json
```

Approximate total raw size for the 30-cell sweep + 2 rescue runs: **~25 GB**.

## Regenerating the raw per-pair data

The sweep is reproducible. On a B200 box (8 GPUs):

```bash
# 1. Clone repo and check out this branch
git clone https://github.com/chungimungi/interp-alignment.git
cd interp-alignment
git checkout b200-sweep

# 2. Install deps (NGC PyTorch container assumed)
pip install --break-system-packages transformers datasets accelerate peft \
    trl sae-lens python-dotenv wandb scikit-learn matplotlib seaborn

# 3. Set HF token (Llama-3.2 is gated)
export HF_TOKEN=hf_...

# 4. Seed-1 sweep, all 15 cells
python3 -u interp_utils/crosscoder/run_all_crosscoders.py \
    --num-gpus 8 --jobs-per-gpu 2

# 5. Seed-2 sweep
CROSSCODER_SEED=99 python3 -u interp_utils/crosscoder/run_all_crosscoders.py \
    --num-gpus 8 --jobs-per-gpu 2 \
    --output-root output/crosscoder-seed2

# 6. Rescue (relaxed config) for qwen-kto, qwen-orpo
CROSSCODER_LAMBDA_SHARED_MULT=0.01 \
CROSSCODER_FORCED_SHARED_FRAC=0.02 \
CROSSCODER_NUM_EPOCHS=8 \
python3 -u interp_utils/crosscoder/run_all_crosscoders.py \
    --only-run-id qwen-kto --only-run-id qwen-orpo \
    --gpu-ids 0,1 --jobs-per-gpu 1 \
    --output-root output/crosscoder-rescue
```

Wall-clock on 8x B200: ~4 hours for seed-1, ~4 hours for seed-2, ~80 min for
the rescue. Each cell produces the per-pair files listed above.

## Re-deriving the aggregate CSVs from raw data

Once the raw `output/crosscoder*/<slug>/L*/features/merged_classification.csv`
files exist, re-derive the summary CSVs with:

```python
import json, glob, os
import pandas as pd
import numpy as np

rows = []
for run_label, root_dir in [("seed1", "output/crosscoder"),
                             ("seed2", "output/crosscoder-seed2"),
                             ("rescue", "output/crosscoder-rescue")]:
    for d in sorted(glob.glob(root_dir + "/*/L*")):
        slug = d.split("/")[-2]
        layer = int(d.split("/")[-1].lstrip("L"))
        mc_path = os.path.join(d, "features/merged_classification.csv")
        agg_path = os.path.join(d, "metrics/aggregate_metrics.json")
        if not (os.path.isfile(mc_path) and os.path.isfile(agg_path)):
            continue
        mc = pd.read_csv(mc_path)
        with open(agg_path) as fp:
            agg = json.load(fp)
        n_total = len(mc)
        for cls, g in mc.groupby("primary_class"):
            sa = g["cf_shift"].abs()
            nr = g["W_aligned_dec_norm"] / g["W_base_dec_norm"].replace(0, np.nan)
            rows.append({
                "run": run_label, "slug": slug, "layer": layer, "class": cls,
                "n": len(g), "frac": len(g) / n_total,
                "shift_mean_abs": sa.mean(), "shift_median_abs": sa.median(),
                "shift_p95_abs": sa.quantile(0.95), "shift_max_abs": sa.max(),
                "shift_mean_signed": g["cf_shift"].mean(),
                "norm_ratio_median": nr.median(),
                "feature_sharing_ratio": agg["feature_sharing_ratio"],
            })
pd.DataFrame(rows).to_csv("per_feature_summary_all_seeds.csv", index=False, float_format="%.6g")
```
