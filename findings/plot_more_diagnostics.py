"""Additional cross-cell diagnostic plots for the workshop writeup.

Reads findings/per_feature_summary_all_seeds.csv and produces:
  fig_heatmap_aligned_only.{pdf,png}    -- 5x3 heatmap of aligned_only counts (seed-1 only)
  fig_class_composition.{pdf,png}       -- stacked bar of feature class fractions per cell
  fig_sharing_ratio.{pdf,png}           -- feature_sharing_ratio per cell, both seeds
  fig_heatmap_shift_p95.{pdf,png}       -- 5x3 heatmap of shared_aligned shift p95 (log scale)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS = Path(__file__).resolve().parent
df = pd.read_csv(THIS / "per_feature_summary_all_seeds.csv")

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "lines.linewidth": 1.5, "lines.markersize": 5,
    "savefig.bbox": "tight",
})


def parse_slug(slug):
    base, algo = slug.split("-", 1)
    return base, algo.upper()


ALGO_ORDER = ["DPO", "SIMPO", "GRPO", "ORPO", "KTO"]
BASE_ORDER = ["smollm", "llama", "qwen"]
BASE_LABEL = {"smollm": "SmolLM3-3B", "llama": "Llama-3.2-3B-Instruct", "qwen": "Qwen3-4B-Instruct-2507"}
CLASS_ORDER = ["base_only", "shared_attenuated", "shared_redirected", "shared_intermediate", "shared_aligned", "aligned_only", "other"]
CLASS_COLOR = {
    "base_only": "#bbbbbb",
    "shared_attenuated": "#9ecae1",
    "shared_redirected": "#fdae6b",
    "shared_intermediate": "#a1d99b",
    "shared_aligned": "#fb6a4a",
    "aligned_only": "#54278f",
    "other": "#000000",
}

# Cells where the crosscoder degenerates -- mark them visually.
DEGENERATE = {("seed1", "qwen-kto"), ("seed1", "qwen-orpo"), ("seed2", "qwen-kto"), ("seed2", "qwen-orpo")}


# ============================================================
# Figure: 5x3 heatmap of aligned_only counts (seed-1)
# ============================================================
ao = df[df["class"] == "aligned_only"].copy()
ao[["base", "algo"]] = ao["slug"].apply(lambda s: pd.Series(parse_slug(s)))

mat_s1 = (
    ao[ao["run"] == "seed1"]
    .pivot_table(index="algo", columns="base", values="n", aggfunc="first")
    .reindex(index=ALGO_ORDER, columns=BASE_ORDER)
)
mat_s2 = (
    ao[ao["run"] == "seed2"]
    .pivot_table(index="algo", columns="base", values="n", aggfunc="first")
    .reindex(index=ALGO_ORDER, columns=BASE_ORDER)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, mat, title, run_label in zip(axes, [mat_s1, mat_s2], ["Seed = 42", "Seed = 99"], ["seed1", "seed2"]):
    im = ax.imshow(np.log10(mat.values.astype(float) + 1e-1), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(BASE_ORDER)))
    ax.set_xticklabels([BASE_LABEL[b] for b in BASE_ORDER], rotation=15, ha="right")
    ax.set_yticks(range(len(ALGO_ORDER)))
    ax.set_yticklabels(ALGO_ORDER)
    ax.set_title(title)
    for i, algo in enumerate(ALGO_ORDER):
        for j, base in enumerate(BASE_ORDER):
            v = mat.iloc[i, j] if not pd.isna(mat.iloc[i, j]) else None
            if v is None:
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=12)
            else:
                marker = "*" if (run_label, f"{base}-{algo.lower()}") in DEGENERATE else ""
                ax.text(j, i, f"{int(v)}{marker}", ha="center", va="center",
                        color="white" if np.log10(v + 0.1) < 3.0 else "black", fontsize=11)
    plt.colorbar(im, ax=ax, label="log10(aligned-only count + 0.1)")
fig.suptitle("Aligned-only feature counts per (algorithm, base) heatmap. * = degenerate crosscoder.", fontsize=14)
fig.savefig(THIS / "fig_heatmap_aligned_only.pdf")
fig.savefig(THIS / "fig_heatmap_aligned_only.png", dpi=150)
plt.close(fig)


# ============================================================
# Figure: stacked bar of feature class fractions per cell (seed-1)
# ============================================================
sub = df[df["run"] == "seed1"].copy()
sub[["base", "algo"]] = sub["slug"].apply(lambda s: pd.Series(parse_slug(s)))
classes_present = [c for c in CLASS_ORDER if c in sub["class"].unique()]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    subb = sub[sub["base"] == base]
    pivot = (
        subb.pivot_table(index="algo", columns="class", values="frac", aggfunc="first", fill_value=0.0)
        .reindex(index=ALGO_ORDER, columns=classes_present, fill_value=0.0)
    )
    bottom = np.zeros(len(ALGO_ORDER))
    x = np.arange(len(ALGO_ORDER))
    for cls in classes_present:
        vals = pivot[cls].values
        ax.bar(x, vals, bottom=bottom, color=CLASS_COLOR[cls], label=cls if base == "smollm" else None,
               edgecolor="black", linewidth=0.4, width=0.8)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("fraction of features" if base == "smollm" else "")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=4, frameon=False, fontsize=9)
fig.suptitle("Feature class composition per (algorithm, base), seed = 42", fontsize=14)
fig.subplots_adjust(bottom=0.22)
fig.savefig(THIS / "fig_class_composition.pdf")
fig.savefig(THIS / "fig_class_composition.png", dpi=150)
plt.close(fig)


# ============================================================
# Figure: feature_sharing_ratio per cell, both seeds
# ============================================================
share = (
    df[df["class"] == "aligned_only"][["run", "slug", "feature_sharing_ratio"]]
    .drop_duplicates()
    .copy()
)
share[["base", "algo"]] = share["slug"].apply(lambda s: pd.Series(parse_slug(s)))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    s1 = share[(share["run"] == "seed1") & (share["base"] == base)].set_index("algo").reindex(ALGO_ORDER)
    s2 = share[(share["run"] == "seed2") & (share["base"] == base)].set_index("algo").reindex(ALGO_ORDER)
    x = np.arange(len(ALGO_ORDER))
    w = 0.4
    ax.bar(x - w / 2, s1["feature_sharing_ratio"], w, label="seed = 42", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, s2["feature_sharing_ratio"], w, label="seed = 99", color="#4c72b0", edgecolor="black", linewidth=0.5, hatch="//", alpha=0.7)
    for i, (v1, v2) in enumerate(zip(s1["feature_sharing_ratio"], s2["feature_sharing_ratio"])):
        if pd.notna(v1):
            ax.text(i - w / 2, v1, f"{v1:.2f}", ha="center", va="bottom", fontsize=8)
        if pd.notna(v2):
            ax.text(i + w / 2, v2, f"{v2:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color="red", linestyle="--", linewidth=0.7, alpha=0.5, label="rho_aligned threshold (0.85)" if base == "smollm" else None)
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("feature sharing ratio" if base == "smollm" else "")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if base == "smollm":
        ax.legend(loc="lower right", frameon=False)
fig.suptitle("Feature sharing ratio per (algorithm, base). Values near 1.0 indicate degenerate crosscoders.", fontsize=14)
fig.savefig(THIS / "fig_sharing_ratio.pdf")
fig.savefig(THIS / "fig_sharing_ratio.png", dpi=150)
plt.close(fig)


# ============================================================
# Figure: 5x3 heatmap of p95 shift on shared_aligned (seed-1, log scale)
# ============================================================
sa = df[df["class"] == "shared_aligned"].copy()
sa[["base", "algo"]] = sa["slug"].apply(lambda s: pd.Series(parse_slug(s)))

mat = (
    sa[sa["run"] == "seed1"]
    .pivot_table(index="algo", columns="base", values="shift_p95_abs", aggfunc="first")
    .reindex(index=ALGO_ORDER, columns=BASE_ORDER)
)

fig, ax = plt.subplots(figsize=(8, 5.5))
masked = np.ma.masked_invalid(mat.values.astype(float))
im = ax.imshow(masked, norm=LogNorm(vmin=max(masked.min(), 1e-3), vmax=masked.max()), cmap="rocket_r" if False else "magma_r", aspect="auto")
ax.set_xticks(range(len(BASE_ORDER)))
ax.set_xticklabels([BASE_LABEL[b] for b in BASE_ORDER], rotation=15, ha="right")
ax.set_yticks(range(len(ALGO_ORDER)))
ax.set_yticklabels(ALGO_ORDER)
for i in range(len(ALGO_ORDER)):
    for j in range(len(BASE_ORDER)):
        v = mat.iloc[i, j]
        if pd.isna(v):
            ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=12)
        else:
            ax.text(j, i, f"{v:.3f}", ha="center", va="bottom" if v < 0.05 else "center",
                    color="white" if v > 0.1 else "black", fontsize=11)
plt.colorbar(im, ax=ax, label="p95 |cf_shift| on shared_aligned (log scale)")
ax.set_title("Per-feature shift magnitude (95th percentile) on shared_aligned class")
fig.savefig(THIS / "fig_heatmap_shift_p95.pdf")
fig.savefig(THIS / "fig_heatmap_shift_p95.png", dpi=150)
plt.close(fig)


print("Wrote 4 new figures to", THIS)
for p in sorted(THIS.glob("fig_*.png")):
    print("  ", p.name, p.stat().st_size // 1024, "KB")
