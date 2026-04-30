"""Render diagnostic plots for the workshop writeup from per_feature_summary.csv."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
OUT = THIS / "crosscoder"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(THIS / "per_feature_summary.csv")

# Standard plot style.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.0, "lines.linewidth": 1.5, "lines.markersize": 5,
    "savefig.bbox": "tight",
})

# Map slug -> (base_short, algo)
def parse_slug(slug):
    base, algo = slug.split("-", 1)
    return base, algo.upper()

ALGO_ORDER = ["DPO", "SIMPO", "GRPO", "ORPO", "KTO"]  # broad first, concentrated last
BASE_ORDER = ["smollm", "llama", "qwen"]
BASE_LABEL = {"smollm": "SmolLM3-3B", "llama": "Llama-3.2-3B-Instruct", "qwen": "Qwen3-4B-Instruct-2507"}
ALGO_COLOR = {"DPO": "#1f77b4", "SIMPO": "#ff7f0e", "GRPO": "#2ca02c", "ORPO": "#d62728", "KTO": "#9467bd"}

# ============================================================
# Figure 1: aligned-only feature counts per (algo, base)
# ============================================================
ao = df[df["class"] == "aligned_only"].copy()
ao[["base", "algo"]] = ao["slug"].apply(lambda s: pd.Series(parse_slug(s)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    sub = ao[ao["base"] == base].set_index("algo").reindex(ALGO_ORDER)
    bars = ax.bar(sub.index, sub["n"], color=[ALGO_COLOR[a] for a in sub.index])
    for bar, n in zip(bars, sub["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(n) if pd.notna(n) else 'N/A'}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("aligned-only feature count" if base == "smollm" else "")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
fig.suptitle("Aligned-only feature recruitment per (algorithm, base) — broad vs concentrated", fontsize=14)
fig.savefig(OUT / "fig_aligned_only_counts.pdf")
fig.savefig(OUT / "fig_aligned_only_counts.png", dpi=150)
plt.close(fig)

# ============================================================
# Figure 2: shared_aligned p95 shift per (algo, base) — log scale
# ============================================================
sa = df[df["class"] == "shared_aligned"].copy()
sa[["base", "algo"]] = sa["slug"].apply(lambda s: pd.Series(parse_slug(s)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    sub = sa[sa["base"] == base].set_index("algo").reindex(ALGO_ORDER)
    bars = ax.bar(sub.index, sub["shift_p95_abs"], color=[ALGO_COLOR[a] for a in sub.index])
    for bar, v in zip(bars, sub["shift_p95_abs"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("p95 |cf_shift| on shared_aligned features" if base == "smollm" else "")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
fig.suptitle("How aggressively each algorithm shifts shared features (95th percentile, log scale)", fontsize=14)
fig.savefig(OUT / "fig_shift_p95.pdf")
fig.savefig(OUT / "fig_shift_p95.png", dpi=150)
plt.close(fig)

# ============================================================
# Figure 3: decoder norm ratio (aligned_only median) per (algo, base)
# ============================================================
ao_norm = df[df["class"] == "aligned_only"].copy()
ao_norm[["base", "algo"]] = ao_norm["slug"].apply(lambda s: pd.Series(parse_slug(s)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    sub = ao_norm[ao_norm["base"] == base].set_index("algo").reindex(ALGO_ORDER)
    bars = ax.bar(sub.index, sub["norm_ratio_median"], color=[ALGO_COLOR[a] for a in sub.index])
    for bar, v in zip(bars, sub["norm_ratio_median"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="parity")
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("median(||W_aligned_dec|| / ||W_base_dec||)" if base == "smollm" else "")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
fig.suptitle("Decoder norm amplification on aligned-only features (median across features)", fontsize=14)
fig.savefig(OUT / "fig_decoder_norm_ratio.pdf")
fig.savefig(OUT / "fig_decoder_norm_ratio.png", dpi=150)
plt.close(fig)

# ============================================================
# Figure 4: 2D scatter — count vs shift, the partition
# ============================================================
sa_pick = sa[["slug", "shift_p95_abs"]].rename(columns={"shift_p95_abs": "shared_aligned_p95_shift"})
joined = ao[["slug", "base", "algo", "n"]].rename(columns={"n": "aligned_only_n"}).merge(sa_pick, on="slug")

fig, ax = plt.subplots(figsize=(9, 7))
for base in BASE_ORDER:
    sub = joined[joined["base"] == base]
    for _, row in sub.iterrows():
        ax.scatter(row["aligned_only_n"], row["shared_aligned_p95_shift"],
                   s=180, color=ALGO_COLOR[row["algo"]], edgecolor="black", linewidth=0.8,
                   marker={"smollm": "o", "llama": "s", "qwen": "^"}[base],
                   label=f"{row['algo']} on {BASE_LABEL[base].split('-')[0]}",
                   alpha=0.85)
        ax.annotate(f"{row['algo'].lower()}", (row["aligned_only_n"], row["shared_aligned_p95_shift"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("aligned-only feature count (log)")
ax.set_ylabel("p95 |cf_shift| on shared_aligned features (log)")
ax.set_title("The partition: broad-recruitment (right-bottom) vs concentrated-modification (left-top)")
ax.grid(True, which="both", alpha=0.3)

# Legend by base shape only (avoid 15-entry legend)
from matplotlib.lines import Line2D
shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="SmolLM3"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Llama-3.2"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Qwen3-4B"),
]
algo_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ALGO_COLOR[a], markersize=10, label=a) for a in ALGO_ORDER]
leg1 = ax.legend(handles=shape_legend, loc="upper right", title="Base", frameon=False)
ax.add_artist(leg1)
ax.legend(handles=algo_legend, loc="lower left", title="Algorithm", frameon=False)

fig.savefig(OUT / "fig_partition_scatter.pdf")
fig.savefig(OUT / "fig_partition_scatter.png", dpi=150)
plt.close(fig)

print("Wrote 4 figures to", OUT)
for p in sorted(OUT.glob("fig_*.png")):
    print("  ", p.name, p.stat().st_size // 1024, "KB")
