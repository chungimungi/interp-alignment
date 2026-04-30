"""Seed-comparison plots for the workshop writeup.

Reads findings/per_feature_summary_all_seeds.csv (rows: run × slug × class)
and renders:
  1. Side-by-side bar chart of aligned_only counts (seed1 vs seed2 per cell)
  2. Family ratio plot (broad-recruitment / concentrated-modification per base)
  3. Updated partition scatter with seed1 and seed2 overlaid
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS = Path(__file__).resolve().parent
OUT = THIS / "crosscoder"
OUT.mkdir(exist_ok=True)
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
ALGO_COLOR = {"DPO": "#1f77b4", "SIMPO": "#ff7f0e", "GRPO": "#2ca02c", "ORPO": "#d62728", "KTO": "#9467bd"}

ao = df[df["class"] == "aligned_only"].copy()
ao[["base", "algo"]] = ao["slug"].apply(lambda s: pd.Series(parse_slug(s)))

# ============================================================
# Figure 1: aligned_only counts paired (seed1 vs seed2) per cell
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, base in zip(axes, BASE_ORDER):
    sub = ao[ao["base"] == base]
    s1 = sub[sub["run"] == "seed1"].set_index("algo").reindex(ALGO_ORDER)
    s2 = sub[sub["run"] == "seed2"].set_index("algo").reindex(ALGO_ORDER)
    x = np.arange(len(ALGO_ORDER))
    w = 0.4
    bar1 = ax.bar(x - w / 2, s1["n"], w, label="seed=42", color=[ALGO_COLOR[a] for a in ALGO_ORDER], edgecolor="black", linewidth=0.6)
    bar2 = ax.bar(x + w / 2, s2["n"], w, label="seed=99", color=[ALGO_COLOR[a] for a in ALGO_ORDER], edgecolor="black", linewidth=0.6, hatch="//", alpha=0.7)
    for i, (v1, v2) in enumerate(zip(s1["n"], s2["n"])):
        if pd.notna(v1):
            ax.text(i - w / 2, v1, f"{int(v1)}", ha="center", va="bottom", fontsize=8)
        if pd.notna(v2):
            ax.text(i + w / 2, v2, f"{int(v2)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_title(BASE_LABEL[base])
    ax.set_ylabel("aligned-only feature count" if base == "smollm" else "")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if base == "smollm":
        ax.legend(loc="upper right", frameon=False)
fig.suptitle("Aligned-only feature recruitment, seed-1 (solid) vs seed-2 (hatched)", fontsize=14)
fig.savefig(OUT / "fig_aligned_only_seeds.pdf")
fig.savefig(OUT / "fig_aligned_only_seeds.png", dpi=150)
plt.close(fig)

# ============================================================
# Figure 2: family-ratio plot — DPO/SIMPO/GRPO mean / KTO/ORPO mean per base
# ============================================================
broad = ["DPO", "SIMPO", "GRPO"]
conc = ["KTO", "ORPO"]
ratios = []
for run in ["seed1", "seed2"]:
    for base in BASE_ORDER:
        sub = ao[(ao["run"] == run) & (ao["base"] == base)].set_index("algo")
        b_mean = sub.loc[[a for a in broad if a in sub.index], "n"].mean()
        c_mean = sub.loc[[a for a in conc if a in sub.index], "n"].mean()
        ratios.append({"run": run, "base": base, "broad_mean": b_mean, "conc_mean": c_mean,
                       "ratio_b_over_c": b_mean / max(c_mean, 1)})
rdf = pd.DataFrame(ratios)
print("Family ratio table:")
print(rdf.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(BASE_ORDER))
w = 0.4
s1 = rdf[rdf["run"] == "seed1"].set_index("base").reindex(BASE_ORDER)
s2 = rdf[rdf["run"] == "seed2"].set_index("base").reindex(BASE_ORDER)
b1 = ax.bar(x - w / 2, s1["ratio_b_over_c"], w, label="seed=42", color="#4c72b0", edgecolor="black")
b2 = ax.bar(x + w / 2, s2["ratio_b_over_c"], w, label="seed=99", color="#4c72b0", edgecolor="black", hatch="//", alpha=0.7)
for i, (v1, v2) in enumerate(zip(s1["ratio_b_over_c"], s2["ratio_b_over_c"])):
    ax.text(i - w / 2, v1, f"{v1:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.text(i + w / 2, v2, f"{v2:.1f}×", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([BASE_LABEL[b] for b in BASE_ORDER])
ax.set_ylabel("mean(DPO,SIMPO,GRPO) / mean(KTO,ORPO) aligned-only count")
ax.set_title("Broad-recruitment vs concentrated-modification family ratio (per base, both seeds)")
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="parity")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(OUT / "fig_family_ratio.pdf")
fig.savefig(OUT / "fig_family_ratio.png", dpi=150)
plt.close(fig)

# ============================================================
# Figure 3: updated partition scatter (seed1 dots + seed2 hollow markers)
# ============================================================
sa = df[df["class"] == "shared_aligned"].copy()
sa[["base", "algo"]] = sa["slug"].apply(lambda s: pd.Series(parse_slug(s)))
joined_s1 = (
    ao[ao["run"] == "seed1"][["slug", "base", "algo", "n"]]
    .rename(columns={"n": "aligned_only_n"})
    .merge(sa[sa["run"] == "seed1"][["slug", "shift_p95_abs"]].rename(columns={"shift_p95_abs": "shift_p95"}), on="slug")
)
joined_s2 = (
    ao[ao["run"] == "seed2"][["slug", "base", "algo", "n"]]
    .rename(columns={"n": "aligned_only_n"})
    .merge(sa[sa["run"] == "seed2"][["slug", "shift_p95_abs"]].rename(columns={"shift_p95_abs": "shift_p95"}), on="slug")
)

fig, ax = plt.subplots(figsize=(10, 7))
shape_map = {"smollm": "o", "llama": "s", "qwen": "^"}
for _, r in joined_s1.iterrows():
    ax.scatter(r["aligned_only_n"], r["shift_p95"], s=170,
               color=ALGO_COLOR[r["algo"]], edgecolor="black", linewidth=0.8,
               marker=shape_map[r["base"]], alpha=0.85)
    ax.annotate(r["algo"].lower(), (r["aligned_only_n"], r["shift_p95"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)
for _, r in joined_s2.iterrows():
    ax.scatter(r["aligned_only_n"], r["shift_p95"], s=170,
               facecolor="none", edgecolor=ALGO_COLOR[r["algo"]], linewidth=2.0,
               marker=shape_map[r["base"]], alpha=0.85)

# Connect seed1<->seed2 pairs with a thin line for the same cell
for slug in joined_s1["slug"].unique():
    p1 = joined_s1[joined_s1["slug"] == slug].iloc[0] if not joined_s1[joined_s1["slug"] == slug].empty else None
    p2 = joined_s2[joined_s2["slug"] == slug].iloc[0] if not joined_s2[joined_s2["slug"] == slug].empty else None
    if p1 is not None and p2 is not None:
        ax.plot([p1["aligned_only_n"], p2["aligned_only_n"]],
                [p1["shift_p95"], p2["shift_p95"]],
                color=ALGO_COLOR[p1["algo"]], alpha=0.4, linewidth=1.0)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("aligned-only feature count (log)")
ax.set_ylabel("p95 |cf_shift| on shared_aligned features (log)")
ax.set_title("Two-axis partition: solid = seed=42, hollow = seed=99 (line connects same cell)")
ax.grid(True, which="both", alpha=0.3)

shape_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="SmolLM3"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Llama-3.2"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Qwen3-4B"),
]
algo_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ALGO_COLOR[a], markersize=10, label=a) for a in ALGO_ORDER]
leg1 = ax.legend(handles=shape_legend, loc="upper right", title="Base", frameon=False)
ax.add_artist(leg1)
ax.legend(handles=algo_legend, loc="lower left", title="Algorithm", frameon=False)

fig.savefig(OUT / "fig_partition_scatter_seeds.pdf")
fig.savefig(OUT / "fig_partition_scatter_seeds.png", dpi=150)
plt.close(fig)

print()
print("Figures written to", OUT)
for p in sorted(OUT.glob("fig_*.png")):
    print("  ", p.name, p.stat().st_size // 1024, "KB")
