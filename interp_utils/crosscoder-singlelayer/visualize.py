import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from . import config


plt.style.use('seaborn-v0_8-whitegrid')

# ICLR-style: publication-ready font sizes and colorblind-friendly palette.
# Each class has a distinct color; only "other" is grey.
ICLR_FONT = {"family": "serif", "size": 11}
ICLR_LABEL_FONT = {"family": "serif", "size": 12, "weight": "bold"}
ICLR_TICK_SIZE = 11
ICLR_DPI = 300

COLORS = {
    "base_only": "#C0392B",   # red
    "aligned_only": "#2980B9",     # blue
    "shared_aligned": "#27AE60",     # green
    "shared_redirected": "#8E44AD",  # purple
    "shared_attenuated": "#D35400",   # orange
    "shared_intermediate": "#C2185B", # magenta
    "other": "#7F8C8D",              # grey
}


def _apply_iclr_style(ax: plt.Axes) -> None:
    """Apply ICLR-standard styling: no title, bold axis labels, clear tick sizes."""
    ax.xaxis.set_tick_params(labelsize=ICLR_TICK_SIZE)
    ax.yaxis.set_tick_params(labelsize=ICLR_TICK_SIZE)
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontdict=ICLR_LABEL_FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontdict=ICLR_LABEL_FONT)
    ax.tick_params(axis="both", which="major", size=5, width=1.25)


def _gmm_boundary_1d(
    pi1: float,
    mu1: float,
    sigma1: float,
    pi2: float,
    mu2: float,
    sigma2: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """
    Find x where pi1*N(x|mu1,sigma1) = pi2*N(x|mu2,sigma2).
    Returns the root in [low, high], or midpoint if brentq fails.
    """
    def f(x: float) -> float:
        return pi1 * norm.pdf(x, mu1, sigma1) - pi2 * norm.pdf(x, mu2, sigma2)

    # Ensure bracket spans the region between means
    a = max(low, min(mu1, mu2) - 0.01)
    b = min(high, max(mu1, mu2) + 0.01)
    if a >= b:
        a, b = low, high

    try:
        return float(brentq(f, a, b))
    except (ValueError, RuntimeError):
        return float((mu1 + mu2) / 2)


def _config_threshold_defaults() -> Dict[str, float]:
    """Fallback when GMM cannot be fit (e.g. empty or degenerate rho)."""
    return {
        "rho_base_only": config.RHO_BASE_ONLY,
        "rho_aligned_only": config.RHO_ALIGNED_ONLY,
        "rho_shared_low": config.RHO_SHARED_LOW,
        "rho_shared_high": config.RHO_SHARED_HIGH,
    }


def compute_adaptive_rho_thresholds(
    classification_df: pd.DataFrame,
    n_components_range: tuple = (1, 4),
    covariance_type: str = "diag",
    min_samples: int = 30,
) -> Dict[str, float]:
    """
    Compute GMM-based rho thresholds for feature classification.
    Uses a Gaussian Mixture Model with BIC model selection on the rho distribution.
    Returns boundaries for uncompressed-only, shared (low, high), and compressed-only.
    Falls back to config defaults when GMM cannot be fit (e.g. degenerate rho).
    Used by feature_classification.csv, plots, and all evaluation.
    Set FORCE_FIXED_RHO=1 to skip GMM and use config defaults directly.
    """
    if os.environ.get("FORCE_FIXED_RHO") == "1":
        return _config_threshold_defaults()

    rho = np.asarray(classification_df["rho"].values, dtype=float)
    rho = rho[(rho >= 0) & (rho <= 1)]
    if len(rho) == 0:
        return _config_threshold_defaults()

    rho_2d = rho.reshape(-1, 1)
    if len(rho) < min_samples:
        p25, p75 = np.percentile(rho, [25, 75])
        return {
            "rho_base_only": float(np.clip(p25 * 0.6, 0.05, 0.35)),
            "rho_aligned_only": float(np.clip(p75 + (1 - p75) * 0.5, 0.65, 0.98)),
            "rho_shared_low": float(np.percentile(rho, 40)),
            "rho_shared_high": float(np.percentile(rho, 75)),
        }

    # Model selection via BIC
    best_bic = np.inf
    best_gmm = None
    for k in range(n_components_range[0], n_components_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=42,
            n_init=3,
            reg_covar=1e-4,
        )
        gmm.fit(rho_2d)
        bic = gmm.bic(rho_2d)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    if best_gmm is None:
        return _config_threshold_defaults()

    k = best_gmm.n_components
    means = best_gmm.means_.flatten()
    weights = best_gmm.weights_
    if covariance_type == "full":
        sigmas = np.sqrt(np.maximum(best_gmm.covariances_.flatten(), 1e-8))
    else:
        sigmas = np.sqrt(np.maximum(best_gmm.covariances_.flatten(), 1e-8))

    # Sort components by mean (left = uncompressed, right = compressed)
    order = np.argsort(means)
    means = means[order]
    weights = weights[order]
    sigmas = sigmas[order]

    p25, p75 = np.percentile(rho, [25, 75])

    if k == 1:
        return {
            "rho_base_only": float(np.clip(p25 * 0.6, 0.05, 0.35)),
            "rho_aligned_only": float(np.clip(p75 + (1 - p75) * 0.5, 0.65, 0.98)),
            "rho_shared_low": float(np.percentile(rho, 40)),
            "rho_shared_high": float(np.percentile(rho, 75)),
        }

    # Compute boundaries between adjacent components
    boundaries: List[float] = []
    for i in range(k - 1):
        b = _gmm_boundary_1d(
            weights[i], means[i], sigmas[i],
            weights[i + 1], means[i + 1], sigmas[i + 1],
            0.0, 1.0,
        )
        boundaries.append(float(np.clip(b, 0.0, 1.0)))

    if k == 2:
        b = boundaries[0]
        return {
            "rho_base_only": float(b),
            "rho_aligned_only": float(b),
            "rho_shared_low": float(b),
            "rho_shared_high": float(b),
        }

    # k >= 3: use first and last boundaries for shared band (covers middle components)
    rho_shared_low = boundaries[0]
    rho_shared_high = boundaries[-1]
    rho_u = min(rho_shared_low, float(means[0] + 2 * sigmas[0]), 0.4)
    rho_c = max(rho_shared_high, float(means[-1] - 2 * sigmas[-1]), 0.5)
    return {
        "rho_base_only": float(np.clip(rho_u, 0.05, 0.4)),
        "rho_aligned_only": float(np.clip(rho_c, 0.5, 0.98)),
        "rho_shared_low": float(rho_shared_low),
        "rho_shared_high": float(rho_shared_high),
    }


def classify_for_plot(rho: float, theta: float, thresh: Dict[str, float]) -> str:
    """
    Classify (rho, theta) using GMM-based rho thresholds and fixed theta thresholds.
    Uses |theta| (absolute value of cosine similarity) for theta-based classification.
    Used for feature_classification.csv, plots, and all evaluation.
    """
    rho_b = thresh["rho_base_only"]
    rho_a = thresh["rho_aligned_only"]
    rho_sl = thresh["rho_shared_low"]
    rho_sh = thresh["rho_shared_high"]
    abs_theta = abs(theta)
    if rho < rho_b:
        return "base_only"
    if rho > rho_a:
        return "aligned_only"
    if rho_sl < rho < rho_sh:
        if abs_theta >= config.THETA_ALIGNED:
            return "shared_aligned"
        if abs_theta < config.THETA_REDIRECTED:
            return "shared_redirected"
        return "shared_intermediate"
    if rho_b <= rho <= rho_sl:
        return "shared_attenuated"
    return "other"


def plot_loss_curves(training_history: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    epochs = training_history["epochs"]

    ax = axes[0, 0]
    ax.plot(epochs, training_history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, training_history["val_loss"], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    _apply_iclr_style(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, training_history["self_recon"], label="Self-recon", linewidth=2)
    ax.plot(epochs, training_history["cross_recon"], label="Cross-recon", linewidth=2)
    ax.plot(epochs, training_history["sparsity"], label="Sparsity", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component")
    _apply_iclr_style(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, training_history["val_fve_base"], label="FVE base", linewidth=2)
    ax.plot(epochs, training_history["val_fve_aligned"], label="FVE aligned", linewidth=2)
    ax.axhline(y=config.FVE_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({config.FVE_THRESHOLD})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FVE")
    _apply_iclr_style(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, training_history["dead_neurons"], label="Dead neurons", linewidth=2, color=COLORS["base_only"])
    ax.axhline(y=config.DEAD_NEURON_THRESHOLD, color=COLORS["shared_attenuated"], linestyle="--", linewidth=1,
               label=f"Threshold ({config.DEAD_NEURON_THRESHOLD})")
    ax2 = ax.twinx()
    ax2.plot(epochs, training_history["l0_base"], label="L0 base", linewidth=2, color=COLORS["aligned_only"], alpha=0.8)
    ax2.plot(epochs, training_history["l0_aligned"], label="L0 aligned", linewidth=2, color=COLORS["shared_aligned"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead neuron fraction", color=COLORS["base_only"], fontdict=ICLR_LABEL_FONT)
    ax2.set_ylabel("L0 sparsity", color=COLORS["aligned_only"], fontdict=ICLR_LABEL_FONT)
    _apply_iclr_style(ax)
    _apply_iclr_style(ax2)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_rho_histogram(classification_df: pd.DataFrame, output_path: Path) -> None:
    """Plot rho histogram with GMM-based threshold lines."""
    fig, ax = plt.subplots(figsize=(6, 4))
    rho_values = np.asarray(classification_df["rho"].values)
    rho_values = rho_values[(rho_values >= 0) & (rho_values <= 1)]
    thresh = compute_adaptive_rho_thresholds(classification_df)
    rho_b, rho_a = thresh["rho_base_only"], thresh["rho_aligned_only"]
    rho_sl, rho_sh = thresh["rho_shared_low"], thresh["rho_shared_high"]

    ax.hist(rho_values, bins=50, range=(0, 1), edgecolor="black", alpha=0.8,
            color=COLORS["shared_aligned"], linewidth=0.5)
    ax.axvline(x=rho_b, color=COLORS["base_only"], linestyle="--", linewidth=1.5,
               label=f"Base-only ({rho_b:.2f})")
    ax.axvline(x=rho_a, color=COLORS["aligned_only"], linestyle="--", linewidth=1.5,
               label=f"Aligned-only ({rho_a:.2f})")
    ax.axvline(x=rho_sl, color=COLORS["shared_attenuated"], linestyle=":", linewidth=1.2, alpha=0.9,
               label=f"Shared ({rho_sl:.2f}–{rho_sh:.2f})")
    ax.axvline(x=rho_sh, color=COLORS["shared_attenuated"], linestyle=":", linewidth=1.2, alpha=0.9)

    ax.set_xlabel(r"decoder norm ratio, $\rho$")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    _apply_iclr_style(ax)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_rho_theta_scatter(classification_df: pd.DataFrame, output_path: Path) -> None:
    """Plot rho vs theta scatter with GMM-based classification colors."""
    fig, ax = plt.subplots(figsize=(6, 5))
    thresh = compute_adaptive_rho_thresholds(classification_df)
    rho_b, rho_a = thresh["rho_base_only"], thresh["rho_aligned_only"]
    rho_sl, rho_sh = thresh["rho_shared_low"], thresh["rho_shared_high"]

    # Classify each point using the same GMM-based thresholds we draw, so colors match the regions
    plot_class = classification_df.apply(
        lambda row: classify_for_plot(row["rho"], row["theta"], thresh), axis=1
    )
    plot_df = classification_df.assign(plot_class=plot_class)

    for class_name, color in COLORS.items():
        class_df = plot_df[plot_df["plot_class"] == class_name]
        if len(class_df) > 0:
            ax.scatter(class_df["rho"], class_df["theta"], c=color,
                       label=f"{class_name} ({len(class_df)})", alpha=0.65, s=18, edgecolors="none")
    ax.axvline(x=rho_b, color=COLORS["base_only"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=rho_a, color=COLORS["aligned_only"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=rho_sl, color=COLORS["shared_attenuated"], linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=rho_sh, color=COLORS["shared_attenuated"], linestyle=":", linewidth=1, alpha=0.7)
    # |theta| thresholds: aligned >= 0.8, redirected < 0.5, intermediate in between
    ax.axhline(y=config.THETA_ALIGNED, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)
    ax.axhline(y=-config.THETA_ALIGNED, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)
    ax.axhline(y=config.THETA_REDIRECTED, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)
    ax.axhline(y=-config.THETA_REDIRECTED, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"decoder norm ratio, $\rho$")
    ax.set_ylabel(r"decoder cosine similarity, $\theta$")
    _apply_iclr_style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)

    # Adaptive ranges from data with small margin
    rho_vals = classification_df["rho"].values
    theta_vals = classification_df["theta"].values
    rho_min, rho_max = np.nanmin(rho_vals), np.nanmax(rho_vals)
    theta_min, theta_max = np.nanmin(theta_vals), np.nanmax(theta_vals)
    rho_margin = max(0.05, (rho_max - rho_min) * 0.05) if rho_max > rho_min else 0.05
    theta_margin = max(0.05, (theta_max - theta_min) * 0.05) if theta_max > theta_min else 0.05
    ax.set_xlim(max(-0.02, rho_min - rho_margin), min(1.02, rho_max + rho_margin))
    ax.set_ylim(max(-1.02, theta_min - theta_margin), min(1.02, theta_max + theta_margin))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_cf_distribution_per_class(merged_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    class_order = ["base_only", "aligned_only", "shared_aligned",
                   "shared_redirected", "shared_attenuated"]
    plot_df = merged_df[merged_df["primary_class"].isin(class_order)]

    ax = axes[0]
    if "cf_base" in plot_df.columns:
        sns.boxplot(data=plot_df, x="primary_class", y="cf_base", order=class_order, palette=COLORS, ax=ax)
        ax.set_xlabel("Feature class")
        ax.set_ylabel("CF base")
        ax.tick_params(axis="x", rotation=45, labelsize=ICLR_TICK_SIZE)
        _apply_iclr_style(ax)
    ax = axes[1]
    if "cf_aligned" in plot_df.columns:
        sns.boxplot(data=plot_df, x="primary_class", y="cf_aligned", order=class_order, palette=COLORS, ax=ax)
        ax.set_xlabel("Feature class")
        ax.set_ylabel("CF aligned")
        ax.tick_params(axis="x", rotation=45, labelsize=ICLR_TICK_SIZE)
        _apply_iclr_style(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_cf_shift_per_class(merged_df: pd.DataFrame, output_path: Path) -> None:
    if "cf_shift" not in merged_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    class_order = ["base_only", "aligned_only", "shared_aligned",
                   "shared_redirected", "shared_attenuated"]
    shift_by_class = merged_df.groupby("primary_class")["cf_shift"].agg(["mean", "std"])
    shift_by_class = shift_by_class.reindex(class_order).dropna()
    colors = [COLORS.get(c, "#7F8C8D") for c in shift_by_class.index]
    ax.bar(range(len(shift_by_class)), shift_by_class["mean"], yerr=shift_by_class["std"],
           color=colors, edgecolor="black", capsize=5, alpha=0.85, linewidth=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(range(len(shift_by_class)))
    ax.set_xticklabels(shift_by_class.index, rotation=45, ha="right")
    ax.set_xlabel("Feature class")
    ax.set_ylabel("CF shift (CF$_c$ $-$ CF$_u$)")
    _apply_iclr_style(ax)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_superposition_analysis(superposition_results: Dict, output_path: Path) -> None:
    features = superposition_results.get("features", {})
    if not features:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    r2_values = [f["r2"] for f in features.values()]
    n_nonzero_values = [f["n_nonzero"] for f in features.values()]
    is_superposition = [f["is_superposition"] for f in features.values()]

    ax = axes[0]
    ax.hist(r2_values, bins=30, edgecolor="black", alpha=0.8, color=COLORS["shared_aligned"], linewidth=0.5)
    ax.axvline(x=config.SUPERPOSITION_R2_THRESHOLD, color=COLORS["base_only"], linestyle="--", linewidth=1.5,
               label=f"R² threshold ({config.SUPERPOSITION_R2_THRESHOLD})")
    ax.set_xlabel("R² score")
    ax.set_ylabel("Count")
    _apply_iclr_style(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = [COLORS["shared_aligned"] if s else COLORS["base_only"] for s in is_superposition]
    ax.scatter(r2_values, n_nonzero_values, c=colors, alpha=0.7, s=35, edgecolors="black", linewidths=0.3)
    ax.axhline(y=config.SUPERPOSITION_MAX_CONSTITUENTS, color=COLORS["shared_attenuated"], linestyle="--", linewidth=1.2,
               label=f"Max constituents ({config.SUPERPOSITION_MAX_CONSTITUENTS})")
    ax.axvline(x=config.SUPERPOSITION_R2_THRESHOLD, color=COLORS["base_only"], linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("R² score")
    ax.set_ylabel("Number of non-zero constituents")
    _apply_iclr_style(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_fsr_comparison(metrics_dict: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    configs = list(metrics_dict.keys())
    fsr_values = [m.get("feature_sharing_ratio", 0) for m in metrics_dict.values()]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(configs)))
    bars = ax.bar(range(len(configs)), fsr_values, color=cmap, edgecolor="black", alpha=0.85, linewidth=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=ICLR_TICK_SIZE)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Feature sharing ratio (FSR)")
    ax.set_ylim(0, 1)
    _apply_iclr_style(ax)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, fsr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_method_comparison(
    wanda_metrics: Dict,
    awq_metrics: Dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    metrics_to_compare = ["feature_sharing_ratio", "semantic_stability_score", "superposition_fraction"]
    ylabels = ["Feature sharing ratio", "Semantic stability score", "Superposition fraction"]
    for ax, metric, ylabel in zip(axes, metrics_to_compare, ylabels):
        wanda_val = wanda_metrics.get(metric, 0)
        awq_val = awq_metrics.get(metric, 0)
        bars = ax.bar(["Wanda", "AWQ"], [wanda_val, awq_val],
                      color=[COLORS["base_only"], COLORS["aligned_only"]],
                      edgecolor="black", alpha=0.85, linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(wanda_val, awq_val) * 1.2 + 0.1)
        _apply_iclr_style(ax)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [wanda_val, awq_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


# Fixed order for class distribution: 6 non-other classes, always shown (count 0 if absent).
CLASS_DISTRIBUTION_ORDER = [
    "base_only",
    "shared_aligned",
    "shared_redirected",
    "shared_attenuated",
    "shared_intermediate",
    "aligned_only",
]


def plot_class_distribution(classification_df: pd.DataFrame, output_path: Path) -> None:
    """Plot class distribution using GMM-based thresholds (consistent with rho_histogram and rho_theta_scatter)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    thresh = compute_adaptive_rho_thresholds(classification_df)
    gmm_class = classification_df.apply(
        lambda row: classify_for_plot(row["rho"], row["theta"], thresh), axis=1
    )
    raw_counts = gmm_class.value_counts()
    # Always show all 6 classes in fixed order; use 0 for missing classes.
    counts = [raw_counts.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER]
    colors = [COLORS[c] for c in CLASS_DISTRIBUTION_ORDER]
    bars = ax.bar(range(len(CLASS_DISTRIBUTION_ORDER)), counts, color=colors,
                  edgecolor="black", alpha=0.85, linewidth=0.8)
    ax.set_xticks(range(len(CLASS_DISTRIBUTION_ORDER)))
    ax.set_xticklabels(CLASS_DISTRIBUTION_ORDER, rotation=45, ha="right", fontsize=ICLR_TICK_SIZE)
    ax.set_xlabel("Feature class")
    ax.set_ylabel("Count")
    _apply_iclr_style(ax)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(val)),
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def plot_shared_geometry(shared_geom_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Histograms of angle_deg and norm_ratio_raw per subclass, and scatter of theta vs norm_ratio_raw.
    """
    shared_classes = [
        "shared_aligned",
        "shared_redirected",
        "shared_attenuated",
        "shared_intermediate",
    ]
    plot_df = shared_geom_df[shared_geom_df["primary_class"].isin(shared_classes)]
    if len(plot_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for col, ax in zip(["angle_deg", "norm_ratio_raw"], axes):
        for cls in shared_classes:
            subset = plot_df[plot_df["primary_class"] == cls]
            if len(subset) > 0 and col in subset.columns:
                ax.hist(
                    subset[col].dropna(),
                    bins=25,
                    alpha=0.6,
                    label=cls,
                    color=COLORS.get(cls, "#7F8C8D"),
                    edgecolor="black",
                    linewidth=0.3,
                )
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        _apply_iclr_style(ax)
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "shared_geometry_histograms.png", dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in shared_classes:
        subset = plot_df[plot_df["primary_class"] == cls]
        if len(subset) > 0 and "theta" in subset.columns and "norm_ratio_raw" in subset.columns:
            ax.scatter(
                subset["theta"],
                subset["norm_ratio_raw"],
                c=COLORS.get(cls, "#7F8C8D"),
                label=f"{cls} ({len(subset)})",
                alpha=0.65,
                s=18,
                edgecolors="none",
            )
    ax.set_xlabel(r"decoder cosine similarity, $\theta$")
    ax.set_ylabel(r"norm ratio $\|W_{\mathrm{aligned}}\|/\|W_{\mathrm{base}}\|$")
    _apply_iclr_style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "shared_geometry_theta_vs_norm_ratio.png", dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    training_history: Dict,
    classification_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    superposition_results: Dict,
    plots_dir: Path,
    features_dir: Optional[Path] = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(training_history, plots_dir / "loss_curves.png")
    plot_rho_histogram(classification_df, plots_dir / "rho_histogram.png")
    plot_rho_theta_scatter(classification_df, plots_dir / "rho_theta_scatter.png")
    plot_cf_distribution_per_class(merged_df, plots_dir / "cf_distribution_per_class.png")
    plot_cf_shift_per_class(merged_df, plots_dir / "cf_shift_per_class.png")
    plot_superposition_analysis(superposition_results, plots_dir / "superposition_analysis.png")
    plot_class_distribution(classification_df, plots_dir / "class_distribution.png")

    geom_path = (features_dir or plots_dir.parent / "features") / "shared_features_geometry.csv"
    if geom_path.exists():
        shared_geom_df = pd.read_csv(geom_path)
        plot_shared_geometry(shared_geom_df, plots_dir)
