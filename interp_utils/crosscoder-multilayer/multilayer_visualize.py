from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ICLR_DPI = 300


def _save(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close(fig)


def _last_series(history: Dict, key: str) -> list[float]:
    value = history.get(key, [])
    if value and isinstance(value[-1], list):
        return value[-1]
    return []


def _layer_labels(history: Dict, aggregate_metrics: Optional[Dict] = None) -> list[int]:
    layers = history.get("layers") or (aggregate_metrics or {}).get("layers") or []
    return [int(layer) for layer in layers]


def plot_multilayer_loss_curves(training_history: Dict, output_path: Path) -> None:
    epochs = training_history.get("epochs", [])
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    axes[0, 0].plot(epochs, training_history.get("train_loss", []), label="train", linewidth=2)
    axes[0, 0].plot(epochs, training_history.get("val_loss", []), label="val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    for key, label in (("self_recon", "train self"), ("cross_recon", "train cross"), ("sparsity", "train sparsity")):
        if training_history.get(key):
            axes[0, 1].plot(epochs, training_history[key], label=label, linewidth=2)
    for key, label in (("val_self_recon", "val self"), ("val_cross_recon", "val cross"), ("val_sparsity", "val sparsity")):
        if training_history.get(key):
            axes[0, 1].plot(epochs, training_history[key], label=label, linewidth=1.5, linestyle="--")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Component loss")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(epochs, training_history.get("val_fve_base", []), label="base", linewidth=2)
    axes[1, 0].plot(epochs, training_history.get("val_fve_aligned", []), label="aligned", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Validation FVE")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, training_history.get("dead_neurons", []), label="dead fraction", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Dead feature fraction")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    _save(fig, output_path)


def plot_fve_l0_by_layer(training_history: Dict, aggregate_metrics: Dict, plots_dir: Path) -> None:
    layers = _layer_labels(training_history, aggregate_metrics)
    if not layers:
        return

    fve_base = _last_series(training_history, "val_fve_base_by_layer") or aggregate_metrics.get("val_fve_base_by_layer", [])
    fve_aligned = _last_series(training_history, "val_fve_aligned_by_layer") or aggregate_metrics.get("val_fve_aligned_by_layer", [])
    if fve_base and fve_aligned:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(layers, fve_base, marker="o", label="base", linewidth=2)
        ax.plot(layers, fve_aligned, marker="o", label="aligned", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Validation FVE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / "fve_by_layer.png")

    l0_base = (
        _last_series(training_history, "val_l0_base_by_layer")
        or aggregate_metrics.get("val_l0_base_by_layer", [])
        or _last_series(training_history, "l0_base_by_layer")
        or aggregate_metrics.get("l0_base_by_layer", [])
    )
    l0_aligned = (
        _last_series(training_history, "val_l0_aligned_by_layer")
        or aggregate_metrics.get("val_l0_aligned_by_layer", [])
        or _last_series(training_history, "l0_aligned_by_layer")
        or aggregate_metrics.get("l0_aligned_by_layer", [])
    )
    if l0_base and l0_aligned:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(layers, l0_base, marker="o", label="base", linewidth=2)
        ax.plot(layers, l0_aligned, marker="o", label="aligned", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("L0")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / "l0_by_layer.png")


def plot_rho_theta_by_layer(profile_df: pd.DataFrame, plots_dir: Path) -> None:
    g = sns.FacetGrid(profile_df, col="layer", col_wrap=3, height=3, sharex=True, sharey=True)
    g.map_dataframe(sns.histplot, x="rho", bins=40, binrange=(0, 1), color="#4C78A8")
    g.set_axis_labels("rho", "Count (log)")
    for ax in g.axes.flat:
        ax.set_yscale("symlog", linthresh=1)
    g.savefig(plots_dir / "rho_histogram_by_layer.png", dpi=ICLR_DPI, bbox_inches="tight")
    plt.close(g.fig)

    sample_df = profile_df
    if len(sample_df) > 30000:
        sample_df = sample_df.sample(30000, random_state=0)
    g = sns.FacetGrid(sample_df, col="layer", col_wrap=3, height=3.2, sharex=True, sharey=True)
    hue = "layer_class" if "layer_class" in sample_df.columns else None
    g.map_dataframe(sns.scatterplot, x="rho", y="theta", hue=hue, s=8, alpha=0.45, linewidth=0)
    g.set_axis_labels("rho", "theta")
    g.set(xlim=(0, 1), ylim=(-1.05, 1.05))
    g.savefig(plots_dir / "rho_theta_scatter_by_layer.png", dpi=ICLR_DPI, bbox_inches="tight")
    plt.close(g.fig)


def plot_class_distribution(profile_df: pd.DataFrame, classification_df: pd.DataFrame, plots_dir: Path) -> None:
    if "layer_class" in profile_df.columns:
        counts = profile_df.groupby(["layer", "layer_class"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=counts, x="layer", y="count", hue="layer_class", ax=ax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature count")
        ax.legend(title="Layer class", fontsize=8)
        _save(fig, plots_dir / "class_distribution_multilayer.png")

    if "primary_class" in classification_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        order = classification_df["primary_class"].value_counts().index.tolist()
        sns.countplot(data=classification_df, x="primary_class", order=order, ax=ax)
        ax.set_xlabel("Primary class")
        ax.set_ylabel("Feature count")
        ax.tick_params(axis="x", rotation=35)
        _save(fig, plots_dir / "class_distribution_primary.png")


def plot_fsr_and_amplification(profile_df: pd.DataFrame, plots_dir: Path) -> None:
    if "layer_class" in profile_df.columns:
        layer_stats = []
        for layer, layer_df in profile_df.groupby("layer", sort=True):
            shared = layer_df["layer_class"].astype(str).str.startswith("shared_").mean()
            layer_stats.append({"layer": int(layer), "feature_sharing_ratio": float(shared)})
        stats_df = pd.DataFrame(layer_stats)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(stats_df["layer"], stats_df["feature_sharing_ratio"], marker="o", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature sharing ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / "feature_sharing_ratio_by_layer.png")

    ratio_df = profile_df.assign(
        decoder_norm_ratio=profile_df["W_aligned_dec_norm"] / (profile_df["W_base_dec_norm"] + 1e-8)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=ratio_df, x="layer", y="decoder_norm_ratio", ax=ax, showfliers=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Aligned/base decoder norm")
    _save(fig, plots_dir / "decoder_norm_ratio_by_layer.png")


def plot_decoder_norm_heatmaps(profile_df: pd.DataFrame, classification_df: pd.DataFrame, plots_dir: Path) -> None:
    total_norm = profile_df.groupby("feature_id")[["W_base_dec_norm", "W_aligned_dec_norm"]].sum().sum(axis=1)
    top_features = total_norm.sort_values(ascending=False).head(250).index.tolist()
    subset = profile_df[profile_df["feature_id"].isin(top_features)]
    order = classification_df.set_index("feature_id").reindex(top_features)
    if "norm_entropy" in order.columns:
        feature_order = order.sort_values(["primary_class", "norm_entropy"], ascending=[True, False]).index.tolist()
    else:
        feature_order = top_features

    for stream, col, name in (
        ("base", "W_base_dec_norm", "base_decoder_norm_heatmap.png"),
        ("aligned", "W_aligned_dec_norm", "aligned_decoder_norm_heatmap.png"),
    ):
        matrix = subset.pivot(index="feature_id", columns="layer", values=col).reindex(feature_order)
        fig, ax = plt.subplots(figsize=(6.5, max(4, min(12, 0.03 * len(matrix)))))
        sns.heatmap(matrix, cmap="viridis", ax=ax, cbar_kws={"label": f"{stream} decoder norm"})
        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature")
        _save(fig, plots_dir / name)


def plot_entropy_and_migration(classification_df: pd.DataFrame, plots_dir: Path) -> None:
    if "norm_entropy" in classification_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        hue = "primary_class" if "primary_class" in classification_df.columns else None
        sns.histplot(data=classification_df, x="norm_entropy", hue=hue, bins=40, multiple="stack", ax=ax)
        ax.set_xlabel("Layer norm entropy")
        ax.set_ylabel("Feature count")
        _save(fig, plots_dir / "layer_concentration_entropy.png")

    if {"max_base_layer", "max_aligned_layer"}.issubset(classification_df.columns):
        migration = pd.crosstab(classification_df["max_base_layer"], classification_df["max_aligned_layer"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(migration, annot=True, fmt="d", cmap="mako", ax=ax)
        ax.set_xlabel("Max aligned layer")
        ax.set_ylabel("Max base layer")
        _save(fig, plots_dir / "max_norm_layer_migration.png")


def plot_feature_trajectories(profile_df: pd.DataFrame, classification_df: pd.DataFrame, plots_dir: Path) -> None:
    top_features = (
        classification_df.assign(total_norm=classification_df["total_base_norm"] + classification_df["total_aligned_norm"])
        .sort_values("total_norm", ascending=False)
        .head(12)["feature_id"]
        .tolist()
    )
    subset = profile_df[profile_df["feature_id"].isin(top_features)]
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for feature_id, feature_df in subset.groupby("feature_id"):
        label = str(int(feature_id))
        axes[0, 0].plot(feature_df["layer"], feature_df["W_base_dec_norm"], marker="o", label=label)
        axes[0, 1].plot(feature_df["layer"], feature_df["W_aligned_dec_norm"], marker="o", label=label)
        axes[1, 0].plot(feature_df["layer"], feature_df["rho"], marker="o", label=label)
        axes[1, 1].plot(feature_df["layer"], feature_df["theta"], marker="o", label=label)
    for ax, ylabel in zip(
        axes.flat,
        ["Base decoder norm", "Aligned decoder norm", "rho", "theta"],
    ):
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(title="Feature", bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=8)
    _save(fig, plots_dir / "feature_layer_trajectories.png")


def plot_theta_by_layer(profile_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=profile_df, x="layer", y="theta", ax=ax, showfliers=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Base/aligned decoder cosine")
    _save(fig, plots_dir / "theta_by_layer.png")


def plot_cross_layer_cosine_drift(drift_df: pd.DataFrame, plots_dir: Path) -> None:
    if drift_df.empty:
        return
    summary = drift_df[drift_df["source_layer_pos"] != drift_df["target_layer_pos"]]
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=summary, x="stream", y="abs_cosine", ax=ax, showfliers=False)
    ax.set_xlabel("Stream")
    ax.set_ylabel("Cross-layer |cosine|")
    _save(fig, plots_dir / "cross_layer_cosine_drift_by_stream.png")


def plot_superposition_summary(superposition_results: Dict, plots_dir: Path) -> None:
    features = list(superposition_results.get("features", {}).values())
    if not features:
        return
    df = pd.DataFrame(
        {
            "feature_id": [item["feature_id"] for item in features],
            "target_layer": [item["target_layer"] for item in features],
            "r2": [item["r2"] for item in features],
            "n_nonzero": [item["n_nonzero"] for item in features],
            "is_superposition": [item["is_superposition"] for item in features],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sns.histplot(data=df, x="r2", hue="is_superposition", bins=30, ax=axes[0])
    axes[0].set_xlabel("Best cross-layer match R2")
    sns.boxplot(data=df, x="target_layer", y="r2", ax=axes[1], showfliers=False)
    axes[1].set_xlabel("Target layer")
    axes[1].set_ylabel("Best match R2")
    _save(fig, plots_dir / "superposition_by_layer.png")


def plot_counterfactual_shift_by_layer(cf_layer_df: pd.DataFrame, profile_df: pd.DataFrame, plots_dir: Path) -> None:
    if cf_layer_df.empty:
        return
    plot_df = cf_layer_df
    if "layer_class" in profile_df.columns:
        plot_df = cf_layer_df.merge(
            profile_df[["feature_id", "layer", "layer_class"]],
            on=["feature_id", "layer"],
            how="left",
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=plot_df, x="layer", y="cf_shift", ax=ax, showfliers=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation usage shift")
    _save(fig, plots_dir / "cf_shift_by_layer.png")

    if "layer_class" in plot_df.columns:
        summary = (
            plot_df.groupby(["layer", "layer_class"])["cf_shift_abs_p95"]
            .quantile(0.95)
            .reset_index(name="p95_abs_shift")
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(data=summary, x="layer", y="p95_abs_shift", hue="layer_class", marker="o", ax=ax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("P95 |usage shift|")
        ax.legend(title="Layer class", fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / "cf_shift_p95_by_layer.png")


def generate_multilayer_plots(
    training_history: Dict,
    classification_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    aggregate_metrics: Dict,
    plots_dir: Path,
    superposition_results: Optional[Dict] = None,
    drift_df: Optional[pd.DataFrame] = None,
    cf_layer_df: Optional[pd.DataFrame] = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_multilayer_loss_curves(training_history, plots_dir / "loss_curves.png")
    plot_fve_l0_by_layer(training_history, aggregate_metrics, plots_dir)
    plot_rho_theta_by_layer(profile_df, plots_dir)
    plot_class_distribution(profile_df, classification_df, plots_dir)
    plot_fsr_and_amplification(profile_df, plots_dir)
    plot_decoder_norm_heatmaps(profile_df, classification_df, plots_dir)
    plot_entropy_and_migration(classification_df, plots_dir)
    plot_feature_trajectories(profile_df, classification_df, plots_dir)
    plot_theta_by_layer(profile_df, plots_dir)
    if drift_df is not None:
        plot_cross_layer_cosine_drift(drift_df, plots_dir)
    if superposition_results is not None:
        plot_superposition_summary(superposition_results, plots_dir)
    if cf_layer_df is not None:
        plot_counterfactual_shift_by_layer(cf_layer_df, profile_df, plots_dir)
