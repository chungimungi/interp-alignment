"""Plot Sparse Autoencoder (SAE) training results grouped by alignment algorithm.

Reads ``output/sae/<model_dir>/<layer_dir>/metrics.jsonl`` files produced by
``run_all_saes.py`` / ``sae.py`` and generates NeurIPS-quality PDF figures:

* Grouped bar plots of final-eval metrics (explained variance, reconstruction
  cosine similarity, MSE, CE-loss preservation, sparsity) with alignment
  algorithms (DPO / GRPO / KTO / ORPO / PPO / SimPO) and the unaligned baseline on the x-axis
  and the base-model family as a seaborn hue.
* Training-curve line plots (explained variance, MSE, overall loss, dead
  features) with one sub-panel per alignment algorithm + a separate baseline
  panel, again using the base-model family as the hue.

Only the probe-optimal layer (``layer_<idx>_best`` directory) is used by
default; pass ``--layer-tag mid`` to plot the mid-network layer instead, or
``--layer-tag both`` to emit both tag suites side-by-side.

Outputs are written to ``output/sae_plots/`` (override with ``--output-dir``).
Fonts / sizes match ``linear-probe.py``'s ``configure_plot_style`` so the PDFs
look consistent with the linear-probe figures in the paper.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

LEGEND_LABEL_BAR = 10.5
LEGEND_TITLE_BAR = 11.5
LEGEND_LABEL_CURVE = 15.0
LEGEND_TITLE_CURVE = 17.0

ALGORITHMS = ["DPO", "GRPO", "KTO", "ORPO", "PPO", "SimPO"]
BASELINE_LABEL = "Baseline"
GROUP_ORDER = [BASELINE_LABEL, "DPO", "GRPO", "KTO", "ORPO", "PPO", "SimPO"]
# Training-curve grid: always one column per method (incl. PPO) so layout stays 4+3.
CURVE_PANEL_ORDER = [BASELINE_LABEL, "DPO", "GRPO", "KTO", "ORPO", "PPO", "SimPO"]
# NeurIPS-ready multi-panel sizing (full-width landscape figure, larger panels).
CURVE_PANEL_W_IN = 4.4
CURVE_PANEL_H_IN = 3.4
CURVE_LEGEND_ROW_RATIO = 1.0  # kept for compatibility with older layouts
BASE_ORDER = ["Llama-3.2-3B-Instruct", "Qwen3-4B-Instruct-2507", "SmolLM3-3B"]
CURVE_LEGEND_LABELS = {
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "SmolLM3-3B": "SmolLM3-3B",
}

BASELINE_DIR_TO_BASE = {
    "HuggingFaceTB_SmolLM3-3B": "SmolLM3-3B",
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B-Instruct-2507",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
}


@dataclass
class SaeRun:
    directory: Path
    model_dir: str
    layer_dir: str
    base_model: str
    algorithm: str
    layer_idx: int
    layer_tag: str 


def _apply_legend_fontsizes(
    legend,
    *,
    label_fontsize: float,
    title_fontsize: float,
) -> None:
    for text in legend.get_texts():
        text.set_fontsize(label_fontsize)
    title = legend.get_title()
    if title is not None:
        title.set_fontsize(title_fontsize)


def configure_plot_style() -> None:
    """Match the NeurIPS-ready style from ``linear-probe.py``."""
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            # NeurIPS print-friendly typography.
            "font.size": 10.5,
            "axes.titlesize": 15.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": LEGEND_LABEL_BAR,
            "legend.title_fontsize": LEGEND_TITLE_BAR,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.0,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )
    sns.set_style("whitegrid", {"grid.alpha": 0.3, "axes.edgecolor": "0.15"})


def normalize_base(name: str) -> str:
    low = name.lower()
    if low.startswith("llama3") or "llama-3" in low:
        return "Llama-3.2-3B-Instruct"
    if "qwen3-4b" in low:
        return "Qwen3-4B-Instruct-2507"
    if "smollm3" in low:
        return "SmolLM3-3B"
    return name


def parse_model_dir(name: str) -> tuple[str, str] | None:
    """Return ``(base_model, algorithm)`` for a folder under ``output/saes``.

    Returns ``None`` if the folder name is not recognised.
    """
    if name in BASELINE_DIR_TO_BASE:
        return BASELINE_DIR_TO_BASE[name], BASELINE_LABEL

    if not name.startswith("MInAlA_"):
        return None

    body = name[len("MInAlA_"):]
    if body.endswith("-merged"):
        body = body[: -len("-merged")]

    body_upper = body.upper()
    # Longer suffixes first so e.g. ...-SimPO never matches ...-PPO.
    for algo in sorted(ALGORITHMS, key=len, reverse=True):
        suffix = "-" + algo
        if body_upper.endswith(suffix):
            base_str = body[: -len(suffix)]
            return normalize_base(base_str), algo

    lower_algo = {a.lower(): a for a in ALGORITHMS}
    for key, algo in sorted(lower_algo.items(), key=lambda kv: len(kv[0]), reverse=True):
        suffix = "-" + key
        if body.lower().endswith(suffix):
            base_str = body[: -len(suffix)]
            return normalize_base(base_str), algo

    return None


def parse_layer_dir(name: str) -> tuple[int, str] | None:
    """Return ``(layer_idx, tag)`` for ``layer_<idx>_<tag>`` directories."""
    if not name.startswith("layer_"):
        return None
    rest = name[len("layer_"):]
    parts = rest.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        idx = int(parts[0])
    except ValueError:
        return None
    tag = parts[1].lower()
    if tag not in {"best", "mid"}:
        return None
    return idx, tag


def discover_runs(root: Path, layer_tags: Iterable[str]) -> list[SaeRun]:
    tag_set = {t.lower() for t in layer_tags}
    runs: list[SaeRun] = []
    if not root.exists():
        raise FileNotFoundError(f"SAE output root does not exist: {root}")
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = parse_model_dir(model_dir.name)
        if parsed is None:
            print(f"[skip] unrecognised model dir: {model_dir.name}")
            continue
        base, algo = parsed
        for layer_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            layer_parsed = parse_layer_dir(layer_dir.name)
            if layer_parsed is None:
                continue
            idx, tag = layer_parsed
            if tag not in tag_set:
                continue
            metrics_file = layer_dir / "metrics.jsonl"
            if not metrics_file.exists():
                print(f"[skip] missing metrics.jsonl: {layer_dir}")
                continue
            runs.append(
                SaeRun(
                    directory=layer_dir,
                    model_dir=model_dir.name,
                    layer_dir=layer_dir.name,
                    base_model=base,
                    algorithm=algo,
                    layer_idx=idx,
                    layer_tag=tag,
                )
            )
    return runs


def load_metrics(run: SaeRun) -> tuple[pd.DataFrame, dict | None]:
    """Return ``(training_curves_df, final_eval_dict_or_None)`` for ``run``."""
    scalar_rows: list[dict] = []
    final_eval: dict | None = None
    path = run.directory / "metrics.jsonl"
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "metric":
                continue
            step = rec.get("step")
            data = rec.get("data", {})
            scalar_row: dict = {"step": step}
            has_eval = "reconstruction_quality" in data
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    scalar_row[key] = value
            scalar_rows.append(scalar_row)

            if has_eval:
                flat: dict = {"step": step}
                for section, sub in data.items():
                    if isinstance(sub, dict):
                        for k, v in sub.items():
                            if isinstance(v, (int, float)):
                                flat[f"{section}.{k}"] = v
                    elif isinstance(sub, (int, float)):
                        flat[section] = sub
                final_eval = flat

    df = pd.DataFrame(scalar_rows).sort_values("step").reset_index(drop=True)
    return df, final_eval


def build_final_eval_table(runs: list[SaeRun]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        _, final = load_metrics(run)
        if final is None:
            print(f"[warn] no eval metrics for {run.directory}")
            continue
        row = {
            "base_model": run.base_model,
            "algorithm": run.algorithm,
            "layer": run.layer_idx,
            "layer_tag": run.layer_tag,
            "model_dir": run.model_dir,
        }
        row.update(final)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


BASE_CURVE_STYLE: dict[str, dict[str, float]] = {
    # Lighter line (Qwen) is drawn first; darker/stronger lines are drawn on top.
    "Qwen3-4B-Instruct-2507": {"alpha": 0.58, "linewidth": 1.9, "zorder": 1},
    "SmolLM3-3B": {"alpha": 0.84, "linewidth": 2.15, "zorder": 2},
    "Llama-3.2-3B-Instruct": {"alpha": 0.95, "linewidth": 2.35, "zorder": 3},
}


def _adjust_lightness(color: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    """Scale color lightness in HLS space."""
    h, l, s = colorsys.rgb_to_hls(*color)
    l = min(1.0, max(0.0, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def _ordered_hue_palette(hue_levels: list[str]) -> dict[str, tuple]:
    # Use perceptually-uniform ``viridis`` with slight trimming at the light
    # end so the brightest bar still contrasts against the white background.
    n = max(len(hue_levels), 3)
    palette = sns.color_palette("viridis", n_colors=n + 1)[:n]
    out = {lvl: palette[i] for i, lvl in enumerate(hue_levels)}
    if "Qwen3-4B-Instruct-2507" in out:
        out["Qwen3-4B-Instruct-2507"] = _adjust_lightness(
            out["Qwen3-4B-Instruct-2507"], 1.18
        )
    if "Llama-3.2-3B-Instruct" in out:
        out["Llama-3.2-3B-Instruct"] = _adjust_lightness(
            out["Llama-3.2-3B-Instruct"], 0.82
        )
    return out


def _present_order(values: Iterable[str], order: Iterable[str]) -> list[str]:
    seen = set(values)
    ordered = [v for v in order if v in seen]
    ordered.extend(sorted(v for v in seen if v not in set(order)))
    return ordered


# Final-eval metrics we render as grouped bar plots.
BAR_METRICS: list[dict] = [
    {
        "column": "reconstruction_quality.explained_variance",
        "label": "Explained Variance",
        "filename": "bar_explained_variance",
        "higher_is_better": True,
    },
    {
        "column": "reconstruction_quality.cossim",
        "label": "Reconstruction Cosine Similarity",
        "filename": "bar_cossim",
        "higher_is_better": True,
    },
    {
        "column": "reconstruction_quality.mse",
        "label": "Reconstruction MSE",
        "filename": "bar_mse",
        "higher_is_better": False,
        "log_y": True,
    },
    {
        "column": "model_performance_preservation.ce_loss_score",
        "label": "CE-Loss Recovery",
        "filename": "bar_ce_loss_score",
        "higher_is_better": True,
    },
    {
        "column": "shrinkage.l2_ratio",
        "label": "L2 Norm Ratio (out/in)",
        "filename": "bar_l2_ratio",
        "higher_is_better": True,
    },
    {
        "column": "sparsity.l1",
        "label": "L1 Sparsity",
        "filename": "bar_l1",
        "higher_is_better": False,
        "log_y": True,
    },
]


def plot_bar(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
    higher_is_better: bool,
    log_y: bool = False,
) -> None:
    if column not in df.columns:
        print(f"[skip] column missing for bar plot: {column}")
        return
    data = df.dropna(subset=[column]).copy()
    if data.empty:
        print(f"[skip] no rows for {column}")
        return

    x_order = _present_order(data["algorithm"], GROUP_ORDER)
    hue_order = _present_order(data["base_model"], BASE_ORDER)
    palette = _ordered_hue_palette(hue_order)

    # Single metric bar figure sized for two-column placement.
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    sns.barplot(
        data=data,
        x="algorithm",
        y=column,
        hue="base_model",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        edgecolor="black",
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("Alignment Algorithm")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=14)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    arrow = "higher is better" if higher_is_better else "lower is better"
    ax.set_title(f"{ylabel} ({arrow})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Keep some headroom above the tallest bar so value labels / legend above
    # the axes never clip the data, and never overlap bars (old ``loc="best"``
    # landed on top of the SmolLM3-3B bars in several panels).
    y_values = data[column].to_numpy()
    if log_y:
        positive = y_values[y_values > 0]
        if positive.size:
            ax.set_yscale("log")
            lo = float(np.nanmin(positive)) / 3.0
            hi = float(np.nanmax(y_values)) * 3.0
            ax.set_ylim(lo, hi)
    elif higher_is_better and np.nanmax(y_values) >= 0:
        ax.set_ylim(0, max(1.05, np.nanmax(y_values) * 1.10))
    else:
        ymin = min(0.0, float(np.nanmin(y_values)))
        ax.set_ylim(ymin, float(np.nanmax(y_values)) * 1.15)

    # Remove the axes-attached legend and put a shared one above the figure
    # instead so it never covers any bars.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=palette[b], edgecolor="black", linewidth=1.0
        )
        for b in hue_order
    ]
    leg = fig.legend(
        handles,
        hue_order,
        title="Base model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(hue_order), 3),
        frameon=False,
        columnspacing=2.2,
        handletextpad=1.0,
        labelspacing=0.6,
    )
    _apply_legend_fontsizes(
        leg,
        label_fontsize=LEGEND_LABEL_BAR,
        title_fontsize=LEGEND_TITLE_BAR,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


TRAINING_METRICS: list[dict] = [
    {
        "column": "metrics/explained_variance",
        "label": "Explained Variance",
        "filename": "curve_explained_variance",
        "smooth": 31,
    },
    {
        "column": "metrics/explained_variance_legacy",
        "label": "Explained Variance (legacy)",
        "filename": "curve_explained_variance_legacy",
        "smooth": 31,
    },
    {
        "column": "losses/mse_loss",
        "label": "MSE Loss",
        "filename": "curve_mse_loss",
        "smooth": 25,
        "yscale": "log",
    },
    {
        "column": "losses/overall_loss",
        "label": "Overall Loss",
        "filename": "curve_overall_loss",
        "smooth": 25,
        "yscale": "log",
    },
    {
        "column": "sparsity/dead_features",
        "label": "Dead Features",
        "filename": "curve_dead_features",
        "smooth": 11,
    },
]


def _ema_time_weighted(y: np.ndarray, steps: np.ndarray, tau_steps: int) -> np.ndarray:
    """Time-weighted EMA using step deltas as elapsed time.

    ``tau_steps`` controls the EMA time constant measured in training steps:
    larger values -> smoother traces.
    """
    if tau_steps <= 1 or y.size <= 2:
        return y
    tau = float(max(tau_steps, 1))
    out = np.empty_like(y, dtype=np.float64)
    out[0] = float(y[0])
    for i in range(1, len(y)):
        dt = float(max(steps[i] - steps[i - 1], 1))
        alpha = 1.0 - np.exp(-dt / tau)
        out[i] = alpha * float(y[i]) + (1.0 - alpha) * out[i - 1]
    return out


def plot_training_curves(
    runs: list[SaeRun],
    curves: dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    output_path: Path,
    smooth: int = 1,
    yscale: str | None = None,
) -> None:
    groups = list(CURVE_PANEL_ORDER)
    by_group: dict[str, list[SaeRun]] = {g: [] for g in groups}
    for r in runs:
        if r.algorithm in by_group:
            by_group[r.algorithm].append(r)

    if not runs:
        print(f"[skip] no runs for curve plot: {column}")
        return

    hue_order = _present_order({r.base_model for r in runs}, BASE_ORDER)
    palette = _ordered_hue_palette(hue_order)

    # Regular 2x4 layout: 7 data panels + legend panel in the second-row 4th slot.
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(CURVE_PANEL_W_IN * ncols, CURVE_PANEL_H_IN * nrows),
        sharey=True,
        squeeze=False,
    )
    axes_data = list(axes.flat[: len(groups)])
    legend_ax = axes[1, 3]
    legend_ax.set_title("")
    legend_ax.set_xlabel("")
    legend_ax.set_ylabel("")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.grid(False)
    legend_ax.set_frame_on(False)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    any_data = False
    # Deterministic within-panel draw order so overlapping curves render
    # consistently (Llama -> Qwen -> SmolLM3) and every base model is visible
    # thanks to the per-base linestyle.
    base_draw_order = {
        b: int(BASE_CURVE_STYLE.get(b, {}).get("zorder", 99))
        for b in hue_order
    }

    for ax, group in zip(axes_data, groups):
        runs_in_group = sorted(
            by_group[group], key=lambda r: base_draw_order.get(r.base_model, 99)
        )
        present_bases: set[str] = set()
        for run in runs_in_group:
            df = curves.get(str(run.directory))
            if df is None or column not in df.columns:
                continue
            s = df[["step", column]].dropna()
            if s.empty:
                continue
            steps = s["step"].to_numpy()
            vals = s[column].to_numpy()
            vals_s = _ema_time_weighted(vals, steps, smooth)
            steps_s = steps
            style = BASE_CURVE_STYLE.get(
                run.base_model, {"alpha": 0.9, "linewidth": 2.8, "zorder": 2}
            )
            ax.plot(
                steps_s,
                vals_s,
                color=palette[run.base_model],
                linestyle="-",
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
                zorder=float(style["zorder"]),
                label=run.base_model,
            )
            present_bases.add(run.base_model)
            any_data = True
        ax.set_title(group)
        ax.set_xlabel("Training Step")
        ax.tick_params(axis="both", labelsize=14)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        if yscale:
            ax.set_yscale(yscale)
            ax.yaxis.set_major_locator(mticker.LogLocator(numticks=4))
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.grid(alpha=0.3)
        if not runs_in_group:
            ax.text(
                0.5,
                0.5,
                "No runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.5,
                color="0.45",
            )
        missing = [b for b in hue_order if b not in present_bases]
        if missing and runs_in_group:
            ax.text(
                0.98,
                0.03,
                "Missing: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="0.35",
                alpha=0.9,
            )

    for idx, ax in enumerate(axes_data):
        if idx in {0, 4}:
            ax.set_ylabel(ylabel)

    if any_data:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=palette[b],
                linestyle="-",
                alpha=float(BASE_CURVE_STYLE.get(b, {}).get("alpha", 0.9)),
                lw=float(BASE_CURVE_STYLE.get(b, {}).get("linewidth", 2.8)),
                label=CURVE_LEGEND_LABELS.get(b, b),
            )
            for b in hue_order
        ]
        # Treat the legend as the fourth panel in the second row.
        leg = legend_ax.legend(
            handles=handles,
            title="Base Model",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=1,
            frameon=True,
            fancybox=True,
            framealpha=0.98,
            edgecolor="0.25",
            columnspacing=1.4,
            handletextpad=0.8,
            labelspacing=0.8,
            borderaxespad=0.2,
            handlelength=2.8,
        )
        _apply_legend_fontsizes(
            leg,
            label_fontsize=LEGEND_LABEL_CURVE,
            title_fontsize=LEGEND_TITLE_CURVE,
        )
        fig.tight_layout()
    else:
        fig.tight_layout()

    fig.savefig(output_path.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/saes"),
        help="Directory containing <model>/<layer_*_best> SAE outputs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sae_plots"),
        help="Directory to write PDF figures and the summary CSV to.",
    )
    p.add_argument(
        "--layer-tag",
        choices=["best", "mid", "both"],
        default="best",
        help="Which layer tag(s) to plot (default: best).",
    )
    return p.parse_args()


def run_for_tag(runs: list[SaeRun], output_dir: Path, tag_label: str) -> None:
    if not runs:
        print(f"[warn] no runs to plot for tag '{tag_label}'")
        return

    print(f"\n=== Plotting {len(runs)} runs for layer tag '{tag_label}' ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = build_final_eval_table(runs)
    if not final_df.empty:
        csv_path = output_dir / f"final_eval_metrics_{tag_label}.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}")

    for spec in BAR_METRICS:
        out = output_dir / f"{spec['filename']}_{tag_label}"
        plot_bar(
            final_df,
            column=spec["column"],
            ylabel=spec["label"],
            output_path=out,
            higher_is_better=spec["higher_is_better"],
            log_y=spec.get("log_y", False),
        )
        print(f"  wrote {out.with_suffix('.pdf')}")

    curves: dict[str, pd.DataFrame] = {}
    for run in runs:
        df, _ = load_metrics(run)
        curves[str(run.directory)] = df

    for spec in TRAINING_METRICS:
        out = output_dir / f"{spec['filename']}_{tag_label}"
        plot_training_curves(
            runs,
            curves,
            column=spec["column"],
            ylabel=spec["label"],
            output_path=out,
            smooth=spec.get("smooth", 1),
            yscale=spec.get("yscale"),
        )
        print(f"  wrote {out.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    configure_plot_style()

    if args.layer_tag == "both":
        tags = ["best", "mid"]
    else:
        tags = [args.layer_tag]

    for tag in tags:
        runs = discover_runs(args.input_dir, [tag])
        run_for_tag(runs, args.output_dir, tag)


if __name__ == "__main__":
    main()
