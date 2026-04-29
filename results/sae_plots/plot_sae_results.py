"""Plot Sparse Autoencoder (SAE) training results.

Reads ``output/sae/<model_dir>/<layer_dir>/metrics.jsonl`` files produced by
``run_all_saes.py`` / ``sae.py`` and generates NeurIPS-quality PDF figures:

* Grouped bar plots of final-eval metrics (explained variance, reconstruction
  cosine similarity, MSE, CE-loss preservation, sparsity) with the base
  model on the x-axis and alignment algorithms (DPO / GRPO / KTO / ORPO / SimPO
  plus unaligned baseline) as the color ``hue`` (magma palette).
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
import numpy as np
import pandas as pd
import seaborn as sns

LEGEND_LABEL_BAR = 26.0
LEGEND_TITLE_BAR = 29.0
BAR_TITLE_SIZE = 50.0
BAR_AXIS_LABEL = 48.0
BAR_TICK = 44.0
BAR_LEGEND_LABEL = 38.0
BAR_LEGEND_TITLE = 40.0
BAR_FIGSIZE = (25.0, 13.5)
LEGEND_LABEL_CURVE = 23.0
LEGEND_TITLE_CURVE = 25.0

ALGORITHMS = ["DPO", "GRPO", "KTO", "ORPO", "SimPO"]
BASELINE_LABEL = "Baseline"
GROUP_ORDER = [BASELINE_LABEL, "DPO", "GRPO", "KTO", "ORPO", "SimPO"]
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
            "font.size": 24,
            "axes.titlesize": 29,
            "axes.labelsize": 28,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": LEGEND_LABEL_BAR,
            "legend.title_fontsize": LEGEND_TITLE_BAR,
            "axes.linewidth": 1.2,
            "lines.linewidth": 3.0,
            "lines.markersize": 8,
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
    for algo in ALGORITHMS:
        suffix = "-" + algo
        if body_upper.endswith(suffix):
            base_str = body[: -len(suffix)]
            return normalize_base(base_str), algo

    lower_algo = {a.lower(): a for a in ALGORITHMS}
    for key, algo in lower_algo.items():
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
    "Qwen3-4B-Instruct-2507": {"alpha": 0.55, "linewidth": 2.4, "zorder": 1},
    "SmolLM3-3B": {"alpha": 0.82, "linewidth": 2.9, "zorder": 2},
    "Llama-3.2-3B-Instruct": {"alpha": 0.95, "linewidth": 3.2, "zorder": 3},
}


def _adjust_lightness(color: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    """Scale color lightness in HLS space."""
    h, l, s = colorsys.rgb_to_hls(*color)
    l = min(1.0, max(0.0, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def _ordered_hue_palette(hue_levels: list[str]) -> dict[str, tuple]:
    # Perceptually-uniform `magma`: skip the lightest slice so end colors stay visible on white.
    n = max(len(hue_levels), 3)
    palette = sns.color_palette("magma", n_colors=n + 1)[:n]
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
# Optional ``legend``: ``loc`` + ``bbox_to_anchor`` (axes coords, 0-1) per panel.
BAR_METRICS: list[dict] = [
    {
        "column": "reconstruction_quality.explained_variance",
        "label": "Explained Variance",
        "filename": "bar_explained_variance",
        "higher_is_better": True,
        "legend": {"loc": "upper center", "bbox_to_anchor": (0.5, 0.995), "ncol": 3},
    },
    {
        "column": "reconstruction_quality.cossim",
        "label": "Reconstruction Cosine Similarity",
        "filename": "bar_cossim",
        "higher_is_better": True,
        "legend": {"loc": "upper center", "bbox_to_anchor": (0.5, 0.995), "ncol": 3},
    },
    {
        "column": "reconstruction_quality.mse",
        "label": "Reconstruction MSE",
        "filename": "bar_mse",
        "higher_is_better": False,
        "log_y": True,
        "log_ylim": {"lo_div": 2.0, "hi_mult": 18.0},
        "legend": {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 0.995),
            "ncol": 3,
        },
    },
    {
        "column": "model_performance_preservation.ce_loss_score",
        "label": "CE-Loss Recovery",
        "filename": "bar_ce_loss_score",
        "higher_is_better": True,
        "legend": {"loc": "upper center", "bbox_to_anchor": (0.5, 0.995), "ncol": 3},
    },
    {
        "column": "shrinkage.l2_ratio",
        "label": "L2 Norm Ratio (out/in)",
        "filename": "bar_l2_ratio",
        "higher_is_better": True,
        "legend": {"loc": "upper center", "bbox_to_anchor": (0.5, 0.995), "ncol": 3},
    },
    {
        "column": "sparsity.l1",
        "label": "L1 Sparsity",
        "filename": "bar_l1",
        "higher_is_better": False,
        "log_y": True,
        "log_ylim": {"lo_div": 2.0, "hi_mult": 14.0},
        "legend": {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 0.995),
            "ncol": 3,
        },
    },
]


def plot_bar(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
    higher_is_better: bool,
    log_y: bool = False,
    *,
    legend: dict | None = None,
    log_ylim: dict | None = None,
) -> None:
    if column not in df.columns:
        print(f"[skip] column missing for bar plot: {column}")
        return
    data = df.dropna(subset=[column]).copy()
    if data.empty:
        print(f"[skip] no rows for {column}")
        return

    x_order = _present_order(data["base_model"], BASE_ORDER)
    hue_order = _present_order(data["algorithm"], GROUP_ORDER)
    palette = _ordered_hue_palette(hue_order)

    bar_rc = {
        "font.size": BAR_AXIS_LABEL,
        "axes.titlesize": BAR_TITLE_SIZE,
        "axes.labelsize": BAR_AXIS_LABEL,
        "xtick.labelsize": BAR_TICK,
        "ytick.labelsize": BAR_TICK,
    }
    with plt.rc_context(bar_rc):
        fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
        sns.barplot(
            data=data,
            x="base_model",
            y=column,
            hue="algorithm",
            order=x_order,
            hue_order=hue_order,
            palette=palette,
            edgecolor="black",
            linewidth=1.0,
            ax=ax,
        )
        n_cat = len(x_order)
        xtick_labels = [CURVE_LEGEND_LABELS.get(m, m) for m in x_order]
        ax.set_xticks(np.arange(n_cat, dtype=float))
        ax.set_xticklabels(
            xtick_labels, rotation=0, ha="center", rotation_mode="default"
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=BAR_AXIS_LABEL, labelpad=12)
        arrow = "higher is better" if higher_is_better else "lower is better"
        ax.set_title(
            f"{ylabel} ({arrow})",
            fontsize=BAR_TITLE_SIZE,
            pad=14,
        )
        ax.tick_params(
            axis="x",
            labelsize=BAR_TICK,
            length=10,
            width=1.0,
        )
        ax.tick_params(
            axis="y",
            labelsize=BAR_TICK,
            length=10,
            width=1.0,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        y_values = data[column].to_numpy(dtype=float)
        if log_y:
            positive = y_values[y_values > 0]
            if positive.size:
                ax.set_yscale("log")
                lo_div = 3.0
                hi_mult = 6.0
                if log_ylim:
                    lo_div = float(log_ylim.get("lo_div", lo_div))
                    hi_mult = float(log_ylim.get("hi_mult", hi_mult))
                lo = float(np.nanmin(positive)) / lo_div
                hi = float(np.nanmax(positive)) * hi_mult
                ax.set_ylim(lo, hi)
        elif higher_is_better and np.nanmax(y_values) >= 0:
            ymax = max(1.05, np.nanmax(y_values) * 1.10)
            head = 1.40 if n_cat <= 3 else 1.30
            ax.set_ylim(0, ymax * head)
        else:
            ymin = min(0.0, float(np.nanmin(y_values)))
            ymax = float(np.nanmax(y_values)) * 1.15
            head = 1.35 if n_cat <= 3 else 1.28
            ax.set_ylim(ymin, ymax * head)

        if ax.get_legend() is not None:
            ax.get_legend().remove()
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=palette[b], edgecolor="black", linewidth=0.8
            )
            for b in hue_order
        ]
        n_leg = len(hue_order)
        if legend is not None and legend.get("ncol") is not None:
            ncol_leg = int(legend["ncol"])
        else:
            ncol_leg = 2 if n_leg > 3 else n_leg
        if legend and "loc" in legend and "bbox_to_anchor" in legend:
            loc = str(legend["loc"])
            bbo = (float(legend["bbox_to_anchor"][0]), float(legend["bbox_to_anchor"][1]))
        elif log_y:
            loc = "upper center"
            bbo = (0.5, 0.995)
        else:
            loc = "upper center"
            bbo = (0.5, 0.995)
        leg = ax.legend(
            handles,
            hue_order,
            title="Alignment algorithm",
            loc=loc,
            bbox_to_anchor=bbo,
            borderaxespad=0.4,
            frameon=True,
            fancybox=False,
            facecolor="white",
            edgecolor="0.55",
            framealpha=0.95,
            ncol=ncol_leg,
            columnspacing=0.9,
            handletextpad=0.35,
            labelspacing=0.35,
        )
        _apply_legend_fontsizes(
            leg,
            label_fontsize=BAR_LEGEND_LABEL,
            title_fontsize=BAR_LEGEND_TITLE,
        )

        fig.tight_layout(pad=0.55)
        fig.savefig(output_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
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
    algos_present = sorted({r.algorithm for r in runs if r.algorithm != BASELINE_LABEL})
    groups = [BASELINE_LABEL] + [a for a in GROUP_ORDER if a in algos_present]
    if not groups:
        print(f"[skip] no groups for curve plot: {column}")
        return

    by_group: dict[str, list[SaeRun]] = {g: [] for g in groups}
    for r in runs:
        if r.algorithm in by_group:
            by_group[r.algorithm].append(r)

    active_groups = [g for g in groups if by_group[g]]
    if not active_groups:
        print(f"[skip] no runs for curve plot: {column}")
        return

    hue_order = _present_order({r.base_model for r in runs}, BASE_ORDER)
    palette = _ordered_hue_palette(hue_order)

    n = len(active_groups)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 5.8 * nrows),
        sharey=True,
        squeeze=False,
    )

    any_data = False
    # Deterministic within-panel draw order so overlapping curves render
    # consistently (Llama -> Qwen -> SmolLM3) and every base model is visible
    # thanks to the per-base linestyle.
    base_draw_order = {
        b: int(BASE_CURVE_STYLE.get(b, {}).get("zorder", 99))
        for b in hue_order
    }

    for ax, group in zip(axes.flat, active_groups):
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
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        missing = [b for b in hue_order if b not in present_bases]
        if missing:
            ax.text(
                0.98,
                0.03,
                "Missing: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                color="0.35",
                alpha=0.9,
            )

    # Hide unused axes.
    unused = list(axes.flat[len(active_groups):])
    for ax in unused:
        ax.set_visible(False)

    for ax in axes[:, 0]:
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
        leg = fig.legend(
            handles=handles,
            title="Base Model",
            loc="upper center",
            ncol=min(len(hue_order), 3),
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.6,
            labelspacing=0.4,
            borderaxespad=0.2,
            handlelength=2.0,
        )
        _apply_legend_fontsizes(
            leg,
            label_fontsize=LEGEND_LABEL_CURVE,
            title_fontsize=LEGEND_TITLE_CURVE,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
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
    p.add_argument(
        "--final-csv",
        type=Path,
        default=None,
        help="Optional final-eval CSV for bar plots only (skips SAE run discovery).",
    )
    return p.parse_args()


def run_bars_from_csv(csv_path: Path, output_dir: Path, tag_label: str = "best") -> None:
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df = pd.read_csv(csv_path)
    if final_df.empty:
        print(f"[warn] CSV is empty: {csv_path}")
        return
    print(f"\n=== Plotting bar charts from CSV: {csv_path} ===")
    for spec in BAR_METRICS:
        out = output_dir / f"{spec['filename']}_{tag_label}"
        plot_bar(
            final_df,
            column=spec["column"],
            ylabel=spec["label"],
            output_path=out,
            higher_is_better=spec["higher_is_better"],
            log_y=spec.get("log_y", False),
            legend=spec.get("legend"),
            log_ylim=spec.get("log_ylim"),
        )
        print(f"  wrote {out.with_suffix('.pdf')}")


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
            legend=spec.get("legend"),
            log_ylim=spec.get("log_ylim"),
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

    if args.final_csv is not None:
        tag = args.layer_tag if args.layer_tag != "both" else "best"
        run_bars_from_csv(args.final_csv, args.output_dir, tag)
        return

    if args.layer_tag == "both":
        tags = ["best", "mid"]
    else:
        tags = [args.layer_tag]

    for tag in tags:
        runs = discover_runs(args.input_dir, [tag])
        run_for_tag(runs, args.output_dir, tag)


if __name__ == "__main__":
    main()
