from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ALGO_ORDER = ["Baseline", "DPO", "GRPO", "KTO", "ORPO", "PPO", "SimPO"]
BASE_ORDER = ["Llama-3.2-3B-Instruct", "Qwen3-4B-Instruct-2507", "SmolLM3-3B"]


@dataclass
class ProbeRun:
    model_dir: str
    base: str
    variant: str
    best_layer: int
    y_true: np.ndarray
    y_prob: np.ndarray
    auroc: float


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 22,
            "axes.titlesize": 36,
            "axes.labelsize": 34,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 24,
            "axes.linewidth": 1.2,
            "lines.linewidth": 3.2,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def normalize_base(model_dir: str) -> str | None:
    low = model_dir.lower()
    if "llama-3.2-3b" in low:
        return "Llama-3.2-3B-Instruct"
    if "qwen3-4b" in low:
        return "Qwen3-4B-Instruct-2507"
    if "smollm3-3b" in low:
        return "SmolLM3-3B"
    return None


def _strip_prefix_suffix(model_dir: str) -> str:
    body = model_dir
    if body.startswith("MInAlA_"):
        body = body[len("MInAlA_") :]
    if body.endswith("-merged"):
        body = body[: -len("-merged")]
    return body


def infer_variant(model_dir: str, base: str) -> str:
    if not model_dir.startswith("MInAlA_"):
        return "Baseline"

    body = _strip_prefix_suffix(model_dir)
    if base == "Llama-3.2-3B-Instruct":
        body = body.replace("Llama-3.2-3B-", "", 1)
        body = body.replace("Instruct-", "", 1)
    elif base == "Qwen3-4B-Instruct-2507":
        body = body.replace("Qwen3-4B-", "", 1)
        body = body.replace("Instruct-2507-", "", 1)
    elif base == "SmolLM3-3B":
        body = body.replace("SmolLM3-3B-", "", 1)

    for algo in ["DPO", "GRPO", "KTO", "ORPO", "PPO", "SimPO"]:
        if body.upper() == algo:
            return algo
        if body.upper().endswith("-" + algo):
            return body.rsplit("-", 1)[0] + "-" + algo
    return body


def variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in ALGO_ORDER:
        return (ALGO_ORDER.index(variant), variant)
    algo = variant.split("-")[-1]
    if algo in ALGO_ORDER:
        return (ALGO_ORDER.index(algo), variant)
    return (len(ALGO_ORDER) + 10, variant)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_auroc_latex_document(runs: list[ProbeRun], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_layer_auroc_table.tex"
    ordered = sorted(runs, key=lambda r: (BASE_ORDER.index(r.base) if r.base in BASE_ORDER else 99, variant_sort_key(r.variant)))

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\begin{document}",
        r"\section*{Best-layer AUROC by model and alignment}",
        r"\begin{longtable}{llrr}",
        r"\toprule",
        r"Base model & Alignment & Best layer & AUROC \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Base model & Alignment & Best layer & AUROC \\",
        r"\midrule",
        r"\endhead",
    ]

    for run in ordered:
        lines.append(
            f"{_latex_escape(run.base)} & {_latex_escape(run.variant)} & {run.best_layer} & {run.auroc:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{document}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_layer_from_metrics(metrics: list[dict]) -> int:
    best = max(metrics, key=lambda m: float(m.get("auroc", float("-inf"))))
    return int(best["layer"])


def discover_runs(probes_dir: Path) -> list[ProbeRun]:
    out: list[ProbeRun] = []
    for model_dir in sorted(p for p in probes_dir.iterdir() if p.is_dir()):
        base = normalize_base(model_dir.name)
        if base is None:
            continue

        metrics_path = model_dir / "layer_metrics.json"
        preds_path = model_dir / "layer_predictions.json"
        probs_path = model_dir / "layer_probabilities.json"
        if not (metrics_path.exists() and preds_path.exists() and probs_path.exists()):
            continue

        metrics = load_json(metrics_path)
        layer_predictions = load_json(preds_path)
        layer_probabilities = load_json(probs_path)
        if not isinstance(metrics, list) or not isinstance(layer_predictions, dict) or not isinstance(layer_probabilities, dict):
            continue

        best_layer = best_layer_from_metrics(metrics)
        key = str(best_layer)
        if key not in layer_predictions or key not in layer_probabilities:
            continue

        y_true = np.asarray(layer_predictions[key]["y_test"], dtype=np.int64)
        y_prob = np.asarray(layer_probabilities[key], dtype=np.float64)
        if y_true.size == 0 or y_true.shape != y_prob.shape:
            continue

        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            continue

        out.append(
            ProbeRun(
                model_dir=model_dir.name,
                base=base,
                variant=infer_variant(model_dir.name, base),
                best_layer=best_layer,
                y_true=y_true,
                y_prob=y_prob,
                auroc=auroc,
            )
        )
    return out


def plot_base_group(base: str, runs: list[ProbeRun], out_dir: Path) -> None:
    if not runs:
        return

    ordered = sorted(runs, key=lambda r: variant_sort_key(r.variant))
    fig, ax = plt.subplots(figsize=(13.5, 11.5))

    colors = plt.cm.magma(np.linspace(0.15, 0.9, max(len(ordered), 3)))
    for i, run in enumerate(ordered):
        fpr, tpr, _ = roc_curve(run.y_true, run.y_prob)
        label = f"{run.variant} (best L{run.best_layer})"
        ax.plot(fpr, tpr, label=label, color=colors[i], linewidth=2.8)

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=2.0, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(base)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="lower right",
        frameon=True,
        ncol=1,
        title="Alignment (best layer)",
        fancybox=False,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base.replace('.', '').replace('-', '_').lower()}_best_layer_roc_alignments.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create combined best-layer ROC figures (one per base model) across alignment variants.",
    )
    parser.add_argument(
        "--probes-dir",
        type=Path,
        default=Path("results/linear-probes"),
        help="Directory containing linear probe JSON outputs per model.",
    )
    parser.add_argument(
        "--linear-probe-figures-dir",
        type=Path,
        default=Path("results/linear-probe-figures"),
        help="Base directory for linear-probe figures.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="_combined-best-layer-roc",
        help="Subdirectory under --linear-probe-figures-dir for combined ROC PDFs.",
    )
    args = parser.parse_args()

    configure_plot_style()
    runs = discover_runs(args.probes_dir)
    if not runs:
        print(f"no runs found under {args.probes_dir}")
        return

    grouped: dict[str, list[ProbeRun]] = {b: [] for b in BASE_ORDER}
    for run in runs:
        grouped.setdefault(run.base, []).append(run)

    out_root = args.linear_probe_figures_dir / args.output_subdir
    for base in BASE_ORDER:
        plot_base_group(base, grouped.get(base, []), out_root)
    write_auroc_latex_document(runs, out_root)


if __name__ == "__main__":
    main()
