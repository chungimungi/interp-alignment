"""
Compare primary_class distributions from two crosscoder result directories.
Example:
  python -m crosscoder.plot_class_comparison --dir-a path/to/run_a --dir-b path/to/run_b
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import get_features_dir
from .visualize import CLASS_DISTRIBUTION_ORDER, ICLR_DPI, ICLR_LABEL_FONT, ICLR_TICK_SIZE, _apply_iclr_style


def load_class_counts(results_dir: Path) -> dict:
    p = get_features_dir(results_dir) / "feature_classification.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return df["primary_class"].value_counts().to_dict()


def plot_comparison(counts_a: dict, counts_b: dict, label_a: str, label_b: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(CLASS_DISTRIBUTION_ORDER))
    w = 0.35
    vals_a = [counts_a.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER]
    vals_b = [counts_b.get(c, 0) for c in CLASS_DISTRIBUTION_ORDER]
    ax.bar([i - w / 2 for i in x], vals_a, width=w, label=label_a, color="#C0392B", edgecolor="black", alpha=0.85)
    ax.bar([i + w / 2 for i in x], vals_b, width=w, label=label_b, color="#2980B9", edgecolor="black", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_DISTRIBUTION_ORDER, rotation=45, ha="right", fontsize=ICLR_TICK_SIZE)
    ax.set_xlabel("Feature class", fontdict=ICLR_LABEL_FONT)
    ax.set_ylabel("Count", fontdict=ICLR_LABEL_FONT)
    _apply_iclr_style(ax)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=ICLR_DPI, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare feature class counts between two result dirs")
    parser.add_argument("--dir-a", type=Path, required=True)
    parser.add_argument("--dir-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="Run A")
    parser.add_argument("--label-b", type=str, default="Run B")
    parser.add_argument("--output", type=Path, default=Path("class_comparison.png"))
    args = parser.parse_args()

    ca = load_class_counts(args.dir_a)
    cb = load_class_counts(args.dir_b)
    if not ca or not cb:
        raise SystemExit("Missing feature_classification.csv in one or both directories.")
    plot_comparison(ca, cb, args.label_a, args.label_b, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
