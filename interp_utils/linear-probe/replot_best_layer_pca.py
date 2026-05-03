#!/usr/bin/env python3
"""Re-render best-layer PCA PDFs into a dedicated subfolder under linear-probe-figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from this directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt

from pca_plot import add_best_tsne_embedding, add_umap_z_fallback, plot_best_layer_pca_figure


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--linear-probes-dir",
        type=Path,
        default=Path("results/linear-probes"),
        help="Directory containing one subfolder per run with best_layer_pca.json.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/linear-probe-figures"),
        help="Root folder for figure outputs.",
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default="best-layer-pca-3d",
        help="Subfolder under --figures-dir for replotted PCA PDFs.",
    )
    p.add_argument(
        "--embedding",
        choices=["tsne-best", "umap", "raw"],
        default="tsne-best",
        help="Embedding used for 3D visualization before plotting.",
    )
    p.add_argument(
        "--run-pattern",
        type=str,
        default=None,
        help="Optional substring filter for run directory names.",
    )
    p.add_argument(
        "--tsne-seed",
        type=int,
        default=42,
        help="Random seed base for t-SNE best-of search.",
    )
    p.add_argument(
        "--no-umap-fallback",
        action="store_true",
        help="When umap_z is missing from JSON, do not fit UMAP on PC1/PC2 (use 2D plot).",
    )
    p.add_argument(
        "--umap-seed",
        type=int,
        default=42,
        help="Random seed for UMAP when using PC1/PC2 fallback.",
    )
    args = p.parse_args()

    _configure_plot_style()

    probes_root: Path = args.linear_probes_dir
    out_root = args.figures_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    if not probes_root.is_dir():
        raise SystemExit(f"Not a directory: {probes_root}")

    n_ok = 0
    for run_dir in sorted(probes_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.run_pattern and args.run_pattern not in run_dir.name:
            continue
        pca_path = run_dir / "best_layer_pca.json"
        if not pca_path.is_file():
            continue
        with pca_path.open() as f:
            payload = json.load(f)
        if args.embedding == "tsne-best":
            if not args.no_umap_fallback:
                payload = add_umap_z_fallback(payload, seed=args.umap_seed)
            payload = add_best_tsne_embedding(payload, seed=args.tsne_seed)
        elif args.embedding == "umap" and not args.no_umap_fallback:
            payload = add_umap_z_fallback(payload, seed=args.umap_seed)
        sub_out = out_root / run_dir.name
        sub_out.mkdir(parents=True, exist_ok=True)
        plot_best_layer_pca_figure(payload, sub_out / "best_layer_pca.pdf")
        extra = ""
        if "separability_auc" in payload:
            extra = (
                f" (AUC={payload['separability_auc']:.3f}, "
                f"perp={payload.get('tsne_perplexity')}, seed={payload.get('tsne_seed')}, "
                f"label_w={payload.get('tsne_label_weight')})"
            )
        print(f"Wrote {sub_out / 'best_layer_pca.pdf'}{extra}")
        n_ok += 1

    print(f"Done. {n_ok} PDF(s) under {out_root}")


if __name__ == "__main__":
    main()
