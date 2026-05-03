#!/usr/bin/env python3
"""Copy ``layerwise_probe_metrics.pdf`` from each model folder into one subfolder."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_SKIP_NAMES = frozenset(
    {
        "best-layer-pca-3d",
        "layerwise-probe-metrics",
        "_combined-best-layer-roc",
    }
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/linear-probe-figures"),
        help="Root folder that contains one subfolder per model run.",
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default="layerwise-probe-metrics",
        help="Subfolder under --figures-dir to create with copies only.",
    )
    args = p.parse_args()

    root: Path = args.figures_dir
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out_root = root / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in _SKIP_NAMES or d.name.startswith("_"):
            continue
        src = d / "layerwise_probe_metrics.pdf"
        if not src.is_file():
            continue
        dest_dir = out_root / d.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "layerwise_probe_metrics.pdf")
        print(f"Copied {src} -> {dest_dir / 'layerwise_probe_metrics.pdf'}")
        n += 1

    print(f"Done. {n} PDF(s) under {out_root}")


if __name__ == "__main__":
    main()
