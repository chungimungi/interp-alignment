"""
Compute metrics (FSR, SSS, CSS) for LLM crosscoder results.
Directory layout: {base_slug}__{aligned_run_id}__L{layer}__{position}
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .classify import load_classification_results
from .metrics import compute_counterfactual_sensitivity_shift, compute_feature_sharing_ratio, compute_semantic_stability_score
from .utils import get_features_dir, get_metrics_dir, load_json
from .visualize import compute_adaptive_rho_thresholds

def _position_cli_from_slug(position_slug: str) -> str:
    if position_slug == "lastprompt":
        return config.POSITION_LAST_PROMPT
    if position_slug == "meanprompt":
        return config.POSITION_MEAN_PROMPT
    return position_slug


def parse_results_dirname(name: str) -> Optional[Dict]:
    m = re.search(r"__L(\d+)__(.+)$", name)
    if not m:
        return None
    layer = int(m.group(1))
    position_slug = m.group(2)
    prefix = name[: m.start()]
    if prefix.endswith("__"):
        prefix = prefix[:-2]
    idx = prefix.rfind("__")
    if idx < 0:
        base_slug = prefix
        aligned_run_id = ""
    else:
        base_slug = prefix[:idx]
        aligned_run_id = prefix[idx + 2 :]
    return {
        "base_slug": base_slug,
        "aligned_run_id": aligned_run_id,
        "layer": layer,
        "position_slug": position_slug,
    }


def _config_if_ready(run_dir: Path) -> Optional[Dict]:
    parsed = parse_results_dirname(run_dir.name)
    if parsed is None:
        return None
    if not (run_dir / "features" / "feature_classification.csv").exists():
        return None
    return {**parsed, "dirname": run_dir.name, "path": run_dir.resolve()}


def discover_configs(results_root: Path) -> List[Dict]:
    """Find result runs under ``results_root``.

    ``--results-root`` may be either:
    - The parent directory containing multiple ``*__L*__*`` run folders, or
    - A single run directory (e.g. ``.../SmolLM3-3B__run__L15__lastprompt``).
    """
    results_root = results_root.resolve()
    if not results_root.is_dir():
        return []

    direct = _config_if_ready(results_root)
    if direct is not None:
        return [direct]

    configs = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue
        c = _config_if_ready(p)
        if c is not None:
            configs.append(c)
    return configs


def explain_no_configs(results_root: Path) -> None:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        print(f"Not a directory: {results_root}")
        return

    parsed = parse_results_dirname(results_root.name)
    if parsed is None:
        print(
            "No runs with features/feature_classification.csv found under this path. "
            "Use --results-root pointing at crosscoder/results (parent of run dirs) "
            "or at a single run directory named like BASE__runid__L15__lastprompt."
        )
        return

    feat_csv = results_root / "features" / "feature_classification.csv"
    if not feat_csv.exists():
        pos = _position_cli_from_slug(parsed["position_slug"])
        print(
            f"This looks like a single run directory ({results_root.name}) but "
            f"{feat_csv} is missing. Run analysis first (needs a trained checkpoint), e.g.:\n"
            f"  python -m crosscoder.main --base-model <BASE_HF_ID> --aligned-model <ALIGNED_HF_OR_PATH> "
            f"--aligned-run-id {parsed['aligned_run_id']} --layer {parsed['layer']} --position {pos} "
            f"--stage analyze --output-dir {results_root}"
        )
        return

    print("No configs to process (unexpected: feature file exists but was not picked up).")


def compute_metrics_for_config(c: Dict) -> Optional[Dict]:
    results_dir = c["path"]
    features_dir = get_features_dir(results_dir)
    classification_path = features_dir / "feature_classification.csv"
    merged_path = features_dir / "merged_classification.csv"

    if not classification_path.exists():
        return None

    classification_df = load_classification_results(str(classification_path))
    fsr = compute_feature_sharing_ratio(classification_df)
    sss = compute_semantic_stability_score(classification_df)

    css = {}
    if merged_path.exists():
        merged_df = pd.read_csv(merged_path)
        css = compute_counterfactual_sensitivity_shift(merged_df)

    row = {
        "dirname": c["dirname"],
        "base_slug": c["base_slug"],
        "aligned_run_id": c["aligned_run_id"],
        "layer": c["layer"],
        "position_slug": c["position_slug"],
        "fsr": fsr,
        "sss": sss,
    }
    for k, v in css.items():
        row[f"css_{k}"] = v
    return row


def compute_classification_thresholds(configs: List[Dict]) -> List[Dict]:
    rows = []
    for c in configs:
        results_dir = c["path"]
        features_dir = get_features_dir(results_dir)
        classification_path = features_dir / "feature_classification.csv"
        if not classification_path.exists():
            continue

        classification_df = load_classification_results(str(classification_path))
        thresh = compute_adaptive_rho_thresholds(classification_df)

        rows.append(
            {
                "dirname": c["dirname"],
                "base_slug": c["base_slug"],
                "aligned_run_id": c["aligned_run_id"],
                "layer": c["layer"],
                "position_slug": c["position_slug"],
                "rho_base_only": thresh["rho_base_only"],
                "rho_aligned_only": thresh["rho_aligned_only"],
                "rho_shared_low": thresh["rho_shared_low"],
                "rho_shared_high": thresh["rho_shared_high"],
                "theta_aligned": config.THETA_ALIGNED,
                "theta_redirected": config.THETA_REDIRECTED,
            }
        )
    return rows


def compute_shared_geometry_rows(configs: List[Dict]) -> List[Dict]:
    from .metrics import SHARED_CLASSES

    rows = []
    for c in configs:
        results_dir = c["path"]
        metrics_dir = get_metrics_dir(results_dir)
        path = metrics_dir / "shared_geometry_metrics.json"
        if not path.exists():
            continue

        data = load_json(path)
        row = {
            "dirname": c["dirname"],
            "base_slug": c["base_slug"],
            "aligned_run_id": c["aligned_run_id"],
            "layer": c["layer"],
            "position_slug": c["position_slug"],
        }
        for cls in list(SHARED_CLASSES) + ["all_shared"]:
            sub = data.get(cls, {})
            if isinstance(sub, dict) and sub.get("n", 0) > 0:
                row[f"{cls}_angle_deg_mean"] = sub.get("angle_deg_mean")
                row[f"{cls}_norm_ratio_raw_mean"] = sub.get("norm_ratio_raw_mean")
                if cls == "all_shared":
                    lm = sub.get("linear_map")
                    if isinstance(lm, dict):
                        row["all_shared_sv_mean"] = lm.get("sv_mean")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute FSR, SSS, CSS for LLM crosscoder result directories."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.CROSSCODER_RESULTS_DIR / "metrics",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=config.CROSSCODER_RESULTS_DIR,
        help=(
            "Parent of run directories (each named BASE__runid__Llayer__position), "
            "or a single run directory path."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = discover_configs(args.results_root)
    if not configs:
        explain_no_configs(args.results_root)
        return

    print(f"Processing {len(configs)} configs...")

    fsr_rows = []
    for c in configs:
        row = compute_metrics_for_config(c)
        if row is not None:
            fsr_rows.append(row)
            print(f"  {c['dirname']}")

    if fsr_rows:
        pd.DataFrame(fsr_rows).to_csv(output_dir / "fsr_sss_css.csv", index=False)
        print(f"Saved: {output_dir / 'fsr_sss_css.csv'}")

    threshold_rows = compute_classification_thresholds(configs)
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(output_dir / "classification_thresholds.csv", index=False)
        print(f"Saved: {output_dir / 'classification_thresholds.csv'}")

    geometry_rows = compute_shared_geometry_rows(configs)
    if geometry_rows:
        pd.DataFrame(geometry_rows).to_csv(
            output_dir / "shared_geometry_metrics_summary.csv", index=False
        )
        print(f"Saved: {output_dir / 'shared_geometry_metrics_summary.csv'}")

    print("Done.")


if __name__ == "__main__":
    main()
