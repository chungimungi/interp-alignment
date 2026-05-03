from typing import Dict, List

import numpy as np
import pandas as pd


def compute_counterfactual_sensitivity(feature_activations: Dict) -> pd.DataFrame:
    """
    Without image counterfactual pairs, use per-feature mean |z| on base vs aligned arms
    (activation usage) and shift = aligned - base.
    """
    z_base = feature_activations["z_base"]
    z_aligned = feature_activations["z_aligned"]

    cf_base = z_base.abs().mean(dim=0)
    cf_aligned = z_aligned.abs().mean(dim=0)
    cf_shift = cf_aligned - cf_base

    num_features = z_base.shape[1]
    records = []
    for i in range(num_features):
        records.append(
            {
                "feature_id": i,
                "cf_base": cf_base[i].item(),
                "cf_aligned": cf_aligned[i].item(),
                "cf_shift": cf_shift[i].item(),
            }
        )

    return pd.DataFrame(records)


def classify_cf_level(cf_scores_df: pd.DataFrame, threshold_type: str = "median") -> pd.DataFrame:
    if threshold_type == "median":
        threshold_base = cf_scores_df["cf_base"].median()
        threshold_aligned = cf_scores_df["cf_aligned"].median()
    else:
        threshold_base = cf_scores_df["cf_base"].mean()
        threshold_aligned = cf_scores_df["cf_aligned"].mean()

    cf_scores_df = cf_scores_df.copy()
    cf_scores_df["cf_level_base"] = cf_scores_df["cf_base"].apply(lambda x: "high" if x > threshold_base else "low")
    cf_scores_df["cf_level_aligned"] = cf_scores_df["cf_aligned"].apply(lambda x: "high" if x > threshold_aligned else "low")
    cf_scores_df["cf_threshold_base"] = threshold_base
    cf_scores_df["cf_threshold_aligned"] = threshold_aligned

    return cf_scores_df


def merge_classification_with_cf(classification_df: pd.DataFrame, cf_scores_df: pd.DataFrame) -> pd.DataFrame:
    merged = classification_df.merge(cf_scores_df, on="feature_id", how="left")
    return merged


def compute_cf_shift_by_class(merged_df: pd.DataFrame) -> Dict[str, Dict]:
    results = {}

    for primary_class in merged_df["primary_class"].unique():
        class_df = merged_df[merged_df["primary_class"] == primary_class]

        cf_shifts = class_df["cf_shift"].values
        cf_base_values = class_df["cf_base"].values
        cf_aligned_values = class_df["cf_aligned"].values

        results[primary_class] = {
            "count": len(class_df),
            "cf_shift_mean": float(np.mean(cf_shifts)),
            "cf_shift_std": float(np.std(cf_shifts)),
            "cf_shift_median": float(np.median(cf_shifts)),
            "cf_base_mean": float(np.mean(cf_base_values)),
            "cf_aligned_mean": float(np.mean(cf_aligned_values)),
            "high_cf_base_count": int((class_df["cf_level_base"] == "high").sum()),
            "low_cf_base_count": int((class_df["cf_level_base"] == "low").sum()),
            "high_cf_aligned_count": int((class_df["cf_level_aligned"] == "high").sum()),
            "low_cf_aligned_count": int((class_df["cf_level_aligned"] == "low").sum()),
        }

    return results


def identify_visual_evidence_features(merged_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Heuristic tags using base/aligned-only classes (legacy JSON key names kept for consumers)."""
    base_only_high = merged_df[(merged_df["primary_class"] == "base_only") & (merged_df["cf_level_base"] == "high")]["feature_id"].tolist()

    shared_redirected_shift = merged_df[(merged_df["primary_class"] == "shared_redirected") & (merged_df["cf_shift"] < 0)]["feature_id"].tolist()

    aligned_only_high = merged_df[(merged_df["primary_class"] == "aligned_only") & (merged_df["cf_level_aligned"] == "high")]["feature_id"].tolist()

    return {
        "lost_visual_evidence": base_only_high,
        "redirected_visual_to_prior": shared_redirected_shift,
        "new_compensatory_visual": aligned_only_high,
    }


def compute_per_sample_feature_activations(feature_activations: Dict) -> Dict[str, Dict]:
    z_base = feature_activations["z_base"]
    z_aligned = feature_activations["z_aligned"]
    sample_ids = feature_activations["sample_ids"]

    per_sample = {}
    for idx, sid in enumerate(sample_ids):
        key = str(sid)
        per_sample[key] = {
            "z_base": z_base[idx],
            "z_aligned": z_aligned[idx],
            "sample_id": sid,
        }

    return per_sample


def save_cf_results(cf_scores_df: pd.DataFrame, output_path: str) -> None:
    cf_scores_df.to_csv(output_path, index=False)


def load_cf_results(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
