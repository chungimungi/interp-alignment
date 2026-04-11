import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from . import config

SHARED_CLASSES = [
    "shared_aligned",
    "shared_redirected",
    "shared_intermediate",
    "shared_attenuated",
]


def compute_feature_sharing_ratio(classification_df: pd.DataFrame) -> float:
    exclusive_classes = ["base_only", "aligned_only"]

    n_shared = classification_df[classification_df["primary_class"].isin(SHARED_CLASSES)].shape[0]
    n_exclusive = classification_df[classification_df["primary_class"].isin(exclusive_classes)].shape[0]
    
    if n_shared + n_exclusive == 0:
        return 0.0
    
    return n_shared / (n_shared + n_exclusive)


def compute_semantic_stability_score(classification_df: pd.DataFrame) -> float:
    """
    Mean theta over shared features (GMM-based classification).
    Shared = primary_class in shared_aligned, shared_redirected, shared_intermediate, shared_attenuated.
    """
    shared_features = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ]
    if len(shared_features) == 0:
        return float("nan")
    return float(shared_features["theta"].mean())


def compute_counterfactual_sensitivity_shift(merged_df: pd.DataFrame) -> Dict[str, float]:
    results = {}

    for primary_class in merged_df["primary_class"].unique():
        class_df = merged_df[merged_df["primary_class"] == primary_class]
        if len(class_df) > 0 and "cf_shift" in class_df.columns:
            results[primary_class] = class_df["cf_shift"].mean()

    return results


def compute_plan_feature_survival_rate(
    feature_activations: Dict,
    threshold: float = 0.5,
) -> float:
    """
    Plan definition (full_project_plan §5.4): % of original features with high
    correlation in compressed model.
    """
    z_base = feature_activations["z_base"]
    z_aligned = feature_activations["z_aligned"]
    if hasattr(z_base, "numpy"):
        z_base = z_base.numpy()
    if hasattr(z_aligned, "numpy"):
        z_aligned = z_aligned.numpy()
    z_base = np.asarray(z_base, dtype=np.float64)
    z_aligned = np.asarray(z_aligned, dtype=np.float64)

    num_features = z_base.shape[1]
    survived = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pearsonr ConstantInputWarning for constant features
        for i in range(num_features):
            try:
                corr, _ = pearsonr(z_base[:, i], z_aligned[:, i])
                if np.isfinite(corr) and corr > threshold:
                    survived += 1
            except (ValueError, RuntimeError):
                pass
    return survived / num_features if num_features > 0 else 0.0


def compute_jaccard_class_distributions(
    classification_a: pd.DataFrame,
    classification_b: pd.DataFrame,
) -> Dict:
    """
    Jaccard similarity of feature class assignments between two configs.
    Per-class: |A_c ∩ B_c| / |A_c ∪ B_c|. Macro-mean over classes.
    """
    # Ensure feature_id alignment (both should have same indices 0..N-1)
    a = classification_a.set_index("feature_id")["primary_class"]
    b = classification_b.set_index("feature_id")["primary_class"]
    all_classes = set(a.unique()) | set(b.unique())
    per_class = {}
    for c in all_classes:
        a_set = set(a[a == c].index.tolist())
        b_set = set(b[b == c].index.tolist())
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        per_class[c] = inter / union if union > 0 else 0.0
    jaccards = list(per_class.values())
    macro_mean = float(np.mean(jaccards)) if jaccards else 0.0
    return {"per_class": per_class, "macro_mean": macro_mean}


def compute_superposition_fraction(superposition_results: Dict) -> float:
    return superposition_results.get("superposition_fraction", 0.0)


def compute_all_primary_metrics(
    classification_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    superposition_results: Dict,
    training_history: Dict,
) -> Dict:
    fsr = compute_feature_sharing_ratio(classification_df)
    sss = compute_semantic_stability_score(classification_df)
    css = compute_counterfactual_sensitivity_shift(merged_df)
    sf = compute_superposition_fraction(superposition_results)
    
    fve_base = training_history["val_fve_base"][-1] if training_history.get("val_fve_base") else 0.0
    fve_aligned = training_history["val_fve_aligned"][-1] if training_history.get("val_fve_aligned") else 0.0
    dead_neurons = training_history["dead_neurons"][-1] if training_history["dead_neurons"] else 0.0
    l0_base = training_history["l0_base"][-1] if training_history.get("l0_base") else 0.0
    l0_aligned = training_history["l0_aligned"][-1] if training_history.get("l0_aligned") else 0.0
    
    class_counts = classification_df["primary_class"].value_counts().to_dict()
    
    return {
        "feature_sharing_ratio": fsr,
        "semantic_stability_score": sss,
        "counterfactual_sensitivity_shift": css,
        "superposition_fraction": sf,
        "fve_base": fve_base,
        "fve_aligned": fve_aligned,
        "dead_neuron_fraction": dead_neurons,
        "l0_sparsity_base": l0_base,
        "l0_sparsity_aligned": l0_aligned,
        "class_counts": class_counts,
        "total_features": len(classification_df),
    }


def test_hypothesis_h1(
    wanda_classification: pd.DataFrame,
    awq_classification: pd.DataFrame,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H1 deprecated: was VLM wanda vs AWQ; use two LLM crosscoder runs and compare CSVs manually.",
    }


def test_hypothesis_h2(merged_df: pd.DataFrame) -> Dict:
    b_only = merged_df[merged_df["primary_class"] == "base_only"]

    if len(b_only) == 0 or "cf_level_base" not in b_only.columns:
        return {
            "high_cf_count": 0,
            "low_cf_count": 0,
            "ratio": 0.0,
            "hypothesis_supported": False,
            "description": "H2: Base-only features with high CF_base (legacy visual-evidence analog)",
        }

    high_cf = (b_only["cf_level_base"] == "high").sum()
    low_cf = (b_only["cf_level_base"] == "low").sum()

    ratio = high_cf / (high_cf + low_cf) if (high_cf + low_cf) > 0 else 0.0

    return {
        "high_cf_count": int(high_cf),
        "low_cf_count": int(low_cf),
        "ratio": float(ratio),
        "hypothesis_supported": ratio > 0.5,
        "description": "H2: Base-only features with high CF_base",
    }


def test_hypothesis_h3(
    wanda_superposition: Dict,
    awq_superposition: Dict,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H3 deprecated: was wanda vs AWQ superposition; compare superposition_analysis.json across runs manually.",
    }


def test_hypothesis_h4(
    cls_classification: pd.DataFrame,
    patch_classification: pd.DataFrame,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H4 deprecated: was CLS vs patch (VLM); not applicable to LLM pipeline.",
    }


def test_hypothesis_h5(
    v_classification: pd.DataFrame,
    p_classification: pd.DataFrame,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H5 deprecated: was V vs P (VLM); not applicable to LLM pipeline.",
    }


def test_hypothesis_h6(p_merged_df: pd.DataFrame) -> Dict:
    redirected = p_merged_df[p_merged_df["primary_class"] == "shared_redirected"]
    
    if len(redirected) == 0 or "cf_shift" not in redirected.columns:
        return {
            "mean_cf_shift": 0.0,
            "negative_shift_count": 0,
            "total_redirected": 0,
            "hypothesis_supported": False,
            "description": "H6: Projector redirected features shift from visual to prior",
        }
    
    mean_shift = redirected["cf_shift"].mean()
    negative_count = (redirected["cf_shift"] < 0).sum()
    
    return {
        "mean_cf_shift": float(mean_shift),
        "negative_shift_count": int(negative_count),
        "total_redirected": len(redirected),
        "hypothesis_supported": mean_shift < 0,
        "description": "H6: Projector redirected features shift from visual to prior",
    }


def test_hypothesis_h7(
    v_metrics: Dict,
    p_metrics: Dict,
    vp_metrics: Dict,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H7 deprecated: was V+P sub-additivity (VLM); not applicable to LLM pipeline.",
    }


def test_hypothesis_h8(
    blip_p_metrics: Dict,
    qwen3vl_p_metrics: Dict,
) -> Dict:
    return {
        "hypothesis_supported": False,
        "description": "H8 deprecated: was BLIP vs Qwen3VL (VLM); not applicable to LLM pipeline.",
    }


def compile_all_hypothesis_results(hypothesis_tests: Dict) -> pd.DataFrame:
    records = []
    for h_name, result in hypothesis_tests.items():
        records.append({
            "hypothesis": h_name,
            "supported": result.get("hypothesis_supported", False),
            "description": result.get("description", ""),
            **{k: v for k, v in result.items() if k not in ["hypothesis_supported", "description"]},
        })
    return pd.DataFrame(records)


def compute_decoder_norm_ratio_raw(
    W_base_dec: Union[torch.Tensor, np.ndarray],
    W_aligned_dec: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Per-feature raw norm ratio: ||W_aligned[:, i]|| / ||W_base[:, i]||."""
    if isinstance(W_base_dec, torch.Tensor):
        W_base_dec = W_base_dec.cpu().numpy()
    if isinstance(W_aligned_dec, torch.Tensor):
        W_aligned_dec = W_aligned_dec.cpu().numpy()
    W_base_dec = np.asarray(W_base_dec, dtype=np.float64)
    W_aligned_dec = np.asarray(W_aligned_dec, dtype=np.float64)
    norms_b = np.linalg.norm(W_base_dec, axis=0)
    norms_a = np.linalg.norm(W_aligned_dec, axis=0)
    eps = 1e-10
    return norms_a / (norms_b + eps)


def compute_linear_map_summary(
    V_A: np.ndarray,
    V_B: np.ndarray,
    k_min: int = 10,
) -> Optional[Dict]:
    """
    Fit T such that T @ V_A ≈ V_B, compute SVD of restricted map.
    Returns dict with sv_mean, sv_std, condition_number, mean_principal_angle; None if cols < k_min.
    """
    k = V_A.shape[1]
    if k < k_min:
        return None
    V_A = np.asarray(V_A, dtype=np.float64)
    V_B = np.asarray(V_B, dtype=np.float64)
    T = V_B @ V_A.T @ np.linalg.pinv(V_A @ V_A.T)
    T_restricted = T  # T maps from col space of V_A to output; we work in feature subspace
    U, s, Vh = np.linalg.svd(T_restricted)
    cond = float(s[0] / (s[-1] + 1e-12)) if len(s) > 0 and s[-1] > 1e-12 else float("inf")
    T_V_A = T @ V_A
    cos_angles = np.diag(V_A.T @ T_V_A) / (
        np.linalg.norm(V_A, axis=0) * np.linalg.norm(T_V_A, axis=0) + 1e-12
    )
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    principal_angles_rad = np.arccos(np.abs(cos_angles))
    mean_angle_deg = float(np.degrees(principal_angles_rad.mean()))
    return {
        "sv_mean": float(np.mean(s)),
        "sv_std": float(np.std(s)),
        "condition_number": cond,
        "mean_principal_angle_deg": mean_angle_deg,
        "singular_values": [float(x) for x in s[:20]],
    }


def summarize_shared_geometry(
    classification_df: pd.DataFrame,
    W_base_dec: Union[torch.Tensor, np.ndarray],
    W_aligned_dec: Union[torch.Tensor, np.ndarray],
    k_min: int = 10,
) -> Dict:
    """
    Per shared subclass: distribution stats (rho, theta, angle_deg, norm_ratio_raw)
    and subspace linear-map SVD summaries when subset size >= k_min.
    """
    shared_df = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ].copy()
    if len(shared_df) == 0:
        return {"all_shared": {"n": 0}}

    if isinstance(W_base_dec, torch.Tensor):
        W_b_np = W_base_dec.cpu().numpy()
        W_a_np = W_aligned_dec.cpu().numpy()
    else:
        W_b_np = np.asarray(W_base_dec, dtype=np.float64)
        W_a_np = np.asarray(W_aligned_dec, dtype=np.float64)

    norm_ratio_raw = compute_decoder_norm_ratio_raw(W_b_np, W_a_np)
    theta_arr = np.asarray(classification_df["theta"].values, dtype=np.float64)
    theta_clipped = np.clip(theta_arr, -1.0, 1.0)
    angle_deg_arr = np.degrees(np.arccos(np.abs(theta_clipped)))

    extra = pd.DataFrame({
        "feature_id": np.arange(len(classification_df)),
        "norm_ratio_raw": norm_ratio_raw,
        "angle_deg": angle_deg_arr,
    })
    shared_df = shared_df.merge(
        extra[["feature_id", "norm_ratio_raw", "angle_deg"]],
        on="feature_id",
        how="left",
    )

    result: Dict = {}
    classes_to_process: List[str] = list(SHARED_CLASSES) + ["all_shared"]
    for cls in classes_to_process:
        if cls == "all_shared":
            subclass_df = shared_df
        else:
            subclass_df = shared_df[shared_df["primary_class"] == cls]

        n = len(subclass_df)
        row: Dict = {"n": n}
        if n == 0:
            result[cls] = row
            continue

        for col in ["rho", "theta", "angle_deg", "norm_ratio_raw"]:
            if col not in subclass_df.columns:
                continue
            vals = subclass_df[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std())

        ids = subclass_df["feature_id"].astype(int).tolist()
        V_A = W_b_np[:, ids]
        V_B = W_a_np[:, ids]
        lin_summary = compute_linear_map_summary(V_A, V_B, k_min=k_min)
        if lin_summary is not None:
            row["linear_map"] = lin_summary

        result[cls] = row
    return result


def get_shared_features_geometry_df(
    classification_df: pd.DataFrame,
    W_base_dec: Union[torch.Tensor, np.ndarray],
    W_aligned_dec: Union[torch.Tensor, np.ndarray],
) -> pd.DataFrame:
    """Per-feature geometry for shared features (used for visualization)."""
    shared_df = classification_df[
        classification_df["primary_class"].isin(SHARED_CLASSES)
    ].copy()
    if len(shared_df) == 0:
        return shared_df

    if isinstance(W_base_dec, torch.Tensor):
        W_b_np = W_base_dec.cpu().numpy()
        W_a_np = W_aligned_dec.cpu().numpy()
    else:
        W_b_np = np.asarray(W_base_dec, dtype=np.float64)
        W_a_np = np.asarray(W_aligned_dec, dtype=np.float64)

    norm_ratio_raw = compute_decoder_norm_ratio_raw(W_b_np, W_a_np)
    theta_arr = np.asarray(classification_df["theta"].values, dtype=np.float64)
    theta_clipped = np.clip(theta_arr, -1.0, 1.0)
    angle_deg_arr = np.degrees(np.arccos(np.abs(theta_clipped)))

    extra = pd.DataFrame({
        "feature_id": np.arange(len(classification_df)),
        "norm_ratio_raw": norm_ratio_raw,
        "angle_deg": angle_deg_arr,
    })
    return shared_df.merge(
        extra[["feature_id", "norm_ratio_raw", "angle_deg"]],
        on="feature_id",
        how="left",
    )


def save_metrics(metrics: Dict, output_path: str) -> None:
    import json
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(input_path: str) -> Dict:
    import json
    with open(input_path) as f:
        return json.load(f)
