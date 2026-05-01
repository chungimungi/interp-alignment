from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from tqdm import tqdm

from . import config
from .model import SPARCCrossCoder


def get_top_activating_samples(
    z_aligned: torch.Tensor,
    feature_id: int,
    top_k: int = config.SUPERPOSITION_TOP_SAMPLES,
) -> List[int]:
    feature_activations = z_aligned[:, feature_id]
    _, top_indices = torch.topk(feature_activations, min(top_k, len(feature_activations)))
    return top_indices.tolist()


def fit_sparse_regression(
    W_aligned_dec_feature: np.ndarray,
    W_base_dec: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 10000,
) -> Tuple[np.ndarray, float, int]:
    lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    lasso.fit(W_base_dec, W_aligned_dec_feature)

    coefficients = lasso.coef_

    y_pred = W_base_dec @ coefficients
    ss_res = np.sum((W_aligned_dec_feature - y_pred) ** 2)
    ss_tot = np.sum((W_aligned_dec_feature - np.mean(W_aligned_dec_feature)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    n_nonzero = np.sum(np.abs(coefficients) > 1e-6)

    return coefficients, r2, n_nonzero


def analyze_superposition_for_feature(
    feature_id: int,
    W_aligned_dec: torch.Tensor,
    W_base_dec: torch.Tensor,
    z_aligned: torch.Tensor,
    z_base: torch.Tensor,
    alpha: float = 0.01,
) -> Dict:
    W_aligned_dec_feature = W_aligned_dec[:, feature_id].cpu().numpy()
    W_base_dec_np = W_base_dec.cpu().numpy()

    coefficients, r2, n_nonzero = fit_sparse_regression(
        W_aligned_dec_feature, W_base_dec_np, alpha=alpha
    )

    is_superposition = (
        r2 > config.SUPERPOSITION_R2_THRESHOLD
        and n_nonzero <= config.SUPERPOSITION_MAX_CONSTITUENTS
        and n_nonzero >= 2
    )

    nonzero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
    constituent_features = [
        {"feature_id": int(idx), "weight": float(coefficients[idx])}
        for idx in nonzero_indices
    ]
    constituent_features.sort(key=lambda x: abs(x["weight"]), reverse=True)

    top_samples = get_top_activating_samples(z_aligned, feature_id)

    return {
        "feature_id": feature_id,
        "r2": float(r2),
        "n_nonzero": int(n_nonzero),
        "is_superposition": is_superposition,
        "constituent_features": constituent_features,
        "top_activating_samples": top_samples[:10],
    }


def analyze_all_aligned_only_features(
    crosscoder: SPARCCrossCoder,
    classification_df: pd.DataFrame,
    feature_activations: Dict,
    aligned_run_id: str,
    n_jobs: int = 1,
) -> Dict:
    aligned_only_features = classification_df[
        classification_df["primary_class"] == "aligned_only"
    ]["feature_id"].tolist()

    if len(aligned_only_features) == 0:
        return {
            "superposition_fraction": 0.0,
            "total_aligned_only": 0,
            "superposition_count": 0,
            "features": {},
        }

    decoder_weights = crosscoder.get_decoder_weights()
    W_base_dec = decoder_weights["W_base_dec"]
    W_aligned_dec = decoder_weights["W_aligned_dec"]

    z_base = feature_activations["z_base"]
    z_aligned = feature_activations["z_aligned"]

    analyses = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(analyze_superposition_for_feature)(
            feature_id, W_aligned_dec, W_base_dec, z_aligned, z_base
        )
        for feature_id in tqdm(aligned_only_features, desc="Analyzing superposition")
    )

    results = {a["feature_id"]: a for a in analyses}
    superposition_count = sum(1 for a in analyses if a["is_superposition"])

    superposition_fraction = superposition_count / len(aligned_only_features)

    return {
        "superposition_fraction": superposition_fraction,
        "total_aligned_only": len(aligned_only_features),
        "superposition_count": superposition_count,
        "features": results,
        "aligned_run_id": aligned_run_id,
    }


def get_superposition_summary(superposition_results: Dict) -> pd.DataFrame:
    records = []
    for feature_id, analysis in superposition_results["features"].items():
        records.append(
            {
                "feature_id": feature_id,
                "r2": analysis["r2"],
                "n_nonzero": analysis["n_nonzero"],
                "is_superposition": analysis["is_superposition"],
                "n_constituents": len(analysis["constituent_features"]),
            }
        )
    return pd.DataFrame(records)


def save_superposition_results(results: Dict, output_path: str) -> None:
    import json

    serializable = {
        "superposition_fraction": float(results["superposition_fraction"]),
        "total_aligned_only": int(results["total_aligned_only"]),
        "superposition_count": int(results["superposition_count"]),
        "aligned_run_id": str(results.get("aligned_run_id", "unknown")),
        "features": {},
    }

    for fid, analysis in results["features"].items():
        serializable["features"][str(fid)] = {
            "feature_id": int(analysis["feature_id"]),
            "r2": float(analysis["r2"]),
            "n_nonzero": int(analysis["n_nonzero"]),
            "is_superposition": bool(analysis["is_superposition"]),
            "constituent_features": [
                {"feature_id": int(c["feature_id"]), "weight": float(c["weight"])}
                for c in analysis["constituent_features"]
            ],
            "top_activating_samples": [int(s) for s in analysis["top_activating_samples"]],
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_superposition_results(input_path: str) -> Dict:
    import json

    with open(input_path) as f:
        return json.load(f)
