from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

from .model import SPARCCrossCoder
from .visualize import classify_for_plot, compute_adaptive_rho_thresholds


def compute_decoder_norm_ratio(W_base_dec: torch.Tensor, W_aligned_dec: torch.Tensor) -> torch.Tensor:
    W_base_norms = W_base_dec.norm(dim=0)
    W_aligned_norms = W_aligned_dec.norm(dim=0)
    rho = W_aligned_norms / (W_base_norms + W_aligned_norms + 1e-8)
    return rho


def compute_decoder_cosine_similarity(W_base_dec: torch.Tensor, W_aligned_dec: torch.Tensor) -> torch.Tensor:
    W_base_normalized = F.normalize(W_base_dec, dim=0)
    W_aligned_normalized = F.normalize(W_aligned_dec, dim=0)
    theta = (W_base_normalized * W_aligned_normalized).sum(dim=0)
    return theta


def classify_all_features(crosscoder: SPARCCrossCoder) -> pd.DataFrame:
    decoder_weights = crosscoder.get_decoder_weights()
    W_base_dec = decoder_weights["W_base_dec"]
    W_aligned_dec = decoder_weights["W_aligned_dec"]

    rho = compute_decoder_norm_ratio(W_base_dec, W_aligned_dec)
    theta = compute_decoder_cosine_similarity(W_base_dec, W_aligned_dec)

    W_base_norms = W_base_dec.norm(dim=0)
    W_aligned_norms = W_aligned_dec.norm(dim=0)

    num_features = rho.shape[0]

    records = []
    for i in range(num_features):
        records.append(
            {
                "feature_id": i,
                "rho": rho[i].item(),
                "theta": theta[i].item(),
                "W_base_dec_norm": W_base_norms[i].item(),
                "W_aligned_dec_norm": W_aligned_norms[i].item(),
                "is_forced_shared": i in crosscoder.forced_shared_indices.tolist(),
            }
        )
    df = pd.DataFrame(records)

    thresh = compute_adaptive_rho_thresholds(df)
    df["primary_class"] = df.apply(
        lambda row: classify_for_plot(row["rho"], row["theta"], thresh), axis=1
    )

    return df


def get_feature_class_counts(classification_df: pd.DataFrame) -> Dict[str, int]:
    return classification_df["primary_class"].value_counts().to_dict()


def get_features_by_class(classification_df: pd.DataFrame, feature_class: str) -> List[int]:
    return classification_df[classification_df["primary_class"] == feature_class]["feature_id"].tolist()


def compute_rho_histogram_data(classification_df: pd.DataFrame, num_bins: int = 50) -> Dict:
    rho_values = classification_df["rho"].values
    hist, bin_edges = torch.histogram(torch.tensor(rho_values), bins=num_bins, range=(0.0, 1.0))
    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)],
    }


def compute_threshold_sensitivity(
    classification_df: pd.DataFrame,
    perturbation: float = 0.05,
) -> Dict:
    original_counts = get_feature_class_counts(classification_df)
    thresh = compute_adaptive_rho_thresholds(classification_df)

    rho_values = classification_df["rho"].values
    theta_values = classification_df["theta"].values

    perturbed_counts = {}
    for delta in [-perturbation, perturbation]:
        adjusted_thresh = {
            "rho_base_only": thresh["rho_base_only"] + delta,
            "rho_aligned_only": thresh["rho_aligned_only"] - delta,
            "rho_shared_low": thresh["rho_shared_low"] + delta,
            "rho_shared_high": thresh["rho_shared_high"] - delta,
        }
        counts = {
            "base_only": 0,
            "aligned_only": 0,
            "shared_aligned": 0,
            "shared_redirected": 0,
            "shared_intermediate": 0,
            "shared_attenuated": 0,
            "other": 0,
        }
        for rho, theta in zip(rho_values, theta_values):
            c = classify_for_plot(rho, theta, adjusted_thresh)
            counts[c] = counts.get(c, 0) + 1
        perturbed_counts[f"delta_{delta:+.2f}"] = counts

    return {
        "original": original_counts,
        "perturbed": perturbed_counts,
        "perturbation": perturbation,
    }


def save_classification_results(
    classification_df: pd.DataFrame,
    output_path: str,
) -> None:
    classification_df.to_csv(output_path, index=False)


def load_classification_results(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
