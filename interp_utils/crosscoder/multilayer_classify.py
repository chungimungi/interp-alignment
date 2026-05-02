from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

from .multilayer_model import MultiLayerSPARCCrossCoder


def _entropy(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] == 1:
        return torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)
    probs = values / values.sum(dim=0, keepdim=True).clamp(min=1e-8)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=0)
    return ent / torch.log(torch.tensor(values.shape[0], device=values.device, dtype=values.dtype))


def classify_multilayer_features(
    crosscoder: MultiLayerSPARCCrossCoder,
    layers: List[int],
    rho_base_only: float = 0.15,
    rho_aligned_only: float = 0.85,
    persistent_entropy_threshold: float = 0.6,
    theta_drift_threshold: float = 0.5,
) -> pd.DataFrame:
    decoder_weights = crosscoder.get_decoder_weights()
    W_base_dec = decoder_weights["W_base_dec"]       # [layers, input_dim, features]
    W_aligned_dec = decoder_weights["W_aligned_dec"] # [layers, input_dim, features]

    base_norm = W_base_dec.norm(dim=1)
    aligned_norm = W_aligned_dec.norm(dim=1)
    total_norm = base_norm + aligned_norm
    rho_by_layer = aligned_norm / (total_norm + 1e-8)

    W_base_normed = F.normalize(W_base_dec, dim=1)
    W_aligned_normed = F.normalize(W_aligned_dec, dim=1)
    theta_by_layer = (W_base_normed * W_aligned_normed).sum(dim=1)

    total_base_norm = base_norm.sum(dim=0)
    total_aligned_norm = aligned_norm.sum(dim=0)
    aggregate_rho = total_aligned_norm / (total_base_norm + total_aligned_norm + 1e-8)
    max_base_idx = base_norm.argmax(dim=0)
    max_aligned_idx = aligned_norm.argmax(dim=0)
    norm_entropy = _entropy(total_norm)
    mean_theta = theta_by_layer.mean(dim=0)
    min_abs_theta = theta_by_layer.abs().min(dim=0).values

    records = []
    num_features = aggregate_rho.shape[0]
    forced = set(crosscoder.forced_shared_indices.detach().cpu().tolist())
    for feature_id in range(num_features):
        rho = aggregate_rho[feature_id].item()
        entropy = norm_entropy[feature_id].item()
        mean_theta_i = mean_theta[feature_id].item()
        min_abs_theta_i = min_abs_theta[feature_id].item()
        is_persistent = entropy >= persistent_entropy_threshold
        if rho < rho_base_only:
            primary_class = "persistent_base_only" if is_persistent else "localized_base_only"
        elif rho > rho_aligned_only:
            primary_class = "persistent_aligned_only" if is_persistent else "localized_aligned_only"
        elif min_abs_theta_i < theta_drift_threshold:
            primary_class = "drifting_or_rotating"
        elif is_persistent:
            primary_class = "persistent_shared"
        else:
            primary_class = "mixed_or_ambiguous"

        records.append(
            {
                "feature_id": feature_id,
                "primary_class": primary_class,
                "aggregate_rho": rho,
                "total_base_norm": total_base_norm[feature_id].item(),
                "total_aligned_norm": total_aligned_norm[feature_id].item(),
                "norm_entropy": entropy,
                "mean_theta": mean_theta_i,
                "min_abs_theta": min_abs_theta_i,
                "max_base_layer": layers[int(max_base_idx[feature_id].item())],
                "max_aligned_layer": layers[int(max_aligned_idx[feature_id].item())],
                "is_forced_shared": feature_id in forced,
            }
        )

    return pd.DataFrame(records)


def multilayer_decoder_profile_df(crosscoder: MultiLayerSPARCCrossCoder, layers: List[int]) -> pd.DataFrame:
    decoder_weights = crosscoder.get_decoder_weights()
    W_base_dec = decoder_weights["W_base_dec"]
    W_aligned_dec = decoder_weights["W_aligned_dec"]

    base_norm = W_base_dec.norm(dim=1)
    aligned_norm = W_aligned_dec.norm(dim=1)
    rho = aligned_norm / (base_norm + aligned_norm + 1e-8)
    theta = (F.normalize(W_base_dec, dim=1) * F.normalize(W_aligned_dec, dim=1)).sum(dim=1)

    records = []
    for layer_pos, layer in enumerate(layers):
        for feature_id in range(base_norm.shape[1]):
            records.append(
                {
                    "feature_id": feature_id,
                    "layer": int(layer),
                    "layer_pos": layer_pos,
                    "rho": rho[layer_pos, feature_id].item(),
                    "theta": theta[layer_pos, feature_id].item(),
                    "W_base_dec_norm": base_norm[layer_pos, feature_id].item(),
                    "W_aligned_dec_norm": aligned_norm[layer_pos, feature_id].item(),
                }
            )
    return pd.DataFrame(records)


def get_multilayer_class_counts(classification_df: pd.DataFrame) -> Dict[str, int]:
    return classification_df["primary_class"].value_counts().to_dict()
