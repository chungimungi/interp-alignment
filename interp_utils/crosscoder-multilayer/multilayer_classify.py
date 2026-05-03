from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from .classify import compute_threshold_sensitivity
from .multilayer_model import MultiLayerSPARCCrossCoder
from .visualize import classify_for_plot, compute_adaptive_rho_thresholds


def _entropy(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] == 1:
        return torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)
    probs = values / values.sum(dim=0, keepdim=True).clamp(min=1e-8)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=0)
    return ent / torch.log(torch.tensor(values.shape[0], device=values.device, dtype=values.dtype))


def classify_multilayer_features(
    crosscoder: MultiLayerSPARCCrossCoder,
    layers: List[int],
    persistent_entropy_threshold: float = 0.6,
    theta_drift_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Classify aggregate multi-layer decoder geometry with the main crosscoder dataflow.

    The main single-layer classifier operates on one rho/theta pair per feature and
    delegates threshold selection to compute_adaptive_rho_thresholds. For multi-layer
    runs, total decoder norms across the selected layer window define the aggregate
    rho, and mean theta is the aggregate theta. The old taxonomy is kept in
    primary_class; multi-layer-only persistence/drift labels live in multilayer_class.
    """
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
    threshold_df = pd.DataFrame(
        {
            "rho": aggregate_rho.detach().cpu().numpy(),
            "theta": mean_theta.detach().cpu().numpy(),
        }
    )
    thresh = compute_adaptive_rho_thresholds(threshold_df)

    records = []
    num_features = aggregate_rho.shape[0]
    forced = set(crosscoder.forced_shared_indices.detach().cpu().tolist())
    for feature_id in range(num_features):
        rho = aggregate_rho[feature_id].item()
        entropy = norm_entropy[feature_id].item()
        mean_theta_i = mean_theta[feature_id].item()
        min_abs_theta_i = min_abs_theta[feature_id].item()
        is_persistent = entropy >= persistent_entropy_threshold
        primary_class = classify_for_plot(rho, mean_theta_i, thresh)

        if primary_class == "base_only":
            multilayer_class = "persistent_base_only" if is_persistent else "localized_base_only"
        elif primary_class == "aligned_only":
            multilayer_class = "persistent_aligned_only" if is_persistent else "localized_aligned_only"
        elif primary_class == "shared_redirected" or min_abs_theta_i < theta_drift_threshold:
            multilayer_class = "drifting_or_rotating"
        elif is_persistent:
            multilayer_class = "persistent_shared"
        else:
            multilayer_class = "mixed_or_ambiguous"

        records.append(
            {
                "feature_id": feature_id,
                "primary_class": primary_class,
                "multilayer_class": multilayer_class,
                "rho": rho,
                "theta": mean_theta_i,
                "W_base_dec_norm": total_base_norm[feature_id].item(),
                "W_aligned_dec_norm": total_aligned_norm[feature_id].item(),
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


def get_multilayer_semantic_class_counts(classification_df: pd.DataFrame) -> Dict[str, int]:
    if "multilayer_class" not in classification_df:
        return {}
    return classification_df["multilayer_class"].value_counts().to_dict()


def get_multilayer_classification_thresholds(classification_df: pd.DataFrame) -> Dict[str, float]:
    return compute_adaptive_rho_thresholds(classification_df)


def get_multilayer_threshold_sensitivity(classification_df: pd.DataFrame) -> Dict:
    return compute_threshold_sensitivity(classification_df)


def derive_layer_classes(
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    df = profile_df.copy()
    threshold_cols = {
        "rho_base_only": "layer_rho_base_only",
        "rho_aligned_only": "layer_rho_aligned_only",
        "rho_shared_low": "layer_rho_shared_low",
        "rho_shared_high": "layer_rho_shared_high",
    }
    df["layer_class"] = ""
    for out_col in threshold_cols.values():
        df[out_col] = float("nan")

    for _, layer_df in df.groupby("layer", sort=True):
        thresh = compute_adaptive_rho_thresholds(layer_df)
        df.loc[layer_df.index, "layer_class"] = layer_df.apply(
            lambda row: classify_for_plot(row["rho"], row["theta"], thresh),
            axis=1,
        )
        for thresh_key, out_col in threshold_cols.items():
            df.loc[layer_df.index, out_col] = thresh[thresh_key]

    return df


def summarize_layer_metrics(profile_df: pd.DataFrame) -> Dict:
    profile_with_class = profile_df.copy()
    if "layer_class" not in profile_with_class:
        profile_with_class = derive_layer_classes(profile_with_class)
    class_counts_by_layer = {}
    fsr_by_layer = {}
    decoder_amplification_by_layer = {}
    classification_thresholds_by_layer = {}
    threshold_sensitivity_by_layer = {}

    for layer, layer_df in profile_with_class.groupby("layer", sort=True):
        counts = layer_df["layer_class"].value_counts().to_dict()
        shared = sum(count for cls, count in counts.items() if cls.startswith("shared_"))
        total = int(len(layer_df))
        ratio = layer_df["W_aligned_dec_norm"] / (layer_df["W_base_dec_norm"] + 1e-8)
        class_counts_by_layer[str(int(layer))] = {str(k): int(v) for k, v in counts.items()}
        fsr_by_layer[str(int(layer))] = float(shared / total) if total else 0.0
        decoder_amplification_by_layer[str(int(layer))] = {
            "median": float(ratio.median()) if total else 0.0,
            "p95": float(ratio.quantile(0.95)) if total else 0.0,
        }
        thresh = compute_adaptive_rho_thresholds(layer_df)
        classification_thresholds_by_layer[str(int(layer))] = {
            str(k): float(v) for k, v in thresh.items()
        }
        sensitivity_df = layer_df.rename(columns={"layer_class": "primary_class"})
        threshold_sensitivity_by_layer[str(int(layer))] = compute_threshold_sensitivity(sensitivity_df)

    return {
        "class_counts_by_layer": class_counts_by_layer,
        "feature_sharing_ratio_by_layer": fsr_by_layer,
        "decoder_amplification_by_layer": decoder_amplification_by_layer,
        "classification_thresholds_by_layer": classification_thresholds_by_layer,
        "threshold_sensitivity_by_layer": threshold_sensitivity_by_layer,
    }


def cross_layer_cosine_drift_df(crosscoder: MultiLayerSPARCCrossCoder, layers: List[int]) -> pd.DataFrame:
    decoder_weights = crosscoder.get_decoder_weights()
    records = []
    for stream, weights in (
        ("base", decoder_weights["W_base_dec"]),
        ("aligned", decoder_weights["W_aligned_dec"]),
    ):
        normalized = F.normalize(weights, dim=1)
        n_layers, _, n_features = normalized.shape
        for feature_id in range(n_features):
            feature_vectors = normalized[:, :, feature_id]
            cosine = feature_vectors @ feature_vectors.T
            for src_pos in range(n_layers):
                for dst_pos in range(n_layers):
                    records.append(
                        {
                            "feature_id": feature_id,
                            "stream": stream,
                            "source_layer": int(layers[src_pos]),
                            "source_layer_pos": src_pos,
                            "target_layer": int(layers[dst_pos]),
                            "target_layer_pos": dst_pos,
                            "cosine": cosine[src_pos, dst_pos].item(),
                            "abs_cosine": cosine[src_pos, dst_pos].abs().item(),
                        }
                    )
    return pd.DataFrame(records)


def model_layer_stream_patterns_df(
    profile_df: pd.DataFrame,
    feature_activations: Optional[Dict] = None,
    norm_threshold: float = 1e-6,
    activation_threshold: float = 0.0,
) -> pd.DataFrame:
    records = []
    activation_stats = {}
    if feature_activations is not None:
        for stream_name, tensor_key in (("base", "z_base"), ("aligned", "z_aligned")):
            z = feature_activations.get(tensor_key)
            if isinstance(z, torch.Tensor):
                active_fraction = (z.abs() > activation_threshold).float().mean(dim=0)
                mean_activation = z.float().mean(dim=0)
                max_activation = z.float().amax(dim=0)
                for layer_pos in range(z.shape[1]):
                    for feature_id in range(z.shape[2]):
                        activation_stats[(stream_name, layer_pos, feature_id)] = {
                            "activation_active_fraction": active_fraction[layer_pos, feature_id].item(),
                            "activation_mean": mean_activation[layer_pos, feature_id].item(),
                            "activation_max": max_activation[layer_pos, feature_id].item(),
                        }

    for row in profile_df.itertuples(index=False):
        for stream_name, norm_col in (
            ("base", "W_base_dec_norm"),
            ("aligned", "W_aligned_dec_norm"),
        ):
            feature_id = int(row.feature_id)
            layer_pos = int(row.layer_pos)
            norm = float(getattr(row, norm_col))
            stats = activation_stats.get((stream_name, layer_pos, feature_id), {})
            is_decoder_alive = norm > norm_threshold
            is_activation_alive = stats.get("activation_active_fraction", 0.0) > 0.0
            records.append(
                {
                    "feature_id": feature_id,
                    "stream": stream_name,
                    "layer": int(row.layer),
                    "layer_pos": layer_pos,
                    "decoder_norm": norm,
                    "decoder_alive": bool(is_decoder_alive),
                    "activation_active_fraction": float(stats.get("activation_active_fraction", 0.0)),
                    "activation_mean": float(stats.get("activation_mean", 0.0)),
                    "activation_max": float(stats.get("activation_max", 0.0)),
                    "stream_state": "active" if is_decoder_alive or is_activation_alive else "dead",
                }
            )
    return pd.DataFrame(records)


def _superposition_feature_worker(
    feature_id: int,
    target_layer: int,
    target_pos: int,
    primary_class: str,
    aligned_normed_target: "object",
    base_flat_np: "object",
    candidate_layers: List[int],
    candidate_features: List[int],
    top_k: int,
    regression_candidates: int,
    lasso_alpha: float,
) -> Dict:
    import numpy as np
    from sklearn.linear_model import Lasso

    target_vec = aligned_normed_target[feature_id]
    cos = base_flat_np @ target_vec
    candidate_count = min(regression_candidates, cos.size)
    abs_cos = np.abs(cos)
    if candidate_count >= cos.size:
        candidate_indices = np.argsort(-abs_cos)
    else:
        part = np.argpartition(-abs_cos, candidate_count - 1)[:candidate_count]
        candidate_indices = part[np.argsort(-abs_cos[part])]
    top_count = min(top_k, candidate_count)
    top_indices = candidate_indices[:top_count]
    matches = []
    for idx in top_indices:
        idx_int = int(idx)
        signed_cos = float(cos[idx_int])
        matches.append(
            {
                "base_feature_id": candidate_features[idx_int],
                "base_layer": candidate_layers[idx_int],
                "cosine": signed_cos,
                "abs_cosine": float(abs(signed_cos)),
                "is_same_feature": candidate_features[idx_int] == feature_id,
                "is_cross_layer": candidate_layers[idx_int] != target_layer,
            }
        )

    X = base_flat_np[candidate_indices].T
    y = target_vec
    lasso = Lasso(alpha=lasso_alpha, max_iter=10000, fit_intercept=False)
    lasso.fit(X, y)
    coefficients = lasso.coef_
    y_pred = X @ coefficients
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    nonzero_local = np.where(np.abs(coefficients) > 1e-6)[0]
    n_nonzero = int(len(nonzero_local))
    constituent_features = []
    for local_idx in nonzero_local:
        global_idx = int(candidate_indices[int(local_idx)])
        constituent_features.append(
            {
                "base_feature_id": candidate_features[global_idx],
                "base_layer": candidate_layers[global_idx],
                "weight": float(coefficients[local_idx]),
                "is_same_feature": candidate_features[global_idx] == feature_id,
                "is_cross_layer": candidate_layers[global_idx] != target_layer,
            }
        )
    constituent_features.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return {
        "feature_id": feature_id,
        "primary_class": primary_class,
        "target_layer": target_layer,
        "r2": float(r2),
        "n_nonzero": n_nonzero,
        "is_superposition": bool(
            r2 >= 0.5
            and n_nonzero >= 2
            and any(item["is_cross_layer"] for item in constituent_features)
        ),
        "constituent_features": constituent_features,
        "top_base_matches": matches,
    }


def multilayer_superposition_analysis(
    crosscoder: MultiLayerSPARCCrossCoder,
    layers: List[int],
    classification_df: pd.DataFrame,
    top_k: int = 5,
    cosine_threshold: float = 0.5,
    regression_candidates: int = 256,
    lasso_alpha: float = 0.01,
    n_jobs: int = 1,
) -> Dict:
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm

    decoder_weights = crosscoder.get_decoder_weights()
    base = decoder_weights["W_base_dec"]
    aligned = decoder_weights["W_aligned_dec"]
    base_normed = F.normalize(base, dim=1)
    aligned_normed = F.normalize(aligned, dim=1)
    n_layers, _, n_features = base_normed.shape

    base_flat_np = (
        base_normed.permute(0, 2, 1).reshape(n_layers * n_features, -1).cpu().numpy()
    )
    aligned_normed_by_pos = [
        aligned_normed[pos].permute(1, 0).cpu().numpy() for pos in range(n_layers)
    ]

    candidate_layers: List[int] = []
    candidate_features: List[int] = []
    for layer in layers:
        for feature_id in range(n_features):
            candidate_layers.append(int(layer))
            candidate_features.append(int(feature_id))

    classes = classification_df.set_index("feature_id")["primary_class"].to_dict()
    target_rows = classification_df[
        classification_df["primary_class"].astype(str).str.contains("aligned_only", na=False)
    ]
    if target_rows.empty:
        target_rows = classification_df

    tasks = []
    for row in target_rows.itertuples(index=False):
        feature_id = int(row.feature_id)
        target_layer = int(row.max_aligned_layer)
        target_pos = layers.index(target_layer) if target_layer in layers else 0
        tasks.append((feature_id, target_layer, target_pos))

    results_iter = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_superposition_feature_worker)(
            feature_id,
            target_layer,
            target_pos,
            str(classes.get(feature_id, "")),
            aligned_normed_by_pos[target_pos],
            base_flat_np,
            candidate_layers,
            candidate_features,
            top_k,
            regression_candidates,
            lasso_alpha,
        )
        for feature_id, target_layer, target_pos in tqdm(
            tasks, desc="Multilayer superposition", unit="feat"
        )
    )

    features = {str(item["feature_id"]): item for item in results_iter}

    return {
        "analysis_kind": "multilayer_decoder_cosine_candidate_matches",
        "description": (
            "Candidate cross-layer superposition screen: aligned decoder vectors are matched against "
            "base decoder vectors across all selected layers/features. This is a geometry-derived "
            "screen, not a causal decomposition."
        ),
        "layers": [int(layer) for layer in layers],
        "top_k": int(top_k),
        "cosine_threshold": float(cosine_threshold),
        "regression_candidates": int(regression_candidates),
        "lasso_alpha": float(lasso_alpha),
        "n_features_analyzed": int(len(features)),
        "features": features,
    }


def top_activating_examples_df(
    feature_activations: Dict,
    classification_df: pd.DataFrame,
    top_n: int = 10,
    max_features: int = 100,
) -> pd.DataFrame:
    if "prompt_texts" not in feature_activations:
        return pd.DataFrame()

    z_base = feature_activations.get("z_base")
    z_aligned = feature_activations.get("z_aligned")
    if not isinstance(z_base, torch.Tensor) or not isinstance(z_aligned, torch.Tensor):
        return pd.DataFrame()

    feature_order = (
        classification_df.assign(
            total_norm=classification_df["total_base_norm"] + classification_df["total_aligned_norm"]
        )
        .sort_values("total_norm", ascending=False)
        .head(max_features)["feature_id"]
        .astype(int)
        .tolist()
    )
    sample_ids = feature_activations.get("sample_ids", [])
    splits = feature_activations.get("splits", [])
    prompts = feature_activations.get("prompt_texts", [])
    layers = [int(layer) for layer in feature_activations.get("layers", [])]
    class_by_feature = classification_df.set_index("feature_id")["primary_class"].to_dict()

    records = []
    for stream, z in (("base", z_base), ("aligned", z_aligned)):
        z = z.float()
        for feature_id in feature_order:
            if feature_id >= z.shape[2]:
                continue
            sample_layer_values = z[:, :, feature_id]
            sample_values, layer_pos = sample_layer_values.max(dim=1)
            k = min(top_n, sample_values.numel())
            top_values, top_indices = torch.topk(sample_values, k)
            for rank, (value, sample_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
                layer_idx = int(layer_pos[sample_idx].item())
                records.append(
                    {
                        "feature_id": int(feature_id),
                        "primary_class": str(class_by_feature.get(feature_id, "")),
                        "stream": stream,
                        "rank": rank,
                        "activation": float(value),
                        "layer": layers[layer_idx] if layer_idx < len(layers) else layer_idx,
                        "layer_pos": layer_idx,
                        "sample_id": sample_ids[sample_idx] if sample_idx < len(sample_ids) else "",
                        "split": splits[sample_idx] if sample_idx < len(splits) else "",
                        "prompt": prompts[sample_idx] if sample_idx < len(prompts) else "",
                    }
                )
    return pd.DataFrame(records)


def multilayer_counterfactual_scores(
    feature_activations: Dict,
    layers: List[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_base = feature_activations.pop("z_base", None)
    z_aligned = feature_activations.pop("z_aligned", None)
    if not isinstance(z_base, torch.Tensor) or not isinstance(z_aligned, torch.Tensor):
        return pd.DataFrame(), pd.DataFrame()

    n_features = z_base.shape[2]
    base_abs = z_base.float().abs_()
    aligned_abs = z_aligned.float().abs_()
    del z_base, z_aligned

    cf_base = base_abs.mean(dim=(0, 1))
    cf_aligned = aligned_abs.mean(dim=(0, 1))
    layer_base = base_abs.mean(dim=0)
    layer_aligned = aligned_abs.mean(dim=0)
    cf_shift = cf_aligned - cf_base
    layer_shift = layer_aligned - layer_base

    diff_abs = aligned_abs.sub_(base_abs).abs_()
    del base_abs, aligned_abs
    cf_shift_abs_p95 = torch.quantile(diff_abs.reshape(-1, n_features), 0.95, dim=0)
    layer_shift_abs_p95 = torch.quantile(diff_abs, 0.95, dim=0)
    del diff_abs

    aggregate_records = []
    for feature_id in range(n_features):
        aggregate_records.append(
            {
                "feature_id": feature_id,
                "cf_base": cf_base[feature_id].item(),
                "cf_aligned": cf_aligned[feature_id].item(),
                "cf_shift": cf_shift[feature_id].item(),
                "cf_shift_abs_p95": cf_shift_abs_p95[feature_id].item(),
            }
        )

    layer_records = []
    for layer_pos, layer in enumerate(layers):
        for feature_id in range(n_features):
            layer_records.append(
                {
                    "feature_id": feature_id,
                    "layer": int(layer),
                    "layer_pos": layer_pos,
                    "cf_base": layer_base[layer_pos, feature_id].item(),
                    "cf_aligned": layer_aligned[layer_pos, feature_id].item(),
                    "cf_shift": layer_shift[layer_pos, feature_id].item(),
                    "cf_shift_abs_p95": layer_shift_abs_p95[layer_pos, feature_id].item(),
                }
            )

    return pd.DataFrame(aggregate_records), pd.DataFrame(layer_records)


def summarize_counterfactual_by_layer(cf_layer_df: pd.DataFrame, classification_df: pd.DataFrame) -> Dict:
    if cf_layer_df.empty:
        return {}
    class_df = classification_df[["feature_id", "primary_class"]]
    merged = cf_layer_df.merge(class_df, on="feature_id", how="left")
    results = {}
    for (layer, primary_class), group in merged.groupby(["layer", "primary_class"], sort=True):
        results.setdefault(str(int(layer)), {})[str(primary_class)] = {
            "mean_shift": float(group["cf_shift"].mean()),
            "median_shift": float(group["cf_shift"].median()),
            "p95_abs_shift": float(group["cf_shift_abs_p95"].quantile(0.95)),
            "count": int(len(group)),
        }
    return results
