#!/usr/bin/env python3
"""Layerwise contrastive logistic probes on (chosen - rejected) preference deltas.

Defaults are tuned for NVIDIA B200 (bfloat16 forward, batch=64). Pass
``--model-dtype float16 --batch-size 8`` to reproduce the original CPU/fp16 baseline
numerics. Probe fitting itself stays on CPU sklearn by default so cross-fold AUROC
matches the reference results regardless of GPU.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

_LP_DIR = Path(__file__).resolve().parent
if str(_LP_DIR) not in sys.path:
    sys.path.insert(0, str(_LP_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from pca_plot import plot_best_layer_pca_figure


def _umap_z_from_scaled(X_scaled: np.ndarray, seed: int) -> np.ndarray:
    """1D UMAP embedding of row-wise scaled activations (third plot axis)."""
    try:
        from umap import UMAP
    except ImportError as e:
        raise SystemExit(
            "umap-learn is required (pip install umap-learn) for the UMAP z-axis."
        ) from e
    n, d = X_scaled.shape
    if n < 3:
        return np.zeros(n, dtype=np.float64)
    Xw = np.asarray(X_scaled, dtype=np.float64)
    if d > 64:
        npc = min(50, n - 1, d)
        Xw = PCA(n_components=npc, random_state=seed).fit_transform(Xw)
    n_neighbors = min(15, max(2, n - 1))
    emb = UMAP(
        n_components=1,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        n_epochs=200,
    ).fit_transform(Xw)
    return emb.ravel()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default="Qwen/Qwen3-4B")
    p.add_argument(
        "--dataset-name",
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="HF dataset id with chosen/rejected message-list columns.",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--probe-backend",
        choices=["auto", "sklearn", "torch", "cuml"],
        default="sklearn",
        help="Backend for the per-layer logistic regression. sklearn (CPU) is the "
        "reference; cuml uses RAPIDS if importable; auto tries cuml then falls back "
        "to sklearn; torch is currently not implemented.",
    )
    p.add_argument(
        "--strict-cuml",
        action="store_true",
        help="Fail if cuML cannot be imported instead of falling back to sklearn.",
    )
    p.add_argument("--max-pairs", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="LM forward batch size for activation extraction. B200 has 179GB VRAM; "
        "64 is comfortable for 3-4B models at seqlen 512. Lower if you OOM on bigger LMs.",
    )
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="LM weight dtype on GPU. bf16 is B200-native (faster matmul, no fp16 overflow).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Root for outputs. Default: <repo_root>/output/linear-probes/",
    )
    return p.parse_args()


def _repo_root() -> Path:
    # interp_utils/linear-probe/linear-probe.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _make_probe_factory(backend: str, *, strict_cuml: bool, seed: int):
    """Return a zero-arg factory that builds a fresh sklearn-API probe pipeline."""
    if backend in ("cuml", "auto"):
        try:
            from cuml.linear_model import LogisticRegression as CumlLR  # type: ignore

            print(f"Probe backend: cuML LogisticRegression (backend={backend})", flush=True)

            def factory():
                return make_pipeline(
                    StandardScaler(),
                    CumlLR(max_iter=5000, class_weight="balanced", random_state=seed),
                )

            return factory
        except Exception as e:  # noqa: BLE001
            if backend == "cuml" or strict_cuml:
                raise SystemExit(
                    f"cuML required but unavailable: {type(e).__name__}: {e}"
                ) from e
            print(
                f"cuML unavailable ({type(e).__name__}); falling back to sklearn.",
                flush=True,
            )
    if backend == "torch":
        raise SystemExit(
            "torch probe backend not implemented; use --probe-backend sklearn (default) or cuml."
        )

    print("Probe backend: sklearn LogisticRegression (CPU)", flush=True)

    def factory():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed,
            ),
        )

    return factory


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


def _set_layer_ticks(ax, layers):
    desired = [10, 20, 30]
    ticks = [t for t in desired if min(layers) <= t <= max(layers)]
    if ticks:
        ax.set_xticks(ticks)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)


def _normalize_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            [str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content]
        )
    return str(content)


def _sanitize_messages(messages):
    return [
        {"role": m["role"], "content": _normalize_content(m["content"])}
        for m in messages
        if "role" in m
    ]


def _build_prompt_text(messages, tokenizer):
    msgs = _sanitize_messages(messages)
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in msgs)


def _coef_norm_from_pipeline(pipe) -> float | None:
    """Pull the LR coefficient norm regardless of step name (sklearn/cuml differ)."""
    for _, step in pipe.named_steps.items():
        coef = getattr(step, "coef_", None)
        if coef is None:
            continue
        arr = np.asarray(coef)
        return float(np.linalg.norm(arr))
    return None


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_root = Path(args.output_root) if args.output_root else _repo_root() / "output" / "linear-probes"
    out_dir = out_root / args.model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(args.model_dtype) if device == "cuda" else torch.float32

    print(f"Model:    {args.model_name}", flush=True)
    print(f"Dataset:  {args.dataset_name}", flush=True)
    print(f"Device:   {device}  dtype={dtype}", flush=True)
    print(f"Batch:    {args.batch_size}  max_len={args.max_length}  max_pairs={args.max_pairs}", flush=True)
    print(f"Output:   {out_dir}", flush=True)

    tok_kwargs: dict = {}
    if args.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    print(f"Loading tokenizer: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model:     {args.model_name}", flush=True)
    model_kwargs: dict = {"torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    layers = model.model.layers
    num_layers = len(layers)
    hidden = model.config.hidden_size
    print(f"Layers:   {num_layers}  hidden={hidden}", flush=True)

    print(f"Loading dataset: {args.dataset_name}", flush=True)
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.select(range(min(args.max_pairs, len(dataset))))

    chosen_texts: list[str] = []
    rejected_texts: list[str] = []
    for item in dataset:
        try:
            chosen_texts.append(_build_prompt_text(item["chosen"], tokenizer))
            rejected_texts.append(_build_prompt_text(item["rejected"], tokenizer))
        except Exception:
            continue
    n_pairs = len(chosen_texts)
    print(f"Valid chosen/rejected pairs: {n_pairs}", flush=True)

    activations: dict[int, np.ndarray] = {}

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, output):
            x = output[0] if isinstance(output, tuple) else output
            # left-padding -> last true token is at index -1
            activations[layer_idx] = x[:, -1, :].detach().float().cpu().numpy()
        return hook

    hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

    def extract(texts: list[str], desc: str) -> dict[int, np.ndarray]:
        feats: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}
        for i in tqdm(range(0, len(texts), args.batch_size), desc=desc):
            batch = texts[i : i + args.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            ).to(device)
            with torch.no_grad():
                _ = model(**inputs)
            for j in range(num_layers):
                feats[j].append(activations[j])
        return {j: np.concatenate(feats[j], axis=0) for j in range(num_layers)}

    chosen_features = extract(chosen_texts, "Chosen")
    rejected_features = extract(rejected_texts, "Rejected")
    for h in hooks:
        h.remove()

    # Free LM VRAM before sklearn fitting so concurrent jobs on other GPUs aren't starved.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_factory = _make_probe_factory(args.probe_backend, strict_cuml=args.strict_cuml, seed=args.seed)
    cv = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    layer_metrics: list[dict] = []
    layer_predictions: dict[int, dict[str, list]] = {}
    layer_probabilities: dict[int, list[float]] = {}

    for layer_idx in tqdm(range(num_layers), desc="Probes"):
        X_c = chosen_features[layer_idx]
        X_r = rejected_features[layer_idx]
        X_diff = X_c - X_r
        X_sym = np.vstack([X_diff, -X_diff])
        y_sym = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
        mask = np.isfinite(X_sym).all(axis=1)
        X_sym, y_sym = X_sym[mask], y_sym[mask]

        oof_preds = np.zeros(len(y_sym))
        oof_probs = np.zeros(len(y_sym))
        coef_norms: list[float] = []

        for tr, te in cv.split(X_sym, y_sym):
            probe = probe_factory()
            probe.fit(X_sym[tr], y_sym[tr])
            oof_preds[te] = probe.predict(X_sym[te])
            oof_probs[te] = probe.predict_proba(X_sym[te])[:, 1]
            cn = _coef_norm_from_pipeline(probe)
            if cn is not None:
                coef_norms.append(cn)

        layer_metrics.append(
            {
                "layer": layer_idx,
                "accuracy": float(accuracy_score(y_sym, oof_preds)),
                "f1": float(f1_score(y_sym, oof_preds)),
                "auroc": float(roc_auc_score(y_sym, oof_probs)),
                "auprc": float(average_precision_score(y_sym, oof_probs)),
                "coef_norm": float(np.mean(coef_norms)) if coef_norms else 0.0,
            }
        )
        layer_predictions[layer_idx] = {"y_test": y_sym.tolist(), "y_pred": oof_preds.tolist()}
        layer_probabilities[layer_idx] = oof_probs.tolist()

    with (out_dir / "layer_metrics.json").open("w") as f:
        json.dump(layer_metrics, f, indent=2)
    with (out_dir / "layer_predictions.json").open("w") as f:
        json.dump(layer_predictions, f, indent=2)
    with (out_dir / "layer_probabilities.json").open("w") as f:
        json.dump(layer_probabilities, f, indent=2)

    best_layer = max(layer_metrics, key=lambda m: m["auroc"])["layer"]
    X_c_b = chosen_features[best_layer]
    X_r_b = rejected_features[best_layer]
    X_pca_in = np.vstack([X_r_b, X_c_b])
    y_pca = np.concatenate([np.zeros(len(X_r_b)), np.ones(len(X_c_b))])
    mask_pca = np.isfinite(X_pca_in).all(axis=1)
    X_pca_in, y_pca = X_pca_in[mask_pca], y_pca[mask_pca]
    n_pca = min(2, X_pca_in.shape[0], X_pca_in.shape[1])
    pca = PCA(n_components=max(1, n_pca), random_state=args.seed)
    X_scaled = StandardScaler().fit_transform(X_pca_in)
    X_pca = pca.fit_transform(X_scaled)
    n_pc_cols = X_pca.shape[1]
    umap_z = _umap_z_from_scaled(X_scaled, args.seed)
    pca_payload = {
        "best_layer": int(best_layer),
        "y_test": y_pca.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1": X_pca[:, 0].tolist(),
        "umap_z": umap_z.tolist(),
    }
    if n_pc_cols >= 2:
        pca_payload["pc2"] = X_pca[:, 1].tolist()
    with (out_dir / "best_layer_pca.json").open("w") as f:
        json.dump(pca_payload, f, indent=2)

    print("Generating PDFs...", flush=True)
    layers_l = [m["layer"] for m in layer_metrics]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(layers_l, [m["accuracy"] for m in layer_metrics], marker="o", label="Accuracy", linewidth=4, markersize=14)
    ax.plot(layers_l, [m["f1"] for m in layer_metrics], marker="^", label="F1", linewidth=4, markersize=14)
    ax.plot(layers_l, [m["auroc"] for m in layer_metrics], marker="P", label="AUROC", linewidth=4, markersize=14)
    ax.plot(layers_l, [m["auprc"] for m in layer_metrics], marker="s", label="AUPRC", linewidth=4, markersize=14)
    ax.set_xlabel("Layer", fontsize=58)
    ax.set_ylabel("Score", fontsize=58)
    ax.tick_params(axis="both", which="major", labelsize=46)
    if len(layers_l) > 20:
        step = 5
    elif len(layers_l) > 10:
        step = 2
    else:
        step = 1
    ax.set_xticks(layers_l[::step])
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_ticks = [y_min + i * 0.03 for i in range(int(y_range / 0.03) + 1)]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(46)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_probe_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(layers_l, [m["coef_norm"] for m in layer_metrics], marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    _set_layer_ticks(ax, layers_l)
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "layerwise_coef_norm.pdf")
    plt.close(fig)

    y_prob_b = np.array(layer_probabilities[best_layer])
    y_test_b = np.array(layer_predictions[best_layer]["y_test"])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(y_prob_b[y_test_b == 0], bins=24, alpha=0.7, density=True, label="Reverse Direction (-Δx)")
    ax.hist(y_prob_b[y_test_b == 1], bins=24, alpha=0.7, density=True, label="Preference Direction (+Δx)")
    ax.set_xlabel("Predicted Probability of Preferred Direction")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "best_layer_probability_hist.pdf")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test_b, y_prob_b)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, label=f"Layer {best_layer}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=3, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "best_layer_roc_curve.pdf")
    plt.close(fig)

    plot_best_layer_pca_figure(pca_payload, out_dir / "best_layer_pca.pdf")

    print(f"Done. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
