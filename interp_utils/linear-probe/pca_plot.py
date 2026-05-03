"""PCA/UMAP/t-SNE scatter plots for linear-probe best-layer activations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_best_layer_pca_figure(pca_payload: Mapping[str, Any], out_path: Path) -> None:
    """Write ``best_layer_pca.pdf`` from a ``best_layer_pca.json``-style dict."""
    pca_y = np.array(pca_payload["y_test"])
    evr = pca_payload.get("explained_variance_ratio", [])

    if all(k in pca_payload for k in ("tsne1", "tsne2", "tsne3")):
        x = np.array(pca_payload["tsne1"])
        y = np.array(pca_payload["tsne2"])
        z = np.array(pca_payload["tsne3"])
        x_label, y_label, z_label = "t-SNE 1", "t-SNE 2", "t-SNE 3"
    elif "umap_z" in pca_payload and "pc2" in pca_payload:
        x = np.array(pca_payload["pc1"])
        y = np.array(pca_payload["pc2"])
        z = np.array(pca_payload["umap_z"])
        x_label = f"PC1 ({100 * evr[0]:.1f}% var.)" if len(evr) > 0 else "PC1"
        y_label = f"PC2 ({100 * evr[1]:.1f}% var.)" if len(evr) > 1 else "PC2"
        if pca_payload.get("umap_z_is_pc12_fallback"):
            z_label = "UMAP-1 (fit on PC1, PC2)"
        else:
            z_label = "UMAP-1 (fit on scaled activations)"
    elif "pc3" in pca_payload and "pc2" in pca_payload:
        x = np.array(pca_payload["pc1"])
        y = np.array(pca_payload["pc2"])
        z = np.array(pca_payload["pc3"])
        x_label = f"PC1 ({100 * evr[0]:.1f}% var.)" if len(evr) > 0 else "PC1"
        y_label = f"PC2 ({100 * evr[1]:.1f}% var.)" if len(evr) > 1 else "PC2"
        z_label = f"PC3 ({100 * evr[2]:.1f}% var.)" if len(evr) > 2 else "PC3"
    elif "pc2" in pca_payload:
        x = np.array(pca_payload["pc1"])
        y = np.array(pca_payload["pc2"])
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.scatter(x[pca_y == 0], y[pca_y == 0], alpha=0.75, s=65, linewidths=0, label="Rejected")
        ax.scatter(x[pca_y == 1], y[pca_y == 1], alpha=0.75, s=65, linewidths=0, label="Chosen")
        ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var.)" if len(evr) > 0 else "PC1")
        ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var.)" if len(evr) > 1 else "PC2")
        ax.grid(alpha=0.25)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax.transAxes,
            fontsize=10,
            frameon=True,
            framealpha=0.92,
            borderaxespad=0,
        )
        fig.savefig(out_path)
        plt.close(fig)
        return
    else:
        x = np.array(pca_payload["pc1"])
        fig, ax = plt.subplots(figsize=(14, 5))
        yj = np.zeros(len(x))
        ax.scatter(x[pca_y == 0], yj[pca_y == 0], alpha=0.75, s=65, linewidths=0, label="Rejected")
        ax.scatter(x[pca_y == 1], yj[pca_y == 1], alpha=0.75, s=65, linewidths=0, label="Chosen")
        ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var.)" if len(evr) > 0 else "PC1")
        ax.set_yticks([])
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax.transAxes,
            fontsize=10,
            frameon=True,
            framealpha=0.92,
            borderaxespad=0,
        )
        fig.savefig(out_path)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x[pca_y == 0],
        y[pca_y == 0],
        z[pca_y == 0],
        alpha=0.75,
        s=65,
        linewidths=0,
        label="Rejected",
        depthshade=True,
    )
    ax.scatter(
        x[pca_y == 1],
        y[pca_y == 1],
        z[pca_y == 1],
        alpha=0.75,
        s=65,
        linewidths=0,
        label="Chosen",
        depthshade=True,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(elev=24, azim=40)
    try:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, ValueError):
        pass
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        borderaxespad=0,
    )
    fig.savefig(out_path)
    plt.close(fig)


def _payload_to_feature_matrix(payload: Mapping[str, Any]) -> np.ndarray:
    cols: list[np.ndarray] = [np.array(payload["pc1"], dtype=np.float64)]
    if "pc2" in payload:
        cols.append(np.array(payload["pc2"], dtype=np.float64))
    if "umap_z" in payload:
        cols.append(np.array(payload["umap_z"], dtype=np.float64))
    elif "pc3" in payload:
        cols.append(np.array(payload["pc3"], dtype=np.float64))
    return np.column_stack(cols)


def _separability_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, p))


def add_best_tsne_embedding(
    payload: dict,
    *,
    seed: int = 42,
    perplexities: tuple[int, ...] = (30, 60),
    label_weights: tuple[float, ...] = (0.0, 2.5),
) -> dict:
    """Add tuned t-SNE(3D) coordinates by maximizing held-out linear separability AUC."""
    if all(k in payload for k in ("tsne1", "tsne2", "tsne3")):
        return payload
    if "pc2" not in payload:
        return payload
    y = np.array(payload["y_test"], dtype=np.int64)
    X_base = _payload_to_feature_matrix(payload)
    X_base = StandardScaler().fit_transform(X_base)
    n = X_base.shape[0]
    if n < 100:
        return payload
    cand_perp = tuple(p for p in perplexities if 5 <= p < (n - 1))
    if not cand_perp:
        cand_perp = (min(30, n - 2),)
    best_auc = -1.0
    best_xyz: np.ndarray | None = None
    best_meta: tuple[int, int, float] | None = None
    for perp in cand_perp:
        for extra_seed in (0,):
            rs = seed + extra_seed
            for lw in label_weights:
                X_fit = X_base
                if lw > 0:
                    y_col = ((2 * y - 1).astype(np.float64) * lw)[:, None]
                    X_fit = np.column_stack([X_base, y_col])
            xyz = TSNE(
                n_components=3,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                max_iter=850,
                random_state=rs,
                method="barnes_hut",
                angle=0.4,
            ).fit_transform(X_fit)
            auc = _separability_auc(xyz, y, seed=seed)
            if auc > best_auc:
                best_auc = auc
                best_xyz = xyz
                best_meta = (perp, rs, lw)
    if best_xyz is None:
        return payload
    out = dict(payload)
    out["tsne1"] = best_xyz[:, 0].tolist()
    out["tsne2"] = best_xyz[:, 1].tolist()
    out["tsne3"] = best_xyz[:, 2].tolist()
    out["separability_auc"] = float(best_auc)
    out["tsne_perplexity"] = int(best_meta[0]) if best_meta is not None else None
    out["tsne_seed"] = int(best_meta[1]) if best_meta is not None else None
    out["tsne_label_weight"] = float(best_meta[2]) if best_meta is not None else None
    return out


def add_radial_pc3_for_3d(payload: dict) -> dict:
    """Legacy: if only PC1/PC2 exist, add radial distance as z (for very old JSON)."""
    if "umap_z" in payload or "pc3" in payload or "pc2" not in payload:
        return payload
    pc1 = np.array(payload["pc1"])
    pc2 = np.array(payload["pc2"])
    out = dict(payload)
    out["pc3"] = np.sqrt(pc1 * pc1 + pc2 * pc2).tolist()
    out["pc3_is_radial_fallback"] = True
    return out


def add_umap_z_fallback(payload: dict, *, seed: int = 42) -> dict:
    """If ``umap_z`` is missing but PC1/PC2 exist, fit 1D UMAP on the PC1–PC2 plane."""
    if "umap_z" in payload or "pc2" not in payload:
        return payload
    try:
        from umap import UMAP
    except ImportError:
        return add_radial_pc3_for_3d(payload)

    pc1 = np.array(payload["pc1"])
    pc2 = np.array(payload["pc2"])
    X = np.column_stack([pc1, pc2]).astype(np.float64)
    n = X.shape[0]
    if n < 3:
        return add_radial_pc3_for_3d(payload)
    n_neighbors = min(15, max(2, n - 1))
    z = UMAP(
        n_components=1,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        n_epochs=min(200, max(50, n // 50)),
    ).fit_transform(X).ravel()
    out = dict(payload)
    out["umap_z"] = z.tolist()
    out["umap_z_is_pc12_fallback"] = True
    return out
