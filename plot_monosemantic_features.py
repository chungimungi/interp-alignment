"""Extract and plot candidate monosemantic SAE features from saved artifacts.

This script reads an existing ``sae-feature.py`` output directory containing
``feature_descriptions.json`` and ``replot_metadata.npz``. It does not load the
base model or SAE.

Use ``--feature-ids`` for manually curated features, or omit it to rank
candidates with a simple evidence heuristic based on repeated keywords in the
top activating examples.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 16.0,
            "axes.titlesize": 18.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 13.0,
            "legend.title_fontsize": 13.0,
            "axes.linewidth": 1.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 7.0,
            "figure.dpi": 600,
            "savefig.dpi": 600,
        }
    )
    sns.set_style("whitegrid", {"grid.alpha": 0.25, "axes.edgecolor": "0.15"})


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
    "are", "was", "were", "have", "has", "had", "but", "not", "can", "will",
    "may", "its", "their", "about", "such", "more", "most", "some", "all",
    "one", "two", "use", "using", "used", "make", "provide", "write", "user",
    "assistant", "system", "code", "math", "solution", "section", "steps", "step",
    "reasoning", "thought", "thinking", "conclusion", "detail", "details",
    "revisiting", "verifying", "exploration", "reassessment", "deem",
}

FORMAT_WORDS = {
    "begin", "text", "date", "system", "metadata", "cutting", "knowledge",
    "today", "start", "user", "assistant", "im_start", "begin_of_text",
    "solution", "section", "reasoning", "thought", "thinking", "conclusion",
    "revisiting", "verifying", "exploration", "reassessment", "deem",
    "metadata", "cutoff", "hugging", "smollm",
}


def tokenize(text: str) -> List[str]:
    return [
        w
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        if w not in STOPWORDS
    ]


def load_artifacts(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    json_path = out_dir / "feature_descriptions.json"
    npz_path = out_dir / "replot_metadata.npz"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(npz_path, allow_pickle=True))
    return meta, arrays


def score_feature(row: Dict[str, Any]) -> Dict[str, Any]:
    examples = row.get("top_examples", [])
    if not examples:
        return {"score": 0.0, "keyword_coverage": 0.0, "top_word_share": 0.0, "is_format": False}

    keywords = [str(k).lower() for k in row.get("keywords", []) if str(k).strip()]
    desc = str(row.get("description", "")).lower()
    fired = " ".join(str(ex.get("fired_token", "")) for ex in examples).lower()
    is_format = any(w in desc or w in fired for w in FORMAT_WORDS)

    example_texts = [str(ex.get("text", "")).lower() for ex in examples]
    keyword_hits = 0
    if keywords:
        keyword_hits = sum(any(k in text for k in keywords) for text in example_texts)
    keyword_coverage = keyword_hits / max(len(example_texts), 1)

    words: List[str] = []
    for text in example_texts:
        words.extend(tokenize(text))
    counts: Dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    top_word_share = max(counts.values(), default=0) / max(len(example_texts), 1)

    density = float(row.get("density", 0.0))
    density_penalty = max(0.0, density - 0.25) * 0.35
    format_penalty = 0.35 if is_format else 0.0
    score = keyword_coverage + min(top_word_share / 3.0, 0.35) - density_penalty - format_penalty

    return {
        "score": float(score),
        "keyword_coverage": float(keyword_coverage),
        "top_word_share": float(top_word_share),
        "is_format": bool(is_format),
    }


def choose_features(
    rows: List[Dict[str, Any]],
    feature_ids: Optional[Set[int]],
    top_k: int,
    include_format: bool,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        metrics = score_feature(row)
        row = {**row, **metrics}
        if feature_ids is not None:
            if int(row["feature_id"]) in feature_ids:
                scored.append(row)
            continue
        if not include_format and row["is_format"]:
            continue
        scored.append(row)
    if feature_ids is not None:
        return scored
    return sorted(scored, key=lambda r: float(r["score"]), reverse=True)[:top_k]


def write_tables(selected: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "monosemantic_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id", "score", "mean_act", "density", "description",
                "keyword_coverage", "top_word_share", "is_format",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    json_path = out_dir / "monosemantic_candidates.json"
    json_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")


def plot_density(
    meta: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    selected: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    mean_act = arrays["mean_act"]
    density = arrays["density"]
    selected_ids = np.array([int(r["feature_id"]) for r in selected], dtype=np.int64)
    alive = mean_act > 0

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    ax.scatter(density[alive], mean_act[alive], s=7, alpha=0.22, color="0.55", label="All active features")
    ax.scatter(
        density[selected_ids],
        mean_act[selected_ids],
        s=78,
        alpha=0.95,
        color="#cc4125",
        edgecolor="black",
        linewidth=0.8,
        label="Candidate monosemantic",
    )
    for fid in selected_ids:
        ax.annotate(str(fid), (density[fid], mean_act[fid]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"Activation density $\left(\mathbb{E}_t[\mathbf{1}\{a_t>0\}]\right)$")
    ax.set_ylabel("Mean activation")
    ax.set_title(f"Candidate Monosemantic Features\n{short_model_name(meta)} Layer {meta.get('layer')}")
    ax.legend(frameon=True, fancybox=True, framealpha=0.95, loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "monosemantic_density_vs_mean.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bars(selected: List[Dict[str, Any]], out_dir: Path) -> None:
    if not selected:
        return
    labels = [f"{int(r['feature_id'])}\n{str(r.get('description', ''))[:26]}" for r in selected]
    means = [float(r.get("mean_act", 0.0)) for r in selected]
    scores = [float(r.get("score", 0.0)) for r in selected]

    fig, axes = plt.subplots(2, 1, figsize=(max(8.2, 0.66 * len(selected) + 3.2), 7.0), layout="constrained")
    x = np.arange(len(selected))
    axes[0].bar(x, means, color=sns.color_palette("viridis", len(selected)), edgecolor="black")
    axes[0].set_ylabel("Mean activation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=58, ha="right", fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, scores, color=sns.color_palette("mako", len(selected)), edgecolor="black")
    axes[1].set_ylabel("Candidate score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=58, ha="right", fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "monosemantic_candidate_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_example_heatmap(
    selected: List[Dict[str, Any]],
    out_dir: Path,
    top_examples: int,
) -> None:
    if not selected:
        return
    matrix = np.zeros((top_examples, len(selected)), dtype=np.float32)
    for col, row in enumerate(selected):
        for ridx, ex in enumerate(row.get("top_examples", [])[:top_examples]):
            matrix[ridx, col] = float(ex.get("activation", 0.0))

    fig, ax = plt.subplots(figsize=(max(7.4, 0.42 * len(selected) + 3.2), 5.4), layout="constrained")
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="magma",
        cbar_kws={"label": "Activation"},
        xticklabels=[str(int(r["feature_id"])) for r in selected],
        yticklabels=[f"ex {i + 1}" for i in range(top_examples)],
    )
    ax.set_xlabel("SAE feature ID")
    ax.set_ylabel("Top firing examples")
    ax.set_title("Candidate Feature Activations")
    fig.savefig(out_dir / "monosemantic_top_examples_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def short_model_name(meta: Dict[str, Any]) -> str:
    base_model = str(meta.get("base_model", "model"))
    name = base_model.rsplit("/", 1)[-1]
    suffix = "-merged"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def main() -> None:
    configure_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="SAE output folder containing feature_descriptions.json")
    parser.add_argument("--feature-ids", nargs="*", type=int, default=None, help="Manual feature IDs to extract")
    parser.add_argument("--top-k", type=int, default=8, help="Number of auto-ranked candidates")
    parser.add_argument("--top-examples", type=int, default=12)
    parser.add_argument("--include-format", action="store_true", help="Keep template/date/chat-token features")
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_dir.expanduser().resolve()
    meta, arrays = load_artifacts(out_root)
    rows = list(meta.get("features", []))
    selected = choose_features(
        rows,
        set(args.feature_ids) if args.feature_ids is not None else None,
        args.top_k,
        args.include_format,
    )

    plot_dir = args.plot_dir or (out_root / "plots" / "monosemantic")
    write_tables(selected, plot_dir)
    plot_density(meta, arrays, selected, plot_dir)
    plot_bars(selected, plot_dir)
    plot_example_heatmap(selected, plot_dir, args.top_examples)

    print(f"wrote {len(selected)} candidate feature(s) to {plot_dir}")
    for row in selected:
        print(
            f"{int(row['feature_id']):>6d}  score={float(row['score']):.3f}  "
            f"mean={float(row['mean_act']):.4f}  density={float(row['density']):.4f}  "
            f"{row.get('description', '')}"
        )


if __name__ == "__main__":
    main()
