from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def _load_sae_feature_module():
    path = Path(__file__).resolve().parent / "sae-feature.py"
    spec = importlib.util.spec_from_file_location("sae_feature_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sf = _load_sae_feature_module()


# Llama 3.2 3B Instruct family: base SAE + aligned HF models + SAE repo (for fallback best layer).
LLAMA_FAMILY: dict[str, Any] = {
    "family_slug": "Llama-3.2-3B-Instruct",
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
}


# Qwen3-4B Instruct family (layer 24 base SAE; per-checkpoint repos for probe fallback).
QWEN_FAMILY: dict[str, Any] = {
    "family_slug": "Qwen3-4B-Instruct-2507",
    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
}


# SmolLM3-3B family (layer 19 base SAE).
SMOLLM_FAMILY: dict[str, Any] = {
    "family_slug": "SmolLM3-3B",
    "base_model": "HuggingFaceTB/SmolLM3-3B",
}


ANCHOR_FAMILIES: dict[str, dict[str, Any]] = {
    "llama": LLAMA_FAMILY,
    "qwen": QWEN_FAMILY,
    "smollm": SMOLLM_FAMILY,
}


class _ResidualCapture:
    def __init__(self) -> None:
        self.residual: torch.Tensor | None = None

    def __call__(self, module, inputs, output):  # noqa: ANN001
        self.residual = output[0] if isinstance(output, tuple) else output


def _cuda_indices_for_empty_cache(devices: list[torch.device]) -> list[int]:
    """Unique CUDA device indices for cache clearing."""
    out: list[int] = []
    seen: set[int] = set()
    for d in devices:
        if d.type != "cuda":
            continue
        idx = torch.cuda.current_device() if d.index is None else int(d.index)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _maybe_empty_cuda_cache(devices: list[torch.device], enabled: bool) -> None:
    if not enabled or not torch.cuda.is_available():
        return
    for idx in _cuda_indices_for_empty_cache(devices):
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()


def _special_token_ids(tokenizer) -> set[int]:  # noqa: ANN001
    ids: set[int] = set()
    for attr in ("all_special_ids",):
        vals = getattr(tokenizer, attr, None)
        if vals:
            ids.update(int(x) for x in vals if x is not None)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            ids.add(int(val))
    return ids


def _add_anchor_candidate_masks(
    *,
    tokenizer,
    batches: list[dict[str, torch.Tensor]],
    skip_first_tokens: int,
    common_position_threshold: float,
) -> None:
    """Mark token positions eligible for anchor discovery.

    Chat templates can create extremely strong but uninteresting anchors at BOS or
    repeated instruction/prefix positions. The anchor mask keeps the fixed sample
    set tied to prompt content rather than tokenizer/template boilerplate.
    """
    input_ids = torch.cat([b["input_ids"] for b in batches], dim=0)
    attention = torch.cat([b["attention_mask"] for b in batches], dim=0).bool()
    eligible = attention.clone()

    for tid in _special_token_ids(tokenizer):
        eligible &= input_ids != tid

    if skip_first_tokens > 0:
        eligible[:, :skip_first_tokens] = False

    threshold = float(common_position_threshold)
    if 0.0 < threshold <= 1.0 and input_ids.shape[0] > 1:
        min_count = max(2, math.ceil(float(input_ids.shape[0]) * threshold))
        for pos in range(input_ids.shape[1]):
            active = attention[:, pos]
            if int(active.sum().item()) < min_count:
                continue
            ids_pos = input_ids[active, pos]
            _, counts = torch.unique(ids_pos, return_counts=True)
            if int(counts.max().item()) >= min_count:
                eligible[:, pos] = False

    start = 0
    for batch in batches:
        end = start + int(batch["input_ids"].shape[0])
        batch["anchor_mask"] = eligible[start:end]
        start = end


def _forward_residual(
    model,
    layer_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_device: torch.device,
) -> torch.Tensor:
    cap = _ResidualCapture()
    h = layer_module.register_forward_hook(cap)
    ids = input_ids.to(model_device, non_blocking=True)
    mask = attention_mask.to(model_device, non_blocking=True)
    try:
        # Backbone-only forward (avoid lm_head logits) to reduce VRAM.
        inner = getattr(model, "model", None)
        if inner is None:
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except TypeError:
                model(input_ids=ids, attention_mask=mask)
        else:
            try:
                inner(input_ids=ids, attention_mask=mask, use_cache=False)
            except TypeError:
                inner(input_ids=ids, attention_mask=mask)
        assert cap.residual is not None
        return cap.residual
    finally:
        h.remove()


def _feature_mean_argmax_scan(
    *,
    model,
    sae,
    layer_module,
    batches: list[dict[str, torch.Tensor]],
    model_device: torch.device,
    sae_device: torch.device,
    sae_dtype: torch.dtype,
    empty_cuda_cache: bool,
) -> tuple[int, torch.Tensor]:
    """Return (feature_id, mean_activations[d_sae]) over all non-pad tokens."""
    d_sae = int(sae.cfg.d_sae)
    sum_act = torch.zeros(d_sae, device=sae_device, dtype=torch.float64)
    n_tok = 0
    cache_devs = [model_device, sae_device]
    with torch.inference_mode():
        for batch in batches:
            resid = _forward_residual(
                model,
                layer_module,
                batch["input_ids"],
                batch["attention_mask"],
                model_device,
            )
            resid = resid.detach().to(device=sae_device, dtype=sae_dtype, non_blocking=True)
            feats = sae.encode(resid)
            mask = batch.get("anchor_mask", batch["attention_mask"]).to(sae_device).bool()
            feats_flat = feats[mask]
            if feats_flat.numel() == 0:
                continue
            sum_act += feats_flat.to(torch.float64).sum(dim=0)
            n_tok += int(feats_flat.shape[0])
            del resid, feats, feats_flat
            _maybe_empty_cuda_cache(cache_devs, empty_cuda_cache)
    mean = (sum_act / max(n_tok, 1)).to(torch.float32)
    fid = int(torch.argmax(mean).item())
    return fid, mean.cpu()


def _topk_positions_for_feature(
    *,
    model,
    sae,
    layer_module,
    batches: list[dict[str, torch.Tensor]],
    model_device: torch.device,
    sae_device: torch.device,
    sae_dtype: torch.dtype,
    feature_id: int,
    k: int,
    max_anchors_per_prompt: int,
    empty_cuda_cache: bool,
) -> list[tuple[float, torch.Tensor, torch.Tensor, int]]:
    """Global top-k (activation, input_ids_row, attention_mask_row, pos) over corpus."""
    heap: list[tuple[float, int, torch.Tensor, torch.Tensor, int]] = []
    uid = 0
    cache_devs = [model_device, sae_device]

    with torch.inference_mode():
        for batch in batches:
            bsz = batch["input_ids"].shape[0]
            resid = _forward_residual(
                model,
                layer_module,
                batch["input_ids"],
                batch["attention_mask"],
                model_device,
            )
            resid = resid.detach().to(device=sae_device, dtype=sae_dtype, non_blocking=True)
            feats = sae.encode(resid)
            mask = batch.get("anchor_mask", batch["attention_mask"]).bool()
            acts = feats[:, :, feature_id]
            for bi in range(bsz):
                row_acts = acts[bi].detach().float().cpu()
                row_mask = mask[bi].cpu().bool()
                valid_pos = torch.nonzero(row_mask, as_tuple=False).flatten()
                if valid_pos.numel() == 0:
                    continue
                valid_acts = row_acts[valid_pos]
                finite = torch.isfinite(valid_acts)
                if not bool(finite.any()):
                    continue
                valid_pos = valid_pos[finite]
                valid_acts = valid_acts[finite]
                per_prompt_k = int(valid_acts.numel())
                if max_anchors_per_prompt > 0:
                    per_prompt_k = min(per_prompt_k, int(max_anchors_per_prompt))
                top_vals, top_idx = torch.topk(valid_acts, k=per_prompt_k)
                row_ids = batch["input_ids"][bi].clone().cpu()
                row_m = batch["attention_mask"][bi].clone().cpu()
                for a_t, idx_t in zip(top_vals, top_idx, strict=False):
                    a = float(a_t.item())
                    if math.isnan(a) or math.isinf(a):
                        continue
                    pos = int(valid_pos[int(idx_t.item())].item())
                    if len(heap) < k:
                        heapq.heappush(heap, (a, uid, row_ids, row_m, pos))
                    elif a > heap[0][0]:
                        heapq.heapreplace(heap, (a, uid, row_ids, row_m, pos))
                    uid += 1
            del resid, feats, acts
            _maybe_empty_cuda_cache(cache_devs, empty_cuda_cache)

    heap.sort(key=lambda t: -t[0])
    return [(t[0], t[2], t[3], t[4]) for t in heap]


def _batched_readout_anchor(
    *,
    model,
    layer_module,
    base_sae,
    samples: list[tuple[torch.Tensor, torch.Tensor, int]],
    model_device: torch.device,
    sae_device: torch.device,
    sae_dtype: torch.dtype,
    feature_id: int,
    microbatch: int,
    empty_cuda_cache: bool,
) -> np.ndarray:
    """Encode residuals at hook layer with base_sae; return activations at anchor feature."""
    out: list[float] = []
    cache_devs = [model_device, sae_device]
    with torch.inference_mode():
        for start in range(0, len(samples), microbatch):
            chunk = samples[start : start + microbatch]
            max_t = max(int(x[0].shape[0]) for x in chunk)
            ids = torch.zeros(len(chunk), max_t, dtype=torch.long)
            msk = torch.zeros(len(chunk), max_t, dtype=torch.long)
            pos = []
            for i, (rid, rmk, p) in enumerate(chunk):
                t = int(rid.shape[0])
                ids[i, :t] = rid
                msk[i, :t] = rmk
                pos.append(p)
            resid = _forward_residual(model, layer_module, ids, msk, model_device)
            resid = resid.detach().to(device=sae_device, dtype=sae_dtype, non_blocking=True)
            feats = base_sae.encode(resid)
            pos_t = torch.tensor(pos, device=sae_device, dtype=torch.long)
            fi = torch.full_like(pos_t, feature_id, dtype=torch.long)
            bi = torch.arange(len(chunk), device=sae_device, dtype=torch.long)
            vals = feats[bi, pos_t, fi]
            out.extend(vals.detach().float().cpu().numpy().tolist())
            del resid, feats, vals
            _maybe_empty_cuda_cache(cache_devs, empty_cuda_cache)
    return np.array(out, dtype=np.float64)


def _batched_anchor_feature_matrix(
    *,
    model,
    layer_module,
    base_sae,
    samples: list[tuple[torch.Tensor, torch.Tensor, int]],
    model_device: torch.device,
    sae_device: torch.device,
    sae_dtype: torch.dtype,
    microbatch: int,
    empty_cuda_cache: bool,
) -> np.ndarray:
    """Encode residuals at saved positions; return [n_samples, d_sae] feature activations."""
    rows: list[np.ndarray] = []
    cache_devs = [model_device, sae_device]
    with torch.inference_mode():
        for start in range(0, len(samples), microbatch):
            chunk = samples[start : start + microbatch]
            max_t = max(int(x[0].shape[0]) for x in chunk)
            ids = torch.zeros(len(chunk), max_t, dtype=torch.long)
            msk = torch.zeros(len(chunk), max_t, dtype=torch.long)
            pos = []
            for i, (rid, rmk, p) in enumerate(chunk):
                t = int(rid.shape[0])
                ids[i, :t] = rid
                msk[i, :t] = rmk
                pos.append(p)
            resid = _forward_residual(model, layer_module, ids, msk, model_device)
            resid = resid.detach().to(device=sae_device, dtype=sae_dtype, non_blocking=True)
            feats = base_sae.encode(resid)
            pos_t = torch.tensor(pos, device=sae_device, dtype=torch.long)
            bi = torch.arange(len(chunk), device=sae_device, dtype=torch.long)
            rows.append(feats[bi, pos_t, :].detach().float().cpu().numpy())
            del resid, feats
            _maybe_empty_cuda_cache(cache_devs, empty_cuda_cache)
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def _feature_scan_summary(
    matrix: np.ndarray,
    *,
    anchor_feature_id: int,
    top_n: int,
) -> dict[str, Any]:
    """Summarize which SAE features are largest on the saved anchor positions."""
    mean = matrix.mean(axis=0, dtype=np.float64)
    std = matrix.std(axis=0, dtype=np.float64)
    median = np.median(matrix, axis=0)
    max_act = matrix.max(axis=0)
    top_n = max(1, min(int(top_n), int(matrix.shape[1])))
    top_mean_ids = np.argsort(-mean)[:top_n]
    top_max_ids = np.argsort(-max_act)[:top_n]
    anchor_mean = float(mean[anchor_feature_id])
    anchor_max = float(max_act[anchor_feature_id])

    def _records(feature_ids: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "feature_id": int(fid),
                "mean": float(mean[fid]),
                "std": float(std[fid]),
                "median": float(median[fid]),
                "max": float(max_act[fid]),
                "is_fixed_anchor": int(fid) == int(anchor_feature_id),
            }
            for fid in feature_ids
        ]

    return {
        "top_n": top_n,
        "fixed_anchor_feature_id": int(anchor_feature_id),
        "fixed_anchor_rank_by_mean": int(np.count_nonzero(mean > anchor_mean) + 1),
        "fixed_anchor_rank_by_max": int(np.count_nonzero(max_act > anchor_max) + 1),
        "fixed_anchor_stats": {
            "mean": anchor_mean,
            "std": float(std[anchor_feature_id]),
            "median": float(median[anchor_feature_id]),
            "max": anchor_max,
        },
        "top_by_mean": _records(top_mean_ids),
        "top_by_max": _records(top_max_ids),
    }


def _store_top_feature_vectors(
    stored_vectors: dict[str, np.ndarray],
    *,
    prefix: str,
    matrix: np.ndarray,
    summary: dict[str, Any],
) -> None:
    mean_ids = np.array([r["feature_id"] for r in summary["top_by_mean"]], dtype=np.int64)
    max_ids = np.array([r["feature_id"] for r in summary["top_by_max"]], dtype=np.int64)
    feature_ids = np.unique(np.concatenate([mean_ids, max_ids]))
    stored_vectors[f"{prefix}_candidate_feature_ids"] = feature_ids
    stored_vectors[f"{prefix}_candidate_feature_acts"] = matrix[:, feature_ids].astype(
        np.float32,
        copy=False,
    )


def _probe_best_layer(probe_roots: list[Path], model_id: str) -> int | None:
    sub = model_id.replace("/", "_")
    for root in probe_roots:
        p = root / sub / "layer_metrics.json"
        if not p.is_file():
            continue
        metrics = json.loads(p.read_text(encoding="utf-8"))
        if not metrics:
            continue
        best = max(metrics, key=lambda m: float(m.get("auroc", 0.0)))
        return int(best["layer"])
    return None


def _fallback_best_layer_from_sae_repo(sae_repo: str) -> int | None:
    return sf._parse_layer_from_repo(sae_repo)


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    sns.set_style("whitegrid", {"grid.alpha": 0.3})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/sae_anchor_study"),
        help="Parent directory; run writes under <root>/<family_slug>/<run_id>/",
    )
    p.add_argument(
        "--family",
        default="llama",
        choices=sorted(ANCHOR_FAMILIES.keys()),
        help="Model family: base model, base SAE, and aligned checkpoint list.",
    )
    p.add_argument("--run-id", default=None, help="Subfolder name; default: timestamp.")
    p.add_argument(
        "--model-device",
        default="cuda:0",
        help="Device for LM weights and forward (e.g. cuda:0).",
    )
    p.add_argument(
        "--sae-device",
        default=None,
        help="Device for base SAE; default: same as --model-device. Use cuda:1 to split VRAM.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="If set, overrides --model-device (legacy; e.g. Slurm --device cuda).",
    )
    p.add_argument("--model-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--sae-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset-split", default="train_sft")
    p.add_argument("--num-prompts", type=int, default=1200)
    p.add_argument(
        "--context-size",
        type=int,
        default=384,
        help="Max sequence length; lower if you OOM on ~24GB GPUs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forward batch size for corpus scan / top-k (use 1 on tight VRAM).",
    )
    p.add_argument("--anchor-k", type=int, default=256, help="Global top-K positions (≥200 recommended).")
    p.add_argument("--anchor-feature-id", type=int, default=None, help="Fix feature id; skip mean scan.")
    p.add_argument(
        "--anchor-skip-first-tokens",
        type=int,
        default=8,
        help="Exclude the first N token positions from anchor discovery to avoid BOS/template anchors.",
    )
    p.add_argument(
        "--anchor-common-position-threshold",
        type=float,
        default=0.8,
        help=(
            "Exclude absolute positions where the same token appears in this fraction of prompts; "
            "set 0 to disable boilerplate filtering."
        ),
    )
    p.add_argument(
        "--max-anchors-per-prompt",
        type=int,
        default=1,
        help="Maximum selected anchor token positions per prompt; <=0 disables this diversity cap.",
    )
    p.add_argument(
        "--candidate-feature-k",
        type=int,
        default=10,
        help="Top features to report when scanning all SAE latents at the saved anchor positions.",
    )
    p.add_argument(
        "--readout-microbatch",
        type=int,
        default=4,
        help="Batch size when re-running the K anchor sequences through each checkpoint.",
    )
    p.add_argument(
        "--empty-cuda-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.empty_cache() between micro-batches (reduces fragmentation).",
    )
    p.add_argument(
        "--linear-probe-root",
        type=Path,
        action="append",
        default=[],
        help="Directory containing <org_model>/layer_metrics.json (repeatable).",
    )
    return p.parse_args()


def main() -> None:
    sf._load_env()
    args = parse_args()
    model_dev_str = args.device if args.device is not None else args.model_device
    model_device = sf._resolve_device(model_dev_str)
    sae_dev_str = (args.sae_device or "").strip()
    sae_device = sf._resolve_device(sae_dev_str) if sae_dev_str else model_device
    model_dtype = sf._resolve_dtype(args.model_dtype)
    sae_dtype = sf._resolve_dtype(args.sae_dtype)

    fam = ANCHOR_FAMILIES[args.family]
    family_slug = str(fam["family_slug"])
    base_model = str(fam["base_model"])
    base_sae_repo = str(fam["base_sae_repo"])
    layer_base = sf._parse_layer_from_repo(base_sae_repo)
    if layer_base is None:
        raise SystemExit(f"Cannot parse layer from {base_sae_repo!r}")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root).expanduser().resolve() / family_slug / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(__file__).resolve().parent / "cache" / "sae_snapshots"
    cache.mkdir(parents=True, exist_ok=True)

    probe_roots = [Path(p).expanduser().resolve() for p in (args.linear_probe_root or [])]
    probe_roots.extend(
        [
            Path(__file__).resolve().parent / "output" / "linear-probes",
            Path(__file__).resolve().parent / "results" / "linear-probes",
            Path(__file__).resolve().parent / "outputs" / "linear-probes",
        ]
    )

    print(
        f"=== sae_anchor_transfer.py ===\n"
        f"  family: {args.family} ({family_slug})\n"
        f"  model_device: {model_device}  sae_device: {sae_device}\n"
        f"  out: {out_dir}",
        flush=True,
    )

    tokenizer = sf.load_tokenizer(base_model)
    print("[data] building prompts...", flush=True)
    prompts = list(
        sf.iter_chat_prompts(args.dataset, args.dataset_split, tokenizer, args.num_prompts)
    )
    if not prompts:
        raise SystemExit("No prompts; check dataset.")
    batches = sf.tokenize_prompts(tokenizer, prompts, args.context_size, args.batch_size)
    _add_anchor_candidate_masks(
        tokenizer=tokenizer,
        batches=batches,
        skip_first_tokens=max(0, int(args.anchor_skip_first_tokens)),
        common_position_threshold=float(args.anchor_common_position_threshold),
    )

    print("[load] base model + base SAE...", flush=True)
    base_lm = sf.load_base_model(base_model, model_device, model_dtype)
    base_layer_mod = sf.get_layer_module(base_lm, layer_base)
    sae_path = sf.download_sae(base_sae_repo, cache)
    base_sae = sf.load_sae(sae_path, sae_device, sae_dtype)

    if args.anchor_feature_id is not None:
        fid = int(args.anchor_feature_id)
        mean_vec = None
        print(f"[anchor] using fixed feature_id={fid}", flush=True)
    else:
        print("[scan] mean activation per feature (base, layer L_base)...", flush=True)
        fid, mean_vec = _feature_mean_argmax_scan(
            model=base_lm,
            sae=base_sae,
            layer_module=base_layer_mod,
            batches=batches,
            model_device=model_device,
            sae_device=sae_device,
            sae_dtype=sae_dtype,
            empty_cuda_cache=bool(args.empty_cuda_cache),
        )
        print(f"[anchor] argmax mean feature_id={fid}", flush=True)

    k = max(int(args.anchor_k), 1)
    print(f"[topk] collecting top-{k} activations for feature {fid}...", flush=True)
    top_rows = _topk_positions_for_feature(
        model=base_lm,
        sae=base_sae,
        layer_module=base_layer_mod,
        batches=batches,
        model_device=model_device,
        sae_device=sae_device,
        sae_dtype=sae_dtype,
        feature_id=fid,
        k=k,
        max_anchors_per_prompt=int(args.max_anchors_per_prompt),
        empty_cuda_cache=bool(args.empty_cuda_cache),
    )
    base_acts = np.array([t[0] for t in top_rows], dtype=np.float64)
    samples = [(t[1], t[2], t[3]) for t in top_rows]
    candidate_top_n = max(1, int(args.candidate_feature_k))
    stored_vectors: dict[str, np.ndarray] = {}

    print(
        f"[scan] full feature activations at base anchor positions (top-{candidate_top_n})...",
        flush=True,
    )
    base_feature_matrix = _batched_anchor_feature_matrix(
        model=base_lm,
        layer_module=base_layer_mod,
        base_sae=base_sae,
        samples=samples,
        model_device=model_device,
        sae_device=sae_device,
        sae_dtype=sae_dtype,
        microbatch=int(args.readout_microbatch),
        empty_cuda_cache=bool(args.empty_cuda_cache),
    )
    base_feature_scan = _feature_scan_summary(
        base_feature_matrix,
        anchor_feature_id=fid,
        top_n=candidate_top_n,
    )
    _store_top_feature_vectors(
        stored_vectors,
        prefix="feature_scan_base_Lbase",
        matrix=base_feature_matrix,
        summary=base_feature_scan,
    )
    del base_feature_matrix

    # Save tensors + JSONL snippets
    ids_stack = torch.stack([s[0] for s in samples], dim=0)
    msk_stack = torch.stack([s[1] for s in samples], dim=0)
    pos_arr = torch.tensor([s[2] for s in samples], dtype=torch.long)
    torch.save(
        {
            "input_ids": ids_stack,
            "attention_mask": msk_stack,
            "positions": pos_arr,
            "activations_base_layer": torch.from_numpy(base_acts),
            "anchor_feature_id": fid,
            "layer_base": layer_base,
        },
        out_dir / "anchor_samples.pt",
    )

    jsonl_path = out_dir / "anchor_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, (a, rid, rmk, pos) in enumerate(top_rows):
            snippet = tokenizer.decode(
                rid[max(0, pos - 32) : min(len(rid), pos + 32)].tolist(),
                skip_special_tokens=True,
            )
            jf.write(
                json.dumps(
                    {"rank": i, "activation": a, "position": pos, "snippet": snippet},
                    ensure_ascii=False,
                )
                + "\n"
            )

    del base_lm
    _maybe_empty_cuda_cache([model_device, sae_device], True)

    results: dict[str, Any] = {
        "anchor_family": args.family,
        "family_slug": family_slug,
        "base_model": base_model,
        "model_device": str(model_device),
        "sae_device": str(sae_device),
        "base_sae_repo": base_sae_repo,
        "sae_training_layer": layer_base,
        "anchor_feature_id": fid,
        "anchor_k": k,
        "anchor_skip_first_tokens": max(0, int(args.anchor_skip_first_tokens)),
        "anchor_common_position_threshold": float(args.anchor_common_position_threshold),
        "max_anchors_per_prompt": int(args.max_anchors_per_prompt),
        "readout_note": (
            "Values are base_sae.encode(residual)[..., anchor_feature_id] on each checkpoint's "
            "hidden states. The SAE was trained on base model layer L_base; readout at L_best "
            "is exploratory when L_best != L_base."
        ),
        "feature_scan_note": (
            "For each checkpoint/layer, the full base-SAE feature vector is also scanned at the "
            "same saved token positions. This tests whether the largest activations remain on "
            "the fixed base anchor feature or shift to other SAE feature indices."
        ),
        "base_reference_activations": {
            "layer": layer_base,
            "mean": float(base_acts.mean()),
            "std": float(base_acts.std()),
            "median": float(np.median(base_acts)),
        },
        "base_anchor_position_feature_scan": base_feature_scan,
        "checkpoints": {},
    }

    aligned_entries = list(fam["aligned"])
    for entry in aligned_entries:
        algo = entry["algo"]
        mid = entry["model"]
        srepo = entry["sae_repo"]
        L_best = _probe_best_layer(probe_roots, mid)
        if L_best is None:
            L_best = _fallback_best_layer_from_sae_repo(srepo)
        if L_best is None:
            print(f"[warn] no best layer for {mid}; skip", flush=True)
            continue

        print(f"\n[{algo}] model={mid}  L_same={layer_base}  L_best={L_best}", flush=True)
        lm = sf.load_base_model(mid, model_device, model_dtype)
        layer_same = sf.get_layer_module(lm, layer_base)
        layer_best = sf.get_layer_module(lm, L_best)

        print(f"[{algo}] scanning all features at L{layer_base} anchor positions...", flush=True)
        matrix_same = _batched_anchor_feature_matrix(
            model=lm,
            layer_module=layer_same,
            base_sae=base_sae,
            samples=samples,
            model_device=model_device,
            sae_device=sae_device,
            sae_dtype=sae_dtype,
            microbatch=int(args.readout_microbatch),
            empty_cuda_cache=bool(args.empty_cuda_cache),
        )
        acts_same = matrix_same[:, fid].astype(np.float64, copy=False)
        feature_scan_same = _feature_scan_summary(
            matrix_same,
            anchor_feature_id=fid,
            top_n=candidate_top_n,
        )
        _store_top_feature_vectors(
            stored_vectors,
            prefix=f"feature_scan_{algo}_same_L{layer_base}",
            matrix=matrix_same,
            summary=feature_scan_same,
        )

        print(f"[{algo}] scanning all features at L{L_best} anchor positions...", flush=True)
        matrix_best = _batched_anchor_feature_matrix(
            model=lm,
            layer_module=layer_best,
            base_sae=base_sae,
            samples=samples,
            model_device=model_device,
            sae_device=sae_device,
            sae_dtype=sae_dtype,
            microbatch=int(args.readout_microbatch),
            empty_cuda_cache=bool(args.empty_cuda_cache),
        )
        acts_best = matrix_best[:, fid].astype(np.float64, copy=False)
        feature_scan_best = _feature_scan_summary(
            matrix_best,
            anchor_feature_id=fid,
            top_n=candidate_top_n,
        )
        _store_top_feature_vectors(
            stored_vectors,
            prefix=f"feature_scan_{algo}_best_L{L_best}",
            matrix=matrix_best,
            summary=feature_scan_best,
        )

        def _summ(x: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
            return {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "median": float(np.median(x)),
                "mean_delta_vs_base_reference": float(x.mean() - ref.mean()),
                "median_delta_vs_base_reference": float(np.median(x) - np.median(ref)),
                "fraction_below_base_median": float(np.mean(x < np.median(ref))),
                "fraction_above_base_median": float(np.mean(x >= np.median(ref))),
            }

        results["checkpoints"][algo] = {
            "model_id": mid,
            "linear_probe_best_layer": L_best,
            "sae_repo_for_layer_fallback": srepo,
            "same_layer_as_sae_training": {
                "layer": layer_base,
                **_summ(acts_same, base_acts),
                "feature_scan_on_anchor_positions": feature_scan_same,
            },
            "at_linear_probe_best_layer": {
                "layer": L_best,
                "mismatched_sae_training_layer": L_best != layer_base,
                **_summ(acts_best, base_acts),
                "feature_scan_on_anchor_positions": feature_scan_best,
            },
        }
        stored_vectors[f"acts_{algo}_L{layer_base}"] = acts_same
        stored_vectors[f"acts_{algo}_L{L_best}_best"] = acts_best
        del matrix_same, matrix_best

        del lm
        _maybe_empty_cuda_cache([model_device, sae_device], True)

    (out_dir / "activation_comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    npz_payload: dict[str, Any] = {
        "base_activations": base_acts,
        "positions": pos_arr.numpy(),
        "anchor_feature_id": np.int64(fid),
        "layer_base": np.int64(layer_base),
        "candidate_feature_k": np.int64(candidate_top_n),
        **stored_vectors,
    }
    if mean_vec is not None:
        npz_payload["scan_feature_means_on_base"] = mean_vec.numpy().astype(np.float32)
    np.savez(out_dir / "replot_arrays.npz", **npz_payload)

    # Plots
    _configure_plots()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    algos = list(results["checkpoints"].keys())
    labels: list[str] = [f"base@{layer_base}"]
    data_cols: list[np.ndarray] = [base_acts]
    for algo in algos:
        Lb = int(results["checkpoints"][algo]["linear_probe_best_layer"])
        labels.append(f"{algo}@L{layer_base}")
        data_cols.append(stored_vectors[f"acts_{algo}_L{layer_base}"])
        labels.append(f"{algo}@L{Lb}(probe)")
        data_cols.append(stored_vectors[f"acts_{algo}_L{Lb}_best"])

    fig, ax = plt.subplots(figsize=(max(10.0, 0.45 * len(labels)), 5.5))
    try:
        ax.boxplot(data_cols, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data_cols, labels=labels, showfliers=False)
    ax.set_ylabel("Base-SAE anchor activation")
    ax.set_title(
        f"Anchor feature {fid} ({family_slug})\n"
        "Readout: base_sae.encode(residual)[..., f]"
    )
    ax.tick_params(axis="x", rotation=55, labelsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    fig.tight_layout()
    fig.savefig(plot_dir / "boxplot_anchor_readouts.pdf")
    plt.close(fig)

    meta = {
        "output_dir": str(out_dir),
        "anchor_family": args.family,
        "family_slug": family_slug,
        "run_id": run_id,
        "anchor_feature_id": fid,
        "anchor_k": k,
        "anchor_skip_first_tokens": max(0, int(args.anchor_skip_first_tokens)),
        "anchor_common_position_threshold": float(args.anchor_common_position_threshold),
        "max_anchors_per_prompt": int(args.max_anchors_per_prompt),
        "candidate_feature_k": candidate_top_n,
        "layer_base": layer_base,
        "model_device": str(model_device),
        "sae_device": str(sae_device),
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "num_prompts": len(prompts),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        os._exit(0)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        sys.stdout.flush()
        os._exit(code)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
