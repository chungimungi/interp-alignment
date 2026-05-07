from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import dotenv
import numpy as np
import torch
from huggingface_hub import snapshot_download

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DEFAULT_SAE_CATALOG: dict[str, str] = {
    # Llama-3.2-3B family.
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-DPO-merged-layer_13_best": "MInAlA/Llama-3.2-3B-DPO-merged",
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-Instruct-PPO-merged-layer_11_best": "MInAlA/Llama-3.2-3B-Instruct-PPO-merged",
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-Instruct-KTO-merged-layer_24_best": "MInAlA/Llama-3.2-3B-Instruct-KTO-merged",
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-Instruct-GRPO-merged-layer_13_best": "MInAlA/Llama-3.2-3B-Instruct-GRPO-merged",
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-ORPO-merged-layer_25_best": "MInAlA/Llama-3.2-3B-ORPO-merged",
    "chungimungi/SAE-MInAlA_Llama-3.2-3B-SimPO-merged-layer_11_best": "MInAlA/Llama-3.2-3B-SimPO-merged",
    "chungimungi/SAE-meta-llama_Llama-3.2-3B-Instruct-layer_11_best": "meta-llama/Llama-3.2-3B-Instruct",
    # Qwen3-4B family.
    "chungimungi/SAE-MInAlA_Qwen3-4B-Instruct-2507-GRPO-merged-layer_20_best": "MInAlA/Qwen3-4B-Instruct-2507-GRPO-merged",
    "chungimungi/SAE-MInAlA_Qwen3-4B-Instruct-2507-DPO-merged-layer_22_best": "MInAlA/Qwen3-4B-Instruct-2507-DPO-merged",
    "chungimungi/SAE-MInAlA_Qwen3-4B-Instruct-2507-SimPO-merged-layer_21_best": "MInAlA/Qwen3-4B-Instruct-2507-SimPO-merged",
    "chungimungi/SAE-MInAlA_Qwen3-4B-Instruct-2507-KTO-merged-layer_24_best": "MInAlA/Qwen3-4B-Instruct-2507-KTO-merged",
    "chungimungi/SAE-MInAlA_Qwen3-4B-Instruct-2507-PPO-merged-layer_21_best": "MInAlA/Qwen3-4B-Instruct-2507-PPO-merged",
    "chungimungi/SAE-MInAlA_Qwen3-4B-ORPO-merged-layer_22_best": "MInAlA/Qwen3-4B-ORPO-merged",
    "chungimungi/SAE-Qwen_Qwen3-4B-Instruct-2507-layer_24_best": "Qwen/Qwen3-4B-Instruct-2507",
    # SmolLM3-3B family.
    "chungimungi/SAE-HuggingFaceTB_SmolLM3-3B-layer_19_best": "HuggingFaceTB/SmolLM3-3B",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-GRPO-merged-layer_17_best": "MInAlA/SmolLM3-3B-GRPO-merged",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-DPO-merged-layer_18_best": "MInAlA/SmolLM3-3B-DPO-merged",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-SimPO-merged-layer_18_best": "MInAlA/SmolLM3-3B-SimPO-merged",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-PPO-merged-layer_18_best": "MInAlA/SmolLM3-3B-PPO-merged",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-ORPO-merged-layer_18_best": "MInAlA/SmolLM3-3B-ORPO-merged",
    "chungimungi/SAE-MInAlA_SmolLM3-3B-KTO-merged-layer_19_best": "MInAlA/SmolLM3-3B-KTO-merged",
}

LAYER_RE = re.compile(r"[-_]layer_(\d+)_(best|mid)$", re.IGNORECASE)

# Stop-words / function tokens we filter out when summarising top contexts.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "as", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "our", "your", "their",
    "do", "does", "did", "doing", "done", "have", "has", "had", "having", "not",
    "no", "yes", "so", "such", "than", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "can", "could", "should", "would", "will",
    "may", "might", "must", "shall", "into", "out", "up", "down", "over",
    "under", "about", "after", "before", "more", "most", "less", "least", "some",
    "any", "all", "each", "every", "few", "many", "much", "one", "two", "first",
    "second", "also", "just", "only", "other", "another", "same", "different",
    "very", "really", "quite", "now", "here", "there", "well", "even", "still",
    "yet", "than", "while", "because", "since", "though", "although", "however",
    "therefore", "thus", "however", "moreover",
}

_CODE_HINTS = {
    "def", "class", "import", "from", "return", "if", "else", "elif", "while",
    "for", "lambda", "yield", "with", "try", "except", "raise", "self", "None",
    "True", "False", "var", "let", "const", "function", "void", "int", "float",
    "string", "public", "private", "static", "==", "!=", "->", "::", "=>", "{",
    "}", ";", "();",
}

_REFUSAL_HINTS = {
    "cannot", "can't", "won't", "unable", "refuse", "refusing", "decline",
    "sorry", "apolog", "i'm sorry", "as an ai", "i do not", "not appropriate",
    "harmful", "illegal", "unsafe", "dangerous", "policy", "safety",
}

_POLITENESS_HINTS = {
    "please", "thank", "thanks", "kindly", "appreciate", "would you", "could you",
    "if you don't mind", "feel free",
}

_MATH_HINTS = {
    "+", "-", "*", "/", "=", "sum", "integral", "derivative", "matrix",
    "equation", "theorem", "proof", "log", "sin", "cos", "tan", "sqrt",
    "lim", "infty", "x_", "y_", "n_",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load HF_TOKEN from .env files near this script (prefer interp-alignment/.env)."""
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            dotenv.load_dotenv(dotenv_path=str(p), override=False)


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _plot_short_model_name(base_model_id: str) -> str:
    """Short label for figure subtitles: drop HF org, strip trailing ``-merged``."""
    tail = base_model_id.rsplit("/", 1)[-1]
    return tail.removesuffix("-merged")


def _iter_sae_feature_output_dirs(root: Path) -> list[Path]:
    """Directories that contain both ``replot_metadata.npz`` and ``feature_descriptions.json``."""
    found: set[Path] = set()
    for npz in root.rglob("replot_metadata.npz"):
        d = npz.parent
        if (d / "feature_descriptions.json").is_file():
            found.add(d)
    return sorted(found)


def replot_sae_feature_plots_from_artifacts(out_root: Path) -> None:
    """Regenerate the three PDFs from saved JSON + NPZ (no model / SAE load)."""
    json_path = out_root / "feature_descriptions.json"
    npz_path = out_root / "replot_metadata.npz"
    if not json_path.is_file():
        raise FileNotFoundError(str(json_path))
    if not npz_path.is_file():
        raise FileNotFoundError(str(npz_path))
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    base_model_id = str(meta["base_model"])
    layer_idx = int(meta["layer"])
    rows: list[dict[str, Any]] = meta["features"]
    z = np.load(npz_path, allow_pickle=True)
    mean_act = z["mean_act"]
    density = z["density"]
    selected = z["selected_feature_ids"]
    top_acts_matrix = z["top_acts_matrix"]
    plot_label = _plot_short_model_name(base_model_id)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_top_features_bar(
        feature_ids=np.array([int(r["feature_id"]) for r in rows], dtype=np.int64),
        mean_acts=np.array([float(r["mean_act"]) for r in rows], dtype=np.float32),
        descriptions=[str(r["description"]) for r in rows],
        title=(
            f"Top {len(rows)} features by mean activation\n"
            f"{plot_label} (layer {layer_idx})"
        ),
        out_path=plots_dir / "top_features_mean_activation",
    )
    plot_density_vs_mean(
        mean_acts=mean_act,
        density=density,
        selected=selected,
        title=(
            f"Feature density vs. mean activation\n{plot_label} (layer {layer_idx})"
        ),
        out_path=plots_dir / "density_vs_mean",
    )
    plot_top_feature_activation_heatmap(
        top_feature_ids=np.array([int(r["feature_id"]) for r in rows], dtype=np.int64),
        top_acts_matrix=top_acts_matrix,
        title=(
            "Top examples × top features (activation magnitude)\n"
            f"{plot_label} (layer {layer_idx})"
        ),
        out_path=plots_dir / "top_examples_heatmap",
    )
    print(f"  replotted PDFs under {plots_dir}", flush=True)


def _outputs_complete(out_root: Path, *, baseline_npz: str | None) -> bool:
    """True when a prior run left all artefacts we would write (skip heavy work)."""
    paths: list[Path] = [
        out_root / "feature_descriptions.json",
        out_root / "feature_descriptions.txt",
        out_root / "replot_metadata.npz",
        out_root / "plots" / "top_features_mean_activation.pdf",
        out_root / "plots" / "density_vs_mean.pdf",
        out_root / "plots" / "top_examples_heatmap.pdf",
    ]
    for p in paths:
        try:
            if not p.is_file() or p.stat().st_size == 0:
                return False
        except OSError:
            return False
    if baseline_npz is not None:
        dpath = out_root / "deltas.json"
        try:
            if not dpath.is_file() or dpath.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _parse_layer_from_repo(repo_id: str) -> int | None:
    m = LAYER_RE.search(repo_id)
    if m is None:
        return None
    return int(m.group(1))


def _resolve_base_model(sae_repo: str, override: str | None) -> str:
    if override:
        return override
    if sae_repo in DEFAULT_SAE_CATALOG:
        return DEFAULT_SAE_CATALOG[sae_repo]
    raise SystemExit(
        f"Unknown SAE repo {sae_repo!r}; pass --base-model explicitly or add it to "
        "DEFAULT_SAE_CATALOG."
    )


def _needs_trust_remote_code(model_id: str) -> bool:
    low = model_id.lower()
    return "smollm" in low


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def download_sae(repo_id: str, cache_root: Path) -> Path:
    """Snapshot the SAE repo locally (uses HF cache); returns the local path."""
    cache_root.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo_id,
        token=_hf_token(),
        cache_dir=str(cache_root),
        allow_patterns=[
            "cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
            "*.json",  # be permissive in case of additional metadata
        ],
    )
    return Path(local)


def load_sae(local_path: Path, device: torch.device, dtype: torch.dtype):
    """Load the SAE from a local snapshot via sae_lens."""
    from sae_lens import SAE  # local import: keeps script importable without sae_lens.

    dtype_str = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]
    sae = SAE.load_from_disk(str(local_path), device=str(device), dtype=dtype_str)
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Tokenizer + base model loading (handles broken Qwen/Llama tokenizer repos).
# ---------------------------------------------------------------------------

_TOKENIZER_FALLBACKS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}


def _tokenizer_fallback_id(model_id: str) -> str | None:
    low = model_id.lower()
    for key, target in _TOKENIZER_FALLBACKS.items():
        if key in low:
            return target
    return None


def load_tokenizer(model_id: str):
    """Load the tokenizer with a chat-template-aware fallback."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    token = _hf_token()
    trust = _needs_trust_remote_code(model_id)
    fallbacks = [model_id]
    fb = _tokenizer_fallback_id(model_id)
    if fb and fb not in fallbacks:
        fallbacks.append(fb)
    last_err: BaseException | None = None
    for src in fallbacks:
        for use_fast in (True, False):
            try:
                tok = AutoTokenizer.from_pretrained(
                    src,
                    use_fast=use_fast,
                    token=token,
                    trust_remote_code=trust,
                )
                if getattr(tok, "chat_template", None):
                    return tok
                last_err = ValueError(f"{src!r}: no chat_template")
            except BaseException as e:  # noqa: BLE001
                last_err = e
        try:
            tok = PreTrainedTokenizerFast.from_pretrained(src, token=token)
            if getattr(tok, "chat_template", None):
                return tok
        except BaseException as e:  # noqa: BLE001
            last_err = e
    assert last_err is not None
    raise last_err


def load_base_model(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=_hf_token(),
        trust_remote_code=_needs_trust_remote_code(model_id),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model


def get_layer_module(model, layer_idx: int):
    """Return the HF submodule that corresponds to ``model.layers[layer_idx]``."""
    inner = getattr(model, "model", None) or model
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            f"Cannot locate ``model.layers`` on {type(model).__name__}; the SAE script "
            "currently assumes the standard HF Llama / Qwen / SmolLM3 layout."
        )
    return layers[layer_idx]


# ---------------------------------------------------------------------------
# Dataset materialisation (chat-formatted prompts).
# ---------------------------------------------------------------------------

def iter_chat_prompts(
    dataset_id: str,
    split: str,
    tokenizer,
    max_prompts: int,
) -> Iterable[str]:
    """Stream ``max_prompts`` chat-formatted strings from a HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    yielded = 0
    for row in ds:
        messages = _row_to_messages(row)
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:  # noqa: BLE001
            continue
        if not text:
            continue
        yield text
        yielded += 1
        if yielded >= max_prompts:
            break


def _row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Best-effort conversion of an HF row into ``[{role, content}, ...]``."""
    # 1) UltraChat: ``data`` is a list of alternating user / assistant strings.
    raw = row.get("data")
    if isinstance(raw, list) and raw:
        msgs: list[dict[str, str]] = []
        for i, piece in enumerate(raw):
            if isinstance(piece, dict):
                content = piece.get("text") or piece.get("content") or str(piece)
            else:
                content = str(piece) if piece is not None else ""
            content = content.strip()
            if not content:
                return None
            msgs.append(
                {"role": "user" if (i % 2 == 0) else "assistant", "content": content}
            )
        return msgs

    # 2) Generic ``messages`` / ``conversation`` columns.
    for key in ("messages", "conversation", "conversations"):
        cell = row.get(key)
        if isinstance(cell, list) and cell:
            msgs = []
            for m in cell:
                if not isinstance(m, dict):
                    return None
                role = m.get("role")
                content = m.get("content") or m.get("value")
                if isinstance(content, list):
                    content = "".join(
                        str(c.get("text", c)) if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if not isinstance(role, str) or not isinstance(content, str):
                    return None
                msgs.append({"role": role, "content": content})
            return msgs or None

    # 3) UltraFeedback: ``chosen`` / ``rejected`` are message lists.
    for key in ("chosen", "rejected"):
        cell = row.get(key)
        if isinstance(cell, list) and cell:
            msgs = _row_to_messages({"messages": cell})
            if msgs:
                return msgs

    # 4) Plain text column.
    text = row.get("text") or row.get("prompt")
    if isinstance(text, str) and text.strip():
        return [{"role": "user", "content": text.strip()}]

    return None


def tokenize_prompts(
    tokenizer,
    prompts: list[str],
    context_size: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Pad prompts into fixed-size tensors batched for the model forward pass."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts,
        truncation=True,
        max_length=context_size,
        padding="max_length",
        return_tensors="pt",
    )
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, encoded["input_ids"].shape[0], batch_size):
        end = start + batch_size
        batches.append(
            {
                "input_ids": encoded["input_ids"][start:end],
                "attention_mask": encoded["attention_mask"][start:end],
            }
        )
    return batches


# ---------------------------------------------------------------------------
# Activation hook + streaming top-k tracker.
# ---------------------------------------------------------------------------

class _ResidualCapture:
    """Forward hook that captures the residual stream output of a HF decoder layer."""

    def __init__(self):
        self.residual: torch.Tensor | None = None

    def __call__(self, module, inputs, output):  # noqa: D401, ANN001
        # HF decoder layers return a tuple ``(hidden_states, ...)``; raw tensor for some.
        residual = output[0] if isinstance(output, tuple) else output
        self.residual = residual


@dataclass
class FeatureStats:
    """Streaming summaries per SAE feature."""

    d_sae: int
    device: torch.device
    sum_act: torch.Tensor = field(init=False)
    sq_sum_act: torch.Tensor = field(init=False)
    max_act: torch.Tensor = field(init=False)
    nonzero_count: torch.Tensor = field(init=False)
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.sum_act = torch.zeros(self.d_sae, device=self.device, dtype=torch.float64)
        self.sq_sum_act = torch.zeros(self.d_sae, device=self.device, dtype=torch.float64)
        self.max_act = torch.full(
            (self.d_sae,), -float("inf"), device=self.device, dtype=torch.float32
        )
        self.nonzero_count = torch.zeros(self.d_sae, device=self.device, dtype=torch.int64)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, mask: torch.Tensor | None) -> None:
        # feats: [N_tokens, d_sae] (already on self.device)
        if mask is not None:
            keep = mask.bool().reshape(-1)
            feats = feats[keep]
        if feats.numel() == 0:
            return
        f64 = feats.to(torch.float64)
        self.sum_act += f64.sum(dim=0)
        self.sq_sum_act += (f64 * f64).sum(dim=0)
        col_max = feats.amax(dim=0).to(self.max_act.dtype)
        torch.maximum(self.max_act, col_max, out=self.max_act)
        self.nonzero_count += (feats > 0).sum(dim=0).to(self.nonzero_count.dtype)
        self.total_tokens += int(feats.shape[0])

    def finalize(self) -> dict[str, np.ndarray]:
        n = max(self.total_tokens, 1)
        mean = (self.sum_act / n).to(torch.float32).cpu().numpy()
        var = (self.sq_sum_act / n).to(torch.float32).cpu().numpy() - mean ** 2
        var = np.clip(var, 0.0, None)
        return {
            "mean_act": mean,
            "std_act": np.sqrt(var),
            "max_act": self.max_act.cpu().numpy(),
            "density": (self.nonzero_count.to(torch.float64) / n).cpu().numpy(),
            "nonzero_count": self.nonzero_count.cpu().numpy(),
            "total_tokens": n,
        }


class TopKExampleTracker:
    """Streaming top-K activating tokens per feature.

    For each feature ``f`` we keep the top ``K`` (activation, example_idx,
    position_idx) triples seen so far. Per-batch updates avoid materialising the
    full ``[total_tokens, d_sae]`` matrix.
    """

    def __init__(
        self,
        d_sae: int,
        top_k: int,
        device: torch.device,
    ) -> None:
        self.d_sae = d_sae
        self.top_k = top_k
        self.device = device
        self.acts = torch.full(
            (top_k, d_sae), -float("inf"), device=device, dtype=torch.float32
        )
        self.example_ids = torch.full(
            (top_k, d_sae), -1, device=device, dtype=torch.int64
        )
        self.position_ids = torch.full(
            (top_k, d_sae), -1, device=device, dtype=torch.int64
        )
        # input_ids / mask buffers per example, kept on CPU for memory.
        self.input_ids: list[torch.Tensor] = []
        self.attention_mask: list[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        feats: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        # feats: [B, T, d_sae]; input_ids/mask: [B, T]
        b, t, d = feats.shape
        ex_offset = len(self.input_ids)
        for i in range(b):
            self.input_ids.append(input_ids[i].cpu())
            self.attention_mask.append(attention_mask[i].cpu())

        flat = feats.reshape(b * t, d)
        # Mask padded tokens by setting their activations to -inf so they never
        # win the top-k tournament.
        keep = attention_mask.bool().reshape(-1)
        flat = flat.clone()
        flat[~keep] = -float("inf")

        k = min(self.top_k, b * t)
        if k == 0:
            return
        top_vals, top_idxs = flat.topk(k, dim=0)  # [k, d_sae]
        top_ex = (top_idxs // t + ex_offset).to(torch.int64)
        top_pos = (top_idxs % t).to(torch.int64)

        combined_acts = torch.cat([self.acts, top_vals], dim=0)
        combined_ex = torch.cat([self.example_ids, top_ex], dim=0)
        combined_pos = torch.cat([self.position_ids, top_pos], dim=0)

        new_vals, new_idx = combined_acts.topk(self.top_k, dim=0)  # [top_k, d_sae]
        col = (
            torch.arange(d, device=self.device).unsqueeze(0).expand_as(new_idx)
        )
        self.acts = new_vals
        self.example_ids = combined_ex.gather(0, new_idx)
        self.position_ids = combined_pos.gather(0, new_idx)
        # ``col`` is computed for parity with the gather indices; no further use.
        del col

    def get_examples(
        self, feature_idx: int, top_n: int
    ) -> list[tuple[float, int, int]]:
        n = min(top_n, self.top_k)
        out: list[tuple[float, int, int]] = []
        acts = self.acts[:, feature_idx].cpu().tolist()
        exs = self.example_ids[:, feature_idx].cpu().tolist()
        poss = self.position_ids[:, feature_idx].cpu().tolist()
        for a, e, p in zip(acts[:n], exs[:n], poss[:n]):
            if math.isinf(a) or math.isnan(a) or e < 0:
                continue
            out.append((float(a), int(e), int(p)))
        return out


# ---------------------------------------------------------------------------
# Core compute pass.
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_feature_activations(
    *,
    model,
    sae,
    layer_module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    sae_dtype: torch.dtype,
    top_k_tracker: TopKExampleTracker,
    feature_stats: FeatureStats,
    progress_every: int = 5,
) -> None:
    """Run the model on every batch, encode through the SAE, update streaming stats.

    No tensors of shape ``[total_tokens, d_sae]`` are ever materialised; each
    batch updates ``feature_stats`` and ``top_k_tracker`` in place.
    """
    capture = _ResidualCapture()
    handle = layer_module.register_forward_hook(capture)
    try:
        n = len(batches)
        for i, batch in enumerate(batches):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except TypeError:
                # Some HF wrappers reject ``use_cache``; retry without it.
                model(input_ids=ids, attention_mask=mask)
            assert capture.residual is not None, "Residual hook did not fire"
            resid = capture.residual.to(sae_dtype)
            feats = sae.encode(resid)  # [B, T, d_sae]
            top_k_tracker.update(feats, ids, mask)
            feature_stats.update(feats.reshape(-1, feats.shape[-1]), mask)
            capture.residual = None
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                print(
                    f"  [activations] batch {i + 1}/{n}  "
                    f"tokens_seen={feature_stats.total_tokens}",
                    flush=True,
                )
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Top-example extraction + interpretation.
# ---------------------------------------------------------------------------

@dataclass
class TopExample:
    activation: float
    text: str
    fired_token: str
    pre_context: str
    post_context: str
    example_idx: int
    position_idx: int


def get_top_examples(
    *,
    feature_idx: int,
    tracker: TopKExampleTracker,
    tokenizer,
    top_n: int,
    window: int,
) -> list[TopExample]:
    """Return up to ``top_n`` decoded contexts for ``feature_idx``."""
    raw = tracker.get_examples(feature_idx, top_n)
    out: list[TopExample] = []
    for activation, ex_idx, pos_idx in raw:
        ids = tracker.input_ids[ex_idx]
        mask = tracker.attention_mask[ex_idx].bool()
        valid_len = int(mask.sum().item())
        if pos_idx >= valid_len:
            continue
        lo = max(0, pos_idx - window)
        hi = min(valid_len, pos_idx + window + 1)
        ids_slice = ids[lo:hi].tolist()
        fired_token = tokenizer.decode([ids[pos_idx].item()], skip_special_tokens=False)
        pre = tokenizer.decode(ids[lo:pos_idx].tolist(), skip_special_tokens=True)
        post = tokenizer.decode(
            ids[pos_idx + 1 : hi].tolist(), skip_special_tokens=True
        )
        text = tokenizer.decode(ids_slice, skip_special_tokens=True)
        out.append(
            TopExample(
                activation=activation,
                text=text,
                fired_token=fired_token,
                pre_context=pre,
                post_context=post,
                example_idx=ex_idx,
                position_idx=pos_idx,
            )
        )
    return out


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']{1,}", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def interpret_feature(
    examples: list[TopExample],
    *,
    max_words: int = 3,
) -> dict[str, Any]:
    """Produce a strictly-evidence-based one-phrase description of a feature.

    Returns a dict with:
        ``description``: short phrase (joined keywords + tag prefix).
        ``keywords``: top content words across firing tokens / context.
        ``tags``: structural tags inferred from token shape.
        ``avg_fired_token_len``: convenience signal for "subword fragment" features.
    """
    if not examples:
        return {
            "description": "(no firing examples)",
            "keywords": [],
            "tags": ["dead"],
            "avg_fired_token_len": 0.0,
        }

    fired = [e.fired_token for e in examples]
    contexts = [f"{e.pre_context} {e.post_context}".strip() for e in examples]

    fired_words = []
    for tok in fired:
        fired_words.extend(_content_words(tok))
    context_words = []
    for ctx in contexts:
        context_words.extend(_content_words(ctx))

    counts: dict[str, int] = {}
    for w in fired_words:
        counts[w] = counts.get(w, 0) + 3  # weight firing-token words higher.
    for w in context_words:
        counts[w] = counts.get(w, 0) + 1
    keywords = [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:max_words]]

    tags: list[str] = []
    fired_concat = " ".join(fired)
    fired_lower = fired_concat.lower()
    if all(re.fullmatch(r"\s+", t) for t in fired):
        tags.append("whitespace")
    if all(re.search(r"\d", t) for t in fired):
        tags.append("numbers")
    if all(re.fullmatch(r"[\W_]+", t.strip()) and t.strip() for t in fired if t.strip()):
        tags.append("punctuation")
    if any(h in fired_lower for h in _CODE_HINTS) or any(
        h in c.lower() for c in contexts for h in _CODE_HINTS
    ):
        tags.append("code")
    if any(h in c.lower() for c in contexts for h in _REFUSAL_HINTS):
        tags.append("refusal/safety")
    if any(h in c.lower() for c in contexts for h in _POLITENESS_HINTS):
        tags.append("politeness")
    if any(h in c for c in contexts for h in _MATH_HINTS):
        tags.append("math")
    title_count = sum(1 for t in fired if t.strip()[:1].isupper())
    if title_count >= max(2, int(0.6 * len(fired))):
        tags.append("title-case")

    if not keywords and not tags:
        description = "(no salient pattern)"
    else:
        head = "/".join(tags) if tags else ""
        body = ", ".join(keywords) if keywords else ""
        description = " | ".join([s for s in (head, body) if s])

    return {
        "description": description,
        "keywords": keywords,
        "tags": tags,
        "avg_fired_token_len": float(np.mean([len(t) for t in fired])) if fired else 0.0,
    }


# ---------------------------------------------------------------------------
# Plotting (NeurIPS-style, large fonts, save replot metadata)
# ---------------------------------------------------------------------------

def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 25.0,
            "axes.titlesize": 28.0,
            "axes.labelsize": 28.0,
            "xtick.labelsize": 25.0,
            "ytick.labelsize": 25.0,
            "legend.fontsize": 25.0,
            "legend.title_fontsize": 25.0,
            "axes.linewidth": 2.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 10.0,
            "figure.dpi": 600,
            "savefig.dpi": 600,
        }
    )
    sns.set_style("whitegrid", {"grid.alpha": 0.3, "axes.edgecolor": "0.15"})


def plot_top_features_bar(
    feature_ids: np.ndarray,
    mean_acts: np.ndarray,
    descriptions: list[str],
    title: str,
    out_path: Path,
) -> None:
    n = len(feature_ids)
    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.45 * n + 4.0), 6.5),
        layout="constrained",
    )
    palette = sns.color_palette("viridis", n)
    bars = ax.bar(np.arange(n), mean_acts, color=palette, edgecolor="black", linewidth=0.8)
    ax.set_xticks(np.arange(n))
    labels = [f"{fid}\n{desc[:24]}" for fid, desc in zip(feature_ids, descriptions)]
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=11)
    ax.set_ylabel("Mean activation")
    ax.set_xlabel("SAE feature ID")
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    # mark vars to satisfy linters
    del bars, palette


def plot_density_vs_mean(
    mean_acts: np.ndarray,
    density: np.ndarray,
    selected: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), layout="constrained")
    alive = mean_acts > 0
    ax.scatter(
        density[alive],
        mean_acts[alive],
        s=10,
        alpha=0.35,
        color="0.55",
        label=f"All features (n={int(alive.sum())})",
    )
    if len(selected):
        ax.scatter(
            density[selected],
            mean_acts[selected],
            s=70,
            alpha=0.95,
            color="#cc4125",
            edgecolor="black",
            linewidth=0.8,
            label=f"Selected (n={len(selected)})",
        )
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(
        r"Activation density $\left(\mathbb{E}_t[\mathbf{1}\{a_t>0\}]\right)$"
    )
    ax.set_ylabel("Mean activation")
    ax.set_title(title)
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )
    ax.grid(alpha=0.3)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_top_feature_activation_heatmap(
    top_feature_ids: np.ndarray,
    top_acts_matrix: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Heatmap of top examples (rows) x top features (cols)."""
    if top_acts_matrix.size == 0:
        return
    n_examples, n_features = top_acts_matrix.shape
    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.45 * n_features + 4.0), max(6.0, 0.32 * n_examples + 2.0)),
        layout="constrained",
    )
    sns.heatmap(
        top_acts_matrix,
        ax=ax,
        cmap="magma",
        cbar_kws={"label": "Activation", "shrink": 0.82, "pad": 0.02},
        linewidths=0.0,
        xticklabels=[str(f) for f in top_feature_ids],
        yticklabels=[f"ex {i + 1}" for i in range(n_examples)],
    )
    ax.set_xlabel("SAE feature ID")
    ax.set_ylabel("Top-N firing examples")
    ax.set_title(title)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sae",
        default=None,
        help="HF repo id of the trained SAE (e.g. chungimungi/SAE-...-layer_19_best). "
        "Not required with --replot-plots-under.",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="HF repo id of the base model. Default: looked up in DEFAULT_SAE_CATALOG.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Decoder layer index to hook (0-based). Default: parsed from --sae name.",
    )
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset-split", default="train_sft")
    p.add_argument("--num-prompts", type=int, default=400)
    p.add_argument("--context-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--top-features",
        type=int,
        default=40,
        help="How many features to keep for interpretation / plotting (3.5: 20-50).",
    )
    p.add_argument(
        "--top-examples",
        type=int,
        default=12,
        help="Top-N activating examples to retain per feature.",
    )
    p.add_argument(
        "--context-window",
        type=int,
        default=20,
        help="+/- tokens around the firing token to decode for context.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--model-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument(
        "--sae-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument(
        "--selection",
        default="mean",
        choices=["mean", "max", "density"],
        help="Strategy used to rank features for interpretation.",
    )
    p.add_argument(
        "--min-density",
        type=float,
        default=1e-4,
        help="Drop features that fire on less than this fraction of tokens "
        "(filters dead / noise features).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-root",
        default="output/sae_features",
        help="All artefacts are written under <output-root>/<sae_repo_sanitized>/.",
    )
    p.add_argument(
        "--sae-cache-dir",
        default="cache/sae_snapshots",
        help="Local cache directory for SAE snapshots.",
    )
    p.add_argument(
        "--baseline-feature-stats",
        default=None,
        help="Optional path to a baseline replot_metadata.npz from a base-model SAE; "
        "if provided we record aggregate Δ statistics into deltas.json.",
    )
    p.add_argument(
        "--replot-plots-only",
        action="store_true",
        help="Regenerate PDFs only from feature_descriptions.json + replot_metadata.npz "
        "under <output-root>/<sae sanitized> (no GPU/model work).",
    )
    p.add_argument(
        "--replot-plots-under",
        type=Path,
        default=None,
        help="Walk this directory tree and replot every sae-feature output that has "
        "replot_metadata.npz and feature_descriptions.json (no --sae needed).",
    )
    return p.parse_args()


def write_readable_table(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    head_examples: int = 4,
) -> None:
    """Plain-text table with feature id, description, top examples."""
    lines: list[str] = []
    head = (
        f"{'feature':>8s}  {'mean':>9s}  {'density':>9s}  {'description':<36s}  examples"
    )
    lines.append(head)
    lines.append("-" * len(head))
    for row in rows:
        ex_lines = []
        for ex in row["top_examples"][:head_examples]:
            snippet = ex["text"].replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            ex_lines.append(f"      [{ex['activation']:.3f}] ...{snippet}...")
        desc = row["description"][:34]
        lines.append(
            f"{row['feature_id']:>8d}  {row['mean_act']:>9.4f}  "
            f"{row['density']:>9.4f}  {desc:<36s}"
        )
        lines.extend(ex_lines)
    out_path.write_text("\n".join(lines) + "\n")


def select_feature_ids(
    stats: dict[str, np.ndarray],
    *,
    selection: str,
    top_features: int,
    min_density: float,
) -> np.ndarray:
    density = stats["density"]
    metric = {
        "mean": stats["mean_act"],
        "max": stats["max_act"],
        "density": density,
    }[selection]
    valid = density >= float(min_density)
    metric_filtered = np.where(valid, metric, -np.inf)
    n = min(int(top_features), int(valid.sum()))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argpartition(-metric_filtered, n - 1)[:n]
    return order[np.argsort(-metric_filtered[order])].astype(np.int64)


def main() -> None:
    _load_env()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.replot_plots_under is not None and args.replot_plots_only:
        raise SystemExit("Use only one of --replot-plots-under or --replot-plots-only")

    if args.replot_plots_under is not None:
        root = args.replot_plots_under.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        configure_plot_style()
        dirs = _iter_sae_feature_output_dirs(root)
        if not dirs:
            raise SystemExit(f"No sae-feature outputs (npz + json) under {root}")
        for d in dirs:
            print(f"[replot] {d}", flush=True)
            replot_sae_feature_plots_from_artifacts(d)
        print(f"\n[done] replotted {len(dirs)} output folder(s)", flush=True)
        return

    if args.replot_plots_only:
        if not args.sae:
            raise SystemExit("--sae is required with --replot-plots-only")
        out_root = Path(args.output_root) / _sanitize(args.sae)
        configure_plot_style()
        print(f"[replot-plots-only] {out_root}", flush=True)
        replot_sae_feature_plots_from_artifacts(out_root)
        print("\n[done]", flush=True)
        return

    if not args.sae:
        raise SystemExit("--sae is required unless using --replot-plots-under")

    sae_repo = args.sae
    layer_idx = args.layer if args.layer is not None else _parse_layer_from_repo(sae_repo)
    if layer_idx is None:
        raise SystemExit(
            f"Could not parse layer from {sae_repo!r}; pass --layer explicitly."
        )
    base_model_id = _resolve_base_model(sae_repo, args.base_model)

    device = _resolve_device(args.device)
    model_dtype = _resolve_dtype(args.model_dtype)
    sae_dtype = _resolve_dtype(args.sae_dtype)

    out_root = Path(args.output_root) / _sanitize(sae_repo)
    out_root.mkdir(parents=True, exist_ok=True)
    sae_cache = Path(args.sae_cache_dir)
    sae_cache.mkdir(parents=True, exist_ok=True)

    print(f"=== sae-feature.py ===", flush=True)
    print(f"  SAE repo:    {sae_repo}", flush=True)
    print(f"  Base model:  {base_model_id}", flush=True)
    print(f"  Layer:       {layer_idx}", flush=True)
    print(f"  Device:      {device} (model dtype={args.model_dtype}, sae dtype={args.sae_dtype})", flush=True)
    print(f"  Dataset:     {args.dataset} (split={args.dataset_split}, prompts={args.num_prompts})", flush=True)
    print(f"  Output:      {out_root}", flush=True)

    started = time.time()
    if _outputs_complete(out_root, baseline_npz=args.baseline_feature_stats):
        need = "json, txt, npz, plots"
        if args.baseline_feature_stats:
            need += ", deltas.json"
        print(
            f"[skip] already have {need} under {out_root}; "
            "delete those files to force a full rerun.",
            flush=True,
        )
        print(f"\n[done] {sae_repo} in {time.time() - started:.1f}s", flush=True)
        return

    print("[load] tokenizer + base model...", flush=True)
    tokenizer = load_tokenizer(base_model_id)
    model = load_base_model(base_model_id, device, model_dtype)
    layer_module = get_layer_module(model, layer_idx)

    print("[load] SAE snapshot...", flush=True)
    sae_local = download_sae(sae_repo, sae_cache)
    # ``snapshot_download`` returns the snapshot root that contains cfg.json
    # directly when ``allow_patterns`` matches files at the repo root.
    sae = load_sae(sae_local, device, sae_dtype)
    d_sae = int(sae.cfg.d_sae)
    print(f"  d_sae={d_sae}, k={getattr(sae.cfg, 'k', 'n/a')}", flush=True)

    print("[data] streaming prompts...", flush=True)
    prompts = list(
        iter_chat_prompts(args.dataset, args.dataset_split, tokenizer, args.num_prompts)
    )
    if not prompts:
        raise SystemExit(
            f"No usable prompts from {args.dataset!r}; try a different --dataset."
        )
    print(f"  prompts: {len(prompts)}", flush=True)
    batches = tokenize_prompts(tokenizer, prompts, args.context_size, args.batch_size)
    print(f"  batches: {len(batches)} of (B={args.batch_size}, T={args.context_size})", flush=True)

    feature_stats = FeatureStats(d_sae=d_sae, device=device)
    tracker = TopKExampleTracker(
        d_sae=d_sae, top_k=max(int(args.top_examples), 1), device=device
    )

    print("[run] forward + SAE encode + streaming top-k...", flush=True)
    get_feature_activations(
        model=model,
        sae=sae,
        layer_module=layer_module,
        batches=batches,
        device=device,
        sae_dtype=sae_dtype,
        top_k_tracker=tracker,
        feature_stats=feature_stats,
    )

    stats = feature_stats.finalize()
    print(
        f"  total tokens encoded: {stats['total_tokens']}  "
        f"(alive features: {(stats['density'] > 0).sum()}/{d_sae})",
        flush=True,
    )

    selected = select_feature_ids(
        stats,
        selection=args.selection,
        top_features=args.top_features,
        min_density=args.min_density,
    )
    print(f"[select] {len(selected)} features retained for interpretation.", flush=True)

    interpreted_rows: list[dict[str, Any]] = []
    top_acts_matrix = np.zeros((args.top_examples, len(selected)), dtype=np.float32)
    for col, fid in enumerate(selected):
        examples = get_top_examples(
            feature_idx=int(fid),
            tracker=tracker,
            tokenizer=tokenizer,
            top_n=args.top_examples,
            window=args.context_window,
        )
        info = interpret_feature(examples)
        for row, ex in enumerate(examples):
            top_acts_matrix[row, col] = ex.activation
        interpreted_rows.append(
            {
                "feature_id": int(fid),
                "description": info["description"],
                "keywords": info["keywords"],
                "tags": info["tags"],
                "mean_act": float(stats["mean_act"][fid]),
                "max_act": float(stats["max_act"][fid]),
                "density": float(stats["density"][fid]),
                "avg_fired_token_len": info["avg_fired_token_len"],
                "top_examples": [
                    {
                        "activation": ex.activation,
                        "text": ex.text,
                        "fired_token": ex.fired_token,
                        "pre_context": ex.pre_context,
                        "post_context": ex.post_context,
                        "example_idx": ex.example_idx,
                        "position_idx": ex.position_idx,
                    }
                    for ex in examples
                ],
            }
        )

    # ------------------- persist results -------------------
    json_path = out_root / "feature_descriptions.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "sae_repo": sae_repo,
                "base_model": base_model_id,
                "layer": layer_idx,
                "selection": args.selection,
                "min_density": args.min_density,
                "num_prompts": len(prompts),
                "context_size": args.context_size,
                "total_tokens": int(stats["total_tokens"]),
                "features": interpreted_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  wrote {json_path}", flush=True)

    table_path = out_root / "feature_descriptions.txt"
    write_readable_table(interpreted_rows, table_path)
    print(f"  wrote {table_path}", flush=True)

    # Replot metadata: every array needed to recreate the figures without rerunning.
    npz_path = out_root / "replot_metadata.npz"
    np.savez(
        npz_path,
        feature_ids_all=np.arange(d_sae, dtype=np.int64),
        mean_act=stats["mean_act"].astype(np.float32),
        std_act=stats["std_act"].astype(np.float32),
        max_act=stats["max_act"].astype(np.float32),
        density=stats["density"].astype(np.float32),
        nonzero_count=stats["nonzero_count"].astype(np.int64),
        total_tokens=np.int64(stats["total_tokens"]),
        selected_feature_ids=selected.astype(np.int64),
        selected_descriptions=np.array(
            [r["description"] for r in interpreted_rows], dtype=object
        ),
        top_acts_matrix=top_acts_matrix.astype(np.float32),
        layer=np.int64(layer_idx),
    )
    print(f"  wrote {npz_path}", flush=True)

    # ------------------- plots -------------------
    configure_plot_style()
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_label = _plot_short_model_name(base_model_id)
    plot_top_features_bar(
        feature_ids=np.array([r["feature_id"] for r in interpreted_rows], dtype=np.int64),
        mean_acts=np.array([r["mean_act"] for r in interpreted_rows], dtype=np.float32),
        descriptions=[r["description"] for r in interpreted_rows],
        title=(
            f"Top {len(interpreted_rows)} features by mean activation\n"
            f"{plot_label} (layer {layer_idx})"
        ),
        out_path=plots_dir / "top_features_mean_activation",
    )
    print(f"  wrote {plots_dir / 'top_features_mean_activation.pdf'}", flush=True)

    plot_density_vs_mean(
        mean_acts=stats["mean_act"],
        density=stats["density"],
        selected=selected,
        title=f"Feature density vs. mean activation\n{plot_label} (layer {layer_idx})",
        out_path=plots_dir / "density_vs_mean",
    )
    print(f"  wrote {plots_dir / 'density_vs_mean.pdf'}", flush=True)

    plot_top_feature_activation_heatmap(
        top_feature_ids=np.array(
            [r["feature_id"] for r in interpreted_rows], dtype=np.int64
        ),
        top_acts_matrix=top_acts_matrix,
        title=(
            "Top examples × top features (activation magnitude)\n"
            f"{plot_label} (layer {layer_idx})"
        ),
        out_path=plots_dir / "top_examples_heatmap",
    )
    print(f"  wrote {plots_dir / 'top_examples_heatmap.pdf'}", flush=True)

    # ------------------- aggregate Δ vs. baseline (optional) -------------------
    if args.baseline_feature_stats:
        try:
            base = np.load(args.baseline_feature_stats, allow_pickle=True)
            delta = {
                "alive_features_aligned": int((stats["density"] > 0).sum()),
                "alive_features_baseline": int((base["density"] > 0).sum()),
                "mean_density_aligned": float(np.mean(stats["density"])),
                "mean_density_baseline": float(np.mean(base["density"])),
                "mean_max_act_aligned": float(np.mean(stats["max_act"])),
                "mean_max_act_baseline": float(np.mean(base["max_act"])),
                "delta_alive": int((stats["density"] > 0).sum())
                - int((base["density"] > 0).sum()),
                "delta_mean_density": float(
                    np.mean(stats["density"]) - np.mean(base["density"])
                ),
                "delta_mean_max_act": float(
                    np.mean(stats["max_act"]) - np.mean(base["max_act"])
                ),
                "baseline_metadata_path": str(args.baseline_feature_stats),
            }
            (out_root / "deltas.json").write_text(json.dumps(delta, indent=2))
            print(f"  wrote {out_root / 'deltas.json'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to compute Δ: {e}", flush=True)

    elapsed = time.time() - started
    print(f"\n[done] {sae_repo} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
