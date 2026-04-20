import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_model_slug(model_id: str) -> str:
    """Filesystem-safe slug from HF id or path (e.g. HuggingFaceTB/SmolLM3-3B)."""
    s = model_id.rstrip("/")
    if "/" in s and not s.startswith("."):
        parts = s.split("/")
        s = parts[-1] if len(parts) <= 2 else "_".join(parts[-2:])
    s = s.replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def get_base_activations_cache_path(
    base_model_id: str,
    layer: int,
    position: str,
    dataset_name: str,
) -> Path:
    base_slug = sanitize_model_slug(base_model_id)
    dataset_slug = sanitize_model_slug(dataset_name)
    pos_slug = position.replace("_", "")
    dir_name = f"{base_slug}__L{layer}__{pos_slug}__{dataset_slug}"
    cache_path = config.BASE_ACTIVATIONS_CACHE_DIR / dir_name / "base_activations.pt"
    return cache_path


def get_results_dir(
    base_model_id: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    base_dir: Optional[Path] = None,
) -> Path:
    base_slug = sanitize_model_slug(base_model_id)
    pos_slug = position.replace("_", "")
    dir_name = f"{base_slug}__{aligned_run_id}__L{layer}__{pos_slug}"
    root = base_dir if base_dir is not None else config.CROSSCODER_RESULTS_DIR
    results_dir = root / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_checkpoint_dir(results_dir: Path) -> Path:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_activations_dir(results_dir: Path) -> Path:
    activations_dir = results_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    return activations_dir


def get_features_dir(results_dir: Path) -> Path:
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir


def get_metrics_dir(results_dir: Path) -> Path:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def get_plots_dir(results_dir: Path) -> Path:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, checkpoint_dir: Path, is_final: bool = False):
    filename = "final.pt" if is_final else f"epoch_{epoch}.pt"
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_dir / filename)


def load_checkpoint(checkpoint_path: Path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_activations(activations: dict, path: Path):
    torch.save(activations, path)


def load_activations(path: Path, map_location=None) -> dict:
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location)


def flush_gpu():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_memory_info() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.2f}/{total:.2f} GB"


def get_topk_llm() -> int:
    return config.TOPK_LLM


def get_expansion_factor_llm() -> int:
    return config.EXPANSION_FACTOR_LLM
