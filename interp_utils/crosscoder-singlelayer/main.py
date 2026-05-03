import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Optional

import torch
from tqdm import tqdm

from . import config
from .utils import (
    flush_gpu,
    get_activations_dir,
    get_base_activations_cache_path_multilayer,
    get_checkpoint_dir,
    get_features_dir,
    get_metrics_dir,
    get_plots_dir,
    get_results_dir,
    get_results_dir_multilayer,
    layers_slug,
    load_activations,
    load_json,
    save_activations,
    save_json,
    sanitize_model_slug,
    set_seed,
)


def _resolve_results_dir(
    base_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is not None:
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    return get_results_dir(base_model, aligned_run_id, layer, position)


def _parse_layers_arg(layers_arg: Optional[str]) -> Optional[list[int]]:
    if not layers_arg:
        return None
    layers = [int(part.strip()) for part in layers_arg.split(",") if part.strip()]
    if not layers:
        raise ValueError("--layers parsed to an empty layer list")
    if len(set(layers)) != len(layers):
        raise ValueError(f"--layers contains duplicates: {layers}")
    return layers


def _resolve_multilayer_layers(
    *,
    layer: Optional[int],
    center_layer: Optional[int],
    layer_window: int,
    layers_arg: Optional[str],
) -> list[int]:
    explicit_layers = _parse_layers_arg(layers_arg)
    if explicit_layers is not None:
        return explicit_layers
    center = center_layer if center_layer is not None else layer
    if center is None:
        raise ValueError("Multi-layer runs require --layers, --center-layer, or --layer")
    if layer_window < 0:
        raise ValueError("--layer-window must be non-negative")
    return list(range(center - layer_window, center + layer_window + 1))


def _explicit_multilayer_layer_args_present(args) -> bool:
    return args.layers is not None or args.center_layer is not None or args.layer is not None


def _load_artifact_layers(results_dir: Path) -> Optional[list[int]]:
    meta_path = results_dir / "run_meta.json"
    if meta_path.is_file():
        meta = load_json(meta_path)
        if "layers" in meta:
            return [int(layer) for layer in meta["layers"]]

    activations_path = get_activations_dir(results_dir) / "activations.pt"
    if activations_path.is_file():
        activations_data = load_activations(activations_path)
        if "layers" in activations_data:
            return [int(layer) for layer in activations_data["layers"]]

    return None


def _resolve_multilayer_layers_for_stage(
    *,
    args,
    output_dir: Optional[Path],
) -> list[int]:
    if _explicit_multilayer_layer_args_present(args):
        cli_layers = _resolve_multilayer_layers(
            layer=args.layer,
            center_layer=args.center_layer,
            layer_window=args.layer_window,
            layers_arg=args.layers,
        )
        if output_dir is not None and args.stage in {"train", "analyze"}:
            artifact_layers = _load_artifact_layers(output_dir)
            if artifact_layers is not None and artifact_layers != cli_layers:
                raise ValueError(
                    f"CLI layers {cli_layers} do not match artifact layers {artifact_layers} in {output_dir}"
                )
        return cli_layers

    if output_dir is not None and args.stage in {"train", "analyze"}:
        artifact_layers = _load_artifact_layers(output_dir)
        if artifact_layers is not None:
            return artifact_layers

    return _resolve_multilayer_layers(
        layer=args.layer,
        center_layer=args.center_layer,
        layer_window=args.layer_window,
        layers_arg=args.layers,
    )


def _validate_multilayer_activations(activations_data: dict, layers: list[int]) -> None:
    import torch

    if "activations_base" not in activations_data or "activations_aligned" not in activations_data:
        raise ValueError("Activation artifact must contain activations_base and activations_aligned")
    x_base = activations_data["activations_base"]
    x_aligned = activations_data["activations_aligned"]
    if not isinstance(x_base, torch.Tensor) or not isinstance(x_aligned, torch.Tensor):
        raise ValueError("activations_base and activations_aligned must be torch tensors")
    if x_base.ndim != 3 or x_aligned.ndim != 3:
        raise ValueError(
            f"Multi-layer activations must be rank-3 [N, L, D], got base={tuple(x_base.shape)}, "
            f"aligned={tuple(x_aligned.shape)}"
        )
    if tuple(x_base.shape) != tuple(x_aligned.shape):
        raise ValueError(
            f"Base/aligned activation shapes must match, got base={tuple(x_base.shape)}, "
            f"aligned={tuple(x_aligned.shape)}"
        )
    artifact_layers = [int(layer) for layer in activations_data.get("layers", [])]
    if not artifact_layers:
        raise ValueError("Multi-layer activation artifact is missing layers metadata")
    if artifact_layers != [int(layer) for layer in layers]:
        raise ValueError(f"Activation artifact layers {artifact_layers} do not match requested layers {layers}")
    if x_base.shape[1] != len(layers):
        raise ValueError(f"Activation tensor has L={x_base.shape[1]} but layers metadata has {len(layers)} entries")


def _load_required_activations(activations_path: Path) -> dict:
    if not activations_path.is_file():
        raise FileNotFoundError(
            f"Missing activations artifact: {activations_path}. Analysis/training restart requires activations.pt."
        )
    return load_activations(activations_path)


def _multilayer_cache_meta(
    *,
    base_model: str,
    layers: list[int],
    position: str,
    dataset_name: str,
    max_prompt_tokens: int,
    hidden_size: Optional[int] = None,
) -> dict:
    meta = {
        "crosscoder_kind": "multilayer_sparc",
        "base_model": base_model,
        "layers": [int(layer) for layer in layers],
        "position": position,
        "dataset_name": dataset_name,
        "max_prompt_tokens": int(max_prompt_tokens),
        "seed": int(config.SEED),
        "val_fraction": float(config.VAL_FRACTION),
        "disable_thinking": bool(config.DISABLE_THINKING),
    }
    if hidden_size is not None:
        meta["hidden_size"] = int(hidden_size)
    return meta


def _cache_meta_matches(stored: dict, expected: dict) -> bool:
    stored_meta = stored.get("cache_meta")
    if not isinstance(stored_meta, dict):
        return False
    for key, value in expected.items():
        if key == "hidden_size":
            continue
        if stored_meta.get(key) != value:
            return False
    return True


def _resolve_multilayer_topk_mode(results_dir: Path, requested_topk_mode: str, prefer_artifact: bool = True) -> str:
    meta_path = results_dir / "run_meta.json"
    if prefer_artifact and meta_path.is_file():
        meta = load_json(meta_path)
        artifact_topk_mode = meta.get("topk_mode")
        if artifact_topk_mode:
            if artifact_topk_mode not in config.MULTILAYER_TOPK_MODES:
                raise ValueError(f"Invalid artifact topk_mode {artifact_topk_mode!r} in {meta_path}")
            if requested_topk_mode != config.MULTILAYER_TOPK_MODE and requested_topk_mode != artifact_topk_mode:
                raise ValueError(
                    f"Requested topk_mode {requested_topk_mode!r} does not match artifact topk_mode "
                    f"{artifact_topk_mode!r}"
                )
            return artifact_topk_mode
    return requested_topk_mode


def _resolve_results_dir_multilayer(
    base_model: str,
    aligned_run_id: str,
    layers: list[int],
    position: str,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is not None:
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    return get_results_dir_multilayer(base_model, aligned_run_id, layers, position)


def run_extract(
    base_model: str,
    aligned_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    dataset_name: str,
    max_prompt_tokens: int,
    trust_remote_code: bool,
    output_dir: Optional[Path] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
):
    import torch
    from .activations import extract_activations_llm
    from .utils import get_base_activations_cache_path

    print(f"\n{'='*60}")
    print(f"EXTRACTION: {base_model} vs {aligned_model} ({aligned_run_id}) L{layer} {position}")
    print(f"{'='*60}")

    results_dir = _resolve_results_dir(
        base_model, aligned_run_id, layer, position, output_dir
    )
    activations_dir = get_activations_dir(results_dir)
    out_path = activations_dir / "activations.pt"
    if out_path.exists():
        print(f"Activations already exist at {out_path}, skipping extraction.")
        return

    # Check for cached base activations — avoids re-running the base LLM for new aligned runs
    base_cache_path = get_base_activations_cache_path(base_model, layer, position, dataset_name)
    base_activations_cache = None
    if base_cache_path.exists():
        print(f"Loading cached base activations: {base_cache_path}")
        base_activations_cache = torch.load(base_cache_path, weights_only=False)

    hf_token = os.environ.get("HF_TOKEN")
    result = extract_activations_llm(
        base_model_id=base_model,
        aligned_model_path=aligned_model,
        aligned_run_id=aligned_run_id,
        layer=layer,
        position=position,
        dataset_name=dataset_name,
        max_prompt_tokens=max_prompt_tokens,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
        prompts_cache_dir=prompts_cache_dir,
        use_prompts_cache=use_prompts_cache,
        extract_batch_size=extract_batch_size,
        base_activations_cache=base_activations_cache,
    )

    # Persist base activations cache for future aligned runs on the same base model
    if base_activations_cache is None:
        base_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations_base": result["activations_base"],
                "sample_ids": result["sample_ids"],
                "splits": result["splits"],
                "base_model": base_model,
                "layer": layer,
                "position": position,
                "dataset_name": dataset_name,
                "hidden_size": result["hidden_size"],
            },
            base_cache_path,
        )
        print(f"Cached base activations: {base_cache_path}")

    save_activations(result, out_path)
    save_json(
        {
            "base_model": base_model,
            "aligned_model": aligned_model,
            "aligned_run_id": aligned_run_id,
            "layer": layer,
            "position": position,
            "dataset_name": dataset_name,
            "peft": result.get("peft", False),
        },
        results_dir / "run_meta.json",
    )
    del result
    flush_gpu()
    print(f"Saved activations: {out_path}")
    print("Extraction complete!")


def run_extract_multilayer(
    base_model: str,
    aligned_model: Optional[str],
    aligned_run_id: str,
    layers: list[int],
    position: str,
    dataset_name: str,
    max_prompt_tokens: int,
    trust_remote_code: bool,
    output_dir: Optional[Path] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
    center_layer: Optional[int] = None,
    layer_window: Optional[int] = None,
    topk_mode: str = config.MULTILAYER_TOPK_MODE,
    extract_side: str = "both",
):
    import torch
    from .activations import extract_activations_llm_multilayer

    if extract_side not in {"both", "base", "aligned"}:
        raise ValueError(f"extract_side must be one of both/base/aligned, got {extract_side!r}")
    if extract_side in {"both", "aligned"} and not aligned_model:
        raise ValueError("--aligned-model is required for extract-side=both or extract-side=aligned")

    print(f"\n{'='*60}")
    print(
        f"EXTRACTION: {base_model} vs {aligned_model or '<base-only>'} "
        f"({aligned_run_id}) {layers_slug(layers)} {position} side={extract_side}"
    )
    print(f"{'='*60}")

    results_dir = _resolve_results_dir_multilayer(
        base_model, aligned_run_id, layers, position, output_dir
    )
    activations_dir = get_activations_dir(results_dir)
    if extract_side == "base":
        out_path = activations_dir / "base_activations.pt"
    elif extract_side == "aligned":
        out_path = activations_dir / "aligned_activations.pt"
    else:
        out_path = activations_dir / "activations.pt"
    if out_path.exists():
        print(f"Activations already exist at {out_path}, skipping extraction.")
        return

    base_cache_path = get_base_activations_cache_path_multilayer(base_model, layers, position, dataset_name)
    base_activations_cache = None
    expected_cache_meta = _multilayer_cache_meta(
        base_model=base_model,
        layers=layers,
        position=position,
        dataset_name=dataset_name,
        max_prompt_tokens=max_prompt_tokens,
    )
    if extract_side == "both" and base_cache_path.exists():
        candidate_cache = torch.load(base_cache_path, weights_only=False)
        if _cache_meta_matches(candidate_cache, expected_cache_meta):
            print(f"Loading cached multi-layer base activations: {base_cache_path}")
            base_activations_cache = candidate_cache
        else:
            print(f"Ignoring stale/incompatible multi-layer base activations cache: {base_cache_path}")

    hf_token = os.environ.get("HF_TOKEN")
    result = extract_activations_llm_multilayer(
        base_model_id=base_model,
        aligned_model_path=aligned_model,
        aligned_run_id=aligned_run_id,
        layers=layers,
        position=position,
        dataset_name=dataset_name,
        max_prompt_tokens=max_prompt_tokens,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
        prompts_cache_dir=prompts_cache_dir,
        use_prompts_cache=use_prompts_cache,
        extract_batch_size=extract_batch_size,
        base_activations_cache=base_activations_cache,
        extract_side=extract_side,
    )

    if extract_side in {"both", "base"} and not base_cache_path.exists():
        base_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations_base": result["activations_base"],
                "sample_ids": result["sample_ids"],
                "splits": result["splits"],
                "base_model": base_model,
                "layers": layers,
                "position": position,
                "dataset_name": dataset_name,
                "hidden_size": result["hidden_size"],
                "crosscoder_kind": "multilayer_sparc",
                "extract_side": "base",
                "cache_meta": _multilayer_cache_meta(
                    base_model=base_model,
                    layers=layers,
                    position=position,
                    dataset_name=dataset_name,
                    max_prompt_tokens=max_prompt_tokens,
                    hidden_size=result["hidden_size"],
                ),
            },
            base_cache_path,
        )
        print(f"Cached multi-layer base activations: {base_cache_path}")

    save_activations(result, out_path)
    save_json(
        {
            "crosscoder_kind": "multilayer_sparc",
            "extract_side": extract_side,
            "base_model": base_model,
            "aligned_model": aligned_model,
            "aligned_run_id": aligned_run_id,
            "layers": layers,
            "center_layer": center_layer,
            "layer_window": layer_window,
            "layer_policy": "matched_aligned_window",
            "position": position,
            "dataset_name": dataset_name,
            "max_prompt_tokens": max_prompt_tokens,
            "peft": result.get("peft", False),
            "topk_mode": topk_mode,
            "activation_artifact": str(out_path),
        },
        results_dir / "run_meta.json",
    )
    del result
    flush_gpu()
    print(f"Saved multi-layer activations: {out_path}")
    print("Multi-layer extraction complete!")


def _resolve_side_activations_path(path: Path, side: str) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / "activations" / f"{side}_activations.pt",
        path / f"{side}_activations.pt",
    ]
    if side == "base":
        candidates.extend([path / "activations" / "activations.pt", path / "activations.pt"])
    if side == "aligned":
        candidates.extend([path / "activations" / "activations.pt", path / "activations.pt"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {side} activations under {path}. "
        f"Expected a .pt file or one of: {', '.join(str(candidate) for candidate in candidates)}"
    )


def _require_multilayer_side_tensor(data: dict, side: str, source: Path) -> torch.Tensor:
    key = f"activations_{side}"
    tensor = data.get(key)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{source} is missing tensor key {key!r}")
    if tensor.ndim != 3:
        raise ValueError(f"{source} {key} must have shape [N, L, D], got {tuple(tensor.shape)}")
    return tensor


def _require_artifact_layers(data: dict, source: Path) -> list[int]:
    if "layers" not in data:
        raise ValueError(f"{source} is missing 'layers' metadata")
    layers = [int(layer) for layer in data["layers"]]
    if not layers:
        raise ValueError(f"{source} has an empty layer list")
    return layers


def _check_matching_artifact_metadata(base_data: dict, aligned_data: dict, base_path: Path, aligned_path: Path) -> None:
    for key in ("sample_ids", "splits"):
        if base_data.get(key) != aligned_data.get(key):
            raise ValueError(f"{key} mismatch between {base_path} and {aligned_path}")

    for key in ("position", "dataset_name", "max_prompt_tokens"):
        base_value = base_data.get(key)
        aligned_value = aligned_data.get(key)
        if base_value is not None and aligned_value is not None and base_value != aligned_value:
            raise ValueError(
                f"{key} mismatch between base ({base_value!r}) and aligned ({aligned_value!r}) artifacts"
            )

    base_hidden = base_data.get("hidden_size")
    aligned_hidden = aligned_data.get("hidden_size")
    if base_hidden is not None and aligned_hidden is not None and int(base_hidden) != int(aligned_hidden):
        raise ValueError(f"hidden_size mismatch: base={base_hidden}, aligned={aligned_hidden}")


def _slice_base_layers_to_aligned(
    base_tensor: torch.Tensor,
    base_layers: list[int],
    aligned_layers: list[int],
    base_path: Path,
) -> torch.Tensor:
    layer_to_idx = {layer: idx for idx, layer in enumerate(base_layers)}
    missing = [layer for layer in aligned_layers if layer not in layer_to_idx]
    if missing:
        raise ValueError(
            f"Aligned layers {aligned_layers} are not a subset of base layers {base_layers}; "
            f"missing {missing} in {base_path}"
        )
    indices = torch.tensor([layer_to_idx[layer] for layer in aligned_layers], dtype=torch.long)
    return base_tensor.index_select(1, indices)


def run_assemble_multilayer(
    *,
    base_activations_path: Path,
    aligned_activations_path: Path,
    output_dir: Path,
    requested_layers: Optional[list[int]] = None,
    topk_mode: str = config.MULTILAYER_TOPK_MODE,
) -> dict:
    base_path = _resolve_side_activations_path(base_activations_path, "base")
    aligned_path = _resolve_side_activations_path(aligned_activations_path, "aligned")

    print(f"\n{'='*60}")
    print(f"ASSEMBLE: {base_path} + {aligned_path}")
    print(f"{'='*60}")

    base_data = load_activations(base_path)
    aligned_data = load_activations(aligned_path)
    base_tensor = _require_multilayer_side_tensor(base_data, "base", base_path)
    aligned_tensor = _require_multilayer_side_tensor(aligned_data, "aligned", aligned_path)
    base_layers = _require_artifact_layers(base_data, base_path)
    aligned_layers = _require_artifact_layers(aligned_data, aligned_path)

    if requested_layers is not None and requested_layers != aligned_layers:
        raise ValueError(f"Requested layers {requested_layers} do not match aligned artifact layers {aligned_layers}")
    if int(aligned_tensor.shape[1]) != len(aligned_layers):
        raise ValueError(
            f"Aligned tensor layer axis {aligned_tensor.shape[1]} does not match layers metadata {aligned_layers}"
        )
    if int(base_tensor.shape[1]) != len(base_layers):
        raise ValueError(f"Base tensor layer axis {base_tensor.shape[1]} does not match layers metadata {base_layers}")

    _check_matching_artifact_metadata(base_data, aligned_data, base_path, aligned_path)
    base_sliced = _slice_base_layers_to_aligned(base_tensor, base_layers, aligned_layers, base_path)
    if tuple(base_sliced.shape) != tuple(aligned_tensor.shape):
        raise ValueError(
            f"Assembled base shape {tuple(base_sliced.shape)} does not match aligned shape {tuple(aligned_tensor.shape)}"
        )

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = get_activations_dir(results_dir)
    out_path = activations_dir / "activations.pt"

    hidden_size = int(aligned_data.get("hidden_size", aligned_tensor.shape[2]))
    assembled = {
        "activations_base": base_sliced.contiguous(),
        "activations_aligned": aligned_tensor.contiguous(),
        "sample_ids": aligned_data["sample_ids"],
        "splits": aligned_data["splits"],
        "base_model": base_data.get("base_model", aligned_data.get("base_model")),
        "aligned_model": aligned_data.get("aligned_model"),
        "aligned_run_id": aligned_data.get("aligned_run_id"),
        "layers": aligned_layers,
        "base_source_layers": base_layers,
        "position": aligned_data.get("position", base_data.get("position", config.POSITION_LAST_PROMPT)),
        "dataset_name": aligned_data.get("dataset_name", base_data.get("dataset_name")),
        "max_prompt_tokens": aligned_data.get("max_prompt_tokens", base_data.get("max_prompt_tokens")),
        "peft": aligned_data.get("peft", False),
        "hidden_size": hidden_size,
        "crosscoder_kind": "multilayer_sparc",
        "extract_side": "assembled",
        "assembly": {
            "base_activations_path": str(base_path),
            "aligned_activations_path": str(aligned_path),
            "base_layers": base_layers,
            "aligned_layers": aligned_layers,
        },
    }
    _validate_multilayer_activations(assembled, aligned_layers)
    save_activations(assembled, out_path)

    save_json(
        {
            "crosscoder_kind": "multilayer_sparc",
            "extract_side": "assembled",
            "base_model": assembled["base_model"],
            "aligned_model": assembled["aligned_model"],
            "aligned_run_id": assembled["aligned_run_id"],
            "layers": aligned_layers,
            "base_source_layers": base_layers,
            "layer_policy": "base_union_sliced_to_aligned_layers",
            "position": assembled["position"],
            "dataset_name": assembled["dataset_name"],
            "max_prompt_tokens": assembled["max_prompt_tokens"],
            "peft": assembled["peft"],
            "topk_mode": topk_mode,
            "base_activations_path": str(base_path),
            "aligned_activations_path": str(aligned_path),
            "activation_artifact": str(out_path),
        },
        results_dir / "run_meta.json",
    )
    print(f"Saved assembled multi-layer activations: {out_path}")
    return assembled


def run_train(
    base_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    output_dir: Optional[Path] = None,
    train_batch_size: Optional[int] = None,
    use_train_amp: Optional[bool] = None,
):
    from .train import train_crosscoder

    print(f"\n{'='*60}")
    print(f"TRAINING: {base_model} / {aligned_run_id} L{layer} {position}")
    print(f"{'='*60}")

    results_dir = _resolve_results_dir(
        base_model, aligned_run_id, layer, position, output_dir
    )
    activations_dir = get_activations_dir(results_dir)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    if (checkpoint_dir / "final.pt").exists():
        print(f"Checkpoint already exists at {checkpoint_dir / 'final.pt'}, skipping training.")
        return None

    activations_path = activations_dir / "activations.pt"
    activations_data = load_activations(activations_path)
    input_dim = int(activations_data.get("hidden_size", activations_data["activations_base"].shape[1]))

    bs = train_batch_size if train_batch_size is not None else config.BATCH_SIZE
    train_result = train_crosscoder(
        activations_data=activations_data,
        input_dim=input_dim,
        base_model_id=base_model,
        aligned_run_id=aligned_run_id,
        layer=layer,
        position=position,
        results_dir=results_dir,
        batch_size=bs,
        use_amp=use_train_amp,
    )

    del activations_data
    flush_gpu()

    print("Training complete!")
    return train_result


def run_train_multilayer(
    base_model: str,
    aligned_run_id: str,
    layers: list[int],
    position: str,
    topk_mode: str = config.MULTILAYER_TOPK_MODE,
    output_dir: Optional[Path] = None,
    train_batch_size: Optional[int] = None,
    use_train_amp: Optional[bool] = None,
):
    from .multilayer_train import train_multilayer_crosscoder

    results_dir = _resolve_results_dir_multilayer(
        base_model, aligned_run_id, layers, position, output_dir
    )
    topk_mode = _resolve_multilayer_topk_mode(results_dir, topk_mode, prefer_artifact=True)

    print(f"\n{'='*60}")
    print(f"TRAINING: {base_model} / {aligned_run_id} {layers_slug(layers)} {position} ({topk_mode})")
    print(f"{'='*60}")

    activations_dir = get_activations_dir(results_dir)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    if (checkpoint_dir / "final.pt").exists():
        print(f"Checkpoint already exists at {checkpoint_dir / 'final.pt'}, skipping training.")
        return None

    activations_data = _load_required_activations(activations_dir / "activations.pt")
    _validate_multilayer_activations(activations_data, layers)
    n_layers = int(activations_data["activations_base"].shape[1])
    input_dim = int(activations_data.get("hidden_size", activations_data["activations_base"].shape[2]))
    bs = train_batch_size if train_batch_size is not None else config.BATCH_SIZE
    train_result = train_multilayer_crosscoder(
        activations_data=activations_data,
        input_dim=input_dim,
        n_layers=n_layers,
        base_model_id=base_model,
        aligned_run_id=aligned_run_id,
        layers=layers,
        position=position,
        topk_mode=topk_mode,
        results_dir=results_dir,
        batch_size=bs,
        use_amp=use_train_amp,
    )

    del activations_data
    flush_gpu()
    print("Multi-layer training complete!")
    return train_result


def run_analyze(
    base_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    output_dir: Optional[Path] = None,
    n_jobs_superposition: int = 1,
):
    from .classify import classify_all_features, save_classification_results
    from .counterfactual import (
        classify_cf_level,
        compute_cf_shift_by_class,
        compute_counterfactual_sensitivity,
        identify_visual_evidence_features,
        merge_classification_with_cf,
        save_cf_results,
    )
    from .metrics import (
        compute_all_primary_metrics,
        get_shared_features_geometry_df,
        save_metrics,
        summarize_shared_geometry,
    )
    from .superposition import analyze_all_aligned_only_features, save_superposition_results
    from .train import compute_all_feature_activations, load_trained_crosscoder

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {base_model} / {aligned_run_id} L{layer} {position}")
    print(f"{'='*60}")

    results_dir = _resolve_results_dir(
        base_model, aligned_run_id, layer, position, output_dir
    )
    activations_dir = get_activations_dir(results_dir)
    features_dir = get_features_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)

    aggregate_path = metrics_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        print(f"Analysis outputs already exist at {aggregate_path}, skipping analysis.")
        return

    activations_data = load_activations(activations_dir / "activations.pt")
    input_dim = int(activations_data.get("hidden_size", activations_data["activations_base"].shape[1]))

    print("Loading trained cross-coder...")
    crosscoder = load_trained_crosscoder(
        input_dim,
        base_model,
        aligned_run_id,
        layer,
        position,
        results_dir=results_dir,
    )

    print("Computing feature activations...")
    feature_activations = compute_all_feature_activations(crosscoder, activations_data)

    # Free GPU memory — remaining crosscoder use is decoder weight reads, not forward passes
    crosscoder.cpu()
    import torch as _torch; _torch.cuda.empty_cache()

    print("Classifying features...")
    classification_df = classify_all_features(crosscoder)
    save_classification_results(classification_df, features_dir / "feature_classification.csv")

    print("Computing sensitivity (base vs aligned latent usage)...")
    cf_scores_df = compute_counterfactual_sensitivity(feature_activations)
    cf_scores_df = classify_cf_level(cf_scores_df)
    save_cf_results(cf_scores_df, features_dir / "counterfactual_scores.csv")

    merged_df = merge_classification_with_cf(classification_df, cf_scores_df)
    merged_df.to_csv(features_dir / "merged_classification.csv", index=False)

    cf_shift_by_class = compute_cf_shift_by_class(merged_df)
    save_json(cf_shift_by_class, metrics_dir / "cf_shift_by_class.json")

    visual_evidence = identify_visual_evidence_features(merged_df)
    save_json(visual_evidence, features_dir / "visual_evidence_features.json")

    print("Analyzing superposition (aligned-only features)...")
    superposition_results = analyze_all_aligned_only_features(
        crosscoder, classification_df, feature_activations, aligned_run_id,
        n_jobs=n_jobs_superposition,
    )
    print("Saving superposition results...")
    save_superposition_results(superposition_results, features_dir / "superposition_analysis.json")

    print("Computing shared feature geometry (CPU: pinv + SVD per class)...")
    decoder_weights = crosscoder.get_decoder_weights()
    shared_geometry = summarize_shared_geometry(
        classification_df, decoder_weights["W_base_dec"], decoder_weights["W_aligned_dec"]
    )
    save_json(shared_geometry, metrics_dir / "shared_geometry_metrics.json")

    print("Building per-feature geometry dataframe...")
    shared_geom_df = get_shared_features_geometry_df(
        classification_df, decoder_weights["W_base_dec"], decoder_weights["W_aligned_dec"]
    )
    if len(shared_geom_df) > 0:
        shared_geom_df.to_csv(features_dir / "shared_features_geometry.csv", index=False)

    print("Computing aggregate metrics...")
    training_history = load_json(metrics_dir / "training_metrics.json")

    aggregate_metrics = compute_all_primary_metrics(
        classification_df, merged_df, superposition_results, training_history
    )
    save_metrics(aggregate_metrics, metrics_dir / "aggregate_metrics.json")

    print(f"\nResults saved to: {results_dir}")
    print("\nAnalysis complete!")


def run_analyze_multilayer(
    base_model: str,
    aligned_run_id: str,
    layers: list[int],
    position: str,
    topk_mode: str = config.MULTILAYER_TOPK_MODE,
    output_dir: Optional[Path] = None,
):
    from .metrics import save_metrics
    from .multilayer_classify import (
        classify_multilayer_features,
        get_multilayer_class_counts,
        multilayer_decoder_profile_df,
    )
    from .multilayer_train import compute_all_multilayer_feature_activations, load_trained_multilayer_crosscoder

    results_dir = _resolve_results_dir_multilayer(
        base_model, aligned_run_id, layers, position, output_dir
    )
    topk_mode = _resolve_multilayer_topk_mode(results_dir, topk_mode, prefer_artifact=True)

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {base_model} / {aligned_run_id} {layers_slug(layers)} {position} ({topk_mode})")
    print(f"{'='*60}")

    activations_dir = get_activations_dir(results_dir)
    features_dir = get_features_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)

    aggregate_path = metrics_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        print(f"Analysis outputs already exist at {aggregate_path}, skipping analysis.")
        return

    activations_data = _load_required_activations(activations_dir / "activations.pt")
    _validate_multilayer_activations(activations_data, layers)
    n_layers = int(activations_data["activations_base"].shape[1])
    input_dim = int(activations_data.get("hidden_size", activations_data["activations_base"].shape[2]))

    print("Loading trained multi-layer cross-coder...")
    crosscoder = load_trained_multilayer_crosscoder(
        results_dir=results_dir,
        input_dim=input_dim,
        n_layers=n_layers,
        topk_mode=topk_mode,
        layers=layers,
    )

    print("Computing multi-layer feature activations...")
    feature_activations = compute_all_multilayer_feature_activations(crosscoder, activations_data)
    import torch as _torch
    _torch.save(feature_activations, features_dir / "feature_activations.pt")

    crosscoder.cpu()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    print("Classifying multi-layer features...")
    classification_df = classify_multilayer_features(crosscoder, layers)
    classification_df.to_csv(features_dir / "feature_classification.csv", index=False)

    print("Writing per-layer decoder profiles...")
    profile_df = multilayer_decoder_profile_df(crosscoder, layers)
    profile_df.to_csv(features_dir / "decoder_layer_profiles.csv", index=False)

    training_metrics_path = metrics_dir / "training_metrics.json"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing training metrics: {training_metrics_path}. Multi-layer analysis requires checkpoint, "
            "activations, and training_metrics.json."
        )
    training_history = load_json(training_metrics_path)
    class_counts = get_multilayer_class_counts(classification_df)
    aggregate_metrics = {
        "crosscoder_kind": "multilayer_sparc",
        "layers": layers,
        "topk_mode": topk_mode,
        "class_counts": class_counts,
        "total_features": int(len(classification_df)),
        "fve_base": training_history["val_fve_base"][-1] if training_history.get("val_fve_base") else 0.0,
        "fve_aligned": training_history["val_fve_aligned"][-1] if training_history.get("val_fve_aligned") else 0.0,
        "fve_base_by_layer": training_history["val_fve_base_by_layer"][-1]
        if training_history.get("val_fve_base_by_layer")
        else [],
        "fve_aligned_by_layer": training_history["val_fve_aligned_by_layer"][-1]
        if training_history.get("val_fve_aligned_by_layer")
        else [],
        "dead_neuron_fraction": training_history["dead_neurons"][-1]
        if training_history.get("dead_neurons")
        else 0.0,
        "l0_sparsity_base": training_history["l0_base"][-1] if training_history.get("l0_base") else 0.0,
        "l0_sparsity_aligned": training_history["l0_aligned"][-1] if training_history.get("l0_aligned") else 0.0,
    }
    save_metrics(aggregate_metrics, aggregate_path)

    print(f"\nResults saved to: {results_dir}")
    print("\nMulti-layer analysis complete!")


def run_visualize(
    base_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    force: bool = False,
    output_dir: Optional[Path] = None,
):
    from .visualize import generate_all_plots
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"VISUALIZATION: {base_model} / {aligned_run_id} L{layer} {position}")
    print(f"{'='*60}")

    results_dir = _resolve_results_dir(
        base_model, aligned_run_id, layer, position, output_dir
    )
    features_dir = get_features_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)
    plots_dir = get_plots_dir(results_dir)

    loss_curves_path = plots_dir / "loss_curves.png"
    if loss_curves_path.exists() and not force:
        print(f"Plots already exist at {plots_dir}, skipping. Use --force to regenerate.")
        return

    training_history = load_json(metrics_dir / "training_metrics.json")
    classification_df = pd.read_csv(features_dir / "feature_classification.csv")
    merged_df = pd.read_csv(features_dir / "merged_classification.csv")
    superposition_results = load_json(features_dir / "superposition_analysis.json")

    generate_all_plots(
        training_history=training_history,
        classification_df=classification_df,
        merged_df=merged_df,
        superposition_results=superposition_results,
        plots_dir=plots_dir,
    )

    print(f"\nPlots saved to: {plots_dir}")
    print("Visualization complete!")


def run_all(
    base_model: str,
    aligned_model: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    dataset_name: str,
    max_prompt_tokens: int,
    trust_remote_code: bool,
    force: bool = False,
    output_dir: Optional[Path] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    use_train_amp: Optional[bool] = None,
    n_jobs_superposition: int = 1,
):
    run_extract(
        base_model,
        aligned_model,
        aligned_run_id,
        layer,
        position,
        dataset_name,
        max_prompt_tokens,
        trust_remote_code,
        output_dir=output_dir,
        prompts_cache_dir=prompts_cache_dir,
        use_prompts_cache=use_prompts_cache,
        extract_batch_size=extract_batch_size,
    )
    flush_gpu()
    run_train(
        base_model,
        aligned_run_id,
        layer,
        position,
        output_dir=output_dir,
        train_batch_size=train_batch_size,
        use_train_amp=use_train_amp,
    )
    run_analyze(base_model, aligned_run_id, layer, position, output_dir=output_dir,
                n_jobs_superposition=n_jobs_superposition)
    run_visualize(
        base_model, aligned_run_id, layer, position, force=force, output_dir=output_dir
    )
    flush_gpu()


def run_all_multilayer(
    base_model: str,
    aligned_model: str,
    aligned_run_id: str,
    layers: list[int],
    position: str,
    dataset_name: str,
    max_prompt_tokens: int,
    trust_remote_code: bool,
    force: bool = False,
    output_dir: Optional[Path] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    use_train_amp: Optional[bool] = None,
    center_layer: Optional[int] = None,
    layer_window: Optional[int] = None,
    topk_mode: str = config.MULTILAYER_TOPK_MODE,
):
    run_extract_multilayer(
        base_model,
        aligned_model,
        aligned_run_id,
        layers,
        position,
        dataset_name,
        max_prompt_tokens,
        trust_remote_code,
        output_dir=output_dir,
        prompts_cache_dir=prompts_cache_dir,
        use_prompts_cache=use_prompts_cache,
        extract_batch_size=extract_batch_size,
        center_layer=center_layer,
        layer_window=layer_window,
        topk_mode=topk_mode,
    )
    flush_gpu()
    run_train_multilayer(
        base_model,
        aligned_run_id,
        layers,
        position,
        topk_mode=topk_mode,
        output_dir=output_dir,
        train_batch_size=train_batch_size,
        use_train_amp=use_train_amp,
    )
    run_analyze_multilayer(
        base_model,
        aligned_run_id,
        layers,
        position,
        topk_mode=topk_mode,
        output_dir=output_dir,
    )
    flush_gpu()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_manifest_gpu_ids(num_gpus: Optional[int], gpu_ids_str: Optional[str]) -> list[int]:
    if gpu_ids_str:
        gpu_ids = [int(part.strip()) for part in gpu_ids_str.split(",") if part.strip()]
        if not gpu_ids:
            raise ValueError(f"--gpu-ids {gpu_ids_str!r} parsed to an empty list")
        return gpu_ids

    detected = 0
    try:
        import torch  # noqa: PLC0415

        detected = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        detected = 0

    if num_gpus is None:
        n = detected if detected > 0 else 1
    else:
        n = max(1, int(num_gpus))
    if detected and n > detected:
        print(f"Requested {n} GPUs but only {detected} visible; clamping to {detected}.")
        n = detected
    return list(range(n))


def _manifest_bool(row: dict, key: str, default: bool = False) -> bool:
    if key not in row:
        return default
    value = row[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _manifest_layers(row: dict) -> Optional[list[int]]:
    layers = row.get("layers")
    if layers is None:
        return None
    if isinstance(layers, str):
        return _parse_layers_arg(layers)
    parsed = [int(layer) for layer in layers]
    if not parsed:
        raise ValueError("manifest row layers parsed to an empty list")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"manifest row layers contains duplicates: {parsed}")
    return parsed


def _manifest_layer_label(row: dict) -> str:
    if row.get("stage") == "assemble" and row.get("layers") is None:
        return "assembled"
    kind = row.get("crosscoder_kind", "single_layer")
    if kind == "multilayer_sparc":
        layers = _manifest_layers(row)
        if layers is not None:
            return layers_slug(layers)
        center = row.get("center_layer", row.get("layer"))
        if center is None:
            return "layers"
        window = int(row.get("layer_window", 1))
        return layers_slug(list(range(int(center) - window, int(center) + window + 1)))
    return f"L{int(row['layer'])}"


def _manifest_default_output_dir(row: dict, output_root: Optional[Path]) -> Optional[Path]:
    if row.get("output_dir"):
        return Path(row["output_dir"])
    if output_root is None:
        return None
    run_id = row.get("aligned_run_id")
    if not run_id:
        return None
    return output_root / run_id / _manifest_layer_label(row)


def _manifest_existing_complete(row: dict, output_root: Optional[Path]) -> bool:
    out_dir = _manifest_default_output_dir(row, output_root)
    stage = row.get("stage", "all")
    if out_dir is None and stage == "assemble":
        return False
    if out_dir is None:
        kind = row.get("crosscoder_kind", "single_layer")
        position = row.get("position", config.POSITION_LAST_PROMPT)
        if kind == "multilayer_sparc":
            layers = _manifest_layers(row)
            if layers is None:
                center = row.get("center_layer", row.get("layer"))
                if center is None:
                    return False
                window = int(row.get("layer_window", 1))
                layers = list(range(int(center) - window, int(center) + window + 1))
            out_dir = get_results_dir_multilayer(row["base_model"], row["aligned_run_id"], layers, position)
        else:
            out_dir = get_results_dir(row["base_model"], row["aligned_run_id"], int(row["layer"]), position)

    if stage == "assemble":
        return (out_dir / "activations" / "activations.pt").is_file()
    if stage == "extract":
        extract_side = row.get("extract_side", "both")
        if extract_side == "base":
            return (out_dir / "activations" / "base_activations.pt").is_file()
        if extract_side == "aligned":
            return (out_dir / "activations" / "aligned_activations.pt").is_file()
        return (out_dir / "activations" / "activations.pt").is_file()
    if stage == "train":
        return (out_dir / "checkpoints" / "final.pt").is_file()
    if stage in {"analyze", "all"}:
        return (out_dir / "metrics" / "aggregate_metrics.json").is_file()
    if stage == "visualize":
        return (out_dir / "plots").is_dir() and any((out_dir / "plots").iterdir())
    return False


def _append_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_manifest_cmd(row: dict, force: bool, output_root: Optional[Path]) -> tuple[list[str], Path]:
    stage = row.get("stage", "all")
    if stage == "manifest":
        raise ValueError("manifest rows cannot use stage=manifest")

    kind = row.get("crosscoder_kind", "single_layer")
    if kind not in {"single_layer", "multilayer_sparc"}:
        raise ValueError(f"Unsupported manifest crosscoder_kind={kind!r}")

    cmd = [
        sys.executable,
        "-m",
        "interp_utils.crosscoder.main",
        "--stage",
        stage,
        "--crosscoder-kind",
        kind,
    ]

    if row.get("base_model") is not None:
        cmd.extend(["--base-model", row["base_model"]])
    if row.get("aligned_run_id") is not None:
        cmd.extend(["--aligned-run-id", row["aligned_run_id"]])
    if row.get("aligned_model") is not None:
        cmd.extend(["--aligned-model", row["aligned_model"]])

    if kind == "multilayer_sparc":
        layers = _manifest_layers(row)
        if layers is not None:
            cmd.extend(["--layers", ",".join(str(layer) for layer in layers)])
        elif row.get("center_layer") is not None:
            cmd.extend(["--center-layer", str(int(row["center_layer"]))])
        elif row.get("layer") is not None:
            cmd.extend(["--layer", str(int(row["layer"]))])
        _append_arg(cmd, "--layer-window", row.get("layer_window"))
        _append_arg(cmd, "--topk-mode", row.get("topk_mode"))
    else:
        cmd.extend(["--layer", str(int(row["layer"]))])

    _append_arg(cmd, "--position", row.get("position"))
    _append_arg(cmd, "--dataset-name", row.get("dataset_name"))
    _append_arg(cmd, "--max-prompt-tokens", row.get("max_prompt_tokens"))
    _append_arg(cmd, "--extract-batch-size", row.get("extract_batch_size"))
    _append_arg(cmd, "--train-batch-size", row.get("train_batch_size"))
    _append_arg(cmd, "--n-jobs-superposition", row.get("n_jobs_superposition"))
    _append_arg(cmd, "--prompts-cache-dir", row.get("prompts_cache_dir"))
    _append_arg(cmd, "--extract-side", row.get("extract_side"))
    _append_arg(cmd, "--base-activations-dir", row.get("base_activations_dir"))
    _append_arg(cmd, "--aligned-activations-dir", row.get("aligned_activations_dir"))

    if _manifest_bool(row, "trust_remote_code", False):
        cmd.append("--trust-remote-code")
    if _manifest_bool(row, "no_prompts_cache", False) or not _manifest_bool(row, "use_prompts_cache", True):
        cmd.append("--no-prompts-cache")
    if _manifest_bool(row, "no_train_amp", False) or row.get("use_train_amp") is False:
        cmd.append("--no-train-amp")
    if force or _manifest_bool(row, "force", False):
        cmd.append("--force")

    out_dir = _manifest_default_output_dir(row, output_root)
    if out_dir is not None:
        cmd.extend(["--output-dir", str(out_dir)])

    return cmd, Path(out_dir) if out_dir is not None else Path(".")


def _manifest_log_path(row: dict, log_dir: Path) -> Path:
    if row.get("log_path"):
        return Path(row["log_path"])
    run_id = row.get("aligned_run_id") or row.get("name") or "manifest-job"
    stage = row.get("stage", "all")
    layer_label = _manifest_layer_label(row)
    safe = sanitize_model_slug(f"{run_id}-{stage}-{layer_label}")
    return log_dir / f"{safe}.log"


def run_manifest(
    manifest_path: Path,
    force: bool = False,
    *,
    num_gpus: Optional[int] = None,
    gpu_ids: Optional[str] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    output_root: Optional[Path] = None,
    log_dir: Optional[Path] = None,
):
    with open(manifest_path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a JSON list of job objects: {manifest_path}")

    output_root = Path(output_root) if output_root is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else _repo_root() / "logs" / "crosscoder"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_pool = _resolve_manifest_gpu_ids(num_gpus, gpu_ids)
    runnable: list[dict] = []
    skipped = 0
    for row in rows:
        if skip_existing and _manifest_existing_complete(row, output_root):
            skipped += 1
            print(f"skip (existing): {row.get('aligned_run_id', row.get('name', 'manifest-job'))} {_manifest_layer_label(row)}")
            continue
        runnable.append(row)

    print(f"Manifest:    {manifest_path}")
    print(f"Jobs:        {len(runnable)} runnable / {len(rows)} total (skipped={skipped})")
    print(f"GPU pool:    {gpu_pool}  (parallelism={len(gpu_pool)})")
    print(f"Log dir:     {log_dir}")
    if output_root is not None:
        print(f"Output root: {output_root}")

    if dry_run:
        for i, row in enumerate(runnable, start=1):
            cmd, _ = _build_manifest_cmd(row, force=force, output_root=output_root)
            print(f"DRY {i}: {' '.join(cmd)}")
        return

    pool: Queue[int] = Queue()
    for gpu in gpu_pool:
        pool.put(gpu)

    def _worker(row: dict) -> tuple[str, str, int, int, str]:
        gpu = pool.get()
        run_id = row.get("aligned_run_id", row.get("name", "manifest-job"))
        layer_label = _manifest_layer_label(row)
        log_path = _manifest_log_path(row, log_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cmd, out_dir = _build_manifest_cmd(row, force=force, output_root=output_root)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w") as f:
                f.write(
                    f"# manifest = {manifest_path}\n"
                    f"# run_id   = {run_id}\n"
                    f"# stage    = {row.get('stage', 'all')}\n"
                    f"# kind     = {row.get('crosscoder_kind', 'single_layer')}\n"
                    f"# layers   = {layer_label}\n"
                    f"# gpu      = {gpu}\n"
                    f"# output   = {out_dir}\n"
                    f"# cmd      = {' '.join(cmd)}\n\n"
                )
                f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            return run_id, layer_label, gpu, int(proc.returncode), str(log_path)
        finally:
            pool.put(gpu)

    failures: list[tuple[str, str, int, str]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
        futures = {executor.submit(_worker, row): row for row in runnable}
        with tqdm(total=len(runnable), desc="manifest", unit="job") as pbar:
            for future in as_completed(futures):
                run_id, layer_label, gpu, rc, log_path = future.result()
                tag = "OK" if rc == 0 else f"FAIL({rc})"
                tqdm.write(f"[GPU{gpu}] {tag} {run_id} {layer_label}  log={log_path}")
                if rc != 0:
                    failures.append((run_id, layer_label, rc, log_path))
                pbar.update(1)

    if failures:
        print("\nSome manifest jobs failed:", file=sys.stderr)
        for run_id, layer_label, rc, log_path in failures:
            print(f"  - {run_id} {layer_label}: exit {rc}  log={log_path}", file=sys.stderr)
        raise SystemExit(1)



def main():
    parser = argparse.ArgumentParser(
        description="SPARC Cross-Coder: base vs aligned LLM activations (GRPO-style preference data)",
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("--base-model", type=str, default=None, help="Base HF model id")
    parser.add_argument(
        "--crosscoder-kind",
        type=str,
        default="single_layer",
        choices=["single_layer", "multilayer_sparc"],
        help="Use the existing single-layer pipeline or the new multi-layer SPARC-style pipeline.",
    )
    parser.add_argument(
        "--aligned-model",
        type=str,
        default=None,
        help="Aligned checkpoint: HF id, local dir (merged weights), or PEFT adapter dir",
    )
    parser.add_argument(
        "--aligned-run-id",
        type=str,
        default=None,
        help="Short slug for artifact directory naming",
    )
    parser.add_argument("--layer", type=int, default=None, help="Decoder layer index for hook")
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated decoder layers for multi-layer runs. Overrides --center-layer/--layer-window.",
    )
    parser.add_argument(
        "--center-layer",
        type=int,
        default=None,
        help="Center decoder layer for multi-layer runs.",
    )
    parser.add_argument(
        "--layer-window",
        type=int,
        default=1,
        help="Layer radius for multi-layer runs; center +/- window.",
    )
    parser.add_argument(
        "--topk-mode",
        type=str,
        default=config.MULTILAYER_TOPK_MODE,
        choices=list(config.MULTILAYER_TOPK_MODES),
        help="TopK support rule for multi-layer crosscoders.",
    )
    parser.add_argument(
        "--position",
        type=str,
        default=config.POSITION_LAST_PROMPT,
        choices=list(config.POSITION_CHOICES),
        help="Pooling over prompt hidden states",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=config.PREFERENCE_DATASET_NAME,
        help="HF dataset for prompts (preference format)",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=config.MAX_PROMPT_TOKENS,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "extract",
            "train",
            "analyze",
            "visualize",
            "assemble",
            "all",
            "manifest",
            "hypothesis_tests",
        ],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON list of jobs for stage=manifest",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="For stage=manifest: number of visible GPUs to schedule across. Defaults to torch.cuda.device_count().",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="For stage=manifest: explicit comma-separated physical GPU ids, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--manifest-dry-run",
        action="store_true",
        help="For stage=manifest: print planned child CLI commands without launching them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="For stage=manifest: skip rows whose expected stage artifact already exists.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="For stage=manifest: root for per-row output dirs when a row omits output_dir.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="For stage=manifest: directory for per-row stdout/stderr logs.",
    )
    parser.add_argument(
        "--extract-side",
        type=str,
        default="both",
        choices=["both", "base", "aligned"],
        help=(
            "For multilayer extraction: run both models, base-only, or aligned-only. "
            "Side-specific extraction writes base_activations.pt or aligned_activations.pt."
        ),
    )
    parser.add_argument(
        "--base-activations-dir",
        type=str,
        default=None,
        help="For stage=assemble: base side artifact dir or .pt path.",
    )
    parser.add_argument(
        "--aligned-activations-dir",
        type=str,
        default=None,
        help="For stage=assemble: aligned side artifact dir or .pt path.",
    )
    parser.add_argument(
        "--prompts-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for reusable normalized-prompt Arrow cache "
            f"(default: {config.NORMALIZED_PROMPTS_CACHE_DIR})"
        ),
    )
    parser.add_argument(
        "--no-prompts-cache",
        action="store_true",
        help="Disable load/save of normalized prompts cache (always normalize from HF)",
    )
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Microbatch size for LLM forward during extraction (default: config.EXTRACT_BATCH_SIZE). "
            "Lower this first if VRAM spikes with two models loaded."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=f"Crosscoder training batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-train-amp",
        action="store_true",
        help="Disable autocast (bf16/fp16) during crosscoder training",
    )
    parser.add_argument(
        "--n-jobs-superposition",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel jobs for superposition analysis (default: 1, use -1 for all cores)",
    )

    args = parser.parse_args()
    set_seed()
    output_dir = Path(args.output_dir) if args.output_dir else None
    prompts_cache_dir = Path(args.prompts_cache_dir) if args.prompts_cache_dir else None
    use_prompts_cache = not args.no_prompts_cache

    if args.stage == "manifest":
        if not args.manifest:
            parser.error("--manifest required for stage=manifest")
        run_manifest(
            Path(args.manifest),
            force=args.force,
            num_gpus=args.num_gpus,
            gpu_ids=args.gpu_ids,
            dry_run=args.manifest_dry_run,
            skip_existing=args.skip_existing,
            output_root=Path(args.output_root) if args.output_root else None,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
        return

    if args.stage == "assemble":
        if args.crosscoder_kind != "multilayer_sparc":
            parser.error("stage=assemble is only implemented for --crosscoder-kind multilayer_sparc")
        if not args.base_activations_dir:
            parser.error("--base-activations-dir is required for stage=assemble")
        if not args.aligned_activations_dir:
            parser.error("--aligned-activations-dir is required for stage=assemble")
        if output_dir is None:
            parser.error("--output-dir is required for stage=assemble")

    # For all non-manifest/non-assemble stages, these args are required
    required = []
    if args.stage != "assemble":
        required.extend([
            ("--base-model", args.base_model),
            ("--aligned-run-id", args.aligned_run_id),
        ])
    if args.stage in {"extract", "all"} and not (
        args.crosscoder_kind == "multilayer_sparc" and args.extract_side == "base"
    ):
        required.append(("--aligned-model", args.aligned_model))
    if args.crosscoder_kind == "single_layer" and args.stage != "assemble":
        required.append(("--layer", args.layer))
    for name, val in required:
        if val is None:
            parser.error(f"{name} is required for stage={args.stage}")
    if args.stage == "all" and args.extract_side != "both":
        parser.error("--extract-side must be both for stage=all")

    use_train_amp = False if args.no_train_amp else None

    if args.crosscoder_kind == "multilayer_sparc":
        if args.stage == "assemble":
            requested_layers = None
            if _explicit_multilayer_layer_args_present(args):
                requested_layers = _resolve_multilayer_layers(
                    layer=args.layer,
                    center_layer=args.center_layer,
                    layer_window=args.layer_window,
                    layers_arg=args.layers,
                )
            run_assemble_multilayer(
                base_activations_path=Path(args.base_activations_dir),
                aligned_activations_path=Path(args.aligned_activations_dir),
                output_dir=output_dir,
                requested_layers=requested_layers,
                topk_mode=args.topk_mode,
            )
            return

        layers = _resolve_multilayer_layers_for_stage(args=args, output_dir=output_dir)
        center_layer = args.center_layer if args.center_layer is not None else args.layer
        if args.stage == "visualize":
            parser.error("visualize is not implemented for --crosscoder-kind multilayer_sparc")
        if args.stage == "extract":
            run_extract_multilayer(
                args.base_model,
                args.aligned_model,
                args.aligned_run_id,
                layers,
                args.position,
                args.dataset_name,
                args.max_prompt_tokens,
                args.trust_remote_code,
                output_dir=output_dir,
                prompts_cache_dir=prompts_cache_dir,
                use_prompts_cache=use_prompts_cache,
                extract_batch_size=args.extract_batch_size,
                center_layer=center_layer,
                layer_window=args.layer_window,
                topk_mode=args.topk_mode,
                extract_side=args.extract_side,
            )
        elif args.stage == "train":
            run_train_multilayer(
                args.base_model,
                args.aligned_run_id,
                layers,
                args.position,
                topk_mode=args.topk_mode,
                output_dir=output_dir,
                train_batch_size=args.train_batch_size,
                use_train_amp=use_train_amp,
            )
        elif args.stage == "analyze":
            run_analyze_multilayer(
                args.base_model,
                args.aligned_run_id,
                layers,
                args.position,
                topk_mode=args.topk_mode,
                output_dir=output_dir,
            )
        elif args.stage == "all":
            run_all_multilayer(
                args.base_model,
                args.aligned_model,
                args.aligned_run_id,
                layers,
                args.position,
                args.dataset_name,
                args.max_prompt_tokens,
                args.trust_remote_code,
                force=args.force,
                output_dir=output_dir,
                prompts_cache_dir=prompts_cache_dir,
                use_prompts_cache=use_prompts_cache,
                extract_batch_size=args.extract_batch_size,
                train_batch_size=args.train_batch_size,
                use_train_amp=use_train_amp,
                center_layer=center_layer,
                layer_window=args.layer_window,
                topk_mode=args.topk_mode,
            )
        return

    if args.stage == "extract":
        run_extract(
            args.base_model,
            args.aligned_model,
            args.aligned_run_id,
            args.layer,
            args.position,
            args.dataset_name,
            args.max_prompt_tokens,
            args.trust_remote_code,
            output_dir=output_dir,
            prompts_cache_dir=prompts_cache_dir,
            use_prompts_cache=use_prompts_cache,
            extract_batch_size=args.extract_batch_size,
        )
    elif args.stage == "train":
        run_train(
            args.base_model,
            args.aligned_run_id,
            args.layer,
            args.position,
            output_dir=output_dir,
            train_batch_size=args.train_batch_size,
            use_train_amp=use_train_amp,
        )
    elif args.stage == "analyze":
        run_analyze(
            args.base_model,
            args.aligned_run_id,
            args.layer,
            args.position,
            output_dir=output_dir,
            n_jobs_superposition=args.n_jobs_superposition,
        )
    elif args.stage == "visualize":
        run_visualize(
            args.base_model,
            args.aligned_run_id,
            args.layer,
            args.position,
            force=args.force,
            output_dir=output_dir,
        )
    elif args.stage == "all":
        run_all(
            args.base_model,
            args.aligned_model,
            args.aligned_run_id,
            args.layer,
            args.position,
            args.dataset_name,
            args.max_prompt_tokens,
            args.trust_remote_code,
            force=args.force,
            output_dir=output_dir,
            prompts_cache_dir=prompts_cache_dir,
            use_prompts_cache=use_prompts_cache,
            extract_batch_size=args.extract_batch_size,
            train_batch_size=args.train_batch_size,
            use_train_amp=use_train_amp,
        )


if __name__ == "__main__":
    main()
