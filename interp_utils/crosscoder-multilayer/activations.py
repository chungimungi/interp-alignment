import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .dataset import PreferenceActivationDataset
from .utils import flush_gpu, get_device, set_seed
from peft import PeftModel


def _model_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _unwrap_base_model(model: nn.Module) -> nn.Module:
    if PeftModel is not None and isinstance(model, PeftModel):
        return model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    return model


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve transformer decoder layers for common causal LM layouts."""
    base = _unwrap_base_model(model)
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return base.transformer.h
    if hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
        return base.gpt_neox.layers
    raise ValueError("Could not find decoder layers on model; expected Llama-like .model.layers or GPT-2-like .transformer.h")


def _is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def _read_adapter_base(path: Path) -> Optional[str]:
    with open(path / "adapter_config.json") as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path")


def load_tokenizer(aligned_path: str, base_model_id: str, trust_remote_code: bool = False):
    p = Path(aligned_path)
    tok_kw = {"trust_remote_code": trust_remote_code}
    if p.is_dir() and any((p / x).exists() for x in ("tokenizer.json", "tokenizer_config.json")):
        tok = AutoTokenizer.from_pretrained(str(p), **tok_kw)
    else:
        tok = AutoTokenizer.from_pretrained(base_model_id, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_llm(base_model_id: str, trust_remote_code: bool = False) -> AutoModelForCausalLM:
    device = get_device()
    dtype = _model_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    return model


def load_aligned_llm(aligned_path: str, base_model_id: str, trust_remote_code: bool = False) -> Tuple[AutoModelForCausalLM, bool]:
    """
    Load aligned checkpoint: full weights or PEFT adapter directory.
    Returns (model, is_peft).
    """
    device = get_device()
    dtype = _model_dtype()
    path = Path(aligned_path)

    if path.is_dir() and _is_peft_adapter_dir(path):
        if PeftModel is None:
            raise ImportError("peft is required for PEFT adapter loading")
        resolved_base = base_model_id
        adapter_base = _read_adapter_base(path)
        if adapter_base and not base_model_id:
            resolved_base = adapter_base
        if not resolved_base:
            raise ValueError("PEFT adapter requires base_model_id or base_model_name_or_path in adapter_config.json")

        base = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            device_map=None,
        )
        model = PeftModel.from_pretrained(base, str(path))
        model = model.to(device)
        model.eval()
        return model, True

    model = AutoModelForCausalLM.from_pretrained(
        aligned_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    return model, False


class LayerActivationExtractor:
    """Captures hidden states at a decoder layer (output tensor)."""

    def __init__(self, model: nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.layers = get_decoder_layers(model)
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(self.layers)})")
        self._stored: Optional[torch.Tensor] = None
        self._handle = self.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            self._stored = out[0].detach()
        else:
            self._stored = out.detach()

    def clear(self):
        self._stored = None

    def get(self) -> torch.Tensor:
        if self._stored is None:
            raise RuntimeError("No activation captured; run forward first")
        return self._stored

    def remove(self):
        self._handle.remove()


class MultiLayerActivationExtractor:
    """Captures hidden states at multiple decoder layers in a fixed order."""

    def __init__(self, model: nn.Module, layer_indices: Sequence[int]):
        self.layer_indices = [int(layer_idx) for layer_idx in layer_indices]
        if not self.layer_indices:
            raise ValueError("layer_indices must be non-empty")
        self.layers = get_decoder_layers(model)
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(self.layers)})")
        self._stored: Dict[int, torch.Tensor] = {}
        self._handles = [
            self.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            for layer_idx in self.layer_indices
        ]

    def _make_hook(self, layer_idx: int):
        def _hook(module, inp, out):
            if isinstance(out, tuple):
                self._stored[layer_idx] = out[0].detach()
            else:
                self._stored[layer_idx] = out.detach()

        return _hook

    def clear(self):
        self._stored.clear()

    def get_all(self) -> List[torch.Tensor]:
        missing = [layer_idx for layer_idx in self.layer_indices if layer_idx not in self._stored]
        if missing:
            raise RuntimeError(f"No activation captured for layer(s): {missing}")
        return [self._stored[layer_idx] for layer_idx in self.layer_indices]

    def remove(self):
        for handle in self._handles:
            handle.remove()


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, position: str) -> torch.Tensor:
    """
    hidden: (batch, seq, dim)
    attention_mask: (batch, seq) 1 = real token
    """
    if position == config.POSITION_MEAN_PROMPT:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (summed / denom).squeeze(1)

    # last non-pad token (left padding: last position is valid)
    if position != config.POSITION_LAST_PROMPT:
        raise ValueError(f"Unknown position {position}")

    positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
    seq_lens = (attention_mask.to(dtype=positions.dtype) * positions).max(dim=1).values
    b = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[b, seq_lens, :]


def _pool_layer_list(
    hidden_layers: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    position: str,
) -> torch.Tensor:
    pooled = [_pool_hidden(hidden, attention_mask, position) for hidden in hidden_layers]
    return torch.stack(pooled, dim=1)


def extract_activations_llm(
    base_model_id: str,
    aligned_model_path: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    dataset_name: str = config.PREFERENCE_DATASET_NAME,
    max_prompt_tokens: int = config.MAX_PROMPT_TOKENS,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
    base_activations_cache: Optional[Dict] = None,
) -> Dict[str, Any]:
    set_seed()
    device = get_device()
    tokenizer = load_tokenizer(aligned_model_path, base_model_id, trust_remote_code=trust_remote_code)

    ds = PreferenceActivationDataset(
        tokenizer,
        dataset_name=dataset_name,
        split="all",
        max_prompt_tokens=max_prompt_tokens,
        hf_token=hf_token,
        prompts_cache_dir=Path(prompts_cache_dir) if prompts_cache_dir else None,
        use_prompts_cache=use_prompts_cache,
    )

    use_base_cache = base_activations_cache is not None
    if use_base_cache:
        print(f"Using cached base activations — skipping base LLM load")
        model_base = None
        ext_base = None
        hidden_size = base_activations_cache["hidden_size"]
    else:
        print(f"Loading base LLM: {base_model_id}")
        model_base = load_base_llm(base_model_id, trust_remote_code=trust_remote_code)
        ext_base = LayerActivationExtractor(model_base, layer)
        hidden_size = model_base.config.hidden_size

    print(f"Loading aligned LLM: {aligned_model_path}")
    model_aligned, is_peft = load_aligned_llm(aligned_model_path, base_model_id, trust_remote_code=trust_remote_code)
    ext_aligned = LayerActivationExtractor(model_aligned, layer)

    if model_aligned.config.hidden_size != hidden_size:
        raise ValueError(f"Hidden size mismatch: base {hidden_size} vs aligned {model_aligned.config.hidden_size}")

    activations_base_list: List[torch.Tensor] = []
    activations_aligned_list: List[torch.Tensor] = []
    sample_ids: List[str] = []
    splits: List[str] = []

    batch_size = extract_batch_size if extract_batch_size is not None else config.EXTRACT_BATCH_SIZE
    flush_interval = getattr(config, "FLUSH_GPU_EVERY_N_BATCHES", 50)
    progress_interval = max(1, int(getattr(config, "PROGRESS_LOG_EVERY_N_BATCHES", 100)))

    items = [ds[i] for i in range(len(ds))]
    total_batches = (len(items) + batch_size - 1) // batch_size

    desc = "Batches (aligned only)" if use_base_cache else "Batches"
    print(f"Extracting LLM activations (batched forward, batch_size={batch_size})...")
    for batch_idx, start in enumerate(tqdm(range(0, len(items), batch_size), desc=desc)):
        chunk = items[start : start + batch_size]
        prompts = [it["prompt"] for it in chunk]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens + 64)
        enc = {k: v.to(device) for k, v in enc.items()}

        if not use_base_cache:
            ext_base.clear()
            with torch.no_grad():
                model_base(**enc)
                h_b = ext_base.get()
                pooled_b = _pool_hidden(h_b, enc["attention_mask"], position)

        ext_aligned.clear()
        with torch.no_grad():
            model_aligned(**enc)
            h_a = ext_aligned.get()
            pooled_a = _pool_hidden(h_a, enc["attention_mask"], position)

        for j in range(pooled_a.shape[0]):
            if not use_base_cache:
                activations_base_list.append(pooled_b[j].cpu().float())
            activations_aligned_list.append(pooled_a[j].cpu().float())
            sample_ids.append(chunk[j]["sample_id"])
            splits.append(chunk[j]["split"])

        if (batch_idx + 1) % flush_interval == 0:
            flush_gpu()
        if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
            print(f"progress: extraction batch {batch_idx + 1}/{total_batches}", flush=True)

    if ext_base is not None:
        ext_base.remove()
    ext_aligned.remove()

    activations_aligned = torch.stack(activations_aligned_list, dim=0)

    if use_base_cache:
        if sample_ids != base_activations_cache["sample_ids"]:
            raise RuntimeError(
                "Sample ID mismatch between base activation cache and current dataset extraction. "
                "Cache may be stale or dataset ordering changed."
            )
        activations_base = base_activations_cache["activations_base"]
    else:
        activations_base = torch.stack(activations_base_list, dim=0)

    models_to_del = [m for m in [model_base, model_aligned] if m is not None]
    del models_to_del, tokenizer
    flush_gpu()

    return {
        "activations_base": activations_base,
        "activations_aligned": activations_aligned,
        "sample_ids": sample_ids,
        "splits": splits,
        "base_model": base_model_id,
        "aligned_model": aligned_model_path,
        "aligned_run_id": aligned_run_id,
        "layer": layer,
        "position": position,
        "dataset_name": dataset_name,
        "peft": is_peft,
        "hidden_size": hidden_size,
    }


def extract_activations_llm_multilayer(
    base_model_id: str,
    aligned_model_path: Optional[str],
    aligned_run_id: str,
    layers: Sequence[int],
    position: str,
    dataset_name: str = config.PREFERENCE_DATASET_NAME,
    max_prompt_tokens: int = config.MAX_PROMPT_TOKENS,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    prompts_cache_dir: Optional[Path] = None,
    use_prompts_cache: bool = True,
    extract_batch_size: Optional[int] = None,
    base_activations_cache: Optional[Dict] = None,
    extract_side: str = "both",
) -> Dict[str, Any]:
    set_seed()
    device = get_device()
    layers = [int(layer) for layer in layers]
    if not layers:
        raise ValueError("layers must be non-empty")
    if extract_side not in {"both", "base", "aligned"}:
        raise ValueError(f"extract_side must be one of both/base/aligned, got {extract_side!r}")
    if extract_side in {"both", "aligned"} and not aligned_model_path:
        raise ValueError("aligned_model_path is required for extract_side='both' or 'aligned'")

    tokenizer_source = aligned_model_path if aligned_model_path else base_model_id
    tokenizer = load_tokenizer(tokenizer_source, base_model_id, trust_remote_code=trust_remote_code)

    ds = PreferenceActivationDataset(
        tokenizer,
        dataset_name=dataset_name,
        split="all",
        max_prompt_tokens=max_prompt_tokens,
        hf_token=hf_token,
        prompts_cache_dir=Path(prompts_cache_dir) if prompts_cache_dir else None,
        use_prompts_cache=use_prompts_cache,
    )

    use_base_cache = base_activations_cache is not None and extract_side == "both"
    if use_base_cache:
        cached_layers = [int(layer) for layer in base_activations_cache.get("layers", [])]
        if cached_layers != layers:
            raise ValueError(f"Base activation cache layers {cached_layers} do not match requested layers {layers}")
        print("Using cached multi-layer base activations — skipping base LLM load")
        model_base = None
        ext_base = None
        hidden_size = base_activations_cache["hidden_size"]
    elif extract_side in {"both", "base"}:
        print(f"Loading base LLM: {base_model_id}")
        model_base = load_base_llm(base_model_id, trust_remote_code=trust_remote_code)
        ext_base = MultiLayerActivationExtractor(model_base, layers)
        hidden_size = model_base.config.hidden_size
    else:
        model_base = None
        ext_base = None
        hidden_size = None

    if extract_side in {"both", "aligned"}:
        print(f"Loading aligned LLM: {aligned_model_path}")
        model_aligned, is_peft = load_aligned_llm(aligned_model_path, base_model_id, trust_remote_code=trust_remote_code)
        ext_aligned = MultiLayerActivationExtractor(model_aligned, layers)

        if hidden_size is None:
            hidden_size = model_aligned.config.hidden_size
        elif model_aligned.config.hidden_size != hidden_size:
            raise ValueError(f"Hidden size mismatch: base {hidden_size} vs aligned {model_aligned.config.hidden_size}")
    else:
        model_aligned = None
        ext_aligned = None
        is_peft = False

    activations_base_list: List[torch.Tensor] = []
    activations_aligned_list: List[torch.Tensor] = []
    sample_ids: List[str] = []
    splits: List[str] = []
    prompt_texts: List[str] = []

    batch_size = extract_batch_size if extract_batch_size is not None else config.EXTRACT_BATCH_SIZE
    flush_interval = getattr(config, "FLUSH_GPU_EVERY_N_BATCHES", 50)
    progress_interval = max(1, int(getattr(config, "PROGRESS_LOG_EVERY_N_BATCHES", 100)))
    items = [ds[i] for i in range(len(ds))]
    total_batches = (len(items) + batch_size - 1) // batch_size

    if extract_side == "base":
        desc = "Batches (base only, multi-layer)"
    elif extract_side == "aligned" or use_base_cache:
        desc = "Batches (aligned only, multi-layer)"
    else:
        desc = "Batches (multi-layer)"
    print(f"Extracting multi-layer LLM activations (side={extract_side}, layers={layers}, batch_size={batch_size})...")
    for batch_idx, start in enumerate(tqdm(range(0, len(items), batch_size), desc=desc)):
        chunk = items[start : start + batch_size]
        prompts = [it["prompt"] for it in chunk]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens + 64)
        enc = {k: v.to(device) for k, v in enc.items()}

        if extract_side in {"both", "base"} and not use_base_cache:
            ext_base.clear()
            with torch.no_grad():
                model_base(**enc)
                pooled_b = _pool_layer_list(ext_base.get_all(), enc["attention_mask"], position)

        if extract_side in {"both", "aligned"}:
            ext_aligned.clear()
            with torch.no_grad():
                model_aligned(**enc)
                pooled_a = _pool_layer_list(ext_aligned.get_all(), enc["attention_mask"], position)

        batch_rows = len(chunk)
        for j in range(batch_rows):
            if extract_side in {"both", "base"} and not use_base_cache:
                activations_base_list.append(pooled_b[j].cpu().float())
            if extract_side in {"both", "aligned"}:
                activations_aligned_list.append(pooled_a[j].cpu().float())
            sample_ids.append(chunk[j]["sample_id"])
            splits.append(chunk[j]["split"])
            prompt_texts.append(chunk[j]["prompt"])

        if (batch_idx + 1) % flush_interval == 0:
            flush_gpu()
        if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
            print(f"progress: multi-layer extraction batch {batch_idx + 1}/{total_batches}", flush=True)

    if ext_base is not None:
        ext_base.remove()
    if ext_aligned is not None:
        ext_aligned.remove()

    result: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "splits": splits,
        "prompt_texts": prompt_texts,
        "base_model": base_model_id,
        "aligned_model": aligned_model_path,
        "aligned_run_id": aligned_run_id,
        "layers": layers,
        "position": position,
        "dataset_name": dataset_name,
        "max_prompt_tokens": max_prompt_tokens,
        "peft": is_peft,
        "hidden_size": hidden_size,
        "crosscoder_kind": "multilayer_sparc",
        "extract_side": extract_side,
    }

    if extract_side in {"both", "aligned"}:
        result["activations_aligned"] = torch.stack(activations_aligned_list, dim=0)

    if extract_side in {"both", "base"}:
        if use_base_cache:
            if sample_ids != base_activations_cache["sample_ids"]:
                raise RuntimeError(
                    "Sample ID mismatch between base activation cache and current dataset extraction. "
                    "Cache may be stale or dataset ordering changed."
                )
            result["activations_base"] = base_activations_cache["activations_base"]
        else:
            result["activations_base"] = torch.stack(activations_base_list, dim=0)

    models_to_del = [m for m in [model_base, model_aligned] if m is not None]
    del models_to_del, tokenizer
    flush_gpu()

    return result
