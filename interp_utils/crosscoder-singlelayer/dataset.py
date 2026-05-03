import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

from . import config

CACHE_META_FILENAME = "cache_meta.json"
NORMALIZED_PROMPTS_FORMAT_VERSION = 1


def _to_text(value: Any, tokenizer) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        is_chat_messages = all(isinstance(item, dict) and "role" in item for item in value)
        if is_chat_messages:
            try:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=not config.DISABLE_THINKING,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                )
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        content = value.get("content", "")
        return str(content)
    return str(value)


def _as_messages(value: Any):
    if isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    ):
        return value
    return None


def _truncate_prompt(prompt_text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0:
        return prompt_text
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt_text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _normalize_prompt_row(example: Dict, tokenizer, max_prompt_tokens: int) -> Dict:
    prompt_messages = _as_messages(example.get("prompt"))
    if prompt_messages:
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not config.DISABLE_THINKING,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt_text = _to_text(example.get("prompt", ""), tokenizer)

    chosen = _to_text(example.get("chosen", ""), tokenizer)
    rejected = _to_text(example.get("rejected", ""), tokenizer)
    prompt_text = _truncate_prompt(prompt_text, tokenizer, max_prompt_tokens)
    return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}


def _tokenizer_cache_key(tokenizer) -> str:
    return str(getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "unknown"))


def normalized_prompts_cache_path(
    cache_root: Path,
    dataset_name: str,
    tokenizer_name_or_path: str,
    max_prompt_tokens: int,
    val_fraction: float,
    seed: int,
) -> Path:
    from .utils import sanitize_model_slug

    d_slug = sanitize_model_slug(dataset_name)
    t_slug = sanitize_model_slug(tokenizer_name_or_path)
    vf = f"{val_fraction:.6f}".rstrip("0").rstrip(".")
    leaf = f"{t_slug}__mt{max_prompt_tokens}__vf{vf}__s{seed}__dt{int(config.DISABLE_THINKING)}"
    return (cache_root / d_slug / leaf).resolve()


def _expected_cache_meta(
    dataset_name: str,
    tokenizer_name_or_path: str,
    max_prompt_tokens: int,
    val_fraction: float,
    seed: int,
) -> Dict:
    return {
        "format_version": NORMALIZED_PROMPTS_FORMAT_VERSION,
        "dataset_name": dataset_name,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "max_prompt_tokens": max_prompt_tokens,
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "disable_thinking": bool(config.DISABLE_THINKING),
    }


def _cache_meta_matches(cache_dir: Path, expected: Dict) -> bool:
    meta_path = cache_dir / CACHE_META_FILENAME
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path) as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if stored.get("format_version") != expected["format_version"]:
        return False
    for k in (
        "dataset_name",
        "tokenizer_name_or_path",
        "max_prompt_tokens",
        "val_fraction",
        "seed",
        "disable_thinking",
    ):
        if stored.get(k) != expected.get(k):
            return False
    return True


def _write_cache_meta(cache_dir: Path, meta: Dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / CACHE_META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)


def build_normalized_dataset_dict(
    tokenizer,
    dataset_name: str,
    max_prompt_tokens: int,
    val_fraction: float,
    seed: int,
    hf_token: Optional[str],
) -> DatasetDict:
    load_kw: Dict = {}
    if hf_token:
        load_kw["token"] = hf_token

    raw = load_dataset(dataset_name, split="train", **load_kw)
    raw = raw.map(
        lambda ex: _normalize_prompt_row(ex, tokenizer, max_prompt_tokens),
        desc="Normalizing prompts",
    )
    keep = [c for c in raw.column_names if c in ("prompt", "chosen", "rejected")]
    raw = raw.remove_columns([c for c in raw.column_names if c not in keep])

    split_ds = raw.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})


def load_or_build_normalized_dataset_dict(
    tokenizer,
    dataset_name: str,
    max_prompt_tokens: int,
    val_fraction: float,
    seed: int,
    hf_token: Optional[str],
    cache_root: Optional[Path],
    use_cache: bool = True,
) -> DatasetDict:
    tok_key = _tokenizer_cache_key(tokenizer)
    expected_meta = _expected_cache_meta(
        dataset_name, tok_key, max_prompt_tokens, val_fraction, seed
    )

    if use_cache and cache_root is not None:
        cache_dir = normalized_prompts_cache_path(
            cache_root, dataset_name, tok_key, max_prompt_tokens, val_fraction, seed
        )
        train_info = cache_dir / "train" / "dataset_info.json"
        if cache_dir.is_dir() and train_info.is_file() and _cache_meta_matches(cache_dir, expected_meta):
            print(f"Loading normalized prompts from cache: {cache_dir}")
            return DatasetDict.load_from_disk(str(cache_dir))

        print(f"Building normalized prompts (will save to cache: {cache_dir})")
        ds_dict = build_normalized_dataset_dict(
            tokenizer, dataset_name, max_prompt_tokens, val_fraction, seed, hf_token
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        ds_dict.save_to_disk(str(cache_dir))
        _write_cache_meta(cache_dir, expected_meta)
        return ds_dict

    print("Building normalized prompts (cache disabled)")
    return build_normalized_dataset_dict(
        tokenizer, dataset_name, max_prompt_tokens, val_fraction, seed, hf_token
    )


class PreferenceActivationDataset(TorchDataset):
    """
    HF preference dataset with GRPO-style prompt normalization (chat template + truncation).
    Train/val split; optional on-disk cache of normalized rows for reuse across extractions.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = config.PREFERENCE_DATASET_NAME,
        split: str = "train",
        max_prompt_tokens: int = config.MAX_PROMPT_TOKENS,
        val_fraction: float = config.VAL_FRACTION,
        seed: int = config.SEED,
        hf_token: Optional[str] = None,
        prompts_cache_dir: Optional[Path] = None,
        use_prompts_cache: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens

        cache_root = Path(prompts_cache_dir) if prompts_cache_dir is not None else None
        if use_prompts_cache and cache_root is None:
            cache_root = config.NORMALIZED_PROMPTS_CACHE_DIR

        ds_dict = load_or_build_normalized_dataset_dict(
            tokenizer,
            dataset_name=dataset_name,
            max_prompt_tokens=max_prompt_tokens,
            val_fraction=val_fraction,
            seed=seed,
            hf_token=hf_token,
            cache_root=cache_root if use_prompts_cache else None,
            use_cache=use_prompts_cache,
        )

        split_ds_train = ds_dict["train"]
        split_ds_test = ds_dict["test"]

        if split == "train":
            self._hf = split_ds_train
            self._splits = ["train"] * len(self._hf)
            self._sample_ids = [f"{dataset_name}::train::{i}" for i in range(len(self._hf))]
        elif split == "val":
            self._hf = split_ds_test
            self._splits = ["val"] * len(self._hf)
            self._sample_ids = [f"{dataset_name}::val::{i}" for i in range(len(self._hf))]
        elif split == "all":
            self._hf = concatenate_datasets([split_ds_train, split_ds_test])
            self._splits = ["train"] * len(split_ds_train) + ["val"] * len(split_ds_test)
            self._sample_ids = (
                [f"{dataset_name}::train::{i}" for i in range(len(split_ds_train))]
                + [f"{dataset_name}::val::{i}" for i in range(len(split_ds_test))]
            )
        else:
            raise ValueError(f"split must be train, val, or all; got {split}")

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> Dict:
        row = self._hf[idx]
        return {
            "sample_id": self._sample_ids[idx],
            "prompt": row["prompt"],
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "split": self._splits[idx],
        }


class PairedActivationDataset(TorchDataset):
    def __init__(
        self,
        activations_base: torch.Tensor,
        activations_aligned: torch.Tensor,
        sample_ids: List[str],
        splits: List[str],
    ):
        self.activations_base = activations_base
        self.activations_aligned = activations_aligned
        self.sample_ids = sample_ids
        self.splits = splits

    def __len__(self) -> int:
        return len(self.activations_base)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "activations_base": self.activations_base[idx],
            "activations_aligned": self.activations_aligned[idx],
            "sample_id": self.sample_ids[idx],
            "split": self.splits[idx],
        }


def create_paired_activation_dataset(
    activations_data: Dict,
    split: str = "train",
) -> PairedActivationDataset:
    mask = [s == split for s in activations_data["splits"]]
    indices = [i for i, m in enumerate(mask) if m]

    return PairedActivationDataset(
        activations_base=activations_data["activations_base"][indices],
        activations_aligned=activations_data["activations_aligned"][indices],
        sample_ids=[activations_data["sample_ids"][i] for i in indices],
        splits=[activations_data["splits"][i] for i in indices],
    )


def collate_activations(batch: List[Dict]) -> Dict:
    return {
        "activations_base": torch.stack([b["activations_base"] for b in batch]),
        "activations_aligned": torch.stack([b["activations_aligned"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch],
        "splits": [b["split"] for b in batch],
    }
