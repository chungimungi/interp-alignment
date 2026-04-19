"""
Train Batch Top-K SAEs on the best (and middle) layers of each model in ``results/``.

For every ``results/<model_dir>/layer_metrics.json`` produced by linear-probe.py, this
launcher:

1. Picks ``best_layer = argmax(auroc)`` and ``mid_layer = num_layers // 2`` (a sanity
baseline so you can compare middle-of-network features against the probe-optimal
layer). Duplicates are de-duplicated.
2. Reverses the directory-name convention back to the HF id (``<org>_<name>`` ->
``<org>/<name>``; the leading underscore is the org separator that linear-probe.py
writes).
3. Builds **one** universal training corpus (default: `openbmb/UltraChat` on Hugging
Face — multi-round dialogues as alternating user/assistant strings in the ``data``
column; see https://huggingface.co/datasets/openbmb/UltraChat) as a local parquet
with a ``messages`` column so sae-lens's ``use_chat_formatting=True`` path can consume
it. Cached under ``data/sae_corpus/<sanitized-id>_full/`` or ``_n<rows>/`` so all models share
the same activations distribution. The HF split is read with ``streaming=True`` and written
to parquet in bounded batches so the full UltraChat split does not OOM the host RAM.
Use ``--sae-corpus`` to point at another HF dataset;
if the id does not match UltraChat, the legacy preference flattening
(chosen/rejected / KTO) from ``linear_probe_datasets.py`` is used instead.
4. Invokes ``sae.py`` as a subprocess per (model, layer) with:
    --model-class-name AutoModelForCausalLM
    --hook-name model.layers.<idx>
    --use-chat-formatting
    --dataset <local parquet dir>
Trust-remote-code is forwarded for SmolLM3 variants (same rule as
``run_all_linear_probes.py``).

Per-model outputs and checkpoints are written under ``output/sae/<model_dir>/layer_<idx>/``
and ``checkpoints/sae/<model_dir>/layer_<idx>/`` respectively. Weights & Biases is
opt-in via ``--log-to-wandb``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "linear-probe"))
from linear_probe_datasets import is_kto_dataset

DEFAULT_SAE_CORPUS = "openbmb/UltraChat"

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _default_probe_results_root() -> Path:
    root = _repo_root()
    primary = root / "output" / "linear-probes"
    secondary = root / "outputs" / "linear-probes"
    if primary.exists():
        return primary
    if secondary.exists():
        return secondary
    return primary


def _sae_output_roots() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "output" / "sae",
        root / "output" / "saes",
    )


def _sae_checkpoint_roots() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "checkpoints" / "sae",
        root / "checkpoints" / "saes",
    )


def _layer_run_dirs(model_dir_name: str, layer_idx: int, layer_tag: str) -> tuple[Path, ...]:
    suffix = f"layer_{layer_idx:02d}_{layer_tag}"
    dirs: list[Path] = []
    for base in _sae_output_roots():
        dirs.append(base / model_dir_name / suffix)
    for base in _sae_checkpoint_roots():
        dirs.append(base / model_dir_name / suffix)
    return tuple(dirs)


def _has_existing_layer_run(model_dir_name: str, layer_idx: int, layer_tag: str) -> Path | None:
    for d in _layer_run_dirs(model_dir_name, layer_idx, layer_tag):
        if d.is_dir() and any(d.iterdir()):
            return d
    return None


def _log_line(msg: str, *, interactive: bool) -> None:
    if interactive:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _load_env() -> None:
    dotenv.load_dotenv(dotenv_path=str(_repo_root() / ".env"))


def _sanitize(path_like: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path_like)


def _needs_trust_remote_code(model_id: str) -> bool:
    lower = model_id.lower()
    return "smollm" in lower or "smol_lm" in lower


def _dir_to_hf_id(dir_name: str) -> str:
    """
    linear-probe.py writes ``results/<org>_<name>`` (single replace of '/' -> '_').
    Reverse that by splitting on the first underscore only.
    """
    if "_" not in dir_name:
        return dir_name
    org, rest = dir_name.split("_", 1)
    return f"{org}/{rest}"


def _discover_models(results_root: Path) -> list[tuple[str, Path]]:
    """Return list of (hf_id, model_results_dir) for every subdir with layer_metrics.json."""
    out: list[tuple[str, Path]] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        metrics_file = child / "layer_metrics.json"
        if metrics_file.is_file():
            out.append((_dir_to_hf_id(child.name), child))
    return out


def _layers_for_model(metrics_path: Path) -> tuple[int, int, int]:
    with metrics_path.open() as f:
        metrics: list[dict[str, Any]] = json.load(f)
    if not metrics:
        raise RuntimeError(f"{metrics_path} is empty")
    num_layers = len(metrics)
    best_layer = int(max(metrics, key=lambda m: m["auroc"])["layer"])
    mid_layer = num_layers // 2
    return best_layer, mid_layer, num_layers


def _normalize_messages(raw: Any) -> list[dict[str, str]] | None:
    """Coerce a dataset cell into a list[{'role', 'content'}]; returns None if unusable."""
    if raw is None:
        return None
    if isinstance(raw, list):
        out: list[dict[str, str]] = []
        for msg in raw:
            if not isinstance(msg, dict):
                return None
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str):
                return None
            if isinstance(content, list):
                content = "".join(
                    str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
                )
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            out.append({"role": role, "content": content})
        return out or None
    return None


def _parquet_stream_write(
    row_iter: Iterator[dict[str, Any]],
    target: Path,
    *,
    batch_size: int = 4096,
    desc: str = "Writing train.parquet",
) -> int:
    """
    Append rows to a single Parquet file using bounded in-memory batches (avoids OOM on
    full-corpus materialization).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    total = 0

    for row in tqdm(row_iter, desc=desc, unit="row"):
        batch.append(row)
        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(str(target), table.schema)
            writer.write_table(table)
            total += len(batch)
            batch.clear()

    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(str(target), table.schema)
        writer.write_table(table)
        total += len(batch)

    if writer is None:
        raise RuntimeError(f"No rows written to {target}")
    writer.close()
    return total


def _iter_ultrachat_message_rows(
    dataset_id: str,
    max_source_rows: int | None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    for i, item in enumerate(ds):
        if max_source_rows is not None and i >= max_source_rows:
            break
        msgs = _ultrachat_row_to_messages(item)
        if msgs is not None:
            yield {"messages": msgs}


def _iter_preference_message_rows(
    dataset_id: str,
    max_source_rows: int | None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)

    if is_kto_dataset(dataset_id):
        for i, item in enumerate(ds):
            if max_source_rows is not None and i >= max_source_rows:
                break
            try:
                prompt = item["prompt"]
                completion = item["completion"]
            except (KeyError, TypeError):
                continue
            msgs = _normalize_messages(
                [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(completion)},
                ]
            )
            if msgs is not None:
                yield {"messages": msgs}
        return

    for i, item in enumerate(ds):
        if max_source_rows is not None and i >= max_source_rows:
            break
        for col in ("chosen", "rejected"):
            if col not in item:
                continue
            msgs = _normalize_messages(item[col])
            if msgs is not None:
                yield {"messages": msgs}


def _ultrachat_row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """
    UltraChat rows use a ``data`` field: list[str] of alternating user / assistant turns
    (see dataset card: https://huggingface.co/datasets/openbmb/UltraChat).
    """
    raw = row.get("data")
    if not isinstance(raw, list) or not raw:
        return None
    pieces: list[str] = []
    for piece in raw:
        if isinstance(piece, dict):
            text = piece.get("text") or piece.get("content") or str(piece)
        else:
            text = str(piece) if piece is not None else ""
        text = text.strip()
        if not text:
            return None
        pieces.append(text)
    return [
        {"role": "user" if (i % 2 == 0) else "assistant", "content": p}
        for i, p in enumerate(pieces)
    ]


def _ultrachat_to_parquet(
    dataset_id: str,
    out_dir: Path,
    max_rows: int | None,
    *,
    parquet_batch: int,
) -> Path:
    """Write ``train.parquet`` with ``messages`` from UltraChat-style ``data`` lists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "train.parquet"
    if target.is_file():
        return target

    n = _parquet_stream_write(
        _iter_ultrachat_message_rows(dataset_id, max_rows),
        target,
        batch_size=parquet_batch,
        desc="UltraChat → parquet",
    )
    if n == 0:
        raise RuntimeError(
            f"No usable rows extracted from {dataset_id!r}; expected a list column 'data' "
            "of alternating user/assistant strings (UltraChat schema)."
        )
    return target


def _preference_dataset_to_parquet(
    dataset_id: str,
    out_dir: Path,
    max_rows: int | None,
    *,
    parquet_batch: int,
) -> Path:
    """
    Materialize ``dataset_id`` as ``<out_dir>/train.parquet`` with a single ``messages``
    column. Chosen and rejected sides of each pair become independent rows so the SAE
    sees both polarities. KTO-style rows (prompt/completion/label) become one row each.
    No-ops if the parquet is already present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "train.parquet"
    if target.is_file():
        return target

    n = _parquet_stream_write(
        _iter_preference_message_rows(dataset_id, max_rows),
        target,
        batch_size=parquet_batch,
        desc="Preferences → parquet",
    )
    if n == 0:
        raise RuntimeError(
            f"No usable rows extracted from {dataset_id!r}; expected 'chosen'/'rejected' or "
            "KTO-style 'prompt'/'completion' columns."
        )
    return target


def _materialize_sae_corpus(
    dataset_id: str,
    out_dir: Path,
    max_rows: int | None,
    *,
    parquet_batch: int,
) -> Path:
    """Dispatch to UltraChat vs preference flattening based on dataset id."""
    if "ultrachat" in dataset_id.lower():
        return _ultrachat_to_parquet(dataset_id, out_dir, max_rows, parquet_batch=parquet_batch)
    return _preference_dataset_to_parquet(dataset_id, out_dir, max_rows, parquet_batch=parquet_batch)


def _invoke_sae_for_layer(
    *,
    python_exe: str,
    script: Path,
    model_id: str,
    layer_idx: int,
    dataset_dir: Path,
    model_dir_name: str,
    layer_tag: str,
    common_args: argparse.Namespace,
) -> tuple[int, str]:
    output_path = _repo_root() / "output" / "sae" / model_dir_name / f"layer_{layer_idx:02d}_{layer_tag}"
    checkpoint_path = (
        _repo_root() / "checkpoints" / "sae" / model_dir_name / f"layer_{layer_idx:02d}_{layer_tag}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_dir_name}-L{layer_idx}-{layer_tag}"

    cmd: list[str] = [
        python_exe,
        str(script),
        "--model-name",
        model_id,
        "--model-class-name",
        "AutoModelForCausalLM",
        "--hook-name",
        f"model.layers.{layer_idx}",
        "--dataset",
        str(dataset_dir),
        "--use-chat-formatting",
        "--context-size",
        str(common_args.context_size),
        "--d-sae",
        str(common_args.d_sae),
        "--k",
        str(common_args.k),
        "--training-tokens",
        str(common_args.training_tokens),
        "--train-batch-size-tokens",
        str(common_args.train_batch_size_tokens),
        "--store-batch-size-prompts",
        str(common_args.store_batch_size_prompts),
        "--n-batches-in-buffer",
        str(common_args.n_batches_in_buffer),
        "--lr",
        str(common_args.lr),
        "--output-path",
        str(output_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--run-name",
        run_name,
        "--save-final-checkpoint",
        "--save-metrics-jsonl",
    ]
    if common_args.model_dtype:
        cmd += ["--model-dtype", common_args.model_dtype]
    if _needs_trust_remote_code(model_id):
        cmd.append("--trust-remote-code")
    if common_args.log_to_wandb:
        cmd += ["--log-to-wandb", "--wandb-project", common_args.wandb_project]
        if common_args.wandb_entity:
            cmd += ["--wandb-entity", common_args.wandb_entity]
    if common_args.autocast:
        cmd.append("--autocast")
    if common_args.autocast_lm:
        cmd.append("--autocast-lm")

    timeout_seconds: int | None = None
    if int(common_args.sae_job_timeout_minutes) > 0:
        timeout_seconds = int(common_args.sae_job_timeout_minutes) * 60

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - started
        return int(proc.returncode), f"elapsed={elapsed:.1f}s"
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        return 124, f"timeout after {elapsed:.1f}s"


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(
        description="Train one SAE per (model, {best, middle}) layer across outputs/linear-probes/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", default=str(_default_probe_results_root()))
    parser.add_argument(
        "--layers",
        choices=["best", "mid", "best_plus_mid"],
        default="best_plus_mid",
        help="Which layers to train SAEs for per model.",
    )
    parser.add_argument(
        "--sae-corpus",
        default=DEFAULT_SAE_CORPUS,
        help="HF dataset id for SAE activations (shared by all models). Default: openbmb/UltraChat. "
        "If the id does not contain 'ultrachat', Argilla preference flattening is used instead.",
    )
    parser.add_argument(
        "--max-corpus-rows",
        type=int,
        default=0,
        help="Cap on HF train rows before writing train.parquet (0 = use full split, no cap).",
    )
    parser.add_argument(
        "--force-rebuild-corpus",
        action="store_true",
        help="Delete cached train.parquet for this corpus + row cap and rebuild.",
    )
    parser.add_argument(
        "--corpus-parquet-batch-rows",
        type=int,
        default=4096,
        help="Rows per PyArrow batch when streaming HF data to train.parquet (lower = less RAM).",
    )
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--training-tokens", type=int, default=2_000_000)
    parser.add_argument("--train-batch-size-tokens", type=int, default=1024)
    parser.add_argument("--store-batch-size-prompts", type=int, default=8)
    parser.add_argument("--n-batches-in-buffer", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--sae-job-timeout-minutes",
        type=int,
        default=120,
        help="Kill an individual sae.py job if it exceeds this duration; 0 disables timeout.",
    )
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="minala_saes")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--only-model",
        action="append",
        default=None,
        help="If given, only run these HF ids (repeatable). Matches against the reverse-mapped id.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (model, layer) if an output/checkpoint run dir already contains files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit without training.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    sae_script = _repo_root() / "sae.py"
    if not sae_script.is_file():
        raise FileNotFoundError(sae_script)

    discovered = _discover_models(results_root)
    if args.only_model:
        wanted = set(args.only_model)
        discovered = [(mid, p) for mid, p in discovered if mid in wanted]
    if not discovered:
        print("No models with layer_metrics.json found; nothing to do.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(discovered)} model(s) with linear-probe results.")
    for mid, p in discovered:
        print(f"  - {mid}  (results: {p.name})")

    dataset_cache_root = _repo_root() / "data" / "sae_corpus"
    dataset_cache_root.mkdir(parents=True, exist_ok=True)

    corpus_id = str(args.sae_corpus)
    cap: int | None = int(args.max_corpus_rows) if int(args.max_corpus_rows) > 0 else None
    suffix = "full" if cap is None else f"n{cap}"
    ds_dir = dataset_cache_root / f"{_sanitize(corpus_id)}_{suffix}"
    parquet_path = ds_dir / "train.parquet"

    print("\nUniversal SAE corpus (one parquet shared by all models):")
    print(f"  {corpus_id}  ->  {ds_dir}" + ("  (full train split)" if cap is None else f"  (first {cap} rows)"))
    if not args.dry_run:
        if args.force_rebuild_corpus and parquet_path.is_file():
            parquet_path.unlink()
        pb = max(256, int(args.corpus_parquet_batch_rows))
        _materialize_sae_corpus(corpus_id, ds_dir, cap, parquet_batch=pb)

    failures: list[tuple[str, int, str]] = []
    python_exe = sys.executable
    total_jobs = 0
    for mid, mdir in discovered:
        best, mid_layer, n_layers = _layers_for_model(mdir / "layer_metrics.json")
        jobs: list[tuple[int, str]] = []
        if args.layers == "best":
            jobs.append((best, "best"))
        elif args.layers == "mid":
            jobs.append((mid_layer, "mid"))
        else:
            jobs.append((best, "best"))
            if mid_layer != best:
                jobs.append((mid_layer, "mid"))
        total_jobs += len(jobs)
    print(f"\nPlanned training jobs: {total_jobs}\n")

    interactive = sys.stdout.isatty()
    pbar = tqdm(total=total_jobs, desc="SAE training", unit="job", disable=not interactive)
    for mid, mdir in discovered:
        best, mid_layer, n_layers = _layers_for_model(mdir / "layer_metrics.json")
        jobs: list[tuple[int, str]] = []
        if args.layers == "best":
            jobs.append((best, "best"))
        elif args.layers == "mid":
            jobs.append((mid_layer, "mid"))
        else:
            jobs.append((best, "best"))
            if mid_layer != best:
                jobs.append((mid_layer, "mid"))

        for layer_idx, tag in jobs:
            _log_line(
                f"{mid}  layer={layer_idx} ({tag})  sae_corpus={corpus_id}  layers={n_layers}"
                + ("  [starting]" if not args.dry_run else "  [dry-run]"),
                interactive=interactive,
            )
            existing_run_dir = _has_existing_layer_run(mdir.name, layer_idx, tag)
            if args.skip_existing and existing_run_dir is not None:
                _log_line(
                    f"  -> skipping: {mid} layer={layer_idx} ({tag}) "
                    f"(existing run dir non-empty: {existing_run_dir})",
                    interactive=interactive,
                )
                pbar.update(1)
                continue
            if args.dry_run:
                pbar.update(1)
                continue
            _log_line(
                f"  -> running: {mid} layer={layer_idx} ({tag})",
                interactive=interactive,
            )
            rc, reason = _invoke_sae_for_layer(
                python_exe=python_exe,
                script=sae_script,
                model_id=mid,
                layer_idx=layer_idx,
                dataset_dir=ds_dir,
                model_dir_name=mdir.name,
                layer_tag=tag,
                common_args=args,
            )
            if rc != 0:
                failures.append((mid, layer_idx, f"exit {rc} ({reason})"))
                _log_line(
                    f"  -> failed: {mid} layer={layer_idx} ({tag}): exit {rc} ({reason})",
                    interactive=interactive,
                )
            else:
                _log_line(
                    f"  -> done: {mid} layer={layer_idx} ({tag}) [{reason}]",
                    interactive=interactive,
                )
            pbar.update(1)
    pbar.close()

    if failures:
        print("\nSome SAE runs failed:", file=sys.stderr)
        for mid, layer, reason in failures:
            print(f"  - {mid} layer={layer}: {reason}", file=sys.stderr)
        raise SystemExit(1)
    print("All SAE runs finished successfully.")


if __name__ == "__main__":
    main()
