"""Run SPARC crosscoders for all (base, aligned) pairs in parallel across N GPUs.

Default plan: 5 alignment algos x 3 base models = 15 (base, aligned) pairs, each
trained at the base model's probe-best layer (so all 5 algos for a given base are
compared at the same depth).

Per-pair stdout/stderr is captured under logs/crosscoder/<run_id>-L<layer>.log to
keep the terminal readable across 8 concurrent jobs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from tqdm import tqdm

# (base_model_id, base_layer, aligned_model_id, aligned_run_id, trust_remote_code)
# base_layer is the base model's probe-best layer from chungimungi's SAE CSV.
DEFAULT_PAIRS: list[tuple[str, int, str, str, bool]] = [
    # SmolLM3-3B @ L19
    ("HuggingFaceTB/SmolLM3-3B", 19, "MInAlA/SmolLM3-3B-DPO-merged",   "smollm-dpo",   True),
    ("HuggingFaceTB/SmolLM3-3B", 19, "MInAlA/SmolLM3-3B-GRPO-merged",  "smollm-grpo",  True),
    ("HuggingFaceTB/SmolLM3-3B", 19, "MInAlA/SmolLM3-3B-KTO-merged",   "smollm-kto",   True),
    ("HuggingFaceTB/SmolLM3-3B", 19, "MInAlA/SmolLM3-3B-ORPO-merged",  "smollm-orpo",  True),
    ("HuggingFaceTB/SmolLM3-3B", 19, "MInAlA/SmolLM3-3B-SimPO-merged", "smollm-simpo", True),
    # Llama-3.2-3B-Instruct @ L11
    ("meta-llama/Llama-3.2-3B-Instruct", 11, "MInAlA/Llama-3.2-3B-DPO-merged",          "llama-dpo",   False),
    ("meta-llama/Llama-3.2-3B-Instruct", 11, "MInAlA/Llama-3.2-3B-Instruct-GRPO-merged","llama-grpo",  False),
    ("meta-llama/Llama-3.2-3B-Instruct", 11, "MInAlA/Llama-3.2-3B-Instruct-KTO-merged", "llama-kto",   False),
    ("meta-llama/Llama-3.2-3B-Instruct", 11, "MInAlA/Llama-3.2-3B-ORPO-merged",         "llama-orpo",  False),
    ("meta-llama/Llama-3.2-3B-Instruct", 11, "MInAlA/Llama-3.2-3B-SimPO-merged",        "llama-simpo", False),
    # Qwen3-4B-Instruct-2507 @ L24
    ("Qwen/Qwen3-4B-Instruct-2507", 24, "MInAlA/Qwen3-4B-Instruct-2507-DPO-merged",  "qwen-dpo",   False),
    ("Qwen/Qwen3-4B-Instruct-2507", 24, "MInAlA/Qwen3-4B-Instruct-2507-GRPO-merged", "qwen-grpo",  False),
    ("Qwen/Qwen3-4B-Instruct-2507", 24, "MInAlA/Qwen3-4B-Instruct-2507-KTO-merged",  "qwen-kto",   False),
    ("Qwen/Qwen3-4B-Instruct-2507", 24, "MInAlA/Qwen3-4B-Instruct-2507-SimPO-merged","qwen-simpo", False),
    ("Qwen/Qwen3-4B-Instruct-2507", 24, "MInAlA/Qwen3-4B-ORPO-merged",                "qwen-orpo",  False),
    # PPO runs (probe-best layers from ppo_runs.json; ultrafeedback-binarized dataset)
    ("HuggingFaceTB/SmolLM3-3B",           19, "MInAlA/SmolLM3-3B-PPO-merged",                    "smollm3-ppo",  True),
    ("meta-llama/Llama-3.2-3B-Instruct",   11, "MInAlA/Llama-3.2-3B-Instruct-PPO-merged",         "llama32-3b-ppo", False),
    ("Qwen/Qwen3-4B-Instruct-2507",        24, "MInAlA/Qwen3-4B-Instruct-2507-PPO-merged",         "qwen3-4b-ppo", True),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_gpu_ids(num_gpus: int | None, gpu_ids_str: str | None) -> list[int]:
    if gpu_ids_str:
        ids = [int(s) for s in gpu_ids_str.split(",") if s.strip()]
        if not ids:
            raise SystemExit(f"--gpu-ids {gpu_ids_str!r} parsed to an empty list")
        return ids
    detected = 0
    try:
        import torch  # noqa: PLC0415

        detected = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        detected = 0
    if num_gpus is None:
        n = detected if detected > 0 else 1
    else:
        n = max(1, int(num_gpus))
    if detected and n > detected:
        print(
            f"Requested {n} GPUs but only {detected} visible; clamping to {detected}.",
            file=sys.stderr,
        )
        n = detected
    return list(range(n))


def _algo_from_run_id(run_id: str) -> str:
    return run_id.rsplit("-", 1)[-1].lower()


def _build_cmd(
    *,
    base: str,
    layer: int,
    aligned: str,
    run_id: str,
    trust_rc: bool,
    dataset: str | None,
    output_root: Path | None,
    extra: list[str],
    n_jobs_superposition: int = 1,
) -> tuple[list[str], Path | None]:
    out_dir: Path | None = None
    if output_root is not None:
        out_dir = output_root / run_id / f"L{layer}"
    cmd: list[str] = [
        sys.executable,
        "-m",
        f"{__package__}.main",
        "--stage",
        "all",
        "--base-model",
        base,
        "--aligned-model",
        aligned,
        "--aligned-run-id",
        run_id,
        "--layer",
        str(layer),
    ]
    if trust_rc:
        cmd.append("--trust-remote-code")
    if dataset:
        cmd += ["--dataset-name", dataset]
    if out_dir is not None:
        cmd += ["--output-dir", str(out_dir)]
    if n_jobs_superposition != 1:
        cmd += ["--n-jobs-superposition", str(n_jobs_superposition)]
    cmd.extend(extra)
    return cmd, out_dir


def _existing_complete(out_dir: Path | None) -> bool:
    if out_dir is None:
        return False
    return (out_dir / "metrics" / "aggregate_metrics.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use in parallel. Defaults to torch.cuda.device_count().",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Explicit comma-separated GPU id list (e.g. '0,1,2,3'). Overrides --num-gpus.",
    )
    parser.add_argument(
        "--only-base",
        action="append",
        default=None,
        help="Restrict to specific base model HF id(s). Repeatable.",
    )
    parser.add_argument(
        "--only-algo",
        action="append",
        default=None,
        help="Restrict to specific algos: dpo, grpo, kto, orpo, simpo. Repeatable.",
    )
    parser.add_argument(
        "--only-run-id",
        action="append",
        default=None,
        help="Restrict to specific aligned-run-id slugs (e.g. smollm-orpo). Repeatable.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Override the preference dataset for activation extraction. "
        "Default: argilla/ultrafeedback-multi-binarized-preferences-cleaned (from config.py).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root dir for crosscoder outputs. Default: <repo_root>/output/crosscoder/",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra raw arg passed through to crosscoder.main. Repeatable. "
        "Use = syntax for values starting with -- (e.g. --extra-arg=--train-batch-size --extra-arg=64).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs whose aggregate_metrics.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit without running.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        metavar="N",
        help="CPUs for superposition analysis per job (--n-jobs-superposition). "
        "Defaults to max(1, cpu_count - 1).",
    )
    args = parser.parse_args()

    n_jobs_superposition = args.num_cpus if args.num_cpus is not None else max(1, (os.cpu_count() or 2) - 1)

    pairs = list(DEFAULT_PAIRS)
    if args.only_base:
        wanted = set(args.only_base)
        pairs = [p for p in pairs if p[0] in wanted]
    if args.only_algo:
        wanted = {a.lower() for a in args.only_algo}
        pairs = [p for p in pairs if _algo_from_run_id(p[3]) in wanted]
    if args.only_run_id:
        wanted = set(args.only_run_id)
        pairs = [p for p in pairs if p[3] in wanted]
    if not pairs:
        print("No pairs match the filters; nothing to do.", file=sys.stderr)
        raise SystemExit(0)

    output_root: Path | None
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = _repo_root() / "output" / "crosscoder"
    output_root.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_arg) if args.extra_arg else []
    gpu_ids = _resolve_gpu_ids(args.num_gpus, args.gpu_ids)

    print(f"Pairs:        {len(pairs)}")
    print(f"GPU pool:     {gpu_ids}  (parallelism={len(gpu_ids)})")
    print(f"CPU jobs:     {n_jobs_superposition}  (superposition analysis per pair)")
    print(f"Output root:  {output_root}")
    if extra_args:
        print(f"Extra args:   {extra_args}")

    log_dir = _repo_root() / "logs" / "crosscoder"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filter skip-existing.
    runnable: list[tuple[str, int, str, str, bool]] = []
    skipped = 0
    for p in pairs:
        base, layer, aligned, run_id, trust_rc = p
        out_dir = output_root / run_id / f"L{layer}"
        if args.skip_existing and _existing_complete(out_dir):
            print(f"  skip (existing): {run_id} L{layer}")
            skipped += 1
            continue
        runnable.append(p)

    if not runnable:
        print(f"\nAll {len(pairs)} pairs already complete (skipped={skipped}).")
        return

    if args.dry_run:
        for p in runnable:
            base, layer, aligned, run_id, trust_rc = p
            cmd, _ = _build_cmd(
                base=base,
                layer=layer,
                aligned=aligned,
                run_id=run_id,
                trust_rc=trust_rc,
                dataset=args.dataset_name,
                output_root=output_root,
                extra=extra_args,
                n_jobs_superposition=n_jobs_superposition,
            )
            print("DRY:", " ".join(cmd))
        print(f"\n{len(runnable)} jobs planned (skipped={skipped}).")
        return

    pool: Queue[int] = Queue()
    for g in gpu_ids:
        pool.put(g)

    def _worker(pair: tuple[str, int, str, str, bool]) -> tuple[str, int, int, int, str]:
        base, layer, aligned, run_id, trust_rc = pair
        gpu = pool.get()
        try:
            cmd, _ = _build_cmd(
                base=base,
                layer=layer,
                aligned=aligned,
                run_id=run_id,
                trust_rc=trust_rc,
                dataset=args.dataset_name,
                output_root=output_root,
                extra=extra_args,
                n_jobs_superposition=n_jobs_superposition,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"{run_id}-L{layer}.log"
            with log_path.open("w") as f:
                f.write(
                    f"# {run_id} L{layer}\n"
                    f"# base    = {base}\n"
                    f"# aligned = {aligned}\n"
                    f"# GPU     = {gpu}\n"
                    f"# cmd     = {' '.join(cmd)}\n\n"
                )
                f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            return run_id, layer, gpu, int(proc.returncode), str(log_path)
        finally:
            pool.put(gpu)

    failures: list[tuple[str, int, int, str]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futures = {ex.submit(_worker, p): p for p in runnable}
        with tqdm(total=len(runnable), desc="crosscoders", unit="pair") as pbar:
            for fut in as_completed(futures):
                run_id, layer, gpu, rc, log = fut.result()
                tag = "OK" if rc == 0 else f"FAIL({rc})"
                tqdm.write(f"[GPU{gpu}] {tag} {run_id} L{layer}  log={log}")
                if rc != 0:
                    failures.append((run_id, layer, rc, log))
                pbar.update(1)

    if failures:
        print("\nSome crosscoder runs failed:", file=sys.stderr)
        for run_id, layer, rc, log in failures:
            print(f"  - {run_id} L{layer}: exit {rc}  log={log}", file=sys.stderr)
        raise SystemExit(1)
    print("All crosscoder runs finished successfully.")


if __name__ == "__main__":
    main()
