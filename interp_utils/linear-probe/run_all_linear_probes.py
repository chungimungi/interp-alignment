"""Run linear-probe.py across MInAlA org checkpoints + base models, in parallel on N GPUs.

Each model gets pinned to one GPU via CUDA_VISIBLE_DEVICES so multiple jobs run
concurrently. Per-model stdout/stderr is captured to logs/linear-probes/<model>.log
to keep terminal output readable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from linear_probe_datasets import infer_dataset_for_model

ORG = "MInAlA"
BASE_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "HuggingFaceTB/SmolLM3-3B",
]


def _needs_trust_remote_code(model_id: str) -> bool:
    lower = model_id.lower()
    return "smollm" in lower or "smol_lm" in lower


def _repo_root() -> Path:
    # interp_utils/linear-probe/run_all_linear_probes.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    return Path(__file__).resolve().parent / "linear-probe.py"


def _load_env() -> None:
    dotenv.load_dotenv(dotenv_path=str(_repo_root() / ".env"))


def _results_roots() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "output" / "linear-probes",
        root / "outputs" / "linear-probes",
    )


def _has_linear_probe_outputs(model_id: str) -> bool:
    required = (
        "layer_metrics.json",
        "layer_predictions.json",
        "layer_probabilities.json",
    )
    model_dir = model_id.replace("/", "_")
    for base in _results_roots():
        out_dir = base / model_dir
        if out_dir.is_dir() and all((out_dir / name).is_file() for name in required):
            return True
    return False


def list_org_model_ids(author: str) -> list[str]:
    url = f"https://huggingface.co/api/models?author={author}"
    req = urllib.request.Request(url, headers={"User-Agent": "run_all_linear_probes/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected API response for {url!r}")
    ids = [item["modelId"] for item in payload if isinstance(item, dict) and "modelId" in item]
    return sorted(set(ids))


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


def _build_cmd(
    *,
    script: Path,
    model_id: str,
    dataset_id: str,
    probe_backend: str,
    strict_cuml: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--model-name",
        model_id,
        "--dataset-name",
        dataset_id,
        "--probe-backend",
        probe_backend,
    ]
    if _needs_trust_remote_code(model_id):
        cmd.append("--trust-remote-code")
    if strict_cuml:
        cmd.append("--strict-cuml")
    cmd.extend(extra_args)
    return cmd


def main(
    *,
    probe_backend: str,
    strict_cuml: bool,
    skip_existing: bool,
    only_models: list[str] | None,
    only_simpo: bool,
    num_gpus: int | None,
    gpu_ids_str: str | None,
    extra_args: list[str],
) -> None:
    _load_env()
    script = _script_path()
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script}")

    org_models = list_org_model_ids(ORG)

    if only_models:
        seen: set[str] = set()
        run_list: list[str] = []
        for mid in only_models:
            mid = mid.strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            run_list.append(mid)
        print(f"Mode: --only-model ({len(run_list)} id(s))")
    elif only_simpo:
        run_list = [m for m in org_models if "simpo" in m.lower()]
        print(f"Mode: --only-simpo (org models matching 'simpo': {len(run_list)})")
        for mid in run_list:
            print(f"  - {mid}")
        if not run_list:
            print("No org models with 'simpo' in the id; nothing to do.", file=sys.stderr)
            raise SystemExit(1)
    else:
        run_list = org_models + BASE_MODELS

    print(f"MInAlA org models: {len(org_models)}")
    print(f"Base models:       {len(BASE_MODELS)}")
    print(f"Total runs:        {len(run_list)}")
    print(f"Probe backend:     {probe_backend}" + (" (strict cuML)" if strict_cuml else ""))

    if skip_existing:
        existing_set = {mid for mid in run_list if _has_linear_probe_outputs(mid)}
        run_list = [mid for mid in run_list if mid not in existing_set]
        print(f"Skipping existing completed models: {len(existing_set)}")
        for mid in sorted(existing_set):
            print(f"  - skipping: {mid}")
        print(f"Models remaining to run: {len(run_list)}")

    if not run_list:
        print("No models left to run after filters.", file=sys.stderr)
        raise SystemExit(0)

    gpu_ids = _resolve_gpu_ids(num_gpus, gpu_ids_str)
    print(f"GPU pool:          {gpu_ids}  (parallelism={len(gpu_ids)})")
    if extra_args:
        print(f"Extra probe args:  {extra_args}")

    log_dir = _repo_root() / "logs" / "linear-probes"
    log_dir.mkdir(parents=True, exist_ok=True)

    pool: Queue[int] = Queue()
    for g in gpu_ids:
        pool.put(g)

    def _worker(model_id: str) -> tuple[str, int, int, str]:
        gpu = pool.get()
        try:
            ds = infer_dataset_for_model(model_id)
            cmd = _build_cmd(
                script=script,
                model_id=model_id,
                dataset_id=ds,
                probe_backend=probe_backend,
                strict_cuml=strict_cuml,
                extra_args=extra_args,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"{model_id.replace('/', '_')}.log"
            with log_path.open("w") as logf:
                logf.write(f"# {model_id}\n# GPU={gpu}\n# cmd={' '.join(cmd)}\n# dataset={ds}\n\n")
                logf.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(_repo_root()),
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            return model_id, gpu, int(proc.returncode), str(log_path)
        finally:
            pool.put(gpu)

    failures: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futures = {ex.submit(_worker, m): m for m in run_list}
        with tqdm(total=len(run_list), desc="probes", unit="model") as pbar:
            for fut in as_completed(futures):
                model_id, gpu, rc, log = fut.result()
                tag = "OK" if rc == 0 else f"FAIL({rc})"
                tqdm.write(f"[GPU{gpu}] {tag} {model_id}  log={log}")
                if rc != 0:
                    failures.append((model_id, rc, log))
                pbar.update(1)

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for mid, rc, log in failures:
            print(f"  - {mid}: exit {rc}  log={log}", file=sys.stderr)
        raise SystemExit(1)
    print("All linear probe runs finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run linear-probe.py on all MInAlA org models plus base checkpoints, in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probe-backend",
        default=os.environ.get("LINEAR_PROBE_BACKEND", "sklearn"),
        choices=["auto", "sklearn", "torch", "cuml"],
        help="Forwarded to linear-probe.py. sklearn (default) preserves the CPU baseline numerics.",
    )
    parser.add_argument(
        "--strict-cuml",
        action="store_true",
        help="Forwarded to linear-probe.py: fail immediately if cuML cannot be imported.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have layer_metrics/layer_predictions/layer_probabilities JSON files.",
    )
    parser.add_argument(
        "--only-model",
        action="append",
        default=None,
        metavar="HF_MODEL_ID",
        help="Run only these HF ids (repeatable). Overrides default org+base list.",
    )
    parser.add_argument(
        "--only-simpo",
        action="store_true",
        help="Run only MInAlA org checkpoints with 'simpo' in the id.",
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
        "--probe-extra-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra raw arg passed through to linear-probe.py. Repeatable. "
        "Example: --probe-extra-arg --batch-size --probe-extra-arg 32",
    )
    launcher_args = parser.parse_args()
    if launcher_args.only_model and launcher_args.only_simpo:
        print("Use either --only-model or --only-simpo, not both.", file=sys.stderr)
        raise SystemExit(2)
    try:
        main(
            probe_backend=str(launcher_args.probe_backend),
            strict_cuml=bool(launcher_args.strict_cuml),
            skip_existing=bool(launcher_args.skip_existing),
            only_models=list(launcher_args.only_model) if launcher_args.only_model else None,
            only_simpo=bool(launcher_args.only_simpo),
            num_gpus=launcher_args.num_gpus,
            gpu_ids_str=launcher_args.gpu_ids,
            extra_args=list(launcher_args.probe_extra_arg) if launcher_args.probe_extra_arg else [],
        )
    except urllib.error.URLError as e:
        print(f"Failed to list org models: {e}", file=sys.stderr)
        raise SystemExit(1) from e
