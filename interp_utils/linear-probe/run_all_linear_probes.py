
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

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
    return Path(__file__).resolve().parents[2]


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


def main(
    *,
    probe_backend: str,
    strict_cuml: bool,
    skip_existing: bool,
    only_models: list[str] | None,
    only_simpo: bool,
) -> None:
    _load_env()
    script = _repo_root() / "linear-probe.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script}")

    org_models = list_org_model_ids(ORG)

    if only_models:
        seen: set[str] = set()
        run_list = []
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
    print(f"Base models: {len(BASE_MODELS)}")
    print(f"Total runs: {len(run_list)}")
    print(f"Probe backend (per subprocess): {probe_backend}" + (" (strict cuML)" if strict_cuml else ""))

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

    failures: list[tuple[str, str]] = []
    for model_id in tqdm(run_list, desc="linear_probe_models", unit="model"):
        ds = infer_dataset_for_model(model_id)
        tqdm.write(f"{model_id}  ->  {ds}")
        cmd = [
            sys.executable,
            str(script),
            "--model-name",
            model_id,
            "--dataset-name",
            ds,
        ]
        if _needs_trust_remote_code(model_id):
            cmd.append("--trust-remote-code")
        cmd.extend(["--probe-backend", probe_backend])
        if strict_cuml:
            cmd.append("--strict-cuml")
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            failures.append((model_id, f"exit {proc.returncode}"))

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for mid, reason in failures:
            print(f"  - {mid}: {reason}", file=sys.stderr)
        raise SystemExit(1)
    print("All linear probe runs finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run linear-probe.py on all MInAlA org models plus base checkpoints.",
    )
    parser.add_argument(
        "--probe-backend",
        default=os.environ.get("LINEAR_PROBE_BACKEND", "auto"),
        choices=["auto", "sklearn", "torch", "cuml"],
        help="Forwarded to linear-probe.py. Use 'cuml' when RAPIDS cuML+CuPy are installed in this Python. "
        "If cuML import fails, linear-probe falls back to torch unless --strict-cuml. "
        "Default: env LINEAR_PROBE_BACKEND if set, else 'auto'.",
    )
    parser.add_argument(
        "--strict-cuml",
        action="store_true",
        help="Forwarded to linear-probe.py: fail immediately if cuML cannot be imported (no torch fallback).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose results already include layer_metrics/layer_predictions/layer_probabilities JSON files.",
    )
    parser.add_argument(
        "--only-model",
        action="append",
        default=None,
        metavar="HF_MODEL_ID",
        help="Run only these Hugging Face model ids (repeatable). Overrides default org+base list.",
    )
    parser.add_argument(
        "--only-simpo",
        action="store_true",
        help="Run only MInAlA org checkpoints whose model id contains 'simpo' (e.g. three SimPO-merged repos).",
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
        )
    except urllib.error.URLError as e:
        print(f"Failed to list org models: {e}", file=sys.stderr)
        raise SystemExit(1) from e
