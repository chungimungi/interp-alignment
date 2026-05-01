import argparse
import os
from pathlib import Path
from typing import Optional

from . import config
from .utils import (
    flush_gpu,
    get_activations_dir,
    get_checkpoint_dir,
    get_features_dir,
    get_metrics_dir,
    get_plots_dir,
    get_results_dir,
    load_activations,
    load_json,
    save_activations,
    save_json,
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
    if not base_cache_path.exists():
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


def run_manifest(manifest_path: Path, force: bool = False):
    import json

    with open(manifest_path) as f:
        rows = json.load(f)
    for i, row in enumerate(rows):
        print(f"\n{'#'*60}\n# Manifest job {i+1}/{len(rows)}\n{'#'*60}")
        out = row.get("output_dir")
        pc = row.get("prompts_cache_dir")
        ebs = row.get("extract_batch_size")
        tbs = row.get("train_batch_size")
        uta = row.get("use_train_amp")
        njs = row.get("n_jobs_superposition")
        run_all(
            base_model=row["base_model"],
            aligned_model=row["aligned_model"],
            aligned_run_id=row["aligned_run_id"],
            layer=int(row["layer"]),
            position=row.get("position", config.POSITION_LAST_PROMPT),
            dataset_name=row.get("dataset_name", config.PREFERENCE_DATASET_NAME),
            max_prompt_tokens=int(row.get("max_prompt_tokens", config.MAX_PROMPT_TOKENS)),
            trust_remote_code=bool(row.get("trust_remote_code", False)),
            force=force,
            output_dir=Path(out) if out else None,
            prompts_cache_dir=Path(pc) if pc else None,
            use_prompts_cache=bool(row.get("use_prompts_cache", True)),
            extract_batch_size=int(ebs) if ebs is not None else None,
            train_batch_size=int(tbs) if tbs is not None else None,
            use_train_amp=None if uta is None else bool(uta),
            n_jobs_superposition=int(njs) if njs is not None else 1,
        )



def main():
    parser = argparse.ArgumentParser(
        description="SPARC Cross-Coder: base vs aligned LLM activations (GRPO-style preference data)",
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("--base-model", type=str, default=None, help="Base HF model id")
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
        run_manifest(Path(args.manifest), force=args.force)
        return

    # For all non-manifest stages, these args are required
    for name, val in [("--base-model", args.base_model), ("--aligned-model", args.aligned_model),
                      ("--aligned-run-id", args.aligned_run_id), ("--layer", args.layer)]:
        if val is None:
            parser.error(f"{name} is required for stage={args.stage}")

    use_train_amp = False if args.no_train_amp else None

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
