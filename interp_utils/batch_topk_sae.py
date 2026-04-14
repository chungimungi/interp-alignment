import argparse
from pathlib import Path
from typing import Any

import torch
from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.load_model import load_model
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_model_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    dtype_lookup: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_lookup:
        raise ValueError(f"Unsupported model dtype: {dtype}")
    return dtype_lookup[dtype]


def _infer_d_in(model_name: str, model_class_name: str, device: str) -> int:
    model = load_model(model_class_name, model_name, device=device)
    width = getattr(model.cfg, "d_model", None)
    if width is None:
        model.to("cpu")
        torch.cuda.empty_cache()
        raise RuntimeError("Could not infer d_model from model config")
    model.to("cpu")
    torch.cuda.empty_cache()
    return int(width)


def _build_runner_config(args: argparse.Namespace, d_in: int) -> LanguageModelSAERunnerConfig[Any]:
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=args.d_sae,
        k=args.k,
        topk_threshold_lr=args.topk_threshold_lr,
        dtype=args.dtype,
        device=args.device,
        decoder_init_norm=args.decoder_init_norm,
        aux_loss_coefficient=args.aux_loss_coeff,
    )

    log_cfg = LoggingConfig(
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        wandb_log_frequency=args.wandb_log_frequency,
    )

    model_from_pretrained_kwargs: dict[str, Any] = {}
    model_dtype = _resolve_model_dtype(args.model_dtype)
    if model_dtype is not None:
        model_from_pretrained_kwargs["torch_dtype"] = model_dtype

    seqpos_slice = (args.sequence_start, args.sequence_end)

    return LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=args.model_name,
        model_class_name=args.model_class_name,
        hook_name=args.hook_name,
        hook_head_index=args.hook_head_index,
        dataset_path=args.dataset,
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        streaming=not args.disable_streaming,
        context_size=args.context_size,
        use_cached_activations=False,
        cached_activations_path=str(Path(args.cached_activations_path))
        if args.cached_activations_path
        else None,
        n_batches_in_buffer=args.n_batches_in_buffer,
        training_tokens=args.training_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        seqpos_slice=seqpos_slice,
        disable_concat_sequences=args.disable_concat_sequences,
        sequence_separator_token=args.sequence_separator_token,
        device=args.device,
        act_store_device=args.act_store_device or "cpu",
        dtype=args.dtype,
        seed=args.seed,
        prepend_bos=args.prepend_bos,
        autocast=args.autocast,
        autocast_lm=args.autocast_lm,
        compile_llm=args.compile_llm,
        compile_sae=args.compile_sae,
        train_batch_size_tokens=args.train_batch_size_tokens,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr=args.lr,
        lr_scheduler_name=args.lr_scheduler,
        lr_warm_up_steps=args.lr_warmup_steps,
        lr_end=args.lr_end,
        lr_decay_steps=args.lr_decay_steps,
        n_restart_cycles=args.n_restart_cycles,
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=args.feature_sampling_window,
        dead_feature_threshold=args.dead_feature_threshold,
        n_eval_batches=args.n_eval_batches,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        logger=log_cfg,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        save_final_checkpoint=args.save_final_checkpoint,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_path=args.output_path,
        verbose=not args.quiet,
        model_kwargs={},
        model_from_pretrained_kwargs=model_from_pretrained_kwargs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Batch Top-K SAE using sae-lens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument(
        "--model-class-name",
        default="HookedTransformer",
        choices=["HookedTransformer", "HookedMamba", "AutoModelForCausalLM"],
    )
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--hook-name", default="blocks.0.hook_mlp_out")
    parser.add_argument("--hook-head-index", type=int)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-trust-remote-code", action="store_true")
    parser.add_argument("--disable-streaming", action="store_true")
    parser.add_argument("--sequence-separator-token", default="bos")
    parser.add_argument("--disable-concat-sequences", action="store_true")
    parser.add_argument("--sequence-start", type=int)
    parser.add_argument("--sequence-end", type=int)

    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--d-in", type=int)
    parser.add_argument("--k", type=float, default=64.0)
    parser.add_argument("--topk-threshold-lr", type=float, default=0.01)
    parser.add_argument("--aux-loss-coeff", type=float, default=1.0)
    parser.add_argument("--decoder-init-norm", type=float, default=0.1)
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--training-tokens", type=int, default=200_000)
    parser.add_argument("--train-batch-size-tokens", type=int, default=2048)
    parser.add_argument("--store-batch-size-prompts", type=int, default=8)
    parser.add_argument("--n-batches-in-buffer", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--lr-end", type=float)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--n-restart-cycles", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--dead-feature-window", type=int, default=1000)
    parser.add_argument("--feature-sampling-window", type=int, default=1000)
    parser.add_argument("--dead-feature-threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-eval-batches", type=int, default=25)
    parser.add_argument("--eval-batch-size-prompts", type=int)

    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sae_lens_training")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-log-frequency", type=int, default=10)
    parser.add_argument("--run-name")

    parser.add_argument("--output-path", default="output/batchtopk")
    parser.add_argument("--checkpoint-path", default="checkpoints/batchtopk")
    parser.add_argument("--n-checkpoints", type=int, default=1)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--cached-activations-path")
    parser.add_argument("--act-store-device", default="cpu")

    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--compile-llm", action="store_true")
    parser.add_argument("--compile-sae", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.device = _resolve_device(args.device)
    if args.lr_end is None:
        args.lr_end = args.lr / 10

    d_in = args.d_in or _infer_d_in(args.model_name, args.model_class_name, args.device)
    cfg = _build_runner_config(args, d_in)

    runner = LanguageModelSAETrainingRunner(cfg=cfg)
    runner.run()

    print(f"Finished training. Inference SAE saved to: {cfg.output_path}")


if __name__ == "__main__":
    main()
