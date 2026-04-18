import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import dotenv
import torch
import wandb
from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.load_model import load_model
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast


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


def _hf_torch_dtype_str(dt: torch.dtype) -> str:
    """HF ``from_pretrained`` accepts dtype strings; sae_lens JSON-saves ``model_from_pretrained_kwargs``."""
    names: dict[torch.dtype, str] = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    if dt not in names:
        raise ValueError(f"Unsupported torch_dtype for HF JSON config: {dt}")
    return names[dt]


def _tokenizer_fallback_ids(model_name: str) -> list[str]:
    """Fallback tokenizers for repos with broken tokenizer metadata."""
    ml = model_name.lower()
    out: list[str] = []
    if "qwen3-4b" in ml:
        out.append("Qwen/Qwen3-4B-Instruct-2507")
    if "llama-3.2-3b" in ml:
        out.append("meta-llama/Llama-3.2-3B-Instruct")
    return out


def _install_tokenizer_fallback_patch(args: argparse.Namespace) -> None:
    """
    Patch AutoTokenizer.from_pretrained used inside sae-lens load_model so
    broken tokenizer repos (TokenizersBackend) can fall back gracefully.
    """
    original_from_pretrained = AutoTokenizer.from_pretrained
    extra_sources: list[str] = []
    if args.tokenizer_name:
        extra_sources.append(args.tokenizer_name)
    for fid in _tokenizer_fallback_ids(args.model_name):
        if fid not in extra_sources:
            extra_sources.append(fid)

    if not extra_sources:
        return

    require_chat_template = bool(args.use_chat_formatting)

    def _has_chat_template(tok: Any) -> bool:
        return bool(getattr(tok, "chat_template", None))

    def _patched_from_pretrained(
        pretrained_model_name_or_path: str, *fa: Any, **fkw: Any
    ):
        base_sources = [pretrained_model_name_or_path, *extra_sources]
        seen: set[str] = set()
        sources = [s for s in base_sources if not (s in seen or seen.add(s))]
        last_err: BaseException | None = None

        for src in sources:
            for use_fast in (True, False):
                try:
                    tok = original_from_pretrained(
                        src,
                        *fa,
                        use_fast=use_fast,
                        **fkw,
                    )
                    if require_chat_template and not _has_chat_template(tok):
                        last_err = ValueError(
                            f"Tokenizer {src!r} loaded but has no chat_template; trying fallback."
                        )
                        continue
                    return tok
                except BaseException as e:
                    last_err = e
            try:
                # Some repos fail AutoTokenizer dispatch but still expose tokenizer files.
                tok = PreTrainedTokenizerFast.from_pretrained(src, **fkw)
                if require_chat_template and not _has_chat_template(tok):
                    last_err = ValueError(
                        f"TokenizerFast {src!r} loaded but has no chat_template; trying fallback."
                    )
                    continue
                return tok
            except BaseException as e:
                last_err = e
                continue
        assert last_err is not None
        raise last_err

    AutoTokenizer.from_pretrained = _patched_from_pretrained  # type: ignore[assignment]


def _infer_d_in(
    model_name: str,
    model_class_name: str,
    device: str,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> int:
    """
    HookedTransformer exposes ``cfg.d_model``; HookedProxyLM (AutoModelForCausalLM wrapper)
    does not, so fall back to the inner HF model's ``config.hidden_size``.
    """
    kwargs = model_from_pretrained_kwargs or {}
    if model_class_name == "AutoModelForCausalLM":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, **kwargs)
        width = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if width is None:
            raise RuntimeError(f"Could not infer hidden size from HF config for {model_name!r}")
        return int(width)

    model = load_model(model_class_name, model_name, device=device)
    try:
        width = getattr(getattr(model, "cfg", None), "d_model", None)
        if width is None:
            inner = getattr(model, "model", model)
            width = getattr(getattr(inner, "config", None), "hidden_size", None)
        if width is None:
            raise RuntimeError("Could not infer d_model from model config")
        return int(width)
    finally:
        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_runner_config(args: argparse.Namespace, d_in: int) -> LanguageModelSAERunnerConfig[Any]:
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=args.d_sae,
        k=int(args.k),
        topk_threshold_lr=args.topk_threshold_lr,
        dtype=args.dtype,
        device=args.device,
        decoder_init_norm=args.decoder_init_norm,
        aux_loss_coefficient=args.aux_loss_coeff,
    )

    # We keep sae-lens "wandb logging path" enabled when local metric capture is requested,
    # because that is where sae-lens computes the rich metric dictionary.
    log_to_wandb_path = args.log_to_wandb or args.save_metrics_jsonl
    log_cfg = LoggingConfig(
        log_to_wandb=log_to_wandb_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        wandb_log_frequency=args.wandb_log_frequency,
    )

    model_from_pretrained_kwargs: dict[str, Any] = {}
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        # `sae_lens.load_model` uses Hugging Face under the hood; `token` is the modern HF arg.
        model_from_pretrained_kwargs["token"] = hf_token
    model_dtype = _resolve_model_dtype(args.model_dtype)
    if model_dtype is not None:
        # Must be JSON-serializable: full ``torch.dtype`` ends up in SAEMetadata and breaks checkpoint save.
        model_from_pretrained_kwargs["torch_dtype"] = _hf_torch_dtype_str(model_dtype)
    if args.trust_remote_code:
        model_from_pretrained_kwargs["trust_remote_code"] = True

    seqpos_slice = (args.sequence_start, args.sequence_end)

    # sae_lens.LanguageModelSAERunnerConfig defaults is_dataset_tokenized=True;
    # __post_init__ rejects use_chat_formatting=True together with that default.
    is_dataset_tokenized = False if args.use_chat_formatting else True

    return LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=args.model_name,
        model_class_name=args.model_class_name,
        hook_name=args.hook_name,
        hook_head_index=args.hook_head_index,
        dataset_path=args.dataset,
        dataset_trust_remote_code=args.dataset_trust_remote_code,
        streaming=not args.disable_streaming,
        is_dataset_tokenized=is_dataset_tokenized,
        use_chat_formatting=args.use_chat_formatting,
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return f"<tensor shape={tuple(value.shape)}>"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _install_local_metrics_sink(metrics_path: Path, forward_to_real_wandb: bool) -> None:
    """
    Intercept wandb logging calls and persist metric dicts as JSONL.
    - If `forward_to_real_wandb` is False, wandb network/artifact calls become no-ops.
    - If True, data is written locally and forwarded to real wandb as well.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fp = metrics_path.open("w", encoding="utf-8")

    real_init = wandb.init
    real_log = wandb.log
    real_finish = wandb.finish
    real_hist = wandb.Histogram
    real_artifact = wandb.Artifact
    real_log_artifact = wandb.log_artifact

    class _DummyArtifact:
        def __init__(self, *_args: Any, **_kwargs: Any):
            pass

        def add_file(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _DummyHistogram:
        def __init__(self, values: Any):
            self.values = values

    def _write_record(kind: str, payload: dict[str, Any]) -> None:
        rec = {"kind": kind, **payload}
        metrics_fp.write(json.dumps(_json_safe(rec), ensure_ascii=True) + "\n")
        metrics_fp.flush()

    def _patched_init(*args: Any, **kwargs: Any):
        _write_record("wandb_init", {"args": list(args), "kwargs": kwargs})
        if forward_to_real_wandb:
            return real_init(*args, **kwargs)
        return None

    def _patched_log(data: dict[str, Any], *args: Any, **kwargs: Any):
        step = kwargs.get("step")
        _write_record("metric", {"step": step, "data": data})
        if forward_to_real_wandb:
            return real_log(data, *args, **kwargs)
        return None

    def _patched_finish(*args: Any, **kwargs: Any):
        _write_record("wandb_finish", {})
        try:
            if forward_to_real_wandb:
                return real_finish(*args, **kwargs)
            return None
        finally:
            metrics_fp.flush()
            metrics_fp.close()

    def _patched_log_artifact(*args: Any, **kwargs: Any):
        _write_record("artifact", {"args": list(args), "kwargs": kwargs})
        if forward_to_real_wandb:
            return real_log_artifact(*args, **kwargs)
        return None

    wandb.init = _patched_init  # type: ignore[assignment]
    wandb.log = _patched_log  # type: ignore[assignment]
    wandb.finish = _patched_finish  # type: ignore[assignment]
    wandb.log_artifact = _patched_log_artifact  # type: ignore[assignment]
    if forward_to_real_wandb:
        wandb.Histogram = real_hist  # type: ignore[assignment]
        wandb.Artifact = real_artifact  # type: ignore[assignment]
    else:
        wandb.Histogram = _DummyHistogram  # type: ignore[assignment]
        wandb.Artifact = _DummyArtifact  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Batch Top-K SAE using sae-lens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument(
        "--model-class-name",
        default="AutoModelForCausalLM",
        choices=["HookedTransformer", "HookedMamba", "AutoModelForCausalLM"],
        help="Use AutoModelForCausalLM for HF models (SmolLM3/Llama/Qwen/MInAlA checkpoints); "
        "HookedTransformer only works for models in TransformerLens's pretrained table.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="HF weight dtype for from_pretrained (JSON-safe string passed through sae-lens metadata).",
    )
    parser.add_argument(
        "--hook-name",
        default="model.layers.0",
        help="For AutoModelForCausalLM, this is the HF submodule path (e.g. 'model.layers.19'); "
        "for HookedTransformer it follows TL conventions (e.g. 'blocks.19.hook_resid_post').",
    )
    parser.add_argument("--hook-head-index", type=int)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to HF from_pretrained for models with custom code (e.g. SmolLM3).",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional HF tokenizer id fallback when model repo tokenizer metadata is broken.",
    )

    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-trust-remote-code", action="store_true")
    parser.add_argument("--disable-streaming", action="store_true")
    parser.add_argument(
        "--use-chat-formatting",
        action="store_true",
        help="Treat dataset rows as chats; columns 'conversation'/'conversations'/'messages'/'text' are supported.",
    )
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
    parser.add_argument(
        "--save-metrics-jsonl",
        action="store_true",
        help="Save full sae-lens metric dictionaries to <output-path>/metrics.jsonl for offline plotting.",
    )
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
    dotenv.load_dotenv(dotenv_path=str(Path(__file__).resolve().parent / ".env"))
    args = _parse_args()
    args.device = _resolve_device(args.device)
    if args.lr_end is None:
        args.lr_end = args.lr / 10

    probe_kwargs: dict[str, Any] = {}
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        probe_kwargs["token"] = hf_token
    if args.trust_remote_code:
        probe_kwargs["trust_remote_code"] = True

    d_in = args.d_in or _infer_d_in(
        args.model_name, args.model_class_name, args.device, probe_kwargs
    )
    _install_tokenizer_fallback_patch(args)
    cfg = _build_runner_config(args, d_in)

    if args.save_metrics_jsonl:
        metrics_path = Path(args.output_path) / "metrics.jsonl"
        _install_local_metrics_sink(
            metrics_path=metrics_path,
            forward_to_real_wandb=bool(args.log_to_wandb),
        )

    runner = LanguageModelSAETrainingRunner(cfg=cfg)
    runner.run()

    print(f"Finished training. Inference SAE saved to: {cfg.output_path}")


if __name__ == "__main__":
    # Force process exit to avoid occasional hangs in interpreter teardown
    # after training (seen with some third-party library background services).
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
