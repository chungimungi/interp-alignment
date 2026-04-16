import contextlib
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import amp as torch_amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import collate_activations, create_paired_activation_dataset
from .model import SPARCCrossCoder, create_crosscoder
from .utils import flush_gpu, get_checkpoint_dir, get_device, get_metrics_dir, get_results_dir, save_checkpoint, save_json, set_seed


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class QualityGateError(Exception):
    pass


def check_quality_gate(fve_base: float, fve_aligned: float, dead_neuron_fraction: float, epoch: int) -> None:
    if fve_base < config.FVE_THRESHOLD:
        raise QualityGateError(f"Quality gate failed at epoch {epoch}: FVE_base={fve_base:.4f} < {config.FVE_THRESHOLD}")
    if fve_aligned < config.FVE_THRESHOLD:
        raise QualityGateError(f"Quality gate failed at epoch {epoch}: FVE_aligned={fve_aligned:.4f} < {config.FVE_THRESHOLD}")
    if dead_neuron_fraction > config.DEAD_NEURON_THRESHOLD:
        raise QualityGateError(f"Quality gate failed at epoch {epoch}: dead_neurons={dead_neuron_fraction:.4f} > {config.DEAD_NEURON_THRESHOLD}")


def _autocast_cm(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch_amp.autocast("cuda", dtype=dtype, enabled=True)


def _train_grad_scaler(device: torch.device, use_amp: bool) -> Optional[torch_amp.GradScaler]:
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return None
    return torch_amp.GradScaler("cuda", enabled=True)


def train_crosscoder(
    activations_data: Dict,
    input_dim: int,
    base_model_id: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    checkpoint_every: int = config.CHECKPOINT_EVERY,
    results_dir: Optional[Path] = None,
    expansion_factor: Optional[int] = None,
    topk: Optional[int] = None,
    use_amp: Optional[bool] = None,
) -> Dict:
    set_seed()
    device = get_device()
    if use_amp is None:
        use_amp = bool(getattr(config, "USE_TRAIN_AMP", True))
    scaler = _train_grad_scaler(device, use_amp)
    if use_amp and device.type == "cuda":
        amp_note = "bf16" if torch.cuda.is_bf16_supported() else "fp16+GradScaler"
        print(f"  Train AMP: {amp_note}")

    if config.CUDA_OPTIMIZATIONS and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    if results_dir is None:
        results_dir = get_results_dir(base_model_id, aligned_run_id, layer, position)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)

    train_dataset = create_paired_activation_dataset(activations_data, split="train")
    val_dataset = create_paired_activation_dataset(activations_data, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_activations,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_activations,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
    )

    crosscoder = create_crosscoder(
        input_dim,
        expansion_factor=expansion_factor,
        topk=topk,
    )
    crosscoder = crosscoder.to(device)

    optimizer = AdamW(crosscoder.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)

    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * config.WARMUP_FRACTION)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    training_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_fve_base": [],
        "train_fve_aligned": [],
        "val_fve_base": [],
        "val_fve_aligned": [],
        "dead_neurons": [],
        "l0_base": [],
        "l0_aligned": [],
        "self_recon": [],
        "cross_recon": [],
        "sparsity": [],
    }

    label = f"{base_model_id}/{aligned_run_id}/L{layer}/{position}"
    print(f"\nTraining cross-coder for {label}")
    print(f"  Dict size: {crosscoder.dict_size}, TopK: {crosscoder.topk}")
    print(f"  Forced shared: {crosscoder.n_forced_shared} ({crosscoder.forced_shared_fraction*100:.1f}%)")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        crosscoder.train()
        train_losses = []
        train_fve_base_list = []
        train_fve_aligned_list = []
        train_dead_list = []
        train_l0_base_list = []
        train_l0_aligned_list = []
        train_self_recon = []
        train_cross_recon = []
        train_sparsity = []

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

        for batch in batch_pbar:
            x_base = batch["activations_base"].to(device, dtype=torch.float32)
            x_aligned = batch["activations_aligned"].to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_cm(device, use_amp):
                outputs = crosscoder(x_base, x_aligned)
                losses = crosscoder.compute_loss(x_base, x_aligned, outputs)

            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(crosscoder.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(crosscoder.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()
            crosscoder.sync_forced_shared_post_step()

            with torch.no_grad():
                with _autocast_cm(device, use_amp):
                    fve = crosscoder.compute_fve(x_base, x_aligned, outputs)
                    dead = crosscoder.compute_dead_neurons(outputs["z_base"], outputs["z_aligned"])
                    l0 = crosscoder.compute_l0_sparsity(outputs["z_base"], outputs["z_aligned"])

            train_losses.append(losses["total"].item())
            train_fve_base_list.append(fve["fve_base"])
            train_fve_aligned_list.append(fve["fve_aligned"])
            train_dead_list.append(dead)
            train_l0_base_list.append(l0["l0_base"])
            train_l0_aligned_list.append(l0["l0_aligned"])
            train_self_recon.append(losses["self_recon"].item())
            train_cross_recon.append(losses["cross_recon"].item())
            train_sparsity.append(losses["sparsity"].item())

            batch_pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "fve_b": f"{fve['fve_base']:.3f}",
                    "fve_a": f"{fve['fve_aligned']:.3f}",
                }
            )
            del x_base, x_aligned, outputs, losses, fve, dead, l0

        if (epoch + 1) % config.FLUSH_GPU_EVERY_N_EPOCHS == 0 and torch.cuda.is_available():
            flush_gpu()

        crosscoder.eval()
        val_losses = []
        val_fve_base_list = []
        val_fve_aligned_list = []

        with torch.no_grad():
            for batch in val_loader:
                x_base = batch["activations_base"].to(device, dtype=torch.float32)
                x_aligned = batch["activations_aligned"].to(device, dtype=torch.float32)

                with _autocast_cm(device, use_amp):
                    outputs = crosscoder(x_base, x_aligned)
                    losses = crosscoder.compute_loss(x_base, x_aligned, outputs)
                    fve = crosscoder.compute_fve(x_base, x_aligned, outputs)

                val_losses.append(losses["total"].item())
                val_fve_base_list.append(fve["fve_base"])
                val_fve_aligned_list.append(fve["fve_aligned"])
                del x_base, x_aligned, outputs, losses, fve

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_train_fve_base = sum(train_fve_base_list) / len(train_fve_base_list)
        avg_train_fve_aligned = sum(train_fve_aligned_list) / len(train_fve_aligned_list)
        avg_val_fve_base = sum(val_fve_base_list) / len(val_fve_base_list) if val_fve_base_list else 0
        avg_val_fve_aligned = (
            sum(val_fve_aligned_list) / len(val_fve_aligned_list) if val_fve_aligned_list else 0
        )
        avg_dead = sum(train_dead_list) / len(train_dead_list)
        avg_l0_base = sum(train_l0_base_list) / len(train_l0_base_list)
        avg_l0_aligned = sum(train_l0_aligned_list) / len(train_l0_aligned_list)
        avg_self_recon = sum(train_self_recon) / len(train_self_recon)
        avg_cross_recon = sum(train_cross_recon) / len(train_cross_recon)
        avg_sparsity = sum(train_sparsity) / len(train_sparsity)

        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["train_fve_base"].append(avg_train_fve_base)
        training_history["train_fve_aligned"].append(avg_train_fve_aligned)
        training_history["val_fve_base"].append(avg_val_fve_base)
        training_history["val_fve_aligned"].append(avg_val_fve_aligned)
        training_history["dead_neurons"].append(avg_dead)
        training_history["l0_base"].append(avg_l0_base)
        training_history["l0_aligned"].append(avg_l0_aligned)
        training_history["self_recon"].append(avg_self_recon)
        training_history["cross_recon"].append(avg_cross_recon)
        training_history["sparsity"].append(avg_sparsity)

        epoch_pbar.set_postfix(
            {
                "train": f"{avg_train_loss:.4f}",
                "val": f"{avg_val_loss:.4f}",
                "fve_b": f"{avg_val_fve_base:.3f}",
                "fve_a": f"{avg_val_fve_aligned:.3f}",
                "dead": f"{avg_dead:.3f}",
            }
        )

        if (epoch + 1) % checkpoint_every == 0:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "fve_base": avg_val_fve_base,
                "fve_aligned": avg_val_fve_aligned,
                "dead_neurons": avg_dead,
            }
            save_checkpoint(crosscoder, optimizer, epoch + 1, metrics, checkpoint_dir)

    final_metrics = {
        "epoch": num_epochs,
        "train_loss": training_history["train_loss"][-1],
        "val_loss": training_history["val_loss"][-1],
        "fve_base": training_history["val_fve_base"][-1],
        "fve_aligned": training_history["val_fve_aligned"][-1],
        "dead_neurons": training_history["dead_neurons"][-1],
        "l0_base": training_history["l0_base"][-1],
        "l0_aligned": training_history["l0_aligned"][-1],
    }
    save_checkpoint(crosscoder, optimizer, num_epochs, final_metrics, checkpoint_dir, is_final=True)

    save_json(training_history, metrics_dir / "training_metrics.json")

    final_fve_base = training_history["val_fve_base"][-1]
    final_fve_aligned = training_history["val_fve_aligned"][-1]
    final_dead = training_history["dead_neurons"][-1]

    check_quality_gate(final_fve_base, final_fve_aligned, final_dead, num_epochs)

    print("\nTraining complete!")
    print(f"  Final FVE_base: {final_fve_base:.4f}, FVE_aligned: {final_fve_aligned:.4f}")
    print(f"  Dead neurons: {final_dead:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")

    return {
        "crosscoder": crosscoder,
        "training_history": training_history,
        "final_metrics": final_metrics,
        "results_dir": results_dir,
    }


def load_trained_crosscoder(
    input_dim: int,
    base_model_id: str,
    aligned_run_id: str,
    layer: int,
    position: str,
    results_dir: Optional[Path] = None,
) -> SPARCCrossCoder:
    device = get_device()
    if results_dir is None:
        results_dir = get_results_dir(base_model_id, aligned_run_id, layer, position)
    checkpoint_dir = get_checkpoint_dir(results_dir)

    crosscoder = create_crosscoder(input_dim)
    crosscoder = crosscoder.to(device)

    checkpoint_path = checkpoint_dir / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    crosscoder.load_state_dict(checkpoint["model_state_dict"])
    crosscoder.eval()

    return crosscoder


def compute_all_feature_activations(crosscoder: SPARCCrossCoder, activations_data: Dict) -> Dict:
    device = get_device()
    crosscoder.eval()

    activations_base = activations_data["activations_base"]
    activations_aligned = activations_data["activations_aligned"]
    if activations_base.device.type == "cuda":
        activations_base = activations_base.cpu()
    if activations_aligned.device.type == "cuda":
        activations_aligned = activations_aligned.cpu()

    all_z_base = []
    all_z_aligned = []
    batch_size = getattr(config, "ANALYZE_FEATURE_BATCH_SIZE", 256)
    num_samples = activations_base.shape[0]

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            x_base = activations_base[i : i + batch_size].to(device, dtype=torch.float32)
            x_aligned = activations_aligned[i : i + batch_size].to(device, dtype=torch.float32)
            outputs = crosscoder(x_base, x_aligned)
            all_z_base.append(outputs["z_base"].cpu())
            all_z_aligned.append(outputs["z_aligned"].cpu())
            del x_base, x_aligned, outputs

    if torch.cuda.is_available():
        flush_gpu()

    return {
        "z_base": torch.cat(all_z_base, dim=0),
        "z_aligned": torch.cat(all_z_aligned, dim=0),
        "sample_ids": activations_data["sample_ids"],
        "splits": activations_data["splits"],
    }
