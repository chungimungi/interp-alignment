from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class MultiLayerSPARCCrossCoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        expansion_factor: int,
        topk: int,
        topk_mode: str = config.MULTILAYER_TOPK_MODE,
        forced_shared_fraction: float = config.FORCED_SHARED_FRACTION,
        seed: int = config.SEED,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if topk_mode not in config.MULTILAYER_TOPK_MODES:
            raise ValueError(f"Unknown topk_mode {topk_mode!r}; expected one of {config.MULTILAYER_TOPK_MODES}")

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.expansion_factor = expansion_factor
        self.dict_size = input_dim * expansion_factor
        self.topk = topk
        self.topk_mode = topk_mode
        self.forced_shared_fraction = forced_shared_fraction

        torch.manual_seed(seed)

        self.encoder_base = nn.ModuleList([nn.Linear(input_dim, self.dict_size) for _ in range(n_layers)])
        self.encoder_aligned = nn.ModuleList([nn.Linear(input_dim, self.dict_size) for _ in range(n_layers)])
        self.decoder_base = nn.ModuleList([nn.Linear(self.dict_size, input_dim, bias=False) for _ in range(n_layers)])
        self.decoder_aligned = nn.ModuleList([nn.Linear(self.dict_size, input_dim, bias=False) for _ in range(n_layers)])

        n_forced_shared = int(self.dict_size * forced_shared_fraction)
        self.register_buffer(
            "forced_shared_indices",
            torch.randperm(self.dict_size)[:n_forced_shared],
        )
        self.n_forced_shared = n_forced_shared

        self._init_weights()
        self._init_forced_shared()

    def _init_weights(self):
        for modules in [self.encoder_base, self.encoder_aligned]:
            for module in modules:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        for modules in [self.decoder_base, self.decoder_aligned]:
            for module in modules:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")

    def _init_forced_shared(self):
        with torch.no_grad():
            for base_dec, aligned_dec in zip(self.decoder_base, self.decoder_aligned):
                aligned_dec.weight.data[:, self.forced_shared_indices] = (
                    base_dec.weight.data[:, self.forced_shared_indices].clone()
                )

    def _encode_stream(self, x: torch.Tensor, encoders: nn.ModuleList) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, n_layers, input_dim], got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_layers or x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected [batch, {self.n_layers}, {self.input_dim}], got shape {tuple(x.shape)}"
            )
        return torch.stack([encoder(x[:, i, :]) for i, encoder in enumerate(encoders)], dim=1)

    def _topk_mask_model_balanced_layer_agg(self, h_base: torch.Tensor, h_aligned: torch.Tensor) -> torch.Tensor:
        k_half = self.topk // 2
        h_base_agg = h_base.mean(dim=1)
        h_aligned_agg = h_aligned.mean(dim=1)

        _, topk_base_indices = torch.topk(h_base_agg, k_half, dim=-1)
        _, topk_aligned_indices = torch.topk(h_aligned_agg, k_half, dim=-1)

        mask = torch.zeros_like(h_base_agg, dtype=torch.bool)
        mask.scatter_(-1, topk_base_indices, True)
        mask.scatter_(-1, topk_aligned_indices, True)
        return mask

    def _topk_mask_global_sum(self, h_base: torch.Tensor, h_aligned: torch.Tensor) -> torch.Tensor:
        h_agg = h_base.sum(dim=1) + h_aligned.sum(dim=1)
        _, topk_indices = torch.topk(h_agg, self.topk, dim=-1)
        mask = torch.zeros_like(h_agg, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        return mask

    def global_topk(self, h_base: torch.Tensor, h_aligned: torch.Tensor) -> torch.Tensor:
        if self.topk_mode == "model_balanced_layer_agg":
            return self._topk_mask_model_balanced_layer_agg(h_base, h_aligned)
        if self.topk_mode == "global_sum":
            return self._topk_mask_global_sum(h_base, h_aligned)
        raise ValueError(f"Unknown topk_mode {self.topk_mode!r}")

    def encode(self, x_base: torch.Tensor, x_aligned: torch.Tensor):
        h_base = self._encode_stream(x_base, self.encoder_base)
        h_aligned = self._encode_stream(x_aligned, self.encoder_aligned)
        mask = self.global_topk(h_base, h_aligned)
        layer_mask = mask.unsqueeze(1).float()
        z_base = F.relu(h_base) * layer_mask
        z_aligned = F.relu(h_aligned) * layer_mask
        return z_base, z_aligned, mask

    def _decode_stream(self, z: torch.Tensor, decoders: nn.ModuleList) -> torch.Tensor:
        return torch.stack([decoder(z[:, i, :]) for i, decoder in enumerate(decoders)], dim=1)

    def decode(self, z_base: torch.Tensor, z_aligned: torch.Tensor):
        x_base_hat = self._decode_stream(z_base, self.decoder_base)
        x_aligned_hat = self._decode_stream(z_aligned, self.decoder_aligned)
        return x_base_hat, x_aligned_hat

    def cross_decode(self, z_base: torch.Tensor, z_aligned: torch.Tensor):
        x_base_from_aligned = self._decode_stream(z_aligned, self.decoder_base)
        x_aligned_from_base = self._decode_stream(z_base, self.decoder_aligned)
        return x_base_from_aligned, x_aligned_from_base

    def forward(self, x_base: torch.Tensor, x_aligned: torch.Tensor):
        z_base, z_aligned, mask = self.encode(x_base, x_aligned)
        x_base_hat, x_aligned_hat = self.decode(z_base, z_aligned)
        x_base_cross, x_aligned_cross = self.cross_decode(z_base, z_aligned)
        return {
            "z_base": z_base,
            "z_aligned": z_aligned,
            "mask": mask,
            "x_base_hat": x_base_hat,
            "x_aligned_hat": x_aligned_hat,
            "x_base_cross": x_base_cross,
            "x_aligned_cross": x_aligned_cross,
        }

    def _decoder_norm_sum(self) -> torch.Tensor:
        norms = []
        for base_dec, aligned_dec in zip(self.decoder_base, self.decoder_aligned):
            norms.append(base_dec.weight.norm(dim=0) + aligned_dec.weight.norm(dim=0))
        return torch.stack(norms, dim=0).sum(dim=0)

    def compute_loss(
        self,
        x_base: torch.Tensor,
        x_aligned: torch.Tensor,
        outputs: dict,
        lambda_sparsity: float = config.LAMBDA_SPARSITY,
        lambda_cross: float = config.LAMBDA_CROSS,
        lambda_shared_multiplier: float = config.LAMBDA_SHARED_MULTIPLIER,
    ) -> dict:
        z_base = outputs["z_base"]
        z_aligned = outputs["z_aligned"]
        x_base_hat = outputs["x_base_hat"]
        x_aligned_hat = outputs["x_aligned_hat"]
        x_base_cross = outputs["x_base_cross"]
        x_aligned_cross = outputs["x_aligned_cross"]

        loss_recon_base = F.mse_loss(x_base, x_base_hat)
        loss_recon_aligned = F.mse_loss(x_aligned, x_aligned_hat)
        loss_self = loss_recon_base + loss_recon_aligned

        loss_cross_base = F.mse_loss(x_base, x_base_cross)
        loss_cross_aligned = F.mse_loss(x_aligned, x_aligned_cross)
        loss_cross = loss_cross_base + loss_cross_aligned

        decoder_norm_sum = self._decoder_norm_sum()
        z_combined = (z_base.abs() + z_aligned.abs()).mean(dim=1) / 2

        forced_mask = torch.zeros(self.dict_size, device=x_base.device, dtype=torch.bool)
        forced_mask[self.forced_shared_indices] = True

        z_forced = z_combined[:, forced_mask]
        decoder_norms_forced = decoder_norm_sum[forced_mask]
        loss_l1_forced = (z_forced * decoder_norms_forced.unsqueeze(0)).sum(dim=-1).mean()

        z_standard = z_combined[:, ~forced_mask]
        decoder_norms_standard = decoder_norm_sum[~forced_mask]
        loss_l1_standard = (z_standard * decoder_norms_standard.unsqueeze(0)).sum(dim=-1).mean()

        lambda_shared = lambda_sparsity * lambda_shared_multiplier
        loss_sparsity = lambda_shared * loss_l1_forced + lambda_sparsity * loss_l1_standard
        total_loss = loss_self + lambda_cross * loss_cross + loss_sparsity

        return {
            "total": total_loss,
            "self_recon": loss_self,
            "cross_recon": loss_cross,
            "sparsity": loss_sparsity,
            "recon_base": loss_recon_base,
            "recon_aligned": loss_recon_aligned,
        }

    def sync_forced_shared_post_step(self):
        with torch.no_grad():
            for base_dec, aligned_dec in zip(self.decoder_base, self.decoder_aligned):
                aligned_dec.weight.data[:, self.forced_shared_indices] = (
                    base_dec.weight.data[:, self.forced_shared_indices].clone()
                )

    def compute_fve(self, x_base: torch.Tensor, x_aligned: torch.Tensor, outputs: dict) -> dict:
        x_base_hat = outputs["x_base_hat"]
        x_aligned_hat = outputs["x_aligned_hat"]
        mse_base_layer = ((x_base - x_base_hat) ** 2).mean(dim=(0, 2))
        mse_aligned_layer = ((x_aligned - x_aligned_hat) ** 2).mean(dim=(0, 2))
        var_base_layer = x_base.var(dim=(0, 2), unbiased=False)
        var_aligned_layer = x_aligned.var(dim=(0, 2), unbiased=False)
        fve_base_layer = 1.0 - mse_base_layer / (var_base_layer + 1e-8)
        fve_aligned_layer = 1.0 - mse_aligned_layer / (var_aligned_layer + 1e-8)
        return {
            "fve_base": fve_base_layer.mean().item(),
            "fve_aligned": fve_aligned_layer.mean().item(),
            "fve_base_by_layer": fve_base_layer.detach().cpu().tolist(),
            "fve_aligned_by_layer": fve_aligned_layer.detach().cpu().tolist(),
        }

    def compute_dead_neurons(self, z_base: torch.Tensor, z_aligned: torch.Tensor) -> float:
        z_combined = z_base + z_aligned
        active_per_feature = (z_combined.abs() > 0).float().sum(dim=(0, 1))
        return (active_per_feature == 0).float().mean().item()

    def compute_l0_sparsity(self, z_base: torch.Tensor, z_aligned: torch.Tensor) -> dict:
        l0_base_layer = (z_base.abs() > 0).float().sum(dim=-1).mean(dim=0)
        l0_aligned_layer = (z_aligned.abs() > 0).float().sum(dim=-1).mean(dim=0)
        return {
            "l0_base": l0_base_layer.mean().item(),
            "l0_aligned": l0_aligned_layer.mean().item(),
            "l0_base_by_layer": l0_base_layer.detach().cpu().tolist(),
            "l0_aligned_by_layer": l0_aligned_layer.detach().cpu().tolist(),
        }

    def get_decoder_weights(self) -> dict:
        return {
            "W_base_dec": torch.stack([dec.weight.data.clone() for dec in self.decoder_base], dim=0),
            "W_aligned_dec": torch.stack([dec.weight.data.clone() for dec in self.decoder_aligned], dim=0),
        }


def create_multilayer_crosscoder(
    input_dim: int,
    n_layers: int,
    expansion_factor: Optional[int] = None,
    topk: Optional[int] = None,
    topk_mode: Optional[str] = None,
    forced_shared_fraction: Optional[float] = None,
) -> MultiLayerSPARCCrossCoder:
    from .utils import get_expansion_factor_llm, get_topk_llm

    if expansion_factor is None:
        expansion_factor = get_expansion_factor_llm()
    if topk is None:
        topk = get_topk_llm()
    if topk_mode is None:
        topk_mode = config.MULTILAYER_TOPK_MODE
    if forced_shared_fraction is None:
        forced_shared_fraction = config.FORCED_SHARED_FRACTION

    return MultiLayerSPARCCrossCoder(
        input_dim=input_dim,
        n_layers=n_layers,
        expansion_factor=expansion_factor,
        topk=topk,
        topk_mode=topk_mode,
        forced_shared_fraction=forced_shared_fraction,
        seed=config.SEED,
    )
