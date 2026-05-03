from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class SPARCCrossCoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        topk: int,
        forced_shared_fraction: float = config.FORCED_SHARED_FRACTION,
        seed: int = config.SEED,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = input_dim * expansion_factor
        self.topk = topk
        self.forced_shared_fraction = forced_shared_fraction

        torch.manual_seed(seed)

        self.encoder_base = nn.Linear(input_dim, self.dict_size)
        self.encoder_aligned = nn.Linear(input_dim, self.dict_size)

        self.decoder_base = nn.Linear(self.dict_size, input_dim, bias=False)
        self.decoder_aligned = nn.Linear(self.dict_size, input_dim, bias=False)

        n_forced_shared = int(self.dict_size * forced_shared_fraction)
        self.register_buffer(
            "forced_shared_indices",
            torch.randperm(self.dict_size)[:n_forced_shared],
        )
        self.n_forced_shared = n_forced_shared

        self._init_weights()
        self._init_forced_shared()

    def _init_weights(self):
        for module in [self.encoder_base, self.encoder_aligned]:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

        for module in [self.decoder_base, self.decoder_aligned]:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")

    def _init_forced_shared(self):
        with torch.no_grad():
            self.decoder_aligned.weight.data[:, self.forced_shared_indices] = (
                self.decoder_base.weight.data[:, self.forced_shared_indices].clone()
            )

    def global_topk_balanced(self, h_base: torch.Tensor, h_aligned: torch.Tensor) -> torch.Tensor:
        k_half = self.topk // 2

        _, topk_base_indices = torch.topk(h_base, k_half, dim=-1)
        _, topk_aligned_indices = torch.topk(h_aligned, k_half, dim=-1)

        mask = torch.zeros_like(h_base, dtype=torch.bool)
        mask.scatter_(-1, topk_base_indices, True)
        mask.scatter_(-1, topk_aligned_indices, True)

        return mask

    def encode(self, x_base: torch.Tensor, x_aligned: torch.Tensor):
        h_base = self.encoder_base(x_base)
        h_aligned = self.encoder_aligned(x_aligned)

        mask = self.global_topk_balanced(h_base, h_aligned)

        z_base = F.relu(h_base) * mask.float()
        z_aligned = F.relu(h_aligned) * mask.float()

        return z_base, z_aligned, mask

    def decode(self, z_base: torch.Tensor, z_aligned: torch.Tensor):
        x_base_hat = self.decoder_base(z_base)
        x_aligned_hat = self.decoder_aligned(z_aligned)
        return x_base_hat, x_aligned_hat

    def cross_decode(self, z_base: torch.Tensor, z_aligned: torch.Tensor):
        x_base_from_aligned = self.decoder_base(z_aligned)
        x_aligned_from_base = self.decoder_aligned(z_base)
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

        W_base_norms = self.decoder_base.weight.norm(dim=0)
        W_aligned_norms = self.decoder_aligned.weight.norm(dim=0)
        decoder_norm_sum = W_base_norms + W_aligned_norms

        z_combined = (z_base.abs() + z_aligned.abs()) / 2

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
            self.decoder_aligned.weight.data[:, self.forced_shared_indices] = (
                self.decoder_base.weight.data[:, self.forced_shared_indices].clone()
            )

    def compute_fve(self, x_base: torch.Tensor, x_aligned: torch.Tensor, outputs: dict) -> dict:
        x_base_hat = outputs["x_base_hat"]
        x_aligned_hat = outputs["x_aligned_hat"]

        var_base = x_base.var()
        var_aligned = x_aligned.var()

        mse_base = F.mse_loss(x_base, x_base_hat)
        mse_aligned = F.mse_loss(x_aligned, x_aligned_hat)

        fve_base = 1.0 - mse_base / (var_base + 1e-8)
        fve_aligned = 1.0 - mse_aligned / (var_aligned + 1e-8)

        return {"fve_base": fve_base.item(), "fve_aligned": fve_aligned.item()}

    def compute_dead_neurons(self, z_base: torch.Tensor, z_aligned: torch.Tensor) -> float:
        z_combined = z_base + z_aligned
        active_per_feature = (z_combined.abs() > 0).float().sum(dim=0)
        dead_fraction = (active_per_feature == 0).float().mean().item()
        return dead_fraction

    def compute_l0_sparsity(self, z_base: torch.Tensor, z_aligned: torch.Tensor) -> dict:
        l0_base = (z_base.abs() > 0).float().sum(dim=-1).mean().item()
        l0_aligned = (z_aligned.abs() > 0).float().sum(dim=-1).mean().item()
        return {"l0_base": l0_base, "l0_aligned": l0_aligned}

    def get_decoder_weights(self) -> dict:
        return {
            "W_base_dec": self.decoder_base.weight.data.clone(),
            "W_aligned_dec": self.decoder_aligned.weight.data.clone(),
        }

    def get_encoder_weights(self) -> dict:
        return {
            "W_base_enc": self.encoder_base.weight.data.clone(),
            "W_aligned_enc": self.encoder_aligned.weight.data.clone(),
            "b_base_enc": self.encoder_base.bias.data.clone(),
            "b_aligned_enc": self.encoder_aligned.bias.data.clone(),
        }


def create_crosscoder(
    input_dim: int,
    expansion_factor: Optional[int] = None,
    topk: Optional[int] = None,
) -> SPARCCrossCoder:
    from .utils import get_expansion_factor_llm, get_topk_llm

    if expansion_factor is None:
        expansion_factor = get_expansion_factor_llm()
    if topk is None:
        topk = get_topk_llm()

    return SPARCCrossCoder(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        topk=topk,
        forced_shared_fraction=config.FORCED_SHARED_FRACTION,
        seed=config.SEED,
    )
