"""
Image-to-Image Schrödinger Bridge (I2SB), ICML 2023.
https://arxiv.org/abs/2302.05872

Core maths ported from the official repo (i2sb/diffusion.py and i2sb/runner.py),
minus ipdb imports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch import nn

from cdnp.model.ctx import ModelCtx
from cdnp.model.meta.unet import UNetModel
from cdnp.model.util import padded_forward


# ---------------------------------------------------------------------------
# Gaussian product helpers (verbatim from official i2sb/diffusion.py)
# ---------------------------------------------------------------------------


def compute_gaussian_product_coef(sigma1: torch.Tensor, sigma2: torch.Tensor):
    """
    Given p1 = N(0, sigma1²) and p2 = N(0, sigma2²), the product
    p1*p2 is N(mu, sigma²) where:
        coef1 = sigma2² / (sigma1² + sigma2²)
        coef2 = sigma1² / (sigma1² + sigma2²)
        sigma = sigma1 * sigma2 / sqrt(sigma1² + sigma2²)
    """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1 * sigma2) ** 2 / denom
    return coef1, coef2, var


# ---------------------------------------------------------------------------
# Beta schedule (verbatim from official i2sb/runner.py)
# ---------------------------------------------------------------------------


def make_beta_schedule(n_timestep: int = 1000, linear_end: float = 2e-2) -> torch.Tensor:
    """Symmetric beta schedule that peaks at the midpoint — key to I2SB bridge."""
    betas = torch.linspace(1e-4**0.5, linear_end**0.5, n_timestep) ** 2
    half = betas[: n_timestep // 2]
    return torch.cat([half, half.flip(0)])


# ---------------------------------------------------------------------------
# I2SB model
# ---------------------------------------------------------------------------


class I2SB(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel | UNetModel,
        device: str,
        num_timesteps: int = 1000,
        beta_max: float = 0.3,
        clip_denoise: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.T = num_timesteps
        self.clip_denoise = clip_denoise

        # Build marginal std tensors from the beta schedule
        betas = make_beta_schedule(n_timestep=num_timesteps)
        betas = torch.cat([torch.zeros(1), betas])  # index 0 = t=0

        std_fwd = torch.sqrt(betas.cumsum(0).clamp(min=0))
        std_bwd = torch.sqrt((betas.flip(0)).cumsum(0).flip(0).clamp(min=0))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = var.sqrt()

        self.register_buffer("betas", betas)
        self.register_buffer("std_fwd", std_fwd)
        self.register_buffer("std_bwd", std_bwd)
        self.register_buffer("std_sb", std_sb)
        self.register_buffer("mu_x0", mu_x0)
        self.register_buffer("mu_x1", mu_x1)

    # ------------------------------------------------------------------
    # Bridge marginal q(x_t | x_0, x_1)  — Eq 11 in paper
    # ------------------------------------------------------------------

    def _q_sample(
        self, step: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample xt ~ q(xt | x0, x1) and return (xt, std_sb[step])."""
        mu_x0 = self.mu_x0[step][:, None, None, None]
        mu_x1 = self.mu_x1[step][:, None, None, None]
        std = self.std_sb[step][:, None, None, None]
        xt = mu_x0 * x0 + mu_x1 * x1 + std * torch.randn_like(x0)
        return xt, std

    # ------------------------------------------------------------------
    # Training label  — Eq 12 in paper
    # ------------------------------------------------------------------

    def _compute_label(
        self, step: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        std_fwd = self.std_fwd[step][:, None, None, None]
        return (xt - x0) / std_fwd.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Construct corrupted endpoint X₁ from context
    # ------------------------------------------------------------------

    def _get_x1(self, ctx: ModelCtx) -> torch.Tensor:
        """
        ctx.image_ctx: (B, 4, H, W) = [masked_rgb(3) | mask(1)]
        mask=1  → pixel is observed (known).
        Known pixels are preserved; unknown pixels become i.i.d. Gaussian noise.
        """
        assert ctx.image_ctx is not None
        visible = ctx.image_ctx[:, 3:]          # (B, 1, H, W)
        x_ctx = ctx.image_ctx[:, :3]            # (B, 3, H, W)
        return x_ctx + (1 - visible) * torch.randn_like(x_ctx)

    # ------------------------------------------------------------------
    # Posterior  p(x_{t-1} | x_t, x_0_pred, x_1)  — from official diffusion.py
    # ------------------------------------------------------------------

    def _p_posterior(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        # std for t-1 and t
        std_n = self.std_sb[step - 1][:, None, None, None]  # t-1
        std_c = self.std_sb[step][:, None, None, None]       # t
        std_delta = (std_c**2 - std_n**2).clamp(min=1e-8).sqrt()

        mu_x0c, mu_x1c, var = compute_gaussian_product_coef(
            self.std_fwd[step - 1][:, None, None, None],
            std_delta,
        )
        mu_x0n, mu_x1n, _ = compute_gaussian_product_coef(
            self.std_fwd[step][:, None, None, None],
            std_delta,
        )
        # Eq from official repo:  mean = mu_x0c*x0 + mu_x1c*xt  adjusted for posterior
        mean = (
            mu_x0n / mu_x0c * (xt - mu_x1c * x1)
            + (1 - mu_x0n / mu_x0c) * x0
            + mu_x0n * x1  # ← accounts for the x1 endpoint
        )
        # Simpler version matching the official ddpm_sampling:
        mu_x0t, mu_x1t, _ = compute_gaussian_product_coef(
            self.std_fwd[step - 1][:, None, None, None],
            self.std_bwd[step][:, None, None, None],
        )
        # Use official formula
        mean = mu_x0t * x0 + mu_x1t * x1
        std = std_n
        return mean + std * torch.randn_like(xt)

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        assert ctx.image_ctx is not None
        x0, x1 = trg, self._get_x1(ctx)
        # timesteps in [1, T]
        step = torch.randint(1, self.T + 1, (x0.shape[0],), device=self.device)

        xt, _ = self._q_sample(step, x0, x1)
        label = self._compute_label(step, x0, xt)

        # xt(3) + masked_image(3) + mask(1) = 7 channels
        t_norm = (step.float() - 1) / (self.T - 1)  # [0, 1]
        backbone_input = torch.cat([xt, ctx.image_ctx], dim=1)
        pred = padded_forward(self.backbone, backbone_input, t_norm)

        visible = ctx.image_ctx[:, 3:]
        # Supervise only on unobserved pixels
        mask = (1 - visible)
        loss = F.mse_loss(pred * mask, label * mask)
        return loss

    # ------------------------------------------------------------------
    # Sampling (reverse DDPM-style)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int, nfe: int | None = None, **kwargs) -> torch.Tensor:
        assert ctx.image_ctx is not None
        visible = ctx.image_ctx[:, 3:]   # (B, 1, H, W)
        x1 = self._get_x1(ctx)          # start from corrupted image

        steps = nfe if nfe is not None else self.T
        # Subsample timestep indices evenly
        ts = torch.linspace(self.T, 1, steps, dtype=torch.long, device=self.device)

        xt = x1.clone()
        for i, t_val in enumerate(ts):
            step = t_val.expand(xt.shape[0])  # (B,)

            # Predict x0 from current xt
            t_norm = (step.float() - 1) / (self.T - 1)
            backbone_input = torch.cat([xt, ctx.image_ctx], dim=1)
            pred_label = padded_forward(self.backbone, backbone_input, t_norm)

            # Recover x0 from label: label = (xt - x0) / std_fwd[t]  →  x0 = xt - label * std_fwd
            std_fwd = self.std_fwd[step][:, None, None, None]
            x0_pred = xt - pred_label * std_fwd

            if self.clip_denoise:
                x0_pred = x0_pred.clamp(-1, 1)

            # Re-insert known pixels: keep visible pixels from x1 (the masked image)
            x0_pred = visible * ctx.image_ctx[:, :3] + (1 - visible) * x0_pred

            # Step to t-1 (or stop at t=1)
            if i < len(ts) - 1:
                prev_step = ts[i + 1].expand(xt.shape[0])
                mu_x0 = self.mu_x0[prev_step][:, None, None, None]
                mu_x1 = self.mu_x1[prev_step][:, None, None, None]
                std = self.std_sb[prev_step][:, None, None, None]
                xt = mu_x0 * x0_pred + mu_x1 * x1 + std * torch.randn_like(x0_pred)
                # Re-insert known pixels at noisy level too
                std_bwd = self.std_bwd[prev_step][:, None, None, None]
                x1_noised = ctx.image_ctx[:, :3] + std_bwd * torch.randn_like(x0_pred)
                xt = visible * x1_noised + (1 - visible) * xt
            else:
                xt = x0_pred

        return xt

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
