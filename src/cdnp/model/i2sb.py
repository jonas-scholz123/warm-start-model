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

        # Build marginal std tensors from the beta schedule.
        # Index 0 = t=0 (clean), index T = t=T (corrupted).
        betas = make_beta_schedule(n_timestep=num_timesteps)
        betas = torch.cat([torch.zeros(1), betas])  # shape: (T+1,)

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
    # Posterior p(x_{t-1} | x_t, x_0) — ported from official p_posterior
    # ------------------------------------------------------------------

    def _p_posterior(
        self,
        nprev: torch.Tensor,
        n: torch.Tensor,
        x_n: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse step: sample x_{t-1} ~ p(x_{t-1} | x_t, x_0_pred).

        Follows official i2sb/diffusion.py p_posterior exactly:
            std_delta = sqrt(std_fwd[n]^2 - std_fwd[nprev]^2)
            coefs via compute_gaussian_product_coef(std_fwd[nprev], std_delta)
        """
        std_n = self.std_fwd[n][:, None, None, None]
        std_nprev = self.std_fwd[nprev][:, None, None, None]
        std_delta = (std_n**2 - std_nprev**2).clamp(min=1e-8).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        return mu_x0 * x0 + mu_xn * x_n + var.sqrt() * torch.randn_like(x0)

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
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        assert ctx.image_ctx is not None
        x0, x1 = trg, self._get_x1(ctx)
        # timesteps in [1, T]
        step = torch.randint(1, self.T + 1, (x0.shape[0],), device=self.device)

        xt, _ = self._q_sample(step, x0, x1)
        label = self._compute_label(step, x0, xt)

        # Pass integer timesteps in [0, T-1] — UNet2DModel sinusoidal embedding
        # is designed for the full integer range, not [0, 1] floats.
        backbone_input = torch.cat([xt, ctx.image_ctx], dim=1)
        pred = padded_forward(self.backbone, backbone_input, step - 1)

        visible = ctx.image_ctx[:, 3:]
        # Supervise only on unobserved pixels
        mask = 1 - visible
        loss = F.mse_loss(pred * mask, label * mask)
        return loss

    # ------------------------------------------------------------------
    # Sampling (reverse DDPM-style, ported from official ddpm_sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int, nfe: int | None = None, **kwargs) -> torch.Tensor:
        assert ctx.image_ctx is not None
        visible = ctx.image_ctx[:, 3:]          # (B, 1, H, W)
        known_pixels = ctx.image_ctx[:, :3]     # observed pixel values
        x1 = self._get_x1(ctx)                 # corrupted image = starting point

        steps = nfe if nfe is not None else self.T
        # Integer timestep sequence: T, T-1, ..., 1  (length = steps)
        ts = torch.linspace(self.T, 1, steps, dtype=torch.long, device=self.device)

        xt = x1.clone()
        for i in range(len(ts) - 1):
            n = ts[i].expand(xt.shape[0])      # current step
            nprev = ts[i + 1].expand(xt.shape[0])  # previous (smaller) step

            # Predict x0 from xt at step n
            backbone_input = torch.cat([xt, ctx.image_ctx], dim=1)
            pred_label = padded_forward(self.backbone, backbone_input, n - 1)

            # Recover x0: label = (xt - x0) / std_fwd[n]  →  x0 = xt - label * std_fwd
            std_fwd = self.std_fwd[n][:, None, None, None]
            x0_pred = xt - pred_label * std_fwd

            if self.clip_denoise:
                x0_pred = x0_pred.clamp(-1, 1)

            # Reverse step for unknown pixels via true posterior p(x_{t-1} | x_t, x0_pred)
            xt_prev = self._p_posterior(nprev, n, xt, x0_pred)

            # Re-insert known pixels at the correct bridge marginal noise level for step t-1:
            # q(x_{t-1} | x_0=known, x_1=known) = N(known, std_sb[t-1]^2)
            std_sb_prev = self.std_sb[nprev][:, None, None, None]
            xt_known = known_pixels + std_sb_prev * torch.randn_like(known_pixels)

            xt = visible * xt_known + (1 - visible) * xt_prev

        # Final denoising step at t=1 → t=0: just return x0 prediction
        n = ts[-1].expand(xt.shape[0])
        backbone_input = torch.cat([xt, ctx.image_ctx], dim=1)
        pred_label = padded_forward(self.backbone, backbone_input, n - 1)
        std_fwd = self.std_fwd[n][:, None, None, None]
        x0_pred = xt - pred_label * std_fwd
        if self.clip_denoise:
            x0_pred = x0_pred.clamp(-1, 1)
        # Restore known pixels cleanly
        return visible * known_pixels + (1 - visible) * x0_pred

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
