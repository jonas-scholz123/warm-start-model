from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import UNet2DModel
from torch import nn
from torch.distributions import Normal

from cdnp.model.ctx import ModelCtx
from cdnp.model.low_rank_cov import nll as lr_nll
from cdnp.model.low_rank_cov import unnormalize as lr_unnormalize
from cdnp.model.util import padded_forward


@dataclass
class CNPPrediction:
    """Prediction from a CNP model.

    Attributes:
        mean: Per-pixel mean, shape (B, C, H, W).
        std: Per-pixel diagonal std, shape (B, C, H, W).
        V: Low-rank correlation factors, shape (B, rank, C, H, W).
            None when correlation_rank=0.
    """

    mean: torch.Tensor
    std: torch.Tensor
    V: Optional[torch.Tensor] = None

    @property
    def stddev(self) -> torch.Tensor:
        """Alias for std, for backward compatibility with Normal interface."""
        return self.std

    @property
    def diagonal_dist(self) -> Normal:
        """Return a diagonal Normal distribution (ignoring V)."""
        return Normal(self.mean, self.std)


class CNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        device: str,
        min_std: float,
        residual: bool = False,
        correlation_rank: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.min_std = min_std
        self.residual = residual
        self.correlation_rank = correlation_rank

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.

        :x: The input data (e.g., images).
        :ctx: The context data (e.g., class labels).
        """

        prd = self.predict(ctx)
        return self.nll(prd, trg)

    def predict(self, ctx: ModelCtx) -> CNPPrediction:
        im_ctx = ctx.image_ctx  # (B, C, H, W)
        labels = ctx.label_ctx

        assert im_ctx is not None, "Image context must be provided for CNP."

        # TODO: this is a hack - we don't need timesteps for CNP
        shape = (im_ctx.shape[0],)
        timesteps = torch.zeros(shape).long().to(self.device)

        pred = padded_forward(self.backbone, im_ctx, timesteps, class_labels=labels)

        if self.correlation_rank > 0:
            # Output channels: 2*C (mean + std) + C*rank (low-rank factors)
            num_out = pred.shape[1]
            num_trg_channels = num_out // (2 + self.correlation_rank)
            mean, std_raw, V_raw = pred.split(
                [num_trg_channels, num_trg_channels, num_trg_channels * self.correlation_rank],
                dim=1,
            )
            # Reshape V from (B, C*rank, H, W) -> (B, rank, C, H, W)
            B, _, H, W = V_raw.shape
            V = V_raw.reshape(B, num_trg_channels, self.correlation_rank, H, W)
            V = V.permute(0, 2, 1, 3, 4)  # (B, rank, C, H, W)
        else:
            mean, std_raw = pred.chunk(2, dim=1)
            V = None

        if self.residual:
            num_trg_channels = mean.shape[1]
            # By convention, the last channels of the image context should be the
            # residuals.
            res = im_ctx[:, -num_trg_channels:, :, :]
            mean = res + mean

        std = nn.functional.softplus(std_raw)
        std = torch.clamp(std, min=self.min_std)
        return CNPPrediction(mean=mean, std=std, V=V)

    def nll(self, prd: CNPPrediction, trg: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood.

        When V is present, uses the full NLL under N(mean, LLᵀ) where
        L = diag(d) + VVᵀ. Otherwise falls back to diagonal NLL.
        """
        if prd.V is not None:
            return lr_nll(trg, prd.mean, prd.std, prd.V)
        return -prd.diagonal_dist.log_prob(trg).mean()

    def _sample(self, prd: CNPPrediction) -> torch.Tensor:
        """Sample from the predicted distribution.

        When V is present, draws z ~ N(0,I) and computes x = Lz + mean
        (i.e. unnormalize). Otherwise samples from the diagonal Normal.
        """
        if prd.V is not None:
            z = torch.randn_like(prd.mean)
            return lr_unnormalize(z, prd.mean, prd.std, prd.V)
        return prd.diagonal_dist.sample()

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process. Shape:
            (num_samples, in_channels, height, width)
        :num_samples: (ignored)
        :return: Generated samples of shape
            (num_samples, out_channels, height, width).
        """
        return self._sample(self.predict(ctx))

    def sample_with_grad(self, ctx: ModelCtx) -> torch.Tensor:
        return self._sample(self.predict(ctx))

    def make_plot(self, ctx: ModelCtx) -> list[torch.Tensor]:
        pred = self.predict(ctx)
        masked_image = ctx.image_ctx[:, -3:, :, :]  # type: ignore
        mask = ctx.image_ctx[:, :1, :, :].repeat(1, 3, 1, 1)  # type: ignore
        return [mask, masked_image, pred.mean, pred.std, self._sample(pred)]
