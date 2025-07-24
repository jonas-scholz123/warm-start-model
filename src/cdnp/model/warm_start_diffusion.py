from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal

from cdnp.model.cnp import CNP
from cdnp.model.ctx import ModelCtx
from cdnp.model.ddpm import DDPM
from cdnp.model.flow_matching.flow_matching import FlowMatching


# TODO move CDNP to use this logic
class WarmStartDiffusion(nn.Module):
    def __init__(
        self,
        warm_start_model: CNP,
        generative_model: DDPM | FlowMatching,
        loss_weighting: bool,
        device: str,
        min_warmth: float = 1.0,
        max_warmth: float = 1.0,
    ):
        super().__init__()
        self.warm_start_model = warm_start_model
        self.generative_model = generative_model
        self.min_warmth = min_warmth
        self.max_warmth = max_warmth

        self.loss_weighting = loss_weighting

        self.scale_warmth = self.min_warmth != 1.0 or self.max_warmth != 1.0
        self.device = device

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        prd_dist = self.warm_start_model.predict(ctx)
        std, warmth = self._get_warm_std(prd_dist.stddev)

        # _n suffix = normalised space
        trg_n = (trg - prd_dist.mean) / std

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, prd_dist.mean, prd_dist.stddev], dim=1),
            warmth=warmth,
        )

        if self.loss_weighting:
            loss_weight = prd_dist.stddev
        else:
            loss_weight = None

        return self.generative_model(gen_model_ctx, trg_n, loss_weight=loss_weight)

    def _get_warm_std(
        self, prd_std: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a warmth-scaled standard deviation.
        """
        if not self.scale_warmth:
            return prd_std, None
        batch_size = prd_std.shape[0]
        base_std = torch.ones_like(prd_std, device=self.device)
        warmth = (
            torch.rand(batch_size, device=self.device)
            * (self.max_warmth - self.min_warmth)
            + self.min_warmth
        )[:, None, None, None]

        scaled_std = warmth * prd_std + (1 - warmth) * base_std
        return scaled_std, warmth.squeeze()

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        """
        # For conditional generation, generate samples based on the context.
        num_samples = ctx.image_ctx.shape[0]
        prd_dist = self.warm_start_model.predict(ctx)
        prd_dist = Normal(prd_dist.mean, prd_dist.stddev)

        if self.scale_warmth:
            # During sampling, for now, we use a constant (full) warmth.
            # TODO: Experiment with different warmth schedules.
            warmth = torch.ones(num_samples, device=self.device) * self.max_warmth
        else:
            warmth = None

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, prd_dist.mean, prd_dist.stddev], dim=1),
            warmth=warmth,
        )

        samples_n = self.generative_model.sample(gen_model_ctx, num_samples, **kwargs)

        # Go back to unnormalised space
        samples = samples_n * prd_dist.stddev + prd_dist.mean

        return samples

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
