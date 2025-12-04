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
        min_std: float = 0.01,
        min_warmth: float = 1.0,
        max_warmth: float = 1.0,
        mean_only_ablation: bool = False,
        feature_only_ablation: bool = False,
    ):
        super().__init__()
        self.warm_start_model = warm_start_model
        self.generative_model = generative_model
        self.min_warmth = min_warmth
        self.max_warmth = max_warmth
        self.mean_only_ablation = mean_only_ablation
        self.feature_only_ablation = feature_only_ablation
        self.min_std = min_std

        self.loss_weighting = loss_weighting

        self.scale_warmth = self.min_warmth != 1.0 or self.max_warmth != 1.0
        self.device = device

        if feature_only_ablation and (min_warmth != 1.0 or max_warmth != 1.0):
            raise ValueError(
                "When using feature only ablation, should always have warmth=1.0."
            )

        if mean_only_ablation and feature_only_ablation:
            raise ValueError("Cannot use both mean-only and feature-only ablation.")

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        prd_dist = self.warm_start_model.predict(ctx)
        wsm_loss = self.warm_start_model.nll(prd_dist, trg)

        # Don't backprop through the entire model.
        mean = prd_dist.mean  # .detach()
        stddev = prd_dist.stddev.detach()

        std, warmth = self._get_warm_std(stddev)
        if self.mean_only_ablation:
            std = torch.ones_like(std, device=self.device)
            warmth = None

        if self.feature_only_ablation:
            trg_n = trg
        else:
            # _n suffix = normalised space
            trg_n = (trg - mean) / std

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, mean, std], dim=1),
            warmth=warmth,
        )

        if self.loss_weighting:
            loss_weight = std
        else:
            loss_weight = None

        generative_loss = self.generative_model(
            gen_model_ctx, trg_n, loss_weight=loss_weight
        )

        return generative_loss
        # return wsm_loss + generative_loss

    def _get_warm_std(
        self, prd_std: torch.Tensor, warmth: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a warmth-scaled standard deviation.
        """
        prd_std = prd_std.clamp(min=self.min_std)
        if not self.scale_warmth:
            return prd_std, None
        batch_size = prd_std.shape[0]
        base_std = torch.ones_like(prd_std, device=self.device)
        if warmth is None:
            warmth = (
                torch.rand(batch_size, device=self.device)
                * (self.max_warmth - self.min_warmth)
                + self.min_warmth
            )
        warmth = warmth[:, None, None, None]

        prd_std = prd_std.clamp(min=1.0 - warmth)

        scaled_std = warmth * prd_std + (1 - warmth) * base_std

        squeezed = warmth.squeeze()
        if squeezed.ndim < 1:
            # For the case where we're doing inference on a single sample (B=1).
            squeezed = squeezed.unsqueeze(0)
        return scaled_std, squeezed

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

        initial_warmth = kwargs.get("warmth", self._get_sample_warmth(kwargs))

        std = prd_dist.stddev
        if self.scale_warmth:
            # During sampling, for now, we use a constant (full) warmth.
            # TODO: Experiment with different warmth schedules.
            warmth = torch.ones(num_samples, device=self.device) * initial_warmth
            std, warmth = self._get_warm_std(std, warmth)
        else:
            warmth = None

        if self.mean_only_ablation:
            std = torch.ones_like(std, device=self.device)
            warmth = None

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, prd_dist.mean, std], dim=1),
            warmth=warmth,
        )

        samples_n = self.generative_model.sample(gen_model_ctx, num_samples, **kwargs)

        if self.feature_only_ablation:
            return samples_n

        # Go back to unnormalised space
        samples = samples_n * std + prd_dist.mean

        return samples

    def _get_sample_warmth(self, kwargs) -> float:
        # TODO
        return self.max_warmth
        if "nfe" not in kwargs:
            return self.max_warmth
        nfe = kwargs["nfe"]
        if nfe <= 10:
            return self.max_warmth
        # For higher NFE, allow the generative model a bit more freedom.
        return 0.8 * (self.max_warmth - self.min_warmth)

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
