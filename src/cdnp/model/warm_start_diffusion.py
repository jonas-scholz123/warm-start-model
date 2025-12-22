from typing import Optional

import numpy as np
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
        warm_start_model: CNP | None,
        generative_model: DDPM | FlowMatching,
        loss_weighting: bool,
        device: str,
        min_std: float = 0.01,
        min_warmth: float = 1.0,
        max_warmth: float = 1.0,
        mean_only_ablation: bool = False,
        feature_only_ablation: bool = False,
        end_to_end: bool = True,
        norm_param_path: str | None = None,
    ):
        super().__init__()
        if warm_start_model is None and norm_param_path is None:
            raise ValueError(
                "Either warm_start_model or norm_param_path must be provided."
            )

        if warm_start_model is None and end_to_end:
            raise ValueError(
                "End-to-end training is not possible without a warm-start model."
            )

        self.warm_start_model = warm_start_model
        self.generative_model = generative_model
        self.min_warmth = min_warmth
        self.max_warmth = max_warmth
        self.mean_only_ablation = mean_only_ablation
        self.feature_only_ablation = feature_only_ablation
        self.min_std = min_std

        if norm_param_path is None:
            self.prd_dist = None
        else:
            norm_params = np.load(norm_param_path)
            mean = torch.tensor(norm_params["mean"], device=device)
            std = torch.tensor(norm_params["std"], device=device)
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
            self.prd_dist = Normal(mean, std)

        self.loss_weighting = loss_weighting
        self.end_to_end = end_to_end

        if self.warm_start_model is not None:
            if end_to_end:
                self.warm_start_model.requires_grad_(True)
            else:
                self.warm_start_model.requires_grad_(False)

        self.scale_warmth = self.min_warmth != 1.0 or self.max_warmth != 1.0
        self.device = device

        if feature_only_ablation and (min_warmth != 1.0 or max_warmth != 1.0):
            raise ValueError(
                "When using feature only ablation, should always have warmth=1.0."
            )

        if mean_only_ablation and feature_only_ablation:
            raise ValueError("Cannot use both mean-only and feature-only ablation.")

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        if self.prd_dist is None:
            prd_dist = self.warm_start_model.predict(ctx)  # type: ignore
        else:
            prd_dist = self.prd_dist
        mean = prd_dist.mean
        std = prd_dist.stddev.detach()

        std, warmth = self._get_warm_std(std)

        if self.mean_only_ablation:
            std = torch.ones_like(std, device=self.device)
            warmth = None

        if self.feature_only_ablation:
            trg_n = trg
        else:
            # _n suffix = normalised space
            trg_n = (trg - mean) / std

        gen_model_ctx = ModelCtx(
            image_ctx=self._build_image_ctx(ctx, mean, std, trg_n.shape[0]),
            warmth=warmth,
        )

        if self.loss_weighting:
            loss_weight = std
        else:
            loss_weight = None

        loss = self.generative_model(gen_model_ctx, trg_n, loss_weight=loss_weight)
        if self.end_to_end and self.warm_start_model is not None:
            # We want the mean to be driven by the generative loss and the
            # std to be driven by the NLL loss.
            detached_prd_dist = Normal(prd_dist.mean.detach(), prd_dist.stddev)
            loss += self.warm_start_model.nll(detached_prd_dist, trg)

        return loss

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
        return scaled_std, warmth.squeeze()

    def _build_image_ctx(
        self, ctx: ModelCtx, mean: torch.Tensor, std: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        shape = [batch_size] + list(mean.shape[1:])

        # Awkward order for backward compatibility.
        image_ctx = []
        if ctx.image_ctx is not None:
            image_ctx.append(ctx.image_ctx)
        image_ctx.append(mean.expand(shape))
        image_ctx.append(std.expand(shape))

        return torch.cat(image_ctx, dim=1)

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        """
        # For conditional generation, generate samples based on the context.
        if ctx.image_ctx is not None:
            num_samples = ctx.image_ctx.shape[0]
        if self.prd_dist is None:
            prd_dist = self.warm_start_model.predict(ctx)  # type: ignore
            prd_dist = Normal(prd_dist.mean, prd_dist.stddev)
        else:
            prd_dist = self.prd_dist

        initial_warmth = kwargs.get("warmth", self._get_sample_warmth(kwargs))

        std = prd_dist.stddev
        mean = prd_dist.mean
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
            image_ctx=self._build_image_ctx(ctx, mean, std, num_samples),
            warmth=warmth,
        )

        samples_n = self.generative_model.sample(gen_model_ctx, num_samples, **kwargs)

        if self.feature_only_ablation:
            return samples_n

        # Go back to unnormalised space
        samples = samples_n * std + prd_dist.mean

        return samples

    def _get_sample_warmth(self, kwargs) -> float:
        if "nfe" not in kwargs:
            return self.max_warmth
        nfe = kwargs["nfe"]
        if nfe <= 10:
            return self.max_warmth
        # For higher NFE, allow the generative model a bit more freedom.
        return 0.8 * (self.max_warmth - self.min_warmth)

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
