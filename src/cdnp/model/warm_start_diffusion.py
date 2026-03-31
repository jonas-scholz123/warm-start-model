from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from cdnp.model.cnp import CNP, CNPPrediction
from cdnp.model.ctx import ModelCtx
from cdnp.model.ddpm import DDPM
from cdnp.model.flow_matching.flow_matching import FlowMatching
from cdnp.model.low_rank_cov import normalize as lr_normalize
from cdnp.model.low_rank_cov import unnormalize as lr_unnormalize


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
        end_to_end_nll_weight: float = 0.01,
        norm_param_path: str | None = None,
        correlation_rank: int = 0,
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
        self.nll_weight = end_to_end_nll_weight
        self.correlation_rank = correlation_rank

        if norm_param_path is None:
            self.prd_dist = None
        else:
            if correlation_rank > 0:
                raise ValueError(
                    "Low-rank correlation is only supported with a warm-start model, "
                    "not with static norm_param_path."
                )
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

    def _predict(self, ctx: ModelCtx) -> CNPPrediction:
        """Get the prediction (from warm-start model or static params)."""
        if self.prd_dist is None:
            return self.warm_start_model.predict(ctx)  # type: ignore
        else:
            # Static norm params: wrap in CNPPrediction (no V factors).
            return CNPPrediction(
                mean=self.prd_dist.mean,
                std=self.prd_dist.stddev,
                V=None,
            )

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        prd = self._predict(ctx)
        mean = prd.mean.detach()
        std = prd.std.detach()
        V = prd.V.detach() if prd.V is not None else None

        std, V, warmth = self._apply_warmth(std, V)

        if self.mean_only_ablation:
            std = torch.ones_like(std, device=self.device)
            V = None
            warmth = None

        if self.feature_only_ablation:
            trg_n = trg
        else:
            # _n suffix = normalised space
            trg_n = self._normalize(trg, mean, std, V)

        gen_model_ctx = ModelCtx(
            image_ctx=self._build_image_ctx(ctx, mean, std, trg_n.shape[0]),
            warmth=warmth,
        )
        # TODO: investigate passing V factors as additional context channels
        # to the generative model.

        if self.loss_weighting:
            loss_weight = std
        else:
            loss_weight = None

        loss = self.generative_model(gen_model_ctx, trg_n, loss_weight=loss_weight)
        if self.end_to_end and self.warm_start_model is not None:
            loss += self.nll_weight * self.warm_start_model.nll(prd, trg)

        return loss

    def _normalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        V: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize data using diagonal std and optional low-rank factors."""
        if V is not None and self.correlation_rank > 0:
            return lr_normalize(x, mean, std, V)
        else:
            return (x - mean) / std

    def _unnormalize(
        self,
        z: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        V: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Unnormalize data using diagonal std and optional low-rank factors."""
        if V is not None and self.correlation_rank > 0:
            return lr_unnormalize(z, mean, std, V)
        else:
            return z * std + mean

    def _apply_warmth(
        self,
        prd_std: torch.Tensor,
        V: Optional[torch.Tensor],
        warmth: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply warmth scaling to std and V.

        Warmth interpolates between the predicted normalisation and
        the identity (standard normal prior):
          d_warm = warmth * d + (1-warmth) * 1
          V_warm = warmth * V  (vanishes at warmth=0)
        """
        prd_std = prd_std.clamp(min=self.min_std)
        if not self.scale_warmth:
            return prd_std, V, None

        batch_size = prd_std.shape[0]
        base_std = torch.ones_like(prd_std, device=self.device)
        if warmth is None:
            warmth = (
                torch.rand(batch_size, device=self.device)
                * (self.max_warmth - self.min_warmth)
                + self.min_warmth
            )
        warmth_expanded = warmth[:, None, None, None]

        scaled_std = warmth_expanded * prd_std + (1 - warmth_expanded) * base_std

        if V is not None:
            # Scale V: vanishes at warmth=0 (standard normal prior).
            V = warmth_expanded.unsqueeze(1) * V  # (B,1,1,1,1) * (B,rank,C,H,W)

        return scaled_std, V, warmth.squeeze()

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

        prd = self._predict(ctx)

        initial_warmth = kwargs.get("warmth", self._get_sample_warmth(kwargs))

        std = prd.std
        mean = prd.mean
        V = prd.V

        if self.scale_warmth:
            # During sampling, for now, we use a constant (full) warmth.
            # TODO: Experiment with different warmth schedules.
            warmth = torch.ones(num_samples, device=self.device) * initial_warmth
            std, V, _warmth = self._apply_warmth(std, V, warmth)
        else:
            std = std.clamp(min=self.min_std)

        if self.mean_only_ablation:
            std = torch.ones_like(std, device=self.device)
            V = None

        gen_model_ctx = ModelCtx(
            image_ctx=self._build_image_ctx(ctx, mean, std, num_samples),
            warmth=_warmth if self.scale_warmth else None,
        )

        samples_n = self.generative_model.sample(gen_model_ctx, num_samples, **kwargs)

        if self.feature_only_ablation:
            return samples_n

        # Go back to unnormalised space
        samples = self._unnormalize(samples_n, mean, std, V)

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
