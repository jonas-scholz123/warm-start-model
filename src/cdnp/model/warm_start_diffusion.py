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
        std_mult: float = 1.0,
    ):
        super().__init__()
        self.warm_start_model = warm_start_model
        self.generative_model = generative_model
        self.std_mult = std_mult

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        prd_dist = self.warm_start_model.predict(ctx)
        prd_dist = Normal(prd_dist.mean, prd_dist.stddev * self.std_mult)

        # _n suffix = normalised space
        trg_n = (trg - prd_dist.mean) / prd_dist.stddev

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, prd_dist.mean, prd_dist.stddev], dim=1),
        )

        return self.generative_model(gen_model_ctx, trg_n)

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        """
        # For conditional generation, generate samples based on the context.
        num_samples = ctx.image_ctx.shape[0]
        prd_dist = self.warm_start_model.predict(ctx)
        prd_dist = Normal(prd_dist.mean, prd_dist.stddev * self.std_mult)

        gen_model_ctx = ModelCtx(
            image_ctx=torch.cat([ctx.image_ctx, prd_dist.mean, prd_dist.stddev], dim=1),
        )

        samples_n = self.generative_model.sample(gen_model_ctx, num_samples)

        # Go back to unnormalised space
        samples = samples_n * prd_dist.stddev + prd_dist.mean

        return samples

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(4)]
