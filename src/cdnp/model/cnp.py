import torch
from diffusers import UNet2DModel
from torch import nn
from torch.distributions import Normal

from cdnp.model.ctx import ModelCtx


class CNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        device: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.

        :x: The input data (e.g., images).
        :ctx: The context data (e.g., class labels).
        """

        prd_dist = self.predict(ctx)
        return self.nll(prd_dist, trg)

    def predict(self, ctx: ModelCtx) -> Normal:
        im_ctx = ctx.image_ctx
        labels = ctx.label_ctx

        assert im_ctx is not None, "Image context must be provided for CNP."

        # TODO: this is a hack - we don't need timesteps for CNP
        shape = (im_ctx.shape[0],)
        timesteps = torch.zeros(shape).long().to(self.device)

        pred = self.backbone(im_ctx, timesteps, class_labels=labels).sample

        mean, std = pred.chunk(2, dim=1)
        std = nn.functional.softplus(std)
        return Normal(mean, std)

    def nll(self, prd_dist: Normal, trg: torch.Tensor) -> torch.Tensor:
        return -prd_dist.log_prob(trg).mean()

    @torch.no_grad()
    def sample(self, num_samples: int, ctx: ModelCtx) -> torch.Tensor:
        """
        Generates samples from the model.

        :num_samples: (ignored)
        :ctx: Context labels for the generation process. Shape:
            (num_samples, in_channels, sidelength, sidelength)
        :return: Generated samples of shape
            (num_samples, out_channels, sidelength, sidelength).
        """
        prd_dist = self.predict(ctx)
        return prd_dist.sample()
