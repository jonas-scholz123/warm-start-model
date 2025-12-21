import torch
from diffusers import UNet2DModel
from torch import nn
from torch.distributions import Normal

from cdnp.model.ctx import ModelCtx
from cdnp.model.util import padded_forward


class CNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        device: str,
        min_std: float,
        residual: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.min_std = min_std
        self.residual = residual

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.

        :x: The input data (e.g., images).
        :ctx: The context data (e.g., class labels).
        """

        prd_dist = self.predict(ctx)
        return self.nll(prd_dist, trg)

    def predict(self, ctx: ModelCtx) -> Normal:
        im_ctx = ctx.image_ctx  # (B, C, H, W)
        labels = ctx.label_ctx

        assert im_ctx is not None, "Image context must be provided for CNP."

        # TODO: this is a hack - we don't need timesteps for CNP
        shape = (im_ctx.shape[0],)
        timesteps = torch.zeros(shape).long().to(self.device)

        pred = padded_forward(self.backbone, im_ctx, timesteps, class_labels=labels)

        mean, std = pred.chunk(2, dim=1)
        if self.residual:
            num_trg_channels = mean.shape[1]
            # By convention, the last channels of the image context should be the
            # residuals.
            res = im_ctx[:, -num_trg_channels:, :, :]
            mean = res + mean
        std = nn.functional.softplus(std)
        std = torch.clamp(std, min=self.min_std)
        return Normal(mean, std)

    def nll(self, prd_dist: Normal, trg: torch.Tensor) -> torch.Tensor:
        return -prd_dist.log_prob(trg).mean()

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
        prd_dist = self.predict(ctx)
        return prd_dist.sample()

    def sample_with_grad(self, ctx: ModelCtx) -> torch.Tensor:
        prd_dist = self.predict(ctx)
        return prd_dist.sample()

    def make_plot(self, ctx: ModelCtx) -> list[torch.Tensor]:
        pred = self.predict(ctx)
        masked_image = ctx.image_ctx[:, -3:, :, :]  # type: ignore
        mask = ctx.image_ctx[:, :1, :, :].repeat(1, 3, 1, 1)  # type: ignore
        return [mask, masked_image, pred.mean, pred.stddev, pred.sample()]
