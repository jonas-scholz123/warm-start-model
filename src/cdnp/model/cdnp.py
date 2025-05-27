import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn

from cdnp.model.cnp import CNP
from cdnp.model.ctx import ModelCtx


class CDNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        loss_fn: nn.Module,
        noise_scheduler: DDPMScheduler,
        cnp: CNP,
        device: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler
        # TODO: This should support all kinds of context data.
        # self.context_embedding = context_embedding
        self.device = device
        self.num_timesteps = noise_scheduler.config.num_train_timesteps  # ty: ignore
        self.cnp = cnp

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.
        """
        labels = ctx.label_ctx

        cnp_sample = self.cnp.sample_with_grad(ctx)

        noise = cnp_sample - trg

        # TODO Refactor to use DDPM class
        shape = (trg.shape[0],)
        timesteps = (
            torch.randint(0, self.num_timesteps - 1, shape).long().to(self.device)
        )
        noisy_x = self.noise_scheduler.add_noise(trg, noise, timesteps)
        model_input = self._cat_ctx(noisy_x, ctx)

        pred = self.backbone(model_input, timesteps, class_labels=labels).sample

        return self.loss_fn(pred, noise)

    def _cat_ctx(self, x: torch.Tensor, ctx: ModelCtx) -> torch.Tensor:
        """
        Concatenates the context data to the input tensor along the channel dimension.

        :x: Noisy input tensor.
        :ctx: Context data.
        :return: Concatenated tensor.
        """
        if ctx.image_ctx is not None:
            return torch.cat([x, ctx.image_ctx], dim=1)
        return x

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        :return: Generated samples of shape `size`.
        """

        # TODO: num samples: extend along batch dimension

        x = self.cnp.sample(ctx)
        labels = ctx.label_ctx

        for t in self.noise_scheduler.timesteps:
            model_input = self._cat_ctx(x, ctx)
            residual = self.backbone(model_input, t, labels).sample
            # Update sample with step
            x = self.noise_scheduler.step(residual, t, x).prev_sample
        return x
