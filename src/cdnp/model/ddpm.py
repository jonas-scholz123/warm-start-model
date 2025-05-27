import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn

from cdnp.model.ctx import ModelCtx


class DDPM(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        loss_fn: nn.Module,
        noise_scheduler: DDPMScheduler,
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

        self.sidelength = backbone.sample_size  # ty: ignore
        self.in_channels = backbone.config.in_channels  # ty: ignore
        self.out_channels = backbone.config.out_channels  # ty: ignore

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.

        :x: The input data (e.g., images).
        :ctx: The context data (e.g., class labels).
        """
        labels = ctx.label_ctx

        noise = torch.randn_like(trg)
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
    def sample(self, num_samples: int, ctx: ModelCtx) -> torch.Tensor:
        """
        Generates samples from the model.

        :num_samples: Number of samples to generate.
        :ctx: Context labels for the generation process.
        :return: Generated samples of shape `size`.
        """

        shape = (num_samples, self.out_channels, self.sidelength, self.sidelength)

        x = torch.randn(*shape).to(self.device)
        labels = ctx.label_ctx

        for t in self.noise_scheduler.timesteps:
            model_input = self._cat_ctx(x, ctx)
            residual = self.backbone(model_input, t, labels).sample
            # Update sample with step
            x = self.noise_scheduler.step(residual, t, x).prev_sample
        return x
