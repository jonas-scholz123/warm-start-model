import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn


class DDPM(nn.Module):
    def __init__(
        self,
        model: UNet2DModel,
        loss_fn: nn.Module,
        noise_scheduler: DDPMScheduler,
        device: str,
    ):
        super(DDPM, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler
        # TODO: This should support all kinds of context data.
        # self.context_embedding = context_embedding
        self.device = device
        self.num_timesteps = noise_scheduler.config.num_train_timesteps

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.

        :param x: The input data (e.g., images).
        :param ctx: The context labels (e.g., class labels).
        """
        noise = torch.randn_like(x)
        shape = (x.shape[0],)
        timesteps = (
            torch.randint(0, self.num_timesteps - 1, shape).long().to(self.device)
        )
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        pred = self.model(noisy_x, timesteps, class_labels=ctx).sample

        return self.loss_fn(pred, noise)

    @torch.no_grad()
    def sample(
        self,
        size: tuple,
        ctx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generates samples from the model.

        :param size: Size of the generated samples, should have shape
            (num_samples, channels, height, width).
        :param ctx: Context labels for the generation process.
        :return: Generated samples of shape `size`.
        """

        x = torch.randn(*size).to(self.device)

        for i, t in enumerate(self.noise_scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                residual = self.model(
                    x, t, ctx
                ).sample  # Again, note that we pass in our labels y

            # Update sample with step
            x = self.noise_scheduler.step(residual, t, x).prev_sample
        return x
