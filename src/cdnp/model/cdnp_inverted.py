import torch
from diffusers import DDPMScheduler, UNet2DModel
from loguru import logger
from torch import nn
from torch.distributions import Normal

from cdnp.model.cnp import CNP
from cdnp.model.ctx import ModelCtx
from cdnp.model.util import padded_forward


class CDNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        cnp: CNP,
        device: str,
        initial_std_mult: float = 1.0,
        # TODO: take a torch.Generator
    ):
        super().__init__()
        self.backbone = backbone
        self.noise_scheduler = noise_scheduler
        # TODO: This should support all kinds of context data.
        # self.context_embedding = context_embedding
        self.device = device
        self.num_timesteps = noise_scheduler.config.num_train_timesteps  # ty: ignore
        self.cnp = cnp
        self.initial_std_mult = initial_std_mult
        self.std_mult = initial_std_mult

    def forward(self, ctx: ModelCtx, trg: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise prediction loss for a batch of data.
        """
        labels = ctx.label_ctx

        prd_dist = self.cnp.predict(ctx)
        prd_dist = Normal(prd_dist.mean, prd_dist.stddev * self.std_mult)

        shape = (trg.shape[0],)
        timesteps = (
            torch.randint(0, self.num_timesteps - 1, shape).long().to(self.device)
        )

        trg_tilde = (trg - prd_dist.mean) / prd_dist.stddev

        noise = torch.randn_like(trg_tilde, device=self.device)
        noisy_x = self.noise_scheduler.add_noise(trg_tilde, noise, timesteps)

        model_input = self._cat_ctx(noisy_x, ctx)
        model_input = torch.cat([model_input, prd_dist.mean, prd_dist.stddev], dim=1)

        pred_noise = padded_forward(
            self.backbone, model_input, timesteps, class_labels=labels
        )

        return self.loss_fn(pred_noise, noise)

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

    def loss_fn(self, pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred_noise, noise, reduction="mean")

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        :return: Generated samples of shape `size`.
        """

        # TODO: num samples: extend along batch dimension
        cnp_dist = self.cnp.predict(ctx)
        cnp_dist = Normal(cnp_dist.mean, cnp_dist.stddev * self.std_mult)
        mean = cnp_dist.mean
        std = cnp_dist.stddev

        x_tilde = torch.randn_like(mean, device=self.device)

        for t in self.noise_scheduler.timesteps:
            model_input = self._cat_ctx(x_tilde, ctx)

            model_input = torch.cat([model_input, mean, std], dim=1)

            prd_noise = padded_forward(
                self.backbone, model_input, t, class_labels=ctx.label_ctx
            )

            out = self.noise_scheduler.step(prd_noise, t, x_tilde)

            x_tilde = out.prev_sample

        # Go back to unnormalised space
        return x_tilde * std + mean

    @torch.no_grad()
    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        cnp_dist = self.cnp.predict(ctx)
        cnp_dist = Normal(cnp_dist.mean, cnp_dist.stddev * self.std_mult)
        mean = cnp_dist.mean
        std = cnp_dist.stddev
        plots = [mean, std]

        x_tilde = torch.randn_like(mean, device=self.device)

        for i, t in enumerate(self.noise_scheduler.timesteps):
            plots.append(x_tilde)
            plots.append(x_tilde * std + mean)
            model_input = self._cat_ctx(x_tilde, ctx)

            model_input = torch.cat([model_input, mean, std], dim=1)

            prd_noise = padded_forward(
                self.backbone, model_input, t, class_labels=ctx.label_ctx
            )

            out = self.noise_scheduler.step(prd_noise, t, x_tilde)
            x_tilde = out.prev_sample

        plots.append(x_tilde)
        plots.append(x_tilde * std + mean)

        return plots

    def set_steps(self, steps: int) -> None:
        stage = steps / 500
        # This helps the model learn at the start by making the noise more obvious.
        self.std_mult = max(self.initial_std_mult / 2**stage, 1.0)
        logger.info("Setting std_mult to {}", self.std_mult)
