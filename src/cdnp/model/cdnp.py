import torch
from diffusers import UNet2DModel
from torch import nn

from cdnp.model.cnp import CNP
from cdnp.model.ctx import ModelCtx
from cdnp.model.noise_scheduler import CDNPScheduler


class CDNP(nn.Module):
    def __init__(
        self,
        backbone: UNet2DModel,
        loss_fn: nn.Module,
        noise_scheduler: CDNPScheduler,
        cnp: CNP,
        device: str,
        # TODO: take a torch.Generator
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

        prd_dist = self.cnp.predict(ctx)

        shape = (trg.shape[0],)
        timesteps = (
            torch.randint(0, self.num_timesteps - 1, shape).long().to(self.device)
        )

        noise = torch.randn_like(trg, device=self.device)
        noisy_x = self.noise_scheduler.add_noise(
            trg, noise, timesteps, x_T_mean=prd_dist.mean, x_T_std=prd_dist.stddev
        )

        model_input = self._cat_ctx(noisy_x, ctx)

        pred_noise = self.backbone(model_input, timesteps, class_labels=labels).sample

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

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int = 0) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        :return: Generated samples of shape `size`.
        """

        # TODO: num samples: extend along batch dimension

        prd_dist = self.cnp.predict(ctx)
        labels = ctx.label_ctx

        x_t = prd_dist.sample()

        for t in self.noise_scheduler.timesteps:
            model_input = self._cat_ctx(x_t, ctx)
            prd_eps = self.backbone(model_input, t, labels).sample
            # Update sample with step

            x_t = self.noise_scheduler.step(
                prd_eps, t, x_t, x_T_mean=prd_dist.mean, x_T_std=prd_dist.stddev
            ).prev_sample
        return x_t

    @torch.no_grad()
    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        cnp_dist = self.cnp.predict(ctx)
        mean = cnp_dist.mean
        std = cnp_dist.stddev
        plots = [mean, std]
        plots.append(torch.zeros_like(mean))
        plots.append(torch.ones_like(mean))

        x = cnp_dist.sample()

        for t in self.noise_scheduler.timesteps:
            plots.append(x)
            model_input = self._cat_ctx(x, ctx)
            prd_noise = self.backbone(model_input, t, ctx.label_ctx).sample

            out = self.noise_scheduler.step(prd_noise, t, x, x_T_mean=mean, x_T_std=std)

            x = out.prev_sample
        plots.append(x)

        return plots
