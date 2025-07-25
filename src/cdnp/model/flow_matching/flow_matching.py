from typing import Optional

import torch
from loguru import logger
from torch import nn

from cdnp.model.ctx import ModelCtx
from cdnp.model.flow_matching.path.affine import CondOTProbPath
from cdnp.model.flow_matching.solver.ode_solver import ODESolver
from cdnp.model.flow_matching.utils import ModelWrapper
from cdnp.model.meta.unet import UNetModel


# TODO, get rid of the whole CFG, not needed.
class CFGScaledModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
        ctx: ModelCtx,
    ):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if ctx.image_ctx is not None:
            x = torch.cat([x, ctx.image_ctx], dim=1)

        extra = {}
        if ctx.warmth is not None:
            extra["warmth"] = ctx.warmth
        result = self.model(x, t, extra=extra)

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class FlowMatching(nn.Module):
    def __init__(
        self,
        backbone: UNetModel,
        skewed_timesteps: bool,
        edm_schedule: bool,
        ode_method: str,
        ode_opts: dict,
        num_channels: int,
        height: int,
        width: int,
        device: str,
    ):
        super().__init__()
        self.skewed_timesteps = skewed_timesteps
        self.device = device
        self.path = CondOTProbPath()
        self.cfg_model = CFGScaledModel(backbone)
        self.solver = ODESolver(velocity_model=self.cfg_model)

        self.ode_method = ode_method
        self.ode_opts = ode_opts

        self.backbone = backbone
        self.num_channels = num_channels
        self.height = height
        self.width = width

        # TODO: Dependency inject these
        if edm_schedule:
            self.time_grid = get_time_discretization(nfes=ode_opts["nfe"])
        else:
            self.time_grid = torch.tensor([0.0, 1.0], device=device)

    def forward(
        self, ctx: ModelCtx, trg: torch.Tensor, loss_weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        noise = torch.randn_like(trg)
        batch_size = trg.shape[0]

        if self.skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device=self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        path_sample = self.path.sample(t=t, x_0=noise, x_1=trg)
        x_t = path_sample.x_t

        if ctx.image_ctx is not None:
            x_t = torch.cat([x_t, ctx.image_ctx], dim=1)
        t = torch.zeros(batch_size, device=self.device) + t

        extra = {}
        if ctx.warmth is not None:
            extra["warmth"] = ctx.warmth

        u_t = path_sample.dx_t

        if ctx.label_ctx:
            # TODO?
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        pred_u = self.backbone(x_t, t, extra=extra)
        loss = torch.pow(pred_u - u_t, 2)
        if loss_weight is not None:
            loss = loss * loss_weight
        return loss.mean()

    @torch.no_grad()
    def sample(self, ctx: ModelCtx, num_samples: int, **kwargs) -> torch.Tensor:
        """
        Generates samples from the model.

        :ctx: Context labels for the generation process.
        :num_samples: Number of samples to generate.
        """

        if ctx.image_ctx is not None:
            num_samples = ctx.image_ctx.shape[0]

        shape = (num_samples, self.num_channels, self.height, self.width)

        x_T = torch.randn(*shape).to(self.device)

        if ctx.label_ctx:
            logger.warning(
                "Conditional flow-matching generation is not yet implemented."
            )

        if "nfe" in kwargs and kwargs["nfe"] is not None:
            time_grid = get_time_discretization(kwargs["nfe"])
        else:
            time_grid = self.time_grid

        return self.solver.sample(
            time_grid=time_grid,
            x_init=x_T,
            method=self.ode_method,
            return_intermediates=False,
            atol=self.ode_opts["atol"],
            rtol=self.ode_opts["rtol"],
            step_size=self.ode_opts["step_size"],
            # TODO remove cfg, not needed
            cfg_scale=0.0,
            label=None,
            ctx=ctx,
        )

    def make_plot(self, ctx: ModelCtx, num_samples: int = 0) -> list[torch.Tensor]:
        return [self.sample(ctx, num_samples) for _ in range(3)]


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time
